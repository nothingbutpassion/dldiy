import numpy as np
import matplotlib.pyplot as plt
import datasets.widerface as widerface
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import layers
import optimizers
import models

def IOU(bx, by, bw, bh, tx, ty, tw, th):
    bx1, bx2 = bx - bw/2, bx + bw/2
    by1, by2 = by - bh/2, by + bh/2
    tx1, tx2 = tx - tw/2, tx + tw/2
    ty1, ty2 = ty - th/2, ty + th/2
    max_x1 = np.maximum(bx1, tx1)
    max_y1 = np.maximum(by1, ty1)
    min_x2 = np.minimum(bx2, tx2)
    min_y2 = np.minimum(by2, ty2)
    intersection = (min_x2 - max_x1)*(min_y2 - max_y1)
    union = bw*bw + tw*th - intersection
    return intersection/union
    
def detection_loss(y_true, y_pred):
    b1, b2, bx, by, bw, bh = y_pred
    t1, t2, tx, ty, tw, th = y_true
    max_b = np.maximum(b1, b2)
    exp_b1 = np.exp(b1 - max_b)
    exp_b2 = np.exp(b2 - max_b)
    exp_s = exp_b1 + exp_b2
    p1 = exp_b1/exp_s
    p2 = 1 - p1
    object_loss = -np.sum(t1*np.log(p1) + t2*np.log(p2))
    box_loss = np.sum((bx-tx)**2 + (by-ty)**2 + (bw-tw)**2 + (bh-th)**2)
    return object_loss + box_loss

def detection_accuracy(y_true, y_pred):
    b1, b2, bx, by, bw, bh = y_pred
    t1, t2, tx, ty, tw, th = y_true
    max_b = np.maximum(b1, b2)
    exp_b1 = np.exp(b1 - max_b)
    exp_b2 = np.exp(b2 - max_b)
    exp_s = exp_b1 + exp_b2
    p1 = exp_b1/exp_s
    m = t1 > 0
    iou = IOU(bx[m], by[m], bw[m], bh[m], tx[m], ty[m], tw[m], th[m])
    iou = np.average(iou)
    rt = np.sum(np.abs(p1 - t1))/t1.size
    return rt*iou

class DetectionLoss:
    def loss(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        batch_loss = [detection_loss(y_true[i], y_pred[i]) for i in range(batch_size)]
        return np.average(batch_loss)
    
    def accuracy(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        batch_accuracy = [detection_accuracy(y_true[i], y_pred[i]) for i in range(batch_size)]
        return np.average(batch_accuracy)

    def grad(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        batch_grad = np.zeros_like(y_pred)
        for i in range(batch_size):
            b1, b2, bx, by, bw, bh = y_pred[i]
            t1, t2, tx, ty, tw, th = y_true[i]
            max_b = np.maximum(b1, b2)
            exp_b1 = np.exp(b1 - max_b)
            exp_b2 = np.exp(b2 - max_b)
            exp_s = exp_b1 + exp_b2
            batch_grad[i] = [
                (t2*exp_b1 - t1*exp_b2)/exp_s,
                (t1*exp_b2 - t2*exp_b1)/exp_s,
                bx-tx, by-ty, bw-tw, bh-th
                ]
        return batch_grad
    
class DataIterator:
    def __init__(self, data, output_size, feature_shape, batch_size):
        self.data = data
        self.output_size = output_size
        self.feature_shape = feature_shape
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.data)//self.batch_size
    
    def __getitem__(self, batch_index):
        if (self.batch_size+1)*batch_index > len(self.data):
            raise IndexError()

        batch_data = self.data[self.batch_size*batch_index : self.batch_size*(batch_index + 1)]
        batch_x = np.zeros((self.batch_size, 3, self.output_size[0], self.output_size[1]), dtype='uint8')
        batch_y = np.zeros(((self.batch_size,) + self.feature_shape), dtype='float')
        for i, sample in enumerate(batch_data):
            image = Image.open(sample["image"])
            x_rate, y_rate = self.output_size[0]/image.size[0], self.output_size[1]/image.size[1]
            image = image.resize(self.output_size, Image.BILINEAR)
            image = np.asarray(image)
            image = image.transpose((2, 0, 1))
            batch_x[i, :, :, :] = image
            boxes = np.array(sample["boxes"])
            for box in boxes:
                box[0] = box[0]*x_rate
                box[2] = box[2]*x_rate
                box[1] = box[1]*y_rate
                box[3] = box[3]*y_rate
            batch_y[i] = encode(self.output_size, boxes, self.feature_shape)
        return batch_x, batch_y

def encode(image_size, boxes, feature_shape=(6,7,7)):
    result = np.zeros(feature_shape)
    result[1,:,:] += 1
    iw, ih = image_size
    oc, oh, ow = feature_shape
    sh, sw = oh/ih, ow/iw
    for box in boxes:
        x, y, w, h = box
        cx = sw*(x+w/2)
        cy = sh*(y+h/2)
        i = int(cx)
        j = int(cy)    
        bx = cx - i
        by = cy - j
        bw = sw*w
        bh = sh*h
        if result[0,j,i] > 0:
            bw0, bh0 = result[4:,j,i]
            if (bw*bh > bw0*bh0):
                result[:,j,i]=(1, 0, bx, by, bw, bh)
        else:
            result[:,j,i]=(1, 0, bx, by, bw, bh)
    return result

def decode(image_size, feature, threshold=1.0):
    boxes = []
    iw, ih = image_size
    oc, oh, ow = feature.shape
    sh, sw = ih/oh, iw/ow
    for j in range(oh):
        for i in range(ow):
            po, pn, bx, by, bw, bh = feature[:,j,i]
            if (po >= threshold):
                w = sw*bw
                h = sh*bh
                x = sw*(i + bx) - w/2
                y = sh*(j + by) - h/2
                boxes.append([x, y, w, h])
    return boxes

def test_data():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    for batch_x, batch_y in DataIterator(train_data, (256, 256), (6,7,7), 4):
        for i in range(4):
            image = batch_x[i].transpose((1,2,0))
            boxes = decode((256,256), batch_y[i])
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title("Sample #{}".format(i))
            ax.axis('off')
            ax.imshow(image)
            for bbox in boxes:
                (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()
        break

def test_codec():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    sample = train_data[2]
    image = Image.open(sample["image"])
    boxes = sample["boxes"]
    feature = encode(image.size, boxes)
    boxes=decode(image.size, feature)
    ax = plt.subplot(1, 1, 1)
    ax.imshow(image)
    for bbox in boxes:
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def test_network():
    modle = models.Sequential()
    modle.add(layers.Conv2D(16, (7, 7), stride=4, pad=0, input_shape=(None, 3, 128, 128))) 
    modle.add(layers.ReLU())
    modle.add(layers.MaxPooling2D((3, 3), stride=2))
    modle.add(layers.Conv2D(32, (5, 5), stride=2, pad=0)) 
    modle.add(layers.ReLU())
    modle.add(layers.MaxPooling2D((2, 2), stride=2))
    modle.add(layers.Conv2D(6, (1, 1), stride=1, pad=0)) 
    # modle.compile(DetectionLoss(), optimizers.SGD(lr=0.001))
    modle.compile(DetectionLoss(), optimizers.Momentum(lr=0.001))
    modle.summary()
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    epochs = 16
    for i in range(epochs):
        for batch_x, batch_y in DataIterator(train_data, (128, 128), (6,3,3), 16):
            modle.train_one_batch(batch_x, batch_y)
            result = modle.evaluate(batch_x, batch_y)
            print("Epoch %d %s" % (i+1, result))
    
    for x_true, y_true in DataIterator(train_data, (128, 128), (6,3,3), 1):
        y_pred = modle.predict(x_true)[0]
        b1, b2 = y_pred[:2,:,:]
        exp_b1 = np.exp(b1)
        exp_b2 = np.exp(b2)
        exp_s = exp_b1 + exp_b2
        y_pred[:2,:,:] = (exp_b1/exp_s, exp_b2/exp_s)
        boxes = decode((128, 128), y_pred, 0.5)
        if len(boxes) > 0:
            ax = plt.subplot(1, 1, 1)
            ax.imshow(x_true[0].transpose((1,2,0)))
            for bbox in boxes:
                (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()
            return

    



if __name__ == "__main__":
    test_network()