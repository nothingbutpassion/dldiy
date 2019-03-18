import numpy as np
import matplotlib.pyplot as plt
import datasets.widerface as widerface
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import layers
import optimizers
import models

def f1_score(y_true, y_pred):
    p = 1/(1 + np.exp(-y_pred[:,0,:,:]))
    tp = y_true[:,0,:,:]
    # NOTES:
    # Precision = TP/(TP + FP)
    # Recall = TP/(TP + FN)
    # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    P = p > 0.5
    F = p <= 0.5
    TP = (tp > 0.5)*P
    FN = (tp > 0.5)*F
    Precision = float(np.sum(TP))/np.sum(P)
    Recall = float(np.sum(TP))/(np.sum(TP) + np.sum(FN))
    F1 = 2*(Recall*Precision)/(Recall + Precision)
    return F1

class DetectLoss:
    def loss(self, y_true, y_pred):
        # y shape: (N, C, H, W)
        p, x, y, w, h = [y_pred[:,i,:,:] for i in range(5)]
        tp, tx, ty, tw, th = [y_true[:,i,:,:] for i in range(5)]
        obj_loss = - tp*np.log(p) - (1-tp)*np.log(1-p)
        loc_loss = np.square(tx-x) + np.square(ty-y) + np.square(tw-w) + np.square(th-h)
        m = tp > 0
        loc_loss *= m
        return np.mean(obj_loss) + np.mean(loc_loss)

    def grad(self, y_true, y_pred):
        p, x, y, w, h = [y_pred[:,i,:,:] for i in range(5)]
        tp, tx, ty, tw, th = [y_true[:,i,:,:] for i in range(5)]
        batch_grad = np.zeros_like(y_pred)
        m = tp > 0
        epsilon = 1e-7
        batch_grad[:,0,:,:] = -tp/(p + epsilon) + (1-tp)/(1-p + epsilon)
        batch_grad[:,1,:,:] = (x - tx)*m
        batch_grad[:,2,:,:] = (y - ty)*m
        batch_grad[:,3,:,:] = (w - tw)*m
        batch_grad[:,4,:,:] = (h - th)*m
        return batch_grad

def is_crop_valid(boxes, rect):
    d = boxes - rect
    xy_valid = (d[:,0] > 0)*(d[:,1] > 0)
    wh_valid = (d[:,0] + d[:,2] < 0)*(d[:,1] + d[:,3] < 0)
    return  np.sum(xy_valid*wh_valid)

def crop(image, boxes, rect):
    image = image.crop(rect)
    boxes[:,:2] -= rect[:2]
    return image, boxes

def resize(image, boxes, size):
    sx = float(size[0])/image.size[0] 
    sy = float(size[1])/image.size[1] 
    image = image.resize(size, Image.LINEAR)
    boxes[:,:2] *= (sx, sy)
    boxes[:,2:] *= (sx, sy)
    return image, boxes

def transform(image, boxes, output_size):
    iw, ih = image.size
    ow, oh = output_size
    if iw < ow or ih < oh:
        return resize(image, boxes, output_size)
    # find the biggest box
    area = np.array([b[2]*b[3] for b in boxes])
    x, y, w, h = boxes[np.argmax(area)]
    x1 = max(x+w/2-ow/2, 0)
    y1 = max(y+h/2-oh/2, 0)
    x2 = min(x+w/2+ow/2, iw)
    y2 = min(y+h/2+oh/2, ih)
    selected = np.array([b for b in boxes if b[0] > x1 and b[1] > y1 and b[0]+b[2] < x2 and b[1]+b[3] < y2])
    if len(selected) > 0:
        image, boxes = crop(image, selected, [x1,y1,x2,y2])
        return resize(image, boxes, output_size)

    if x1 == 0:
        x2 = ow
    if x2 == iw:
        x1 = iw - ow
    if y1 == 0:
        y2 = oh
    if y2 == ih:
        y1 = ih - oh 
    selected = np.array([b for b in boxes if b[0] > x1 and b[1] > y1 and b[0]+b[2] < x2 and b[1]+b[3] < y2])
    if len(selected) > 0:
        image, boxes = crop(image, selected, [x1,y1,x2,y2])
        return resize(image, boxes, output_size)

    s = max(w, h)
    x1 = max(x+w/2-3*s, 0)
    y1 = max(y+h/2-3*s, 0)
    x2 = min(x+w/2+3*s, iw)
    y2 = min(y+h/2+3*s, ih)
    selected = np.array([b for b in boxes if b[0] >= x1 and b[1] >= y1 and b[0]+b[2] <= x2 and b[1]+b[3] <= y2])
    image, boxes = crop(image, selected, [x1,y1,x2,y2])
    return resize(image, boxes, output_size)

class DataIterator:
    def __init__(self, data, image_size, feature_shape, batch_size):
        self.data = data
        self.image_size = image_size
        self.feature_shape = feature_shape
        self.batch_size = batch_size
    
    def __len__(self):
        steps = len(self.data)//self.batch_size
        if len(self.data) % self.batch_size > 0:
            steps += 1
        return steps
    
    def __getitem__(self, batch_index):
        start = self.batch_size*batch_index
        if start >= len(self.data):
            raise IndexError()
        end = min(start + self.batch_size, len(self.data))
        batch_data = self.data[start : end]
        # NOTES:
        # image_size: W, H
        # batch_x shape: N, 3, H, W
        # batch_y shape: N, C, H, W
        batch_x = np.zeros((self.batch_size, 3, self.image_size[1], self.image_size[0]))
        batch_y = np.zeros(((self.batch_size,) + self.feature_shape))
        for i, sample in enumerate(batch_data):
            image = Image.open(sample["image"])
            scale = np.array(self.image_size[:2], dtype='float')/np.array(image.size[:2])
            image = image.resize(self.image_size, Image.BICUBIC)   # image_size: W, H
            image = np.asarray(image)                               # numpy array shape: H, W, C
            image = image.transpose((2, 0, 1))                      # after transposed: C, H, W
            batch_x[i, :, :, :] = (image - 127.5)/255
            boxes = np.array(sample["boxes"])
            for box in boxes:
                box[:2] *= scale
                box[2:] *= scale
            batch_y[i] = encode(self.image_size, boxes, self.feature_shape)
        return batch_x, batch_y


def iou(oh, ow, box1, box2 = [0.5, 0.5, 1.0, 1.0]):
    # NOTES:
    # x, y is based on grid
    # w, h is based on image, need adjusted to align grid
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w1, h1 = ow*w1, oh*h1
    x11, x12, y11, y12  = x1-w1/2, x1+w1/2, y1-h1/2, y1+h1/2
    x21, x22, y21, y22  = x2-w2/2, x2+w2/2, y2-h2/2, y2+h2/2
    max_x, max_y = max(x11, x21), min(y11, y21)
    min_x, min_y = max(x12, x22), min(y12, y22)
    assert(0 <= x1 and x1 < 1 and 0 <= y1 and y1 < 1 and w1 > 0 and h1 > 0)
    assert(min_x > max_x and min_y > max_y)
    I = (min_x - max_x)*(min_y - max_y)
    U = w1*h1 + w2*h2 - I
    return I/U

# NOTES:
# image_size    = (width, height)
# feature_shape = (channel, height, width)
def encode(image_size, boxes, feature_shape):
    result = np.zeros(feature_shape)
    iw, ih = image_size
    oc, oh, ow = feature_shape
    sh, sw = float(oh)/ih, float(ow)/iw
    for box in boxes:
        x, y, w, h = box
        cx, cy = sw*(x+w/2), sh*(y+h/2)
        i, j = int(cx), int(cy)    
        bx, by = cx-i, cy-j
        bw, bh = float(w)/iw, float(h)/ih
        if result[0,j,i] > 0 and iou(oh,ow,[bx, by, bw, bh]) > iou(oh,ow,result[1:,j,i]):
            # NOTES: select the bunding box that has biggest IOU with the grid
            result[:,j,i]=(1, bx, by, bw, bh)
        else:
            result[:,j,i]=(1, bx, by, bw, bh)
    return result

def decode(image_size, feature, threshold=1.0):
    boxes = []
    iw, ih = image_size
    oc, oh, ow = feature.shape
    sh, sw = ih/oh, iw/ow
    for j in range(oh):
        for i in range(ow):
            p, bx, by, bw, bh = feature[:,j,i]
            if (p >= threshold):
                w = iw*bw
                h = ih*bh
                x = sw*(i + bx) - w/2
                y = sh*(j + by) - h/2
                boxes.append([x, y, w, h])
    return boxes

def show_images(batch_x, batch_y, image_size):
    batch_size = batch_x.shape[0]
    for i in range(batch_size):
        image = batch_x[i].transpose((1,2,0))
        ax = plt.subplot(1, batch_size, i + 1)
        plt.tight_layout()
        ax.set_title("Sample #{}".format(i))
        ax.axis('off')
        ax.imshow((image*255 + 127.5).astype('uint8'))
        boxes = decode(image_size, batch_y[i])
        for box in boxes:
            (x, y, w, h) = box[:4]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()

def test_data():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    image_size = (128, 128)
    feature_shape = (5, 7, 7)
    batch_size = 4
    for batch_x, batch_y in DataIterator(train_data, image_size, feature_shape, batch_size):
        show_images(batch_x, batch_y, image_size)
        break

def draw_grids(image, grid_shape):
    d = ImageDraw.Draw(image)
    gw, gh = grid_shape
    w, h = image.size[0], image.size[1]
    rw, rh = w/gw, h/gh
    for j in range(gh):
        d.line([1, j*rh, w-1, j*rh], fill=(255,0,0), width=2)
    for i in range(gw):
        d.line([i*rw, 1, i*rw, h-1], fill=(255,0,0), width=2)

def draw_boxes(image, boxes):
    d = ImageDraw.Draw(image)
    for box in boxes:
        (x, y, w, h) = box[:4]
        d.rectangle([x,y,x+w,y+h], outline=(255,0,0))
        d.rectangle([x+1,y+1,x+w-1,y+h-1], outline=(255,0,0))

def test_codec():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    sample = train_data[2]
    image = Image.open(sample["image"])
    boxes = sample["boxes"]
    feature = encode(image.size, boxes, (5,7,7))
    print(feature)
    boxes=decode(image.size, feature, 1.0)
    draw_grids(image, (7,7))
    draw_boxes(image, boxes)
    plt.imshow(image)
    plt.show()

def test_transform():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", illumination="0", occlusion="0", invalid="0", min_size=16)
    for sample in train_data:
        image = Image.open(sample["image"])
        boxes = np.array(sample["boxes"])
        image, boxes = transform(image, boxes, (128, 128))
        # draw_grids(image, (7,7))
        draw_boxes(image, boxes)
        plt.imshow(image)
        plt.show(block=True)

def test_model():
    modle = models.Sequential()
    modle.add(layers.Conv2D(16, (3, 3), stride=1, pad=1, input_shape=(None, 3, 128, 128))) 
    modle.add(layers.ReLU())
    modle.add(layers.MaxPooling2D((2, 2), stride=2))

    modle.add(layers.Conv2D(32, (3, 3), stride=1)) 
    modle.add(layers.ReLU())
    modle.add(layers.MaxPooling2D((2, 2), stride=2))

    modle.add(layers.Conv2D(48, (3, 3), stride=1)) 
    modle.add(layers.ReLU())
    modle.add(layers.MaxPooling2D((2, 2), stride=2))

    modle.add(layers.Conv2D(64, (3, 3), stride=1)) 
    modle.add(layers.ReLU())
    modle.add(layers.MaxPooling2D((2, 2), stride=2))

    modle.add(layers.Flatten())
    modle.add(layers.Linear(5*7*7))
    modle.add(layers.Sigmoid())
    modle.add(layers.Reshape((None, 5, 7, 7)))
    detect_loss = DetectLoss()
    modle.compile(detect_loss, optimizers.SGD(lr=0.001))
    modle.summary()

    # training
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    epochs = 32
    generator = DataIterator(train_data, (128, 128), (5,7,7), 32)
    for i in range(epochs):
        for j in range(len(generator)):
            batch_x,batch_y = generator[j]
            modle.train_one_batch(batch_x, batch_y)
            y = modle.predict(batch_x)
            loss = detect_loss.loss(batch_y,y)
            score = f1_score(batch_y,y)
            print("Epoch=%d, step=%d/%d, loss=%f, f1=%f" % (i+1, j+1, len(generator), loss, score))

    # predict
    x_true, y_true = generator[0]
    y_pred = modle.predict(x_true)
    boxes = decode((128, 128), y_pred[0], 0.5)
    if len(boxes) > 0:
        ax = plt.subplot(1, 1, 1)
        ax.imshow(x_true[0].transpose((1,2,0)*255+128))
        for bbox in boxes:
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()


if __name__ == "__main__":
    test_transform()