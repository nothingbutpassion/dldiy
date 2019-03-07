import numpy as np
import matplotlib.pyplot as plt
import datasets.widerface as widerface
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Iterator:
    def __init__(self, data, output_size, batch_size):
        self.data = data
        self.output_size = output_size
        self.batch_size = batch_size
    
    def __getitem__(self, batch_index):
        if (self.batch_size+1)*batch_index > len(self.data):
            raise IndexError()
        result = {
            "image": np.zeros((self.batch_size, 3, self.output_size[0], self.output_size[1]), dtype='uint8'),
            "boxes": []
        }
        batch_data = self.data[self.batch_size*batch_index : self.batch_size*(batch_index + 1)]
        for i, sample in enumerate(batch_data):
            image = Image.open(sample["image"])
            x_rate, y_rate = self.output_size[0]/image.size[0], self.output_size[1]/image.size[1]
            image = image.resize(self.output_size, Image.BILINEAR)
            image = np.asarray(image)
            image = image.transpose((2, 0, 1))
            result["image"][i, :, :, :] = image
            boxes = np.array(sample["boxes"])
            for box in boxes:
                box[0] = box[0]*x_rate
                box[2] = box[2]*x_rate
                box[1] = box[1]*y_rate
                box[3] = box[3]*y_rate
            result["boxes"].append(boxes)
        return result

def encode(image_size, boxes, feature_shape=(6,7,7)):
    result = np.zeros(feature_shape)
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

def loss(y_true, y_pred):
    epsilon = 1e-7
    po, pn, bx, by, bw, bh = y_true
    o, n, x, y, w, h = y_pred
    object_loss = -np.sum(po*np.log(o + epsilon) + pn*np.log(n + epsilon))
    box_loss = np.sum((bx-x)**2 + (by-y)**2 + (bw-w)**2 + (bh-h)**2)
    return object_loss * box_loss


def test_data():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    for batch in Iterator(train_data, (512, 512), 4):
        for i in range(4):
            image = batch["image"][i].transpose((1,2,0))
            boxes = batch["boxes"][i]
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title("Sample #{}".format(i))
            ax.axis('off')
            ax.imshow(image)
            for bbox in boxes:
                (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show(block=True)
        break

def test_codec():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    sample = train_data[0]
    image = Image.open(sample["image"])
    boxes = sample["boxes"]
    feature = encode(image.size, boxes)  
    # feature2 = encode(image.size, boxes)
    # print(loss(feature, feature2))
    boxes=decode(image.size, feature)
    ax = plt.subplot(1, 1, 1)
    ax.imshow(image)
    for bbox in boxes:
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    test_codec()