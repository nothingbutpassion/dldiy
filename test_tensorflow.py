from PIL import Image, ImageDraw
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datasets.widerface as widerface


class DataGenerator:
    def __init__(self, data, output_size, feature_shape, batch_size):
        self.data = data
        self.output_size = output_size
        self.feature_shape = feature_shape
        self.batch_size = batch_size
    
    def __len__(self):
        batches = len(self.data)//self.batch_size
        if len(self.data)%self.batch_size > 0:
            batches += 1
        return batches
    
    def __getitem__(self, batch_index):
        start_index = self.batch_size*batch_index
        if start_index >= len(self.data):
            raise IndexError()
        end_index = np.minimum(start_index + self.batch_size, len(self.data))
        batch_data = self.data[start_index : end_index]
        batch_x = np.zeros((self.batch_size, self.output_size[0], self.output_size[1], 3), dtype='float32')
        batch_y = np.zeros(((self.batch_size,) + self.feature_shape), dtype='float32')
        for i, sample in enumerate(batch_data):
            image = Image.open(sample["image"])
            x_rate, y_rate = self.output_size[0]/image.size[0], self.output_size[1]/image.size[1]
            image = image.resize(self.output_size, Image.BILINEAR)
            image = np.asarray(image)
            batch_x[i, :, :, :] = (image - 128)/255
            boxes = np.array(sample["boxes"])
            for box in boxes:
                box[0] = box[0]*x_rate
                box[2] = box[2]*x_rate
                box[1] = box[1]*y_rate
                box[3] = box[3]*y_rate
            batch_y[i] = encode(self.output_size, boxes, self.feature_shape)
        return batch_x, batch_y

def iou(box1, box2 = [0.5, 0.5, 1, 1]):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x11, x12, y11, y12  = x1-w1/2, x2+w1/2, y1-h1/2, y1+h1/2
    x21, x22, y21, y22  = x2-w2/2, x2+w2/2, y2-h2/2, y2+h2/2
    max_x, max_y = np.maximum(x11, x21), np.maximum(y11, y21)
    min_x, min_y = np.minimum(x12, x22), np.minimum(y12, y22)
    if min_x < max_x or min_y < max_y:
        print("IOU Error: box1=%s, box2=%s" % (box1, box2))
        return 0
    I = (min_x - max_x)*(min_y - max_y)
    U = w1*h1 + w2*h2 - I
    return I/U

def encode(image_size, boxes, feature_shape=(7,7,5)):
    result = np.zeros(feature_shape)
    iw, ih = image_size
    oh, ow, oc = feature_shape
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
        if result[j,i,0] > 0:
            # NOTES: Select the bunding box that has biggest IOU with grid
            if iou([bx, by, bw, bh]) > iou(result[j,i,1:]):
                result[j,i,:]=(1, bx, by, bw, bh)
        else:
            result[j,i,:]=(1, bx, by, bw, bh)
    return result

def decode(image_size, feature, threshold=1.0):
    boxes = []
    iw, ih = image_size
    oh, ow, oc = feature.shape
    sh, sw = ih/oh, iw/ow
    for j in range(oh):
        for i in range(ow):
            p, bx, by, bw, bh = feature[j,i,:]
            if (p >= threshold):
                w = sw*bw
                h = sh*bh
                x = sw*(i + bx) - w/2
                y = sh*(j + by) - h/2
                boxes.append([x, y, w, h])
    return boxes

def draw_grids(image, grid_shape):
    d = ImageDraw.Draw(image)
    gw, gh = grid_shape
    w, h = image.size[0], image.size[1]
    rw, rh = w/gw, h/gh
    for j in range(gh):
        d.line([1, j*rh, w-1, j*rh], fill=(255,0,0), width=4)
    for i in range(gw):
        d.line([i*rw, 1, i*rw, h-1], fill=(255,0,0), width=4)

def draw_boxes(image, boxes):
    d = ImageDraw.Draw(image)
    for box in boxes:
        (x, y, w, h) = box[:4]
        d.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=4)

def test_codec():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    sample = train_data[2]
    image = Image.open(sample["image"])
    boxes = sample["boxes"]
    feature = encode(image.size, boxes, (7,7,5))
    print(feature)
    boxes=decode(image.size, feature)
    draw_grids(image, (7,7))
    draw_boxes(image, boxes)
    plt.imshow(image)
    plt.show()

def show_images(batch_x, batch_y, image_size=(256,256)):
    batch_size = batch_x.shape[0]
    for i in range(batch_size):
        boxes = decode(image_size, batch_y[i])
        ax = plt.subplot(1, batch_size, i + 1)
        plt.tight_layout()
        ax.set_title("Sample #{}".format(i))
        ax.axis('off')
        ax.imshow(batch_x[i])
        for box in boxes:
            (x, y, w, h) = box[:4]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()

def test_data():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    for batch_x, batch_y in DataGenerator(train_data, (256, 256), (7,7,5), 4):
        show_images(batch_x, batch_y)
        break

if __name__ == "__main__":

    test_data()
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=(1,1), padding='valid', activation="relu", input_shape=(256, 256, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))

    model.add(layers.Conv2D(32, (3, 3), strides=(1,1), padding='valid', activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))

    model.add(layers.Conv2D(64, (3, 3), strides=(1,1), padding='valid', activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(layers.Conv2D(32, (3, 3), strides=(1,1), padding='valid', activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(7*7*5, activation="relu"))
    model.add(layers.Reshape((7,7,5)))

    model.compile(optimizer=optimizers.SGD(lr=0.001), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    