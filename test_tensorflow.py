from PIL import Image, ImageDraw
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datasets.widerface as widerface

# NOTES:
# custom generator must extends keras.utils.Sequence
# output_size: W, H
# feature_shape: H, W, C
class DataGenerator(utils.Sequence):
    def __init__(self, data, output_size, feature_shape, batch_size):
        self.output_size = output_size
        self.feature_shape = feature_shape
        self.batch_size = batch_size

        # NOTES
        # self.x.shape: N, H, W, C
        # self.y.shape: N, H, W, C
        self.x = np.zeros((len(data), output_size[1], output_size[0], 3))
        self.y = np.zeros((len(data),) + feature_shape)
        # NOTES: 
        # consider data augmentation if possible
        box_num = 0
        for i, sample in enumerate(data):
            image = Image.open(sample["image"])
            scale = np.array(output_size[:2])/np.array(image.size[:2])
            image = np.array(image.resize(output_size, Image.BICUBIC))
            self.x[i] = (image - 127.5)/255
            boxes = np.array(sample["boxes"])
            for box in boxes:
                box[:2] *= scale
                box[2:] *= scale
            self.y[i] = encode(output_size, boxes, feature_shape)
            box_num += np.sum(self.y[i,:,:,0])
            print("loaded sample=%d boxes=%d, total=%s" % (i, box_num, self.x.shape[0])) 

    def __len__(self):
        batches = self.x.shape[0]//self.batch_size
        if self.x.shape[0] % self.batch_size > 0:
            batches += 1
        return batches
    
    def __getitem__(self, batch_index):
        start_index = self.batch_size*batch_index
        if start_index >= self.x.shape[0]:
            raise IndexError()
        end_index = min(start_index + self.batch_size, self.x.shape[0])
        return self.x[start_index:end_index], self.y[start_index:end_index]

def iou(box1, box2 = [0.5, 0.5, 1, 1]):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x11, x12, y11, y12  = x1-w1/2, x1+w1/2, y1-h1/2, y1+h1/2
    x21, x22, y21, y22  = x2-w2/2, x2+w2/2, y2-h2/2, y2+h2/2
    max_x, max_y = np.maximum(x11, x21), np.maximum(y11, y21)
    min_x, min_y = np.minimum(x12, x22), np.minimum(y12, y22)
    assert(min_x > max_x and min_y > max_y)
    I = (min_x - max_x)*(min_y - max_y)
    U = w1*h1 + w2*h2 - I
    return I/U

# NOTES:
# image_size    = (width, height)
# feature_shape = (height, width, channel)
def encode(image_size, boxes, feature_shape):
    result = np.zeros(feature_shape)
    iw, ih = image_size
    oh, ow, oc = feature_shape
    sh, sw = oh/ih, ow/iw
    for box in boxes:
        x, y, w, h = box
        cx, cy = sw*(x+w/2), sh*(y+h/2)
        i, j = int(cx), int(cy)    
        bx, by = cx-i, cy-j
        bw, bh = sw*w, sh*h
        if result[j,i,0] > 0 and iou([bx, by, bw, bh]) > iou(result[j,i,1:]):
            # NOTES: select the bunding box that has biggest IOU with the grid
            result[j,i,:]=(1, bx, by, bw, bh)
        else:
            result[j,i,:]=(1, bx, by, bw, bh)
    return result

def decode(image_size, feature, threshold):
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

def detect_loss(y_true, y_pred):
    p, x, y, w, h = [y_pred[:,:,:,i] for i in range(5)]
    tp, tx, ty, tw, th = [y_true[:,:,:,i] for i in range(5)]
    p = 1/(1+K.exp(-p))
    obj_loss = - tp*K.log(p) - (1-tp)*K.log(1-p)
    loc_loss = K.square(tx-x) + K.square(ty-y) + K.square(tw-w) + K.square(th-h)
    loc_loss *= K.cast(tp > 0, dtype='float32')
    return K.mean(obj_loss) + K.mean(loc_loss)
    
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
    boxes=decode(image.size, feature, 1.0)
    draw_grids(image, (7,7))
    draw_boxes(image, boxes)
    plt.imshow(image)
    plt.show()

def test_data():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    image_size = (300, 200)
    batch_size = 4
    feature_shape = (7,7,5)
    for batch_x, batch_y in DataGenerator(train_data, image_size, feature_shape, batch_size):
        for i in range(batch_size):
            boxes = decode(image_size, batch_y[i], 1.0)
            ax = plt.subplot(1, batch_size, i + 1)
            plt.tight_layout()
            ax.set_title("Sample %s" % i)
            ax.axis('off')
            ax.imshow(np.array(batch_x[i]*255+127, dtype='uint8'))
            for box in boxes:
                (x, y, w, h) = box[:4]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()
        break

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=(1,1), activation="relu", input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))
    model.add(layers.Conv2D(32, (3, 3), strides=(1,1), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))
    model.add(layers.Conv2D(48, (3, 3), strides=(1,1), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Conv2D(64, (3, 3), strides=(1,1), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(7*7*5))
    model.add(layers.Reshape((7,7,5)))
    model.compile(optimizer=optimizers.SGD(lr=0.001), loss=detect_loss)
    model.summary()
    return model


def test_model():
    # build model
    model_file = os.path.dirname(os.path.abspath(__file__)) + "/datasets/widerface/face_model.h5"
    model = models.load_model(model_file, custom_objects={"detect_loss":detect_loss})
    model.summary()
    # model = build_model()

    # load train data
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", occlusion="0", pose="0", invalid="0")
    generator = DataGenerator(train_data, (256, 256), (7,7,5), 32)

    # train model
    model.fit_generator(generator, epochs=20)
    models.save(model_file)

    # predict sample
    batch_x, batch_y = generator[0]
    batch_x, batch_y = batch_x[:11], batch_y[:11]
    y = model.predict(batch_x)
    for i in range(len(y)):
        boxes = decode((256, 256), y[i], 0.5)
        ax = plt.subplot(1, 11, i + 1)
        plt.tight_layout()
        ax.set_title("Sample %d" % i)
        ax.axis('off')
        ax.imshow(np.array(batch_x[i]*255+127.5, dtype='uint8'))
        for box in boxes:
            (x, y, w, h) = box[:4]
            print("Sample %d, box=(%d,%d,%d,%d)", (i, x, y, w, h))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()

if __name__ == "__main__":
    test_model()


    