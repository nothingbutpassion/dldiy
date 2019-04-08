import os
import sys
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/.." )
import datasets.widerface as widerface
import preprocessing.imgkit as imgkit


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w = max(0, min(x1+w1/2, x2+w2/2) - max(x1-w1/2, x2-w2/2))
    h = max(0, min(y1+h1/2, y2+h2/2) - max(y1-h1/2, y2-h2/2))
    I = w*h
    U = w1*h1 + w2*h2 - I
    return I/U

def generate_dboxes(scale, size, aspects):
    rows, cols = size
    dboxes = np.zeros((rows, cols, len(aspects), 4))
    for i in range(rows):
        for j in range(cols):
            for k in range(len(aspects)):
                bx, by, bw, bh = (j+0.5)/cols, (i+0.5)/rows, scale*np.sqrt(aspects[k]), scale/np.sqrt(aspects[k])
                dboxes[i,j,k,:] = [bx, by, bw, bh]
    dboxes = dboxes.reshape(-1, 4)
    return dboxes

# NOTES:
# Each box in gboxes is a tuple with 4 value: bx, by, bw, bh
# bx, by is the center point of the box (nomalized to image size)
# bw, by is the width, heigh of the box (nomalized to image size)
def encode(gboxes, scales=[0.3, 0.5, 0.7, 0.9], sizes=[[10,10], [5,5], [3,3], [1,1]], aspects=[1.0, 1.5, 2.0]):
    dboxes = []
    for i in range(len(scales)):
        dboxes.append(generate_dboxes(scales[i], sizes[i], aspects))
    dboxes = np.concatenate(dboxes)
    features = np.zeros((dboxes.shape[0], 6))
    features[:,:2] = [0, 1]
    # For each ground truth box, match a default box with max IOU
    for gb in gboxes:
        ious = [iou(gb, db) for db in dboxes]
        i = np.argmax(ious)
        dx, dy, dw, dh = dboxes[i]
        gx, gy, gw, gh = gb
        features[i] = [1, 0] + [(gx-dx)/dw, (gy-dy)/dh, np.log(gw/dw), np.log(gh/dh)]
    # For each default box, match a groud truth box with IOU > 0.5
    for i in range(len(dboxes)):
        gbs = [gb for gb in gboxes if iou(gb, dboxes[i]) > 0.5]
        if len(gbs) > 0:
            ious = [iou(gb, dboxes[i]) for gb in gbs]
            dx, dy, dw, dh = dboxes[i]
            gx, gy, gw, gh = gbs[np.argmax(ious)]
            features[i] = [1, 0] + [(gx-dx)/dw, (gy-dy)/dh, np.log(gw/dw), np.log(gh/dh)]
    return features

def get_dbox(feature_index, scales=[0,3, 0.5, 0.7, 0.9], sizes=[[10,10], [5,5], [3,3], [1,1]], aspects=[1.0, 1.5, 2.0]):
    aspect_num = len(aspects)
    feature_nums = [s[0]*s[1]*aspect_num for s in sizes]
    for i in range(1, len(feature_nums)):
        feature_nums[i] += feature_nums[i-1]
    i = 0
    while feature_index >= feature_nums[i]:
        i += 1
    if i > 0:
        feature_index -= feature_nums[i-1]
    rows, cols = sizes[i]
    scale = scales[i]
    i = feature_index//(cols*aspect_num)
    j = (feature_index - i*cols*aspect_num)//3
    k = feature_index - i*cols*aspect_num - j*aspect_num
    dx, dy, dw, dh = (j+0.5)/cols, (i+0.5)/rows, scale*np.sqrt(aspects[k]), scale/np.sqrt(aspects[k])
    return [dx, dy, dw, dh]


def decode(features, scales=[0,3, 0.5, 0.7, 0.9], sizes=[[10,10], [5,5], [3,3], [1,1]], aspects=[1.0, 1.5, 2.0]):
    gboxes = []
    for i in range(len(features)):
        c0, c1, x, y, w, h = features[i]
        if c0 > c1:
            dx, dy, dw, dh = get_dbox(i)
            gx, gy, gw, gh = x*dw+dx, y*dh+dy, np.exp(w)*dw, np.exp(h)*dh
            gboxes.append([gx, gy, gw, gh])
    return gboxes

def test_codec():
    image_size = (160, 160)
    data = widerface.load_data()
    train_data = widerface.select(data[0], blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    train_data = widerface.crop(train_data, 100, image_size)
    for sample in train_data:
        image = imgkit.crop(Image.open(sample["image"]), sample["crop"])
        w, h = image_size
        boxes = np.array(sample["boxes"])
        print("before encode: boxes=%s" % str(boxes))
        boxes = [[(b[0]+0.5*b[2])/w, (b[1]+0.5*b[3])/h, b[2]/w, b[3]/h] for b in boxes]
        features = encode(boxes)
        boxes=decode(features)
        boxes = [[(b[0]-0.5*b[2])*w, (b[1]-0.5*b[3])*h, b[2]*w, b[3]*h] for b in boxes]
        print("after decode: boxes=%s" % str(boxes))
        imgkit.draw_boxes(image, boxes, color=(0,255,0))
        plt.imshow(image)
        plt.show()

def confidence_loss(y_true, y_pred):
    c = K.softmax(y_pred[:,:,:2])
    return K.sum(-y_true[:,:,0]*K.log(c[:,:,0]) - y_true[:,:,1]*K.log(c[:,:,1]))

def localization_loss(y_true, y_pred):
    return 0

def detection_loss(y_true, y_pred):
    N = K.sum(K.cast(y_true[:,:,0] > 0, dtype='float32'))
    return (confidence_loss(y_true, y_pred) + localization_loss(y_true, y_pred))/N


def build_modle():
    base = MobileNetV2(input_shape=(160, 160, 3), alpha=0.5, include_top=False)
    for layer in base.layers:
        layer.trainable = False
    # (10, 10, 288) -> (10, 10, 5K)
    x1 = [l for l in base.layers if l.name == "block_13_expand_relu"][0]
    y1 = layers.Conv2D(5, 3, padding="same", activation="sigmoid")(x1.output)

    # (5, 5, 480) -> (5, 5, 5K)
    x2 = [l for l in base.layers if l.name == "block_15_expand_relu"][0]
    y2 = layers.Conv2D(5, 3, padding="same", activation="sigmoid")(x2.output)

    # (5, 5, 1280) -> (3, 3, 5K)
    x3 = [l for l in base.layers if l.name == "out_relu"][0]
    y3 = layers.Conv2D(5, 3, activation="sigmoid")(x3.output)

    # (3, 3, 5k) -> (1, 1, 256) -> (1, 1, 5K)
    x4 = layers.Conv2D(256, 1, activation="relu")(y3)
    y4 = layers.Conv2D(5, 3, activation="sigmoid")(x4)

    model = models.Model(inputs=base.input, outputs=[y1, y2, y3, y4])
    # model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.00005), loss=detection_loss, metrics=[confidence_loss, localization_loss])
    model.summary()
    return model

if __name__ == "__main__":
    test_codec()