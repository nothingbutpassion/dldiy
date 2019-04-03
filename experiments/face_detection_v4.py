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


def confidence_loss(y_true, y_pred):
    return 0

def localization_loss(y_true, y_pred):
    return 0

def detection_loss(y_true, y_pred):
    return confidence_loss(y_true, y_pred) + localization_loss(y_true, y_pred)


def build_modle():
    base = MobileNetV2(input_shape=(160, 160, 3), alpha=0.5, include_top=False)
    for layer in base.layers:
        layer.trainable = False
    # (10, 10, 288) -> (10, 10, 5K)
    x1 = [l for l in base.layers if l.name == "block_13_expand_relu"][0]
    y1 = layers.Conv2D(5, 3, padding="same", activation="sigmoid")(x1)

    # (5, 5, 480) -> (5, 5, 5K)
    x2 = [l for l in base.layers if l.name == "block_15_expand_relu"][0]
    y2 = layers.Conv2D(5, 3, padding="same", activation="sigmoid")(x2)

    # (5, 5, 1280) -> (3, 3, 5K)
    x3 = [l for l in base.layers if l.name == "out_relu"][0]
    y3 = layers.Conv2D(5, 3, activation="sigmoid")(x3)

    # (3, 3, 5k) -> (1, 1, 256) -> (1, 1, 5K)
    x4 = layers.Conv2D(256, 1, activation="relu")(y3)
    y4 = layers.Conv2D(5, 3, activation="sigmoid")(x4)

    model = models.Model(inputs=base.input, outputs=[y1, y2, y3, y4])
    # model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.00005), loss=detection_loss, metrics=[confidence_loss, localization_loss])
    model.summary()
    return model

if __name__ == "__main__":
    build_modle