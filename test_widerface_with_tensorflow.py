import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datasets.widerface as widerface
import preprocessing.imgkit as imgkit
import PIL.Image as Image
import tensorflow

models = tensorflow.keras.models
layers = tensorflow.keras.layers
optimizers = tensorflow.keras.optimizers
utils = tensorflow.keras.utils
K = tensorflow.keras.backend

# NOTES:
# custom generator must extends keras.utils.Sequence
# output_size: W, H
# feature_shape: H, W, C
class DataGenerator(utils.Sequence):
    def __init__(self, data, output_size, feature_shape, batch_size):
        self.data = data
        self.output_size = output_size
        self.feature_shape = feature_shape
        self.batch_size = batch_size
    def __len__(self):
        samples = len(self.data)
        batches = samples//self.batch_size
        if samples % self.batch_size > 0:
            batches += 1
        return batches
    def __getitem__(self, batch_index):
        start_index = self.batch_size*batch_index
        if start_index >= len(self.data):
            raise IndexError()
        end_index = min(start_index + self.batch_size, len(self.data))
        # NOTES
        # batch_x.shape: N, H, W, C
        # batch_y.shape: N, H, W, C
        batch_size = end_index - start_index
        batch_x = np.zeros((batch_size, self.output_size[1], self.output_size[0], 3))
        batch_y = np.zeros((batch_size,) + self.feature_shape)
        for i in range(start_index, end_index):
            sample = self.data[i]
            image = imgkit.crop(Image.open(sample["image"]), sample["crop"])
            boxes = np.array(sample["boxes"])
            batch_x[i-start_index] = (np.array(image) - 127.5)/255
            batch_y[i-start_index] = encode(self.output_size, boxes, self.feature_shape)
        return batch_x, batch_y

def iou(oh, ow, box1, box2=[0.5, 0.5, 1.0, 1.0]):
    x1, y1, w1, h1 = box1
    w1, h1 = ow*w1, oh*h1
    x2, y2, w2, h2 = box2
    x11, x12, y11, y12  = x1-w1/2, x1+w1/2, y1-h1/2, y1+h1/2
    x21, x22, y21, y22  = x2-w2/2, x2+w2/2, y2-h2/2, y2+h2/2
    max_x, max_y = max(x11, x21), max(y11, y21)
    min_x, min_y = min(x12, x22), min(y12, y22)
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
    sh, sw = float(oh)/ih, float(ow)/iw
    for box in boxes:
        x, y, w, h = box
        cx, cy = sw*(x+w/2), sh*(y+h/2)
        i, j = int(cx), int(cy)    
        bx, by = cx-i, cy-j
        bw, bh = float(w)/iw, float(h)/ih
        if result[j,i,0] > 0 and iou(oh, ow, [bx, by, bw, bh]) > iou(oh, ow, result[j,i,1:]):
            result[j,i,:]=(1, bx, by, bw, bh)
        else:
            result[j,i,:]=(1, bx, by, bw, bh)
        if bx < 0.01 and i > 0:
            if result[j,i-1,0] > 0 and iou(oh, ow, [1-bx, by, bw, bh]) > iou(oh, ow, result[j,i-1,1:]):
                result[j,i-1,:]=(1-bx, 1-bx, by, bw, bh)
            else:
                result[j,i-1,:]=(1-bx, 1-bx, by, bw, bh)
        if bx > 0.99 and i < ow-1:
            if result[j,i+1,0] > 0 and iou(oh, ow, [1-bx, by, bw, bh]) > iou(oh, ow, result[j,i+1,1:]):
                result[j,i+1,:]=(bx, 1-bx, by, bw, bh)
            else:
                result[j,i+1,:]=(bx, 1-bx, by, bw, bh)
        if by < 0.01 and j > 0:
            if result[j-1,i,0] > 0 and iou(oh, ow, [bx, 1-by, bw, bh]) > iou(oh, ow, result[j-1,i,1:]):
                result[j-1,i,:]=(1-by, bx, 1-by, bw, bh)
            else:
                result[j-1,i,:]=(1-by, bx, 1-by, bw, bh)
        if by > 0.99 and j < oh-1:
            if result[j+1,i,0] > 0 and iou(oh, ow, [bx, 1-by, bw, bh]) > iou(oh, ow, result[j+1,i,1:]):
                result[j+1,i,:]=(by, bx, 1-by, bw, bh)
            else:
                result[j+1,i,:]=(by, bx, 1-by, bw, bh)
    return result

def decode(image_size, feature, threshold):
    boxes = []
    iw, ih = image_size
    oh, ow, oc = feature.shape
    sh, sw = float(ih)/oh, float(iw)/ow
    for j in range(oh):
        for i in range(ow):
            p, bx, by, bw, bh = feature[j,i,:]
            if (p >= threshold):
                w = iw*bw
                h = ih*bh
                x = sw*(i + bx) - w/2
                y = sh*(j + by) - h/2
                boxes.append([x, y, w, h, p])
    return boxes

def object_loss(y_true, y_pred):
    p1 = y_true[:,:,:,0]
    p2 = y_pred[:,:,:,0]
    epsilon = 1e-7
    object_loss = - p1*K.log(p2+epsilon) - 0.5*(1-p1)*K.log(1-p2+epsilon)
    return K.sum(object_loss)

def detect_loss(y_true, y_pred):
    p1, x1, y1, w1, h1 = [y_true[:,:,:,i] for i in range(5)]
    p2, x2, y2, w2, h2 = [y_pred[:,:,:,i] for i in range(5)]
    epsilon = 1e-7
    object_loss = - p1*K.log(p2+epsilon) - 0.5*(1-p1)*K.log(1-p2+epsilon)
    location_loss = K.square(x1-x2) + K.square(y1 - y2) + K.square(K.sqrt(w1) - K.sqrt(w2)) + K.square(K.sqrt(h1) - K.sqrt(h2))
    location_loss *= K.cast(p1 > 0.5, dtype='float32')
    return K.sum(object_loss) + 10*K.sum(location_loss)

# NOTES:
# Precision = TP/(TP + FP)
def precision(y_true, y_pred):
    p1 = y_pred[:,:,:,0]
    p2 = y_true[:,:,:,0]
    P = K.cast(p1 > 0.5, dtype='float32')
    TP = K.cast(p2 > 0.5, dtype='float32')*P
    epsilon = 1e-7
    return K.sum(TP)/(K.sum(P) + epsilon)

# NOTES:
# Recall = TP/(TP + FN)
def recall(y_true, y_pred):
    p1 = y_pred[:,:,:,0]
    p2 = y_true[:,:,:,0]
    TP = K.cast(p2 > 0.5, dtype='float32')*K.cast(p1 > 0.5, dtype='float32')
    FN = K.cast(p2 > 0.5, dtype='float32')*K.cast(p1 <= 0.5, dtype='float32')
    epsilon = 1e-7
    return K.sum(TP)/(K.sum(TP) + K.sum(FN) + epsilon)

def test_codec():
    image_size = (160, 160)
    data = widerface.load_data()
    train_data = widerface.select(data[0], blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    train_data = widerface.crop(train_data, 100, image_size)
    for sample in train_data:
        image = imgkit.crop(Image.open(sample["image"]), sample["crop"])
        boxes = np.array(sample["boxes"])
        feature = encode(image.size, boxes, (3,3,5))
        print(feature)
        boxes=decode(image.size, feature, 0.9)
        imgkit.draw_grids(image, (3,3))
        imgkit.draw_boxes(image, boxes, color=(0,255,0))
        plt.imshow(image)
        plt.show()

def test_generator():
    image_size = (160, 160)
    batch_size = 4
    feature_shape = (3, 3, 5)
    data = widerface.load_data()
    train_data = widerface.select(data[0], blur="0", illumination="0", occlusion="0", invalid="0", min_size=32)
    train_data = widerface.crop(train_data, 100, image_size)
    for batch_x, batch_y in DataGenerator(train_data, image_size, feature_shape, batch_size):
        for i in range(batch_size):
            boxes = decode(image_size, batch_y[i], 0.5)
            ax = plt.subplot(1, batch_size, i + 1)
            plt.tight_layout()
            ax.set_title("Sample %s" % i)
            ax.axis('off')
            ax.imshow(np.array(batch_x[i]*255+127.5, dtype='uint8'))
            for box in boxes:
                (x, y, w, h) = box[:4]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()

def build_model():
    x = layers.Input(shape=(160, 160, 3))
    conv_1 = layers.SeparableConv2D(32, 3, padding="same", activation="relu", input_shape=(160, 160, 3))(x)
    maxpool_1= layers.MaxPool2D(2)(conv_1)
    conv_2 = layers.SeparableConv2D(64, 3, padding="same", activation="relu")(maxpool_1)
    maxpool_2= layers.MaxPool2D(2)(conv_2)
    conv_3 = layers.SeparableConv2D(128, 3, padding="same", activation="relu")(maxpool_2)
    maxpool_3= layers.MaxPool2D(2)(conv_3)
    conv_4 = layers.SeparableConv2D(256, 3, padding="same", activation="relu")(maxpool_3)
    maxpool_4= layers.MaxPool2D(2)(conv_4)
    conv_5 = layers.SeparableConv2D(512, 3, padding="same", activation="relu")(maxpool_4)
    maxpool_5= layers.MaxPool2D(2)(conv_5)
    y = layers.SeparableConv2D(5, 3, activation="sigmoid")(maxpool_5)
    model = models.Model(inputs=x, outputs=y)
    model.compile(optimizer=optimizers.Adagrad(), loss=detect_loss, metrics=[object_loss, precision, recall])
    model.summary()
    return model

def load_model(model_file):
    
    model = models.load_model(model_file, custom_objects={
        "detect_loss": detect_loss, 
        "object_loss": object_loss, 
        "precision": precision, 
        "recall": recall})
    model.summary()
    return model

def train_model(model, train_data, image_size, feature_shape, num_sample, batch_size, save_file):
    train_data = widerface.select(train_data, blur="0", illumination="0", occlusion="0", invalid="0", min_size=32)
    train_data = widerface.crop(train_data, num_sample, image_size)
    generator = DataGenerator(train_data, image_size, feature_shape, batch_size)
    pos=save_file.rfind('_')
    start = int(save_file[pos+1: len(save_file)-3])+1
    for i in range(start,111):
        model.fit_generator(generator, epochs=20, workers=2, use_multiprocessing=True)
        model.save(save_file[:pos+1] + str(i*20) + ".h5")

def predict_model(model, val_data, image_size, feature_shape):
    val_data = widerface.select(val_data, blur="0", illumination="0", occlusion="0", invalid="0", min_size=32)
    val_data = widerface.crop(val_data, 9, (160,160))
    generator = DataGenerator(val_data, image_size, feature_shape, 9)
    batch_x, batch_y = generator[0]
    y_pred = model.predict(batch_x) 
    for i in range(len(y_pred)):
        ax = plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        ax.set_title("Sample %d" % i)
        ax.axis('off')
        ax.imshow(np.array(batch_x[i]*255+127.5, dtype='uint8'))
        boxes = decode(image_size, y_pred[i], 0.5)
        for box in boxes:
            (x, y, w, h, p) = box
            print("Sample %d, predicted_box=(%d,%d,%d,%d), score=%f" % (i, x, y, w, h, p))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        boxes = decode(image_size, batch_y[i], 0.5)
        for box in boxes:
            (x, y, w, h, p) = box
            print("Sample %d, true_box=(%d,%d,%d,%d), score=%f" % (i, x, y, w, h, p))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    plt.show()

def test_model():
    image_size=(160, 160)
    feature_shape=(3,3,5)
    num_sample=6400
    batch_size=64

    # build model
    model_file = os.path.dirname(os.path.abspath(__file__)) + "/datasets/widerface/face_model_v5_160.h5"
    model=load_model(model_file)
    #model=build_model()

    # load widerface data
    data = widerface.load_data()

    # train model
    train_model(model, data[0], image_size, feature_shape, num_sample, batch_size, model_file)

    # predict model
    predict_model(model, data[1], image_size, feature_shape)


if __name__ == "__main__":
    test_model()