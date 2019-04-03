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

def transform(image, boxes, output_size):
    iw, ih = image.size
    ow, oh = output_size
    # image size is smaller than output size
    if iw < ow or ih < oh:
        return imgkit.resize(image, output_size, boxes)

    # find the box close to 32*32, try crop based on the center of the biggest box
    area = np.array([abs(b[2]*b[3] - 32*32) for b in boxes])
    x, y, w, h = boxes[np.argmin(area)]
    # the box size is greater than output size
    # crop a random rect with about 2-times of the biggest box' size
    if w > ow/2 or h > oh/2:
        r = max(w, h)*np.random.rand()
        x1 = max(x-r, 0)
        y1 = max(y-r, 0)
        x2 = min(max(x1+ow, x+w+r), iw)
        y2 = min(max(y1+oh, y+h+r), ih)
        selected = np.array([b for b in boxes if b[0] >= x1 and b[1] >= y1 and b[0]+b[2] <= x2 and b[1]+b[3] <= y2])
        assert(selected.shape[0] > 0)
        image, boxes = imgkit.crop(image, [x1,y1,x2,y2], selected)
        return imgkit.resize(image, output_size, boxes)

    # crop a random rect with output size
    rx = (ow - w)*np.random.rand()
    ry = (oh - h)*np.random.rand()
    x1 = max(x - rx, 0)
    y1 = max(y - ry, 0)
    x2 = min(x1 + ow, iw)
    y2 = min(y1 + oh, ih)
    selected = np.array([b for b in boxes if b[0] >= x1 and b[1] >= y1 and b[0]+b[2] <= x2 and b[1]+b[3] <= y2])
    assert(selected.shape[0] > 0)
    image, boxes = imgkit.crop(image, [x1,y1,x2,y2], selected)
    return imgkit.resize(image, output_size, boxes)

    
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
            image = Image.open(sample["image"])
            boxes = np.array(sample["boxes"])
            image, boxes = transform(image, boxes, self.output_size)
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
            # NOTES: select the bunding box that has biggest IOU with the grid
            result[j,i,:]=(1, bx, by, bw, bh)
        else:
            result[j,i,:]=(1, bx, by, bw, bh)
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

def giou_loss(y_true, y_pred):
    p1, x1, y1, w1, h1 = [y_true[:,:,:,i] for i in range(5)]
    p2, x2, y2, w2, h2 = [y_pred[:,:,:,i] for i in range(5)]
    # NOTES:
    # x/y is rate of image width/height
    # w/h is rate of grid  width/height
    # So, adjust w/h is align with grid  width/height
    w1, h1, w2, h2 = 5*w1, 5*h1, 5*w2, 5*h2
    x11, x12, y11, y12 = x1-w1/2, x1+w1/2, y1-h1/2, y1+h1/2
    x21, x22, y21, y22 = x2-w2/2, x2+w2/2, y2-h2/2, y2+h2/2
    max_x, max_y = K.maximum(x11, x21), K.maximum(y11, y21)
    min_x, min_y = K.minimum(x12, x22), K.minimum(y12, y22)
    I = (min_x - max_x)*(min_y - max_y)*K.cast(min_x > max_x, dtype='float32')*K.cast(min_y > max_y, dtype='float32')
    U = w1*h1 + w2*h2 - I
    min_x, min_y = K.minimum(x11, x21), K.minimum(y11, y21)
    max_x, max_y = K.maximum(x12, x22), K.maximum(y12, y22)
    C = (max_x - min_x)*(max_y - min_y)
    giou = I/U - (C-U)/C
    loss = (1- giou)*K.cast(p1 > 0.5, dtype='float32')
    return K.sum(loss)

def object_loss(y_true, y_pred):
    p1 = y_true[:,:,:,0]
    p2 = y_pred[:,:,:,0]
    object_loss = - p1*K.log(p2) - 0.5*(1-p1)*K.log(1-p2)
    return K.sum(object_loss)

def detect_loss(y_true, y_pred):
    p1, x1, y1, w1, h1 = [y_true[:,:,:,i] for i in range(5)]
    p2, x2, y2, w2, h2 = [y_pred[:,:,:,i] for i in range(5)]
    object_loss = - p1*K.log(p2) - 0.5*(1-p1)*K.log(1-p2)
    location_loss = K.square(x1-x2) + K.square(y1 - y2) + K.square(K.sqrt(w1) - K.sqrt(w2)) + K.square(K.sqrt(h1) - K.sqrt(h2))
    location_loss *= K.cast(p1 > 0.5, dtype='float32')
    return K.sum(object_loss) + 5*K.sum(location_loss)

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

# NOTES:
# Precision = TP/(TP + FP)
# Recall = TP/(TP + FN)
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
def f1_score(y_true, y_pred):
    Precision = precision(y_true, y_pred)
    Recall = recall(y_true, y_pred)
    epsilon = 1e-7
    return 2*(Recall*Precision)/(Recall + Precision + epsilon)

def test_codec():
    train_data = widerface.load_data()
    train_data = widerface.select(train_data[0], blur="0", illumination="0", occlusion="0", invalid="0", min_size=30)
    for sample in train_data:
        image = Image.open(sample["image"])
        boxes = np.array(sample["boxes"])
        image, boxes = transform(image, boxes, (160, 160))
        feature = encode(image.size, boxes, (5,5,5))
        print(feature)
        boxes=decode(image.size, feature, 1.0)
        imgkit.draw_grids(image, (5,5))
        imgkit.draw_boxes(image, boxes, color=(0,255,0))
        plt.imshow(image)
        plt.show()

def test_generator():
    data = widerface.load_data()
    train_data = widerface.select(data[0], blur="0", occlusion="0", pose="0", invalid="0")
    image_size = (160, 160)
    batch_size = 4
    feature_shape = (5, 5, 5)
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
    model = models.Sequential()
    model.add(layers.SeparableConv2D(32, 3, padding="same", activation="relu", input_shape=(160, 160, 3)))
    model.add(layers.MaxPool2D(2))
    model.add(layers.SeparableConv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.SeparableConv2D(128, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.SeparableConv2D(256, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.SeparableConv2D(512, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.SeparableConv2D(5, 3, padding="same", activation="sigmoid"))
    model.compile(optimizer=optimizers.Adagrad(), loss=detect_loss, metrics=[object_loss, giou_loss, precision, recall, f1_score])
    return model

def train_model(model, train_data, image_size, feature_shape, batch_size):
    train_data = widerface.select(train_data, blur="0", illumination="0", occlusion="0", invalid="0", min_size=32)
    generator = DataGenerator(train_data, image_size, feature_shape, batch_size)
    for i in range(111):
        model.fit_generator(generator, epochs=20, workers=2, use_multiprocessing=True)
        model_file = os.path.dirname(os.path.abspath(__file__)) + "/datasets/widerface/face_model_v4"
        model.save(model_file + "_" + str(20+(i+1)*20) + ".h5")

def predict_model(model, val_data, image_size, feature_shape):
    val_data = widerface.select(val_data, blur="0", illumination="0", occlusion="0", invalid="0", min_size=32)
    generator = DataGenerator(val_data, image_size, feature_shape, 32)
    batch_x, batch_y = generator[11]
    batch_x, batch_y = batch_x[:9], batch_y[:9]
    y_pred = model.predict(batch_x) 
    for i in range(len(y_pred)):
        ax = plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        ax.set_title("Sample %d" % i)
        ax.axis('off')
        ax.imshow(np.array(batch_x[i]*255+127.5, dtype='uint8'))
        boxes = decode(image_size, y_pred[i], 0.6)
        for box in boxes:
            (x, y, w, h, p) = box
            print("Sample %d, box=(%d,%d,%d,%d), score=%f" % (i, x, y, w, h, p))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        boxes = decode(image_size, batch_y[i], 0.5)
        for box in boxes:
            (x, y, w, h, p) = box
            print("Sample %d, box=(%d,%d,%d,%d), score=%f" % (i, x, y, w, h, p))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    plt.show()

def test_model():
    image_size=(160, 160)
    feature_shape=(5,5,5)
    batch_size=100

    # build model
    model_file = os.path.dirname(os.path.abspath(__file__)) + "/datasets/widerface/face_model_v4_960.h5"
    model = models.load_model(model_file, custom_objects={
        "detect_loss": detect_loss, 
        "object_loss": object_loss,
        "giou_loss": giou_loss, 
        "f1_score": f1_score, 
        "precision": precision, 
        "recall": recall})

    # model = build_model()
    model.summary()

    # load widerface data
    data = widerface.load_data()

    # train model
    train_model(model, data[0], image_size, feature_shape, batch_size)

    # predict model
    predict_model(model, data[1], image_size, feature_shape)


if __name__ == "__main__":
    test_model()