import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import models, layers, utils, optimizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

sys.path.append(Path(__file__).absolute().parents[1].as_posix())
from datasets import widerface

g_scales = [0.3, 0.5, 0.7, 0.9]
g_sizes = [[10,10], [5,5], [3,3], [1,1]]
g_aspects = [0.5, 0.8, 1.0]

def image_crop(image, rect, boxes=None):
    image = image.crop(rect)
    if boxes is None:
        return image
    boxes=np.array([[b[0]-rect[0], b[1]-rect[1], b[2], b[3]] for b in boxes])
    return image, boxes
def image_resize(image, size, boxes=None):
    sx = float(size[0])/image.size[0]
    sy = float(size[1])/image.size[1]
    image = image.resize(size, Image.LINEAR)
    if boxes is None:
        return image
    boxes = np.array([[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes])
    return image, boxes
def image_flip(image, boxes=None):
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes is None:
        return image
    w = image.size[0]
    boxes = np.array([[w-b[0]-b[2], b[1], b[2], b[3]] for b in boxes])
    return image, boxes

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
# gboxes - ground-truth bunding boxes, dboxes - default bunding boxes
# Each box in gboxes is a tuple with 4 value: bx, by, bw, bh
# bx, by is the center point of the box (nomalized to image size)
# bw, by is the width, heigh of the box (nomalized to image size)
def encode(gboxes, scales=g_scales, sizes=g_sizes, aspects=g_aspects):
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
        # print("encode: index=%d, dbox=%s, gbox=%s" % (i, str([dx, dy, dw, dh]), str([gx, gy, gw, gh])))
    # For each default box, match a groud truth box with IOU > 0.5
    for i in range(len(dboxes)):
        if features[i][0] > 0:
            continue
        gbs = [gb for gb in gboxes if iou(gb, dboxes[i]) > 0.5]
        if len(gbs) > 0:
            ious = [iou(gb, dboxes[i]) for gb in gbs]
            dx, dy, dw, dh = dboxes[i]
            gx, gy, gw, gh = gbs[np.argmax(ious)]
            features[i] = [1, 0] + [(gx-dx)/dw, (gy-dy)/dh, np.log(gw/dw), np.log(gh/dh)]
            # print("encode: index=%d, dbox=%s, gbox=%s" % (i, str([dx, dy, dw, dh]), str([gx, gy, gw, gh])))
    return features

def get_dbox(feature_index, scales=g_scales, sizes=g_sizes, aspects=g_aspects):
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
    j = (feature_index - i*cols*aspect_num)//aspect_num
    k = feature_index - i*cols*aspect_num - j*aspect_num
    dx, dy, dw, dh = (j+0.5)/cols, (i+0.5)/rows, scale*np.sqrt(aspects[k]), scale/np.sqrt(aspects[k])
    return [dx, dy, dw, dh]

def decode(features, threshold=0.3, scales=g_scales, sizes=g_sizes, aspects=g_aspects):
    boxes = []
    for i in range(len(features)):
        c0, c1, x, y, w, h = features[i]
        max_c = max(c0, c1)
        c0, c1 = c0 - max_c, c1 - max_c
        c = np.exp(c0-max_c)/(np.exp(c0-max_c) + np.exp(c1-max_c))
        if c > threshold:
            dx, dy, dw, dh = get_dbox(i)
            gx, gy, gw, gh = x*dw+dx, y*dh+dy, np.exp(w)*dw, np.exp(h)*dh
            # print("decode: index=%d, dbox=%s, gbox=%s" % (i, str([dx, dy, dw, dh]), str([gx, gy, gw, gh])))
            boxes.append([gx, gy, gw, gh, c])
    return boxes

def precision(y_true, y_pred):
    c = K.softmax(y_pred[:,:,:2])
    P = K.cast(c[:,:,0] > 0.5, dtype='float32')
    TP = K.cast(y_true[:,:,0] > 0.5, dtype='float32')*P
    epsilon = 1e-7
    return K.sum(TP)/(K.sum(P) + epsilon)

def recall(y_true, y_pred):
    c = K.softmax(y_pred[:,:,:2])
    P = K.cast(y_true[:,:,0] > 0.5, dtype='float32')
    TP = K.cast(c[:,:,0] > 0.5, dtype='float32')*P
    FN = K.cast(c[:,:,1] > 0.5, dtype='float32')*P
    epsilon = 1e-7
    return K.sum(TP)/(K.sum(TP) + K.sum(FN) + epsilon)

def confidence_loss(y_true, y_pred):
    c = K.softmax(y_pred[:,:,:2])
    # NOTES: pos_num MUST NOT be 0 in training
    pos_num = tf.math.reduce_sum(tf.cast(y_true[:,:,0] > 0.5, tf.int32))
    pos_loss = -y_true[:,:,0]*K.log(c[:,:,0])
    neg_loss = -y_true[:,:,1]*K.log(c[:,:,1])
    neg_loss, _ = tf.math.top_k(tf.reshape(neg_loss, (-1,)), 3*pos_num)
    return (K.sum(pos_loss) + K.sum(neg_loss))/tf.cast(pos_num, tf.float32)

# Smooth L1-loss: 
# f(x) = 0.5*x^2 		if abs(x) <= 1  (Similar to L2-loss)
# f(x) = abs(x) - 0.5 	if abs(x) > 1	(Similar to L1-loss)
def smooth_l1_loss(x):
    abs_x = K.abs(x)
    m = K.cast(abs_x <= 1, dtype='float32')
    return K.sum(0.5*x*x*m + (abs_x - 0.5)*(1-m))

def localization_loss(y_true, y_pred):
    m = K.cast(y_true[:,:,0] > 0.5, dtype='float32')
    d = y_true[:,:,2:] - y_pred[:,:,2:]
    l1_loss = smooth_l1_loss(m*d[:,:,0]) + smooth_l1_loss(m*d[:,:,1]) + smooth_l1_loss(m*d[:,:,2]) + smooth_l1_loss(m*d[:,:,3])
    return l1_loss

def detection_loss(y_true, y_pred):
    return confidence_loss(y_true, y_pred) + localization_loss(y_true, y_pred)

def build_modle():
    base = MobileNetV2(input_shape=(160, 160, 3), alpha=0.5, include_top=False)
    trainable = False
    for layer in base.layers:
        if layer.name == "block_12_add":
            trainable = True
        layer.trainable = trainable
    # (10, 10, 288) -> (10, 10, 3*6)
    x1 = base.get_layer("block_13_expand_relu")
    x1 = layers.SeparableConv2D(256, 1, padding="same", activation="relu")(x1.output)
    x1 = layers.Conv2D(3*6, 3, padding="same")(x1)
    x1 = layers.Reshape((3*10*10, 6))(x1)
    # (5, 5, 480) -> (5, 5, 3*6)
    x2 = base.get_layer("block_14_expand_relu")
    x2 = layers.SeparableConv2D(256, 1, padding="same", activation="relu")(x2.output)
    x2 = layers.Conv2D(3*6, 3, padding="same")(x2)
    x2 = layers.Reshape((3*5*5, 6))(x2)
    # (5, 5, 480) -> (3, 3, 3*6)
    x3 = base.get_layer("block_15_expand_relu")
    x3 = layers.SeparableConv2D(256, 1, padding="same", activation="relu")(x3.output)
    x3 = layers.Conv2D(3*6, 3, padding="valid")(x3)
    x3 = layers.Reshape((3*3*3, 6))(x3)
    # (5, 5, 480) -> (3, 3, 256) -> (1, 1, 3*6)
    x4 = base.get_layer("block_16_expand_relu")
    x4 = layers.SeparableConv2D(256, 3, padding="valid", activation="relu")(x4.output)
    x4 = layers.Conv2D(3*6, 3, padding="valid")(x4)
    x4 = layers.Reshape((3*1*1, 6))(x4)
    model = models.Model(inputs=base.input, outputs=layers.Concatenate(axis=1)([x1,x2,x3,x4]))
    model.summary()
    return model

class DataGenerator(utils.Sequence):
    def __init__(self, data, image_size, feature_shape, batch_size):
        self.data = data
        self.image_size = image_size
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
        batch_size = end_index - start_index
        w, h = self.image_size
        batch_x = np.zeros((batch_size, h, w, 3))
        batch_y = np.zeros((batch_size,) + self.feature_shape)
        for i in range(start_index, end_index):
            sample = self.data[i]
            image = image_crop(Image.open(sample["image"]), sample["crop"])
            boxes = np.array(sample["boxes"])
            if sample["resize"]:
                image, boxes = image_resize(image, (w, h), boxes)
            if sample["flip"]:
                image, boxes = image_flip(image, boxes)
            boxes = [[(b[0]+0.5*b[2])/w, (b[1]+0.5*b[3])/h, b[2]/w, b[3]/h] for b in boxes]
            batch_x[i-start_index] = (np.array(image) - 127.5)/255
            batch_y[i-start_index] = encode(boxes)
        return batch_x, batch_y

def transform(data, num_sample, crop_size, output_size, resize_rate=0.5, flip_rate=0.5):
    result = []
    index = int(np.random.rand()*len(data))
    while (len(result) < num_sample):
        index = index+1 if index < len(data)-1 else 0
        sample = data[index]
        image = sample["image"]
        iw, ih = Image.open(image).size
        for i in range(11):
            resize = True if np.random.rand() < resize_rate else False
            if resize:
                cw, ch = crop_size
            else:
                cw, ch = output_size
            if iw < cw  or ih < ch:
                continue
            x = int((iw - cw)*np.random.rand())
            y = int((ih - ch)*np.random.rand())
            candidates = [b for b in sample["boxes"] if x < b[0]+b[2]/2 and b[0]+b[2]/2 < x+cw and y < b[1]+b[3]/2 and b[1]+b[3]/2 < y+ch]
            boxes = [[b[0]-x, b[1]-y, b[2], b[3]] for b in candidates if b[0] > x and b[1] > y and b[0]+b[2] < x+cw and b[1]+b[3] < y+ch]
            if len(candidates) == 0 or len(candidates) != len(boxes):
                continue
            flip = True if np.random.rand() < flip_rate else False
            result.append({"image": image, "crop": [x, y, x+cw, y+ch], "boxes": boxes, "resize": resize, "flip": flip})
            if len(result) % 100 == 0:
                print("croped %d samples" % len(result))
            break
    return result

def train_model(dataset_dir, model_path):
    crop_size = (240, 180)
    image_size = (160, 160)
    resize_rate = 1.0
    sample_num = 32000
    batch_size = 256
    feture_shape = (3*(10*10+5*5+3*3+1), 6)
    if Path(model_path).is_file():
        model = models.load_model(model_path, compile=False)
    else:
        model = build_modle()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    # NOTES: model path is like <prefix>_<epoch>.h5
    pos = model_path.rfind('_')
    model_prefix = model_path[:pos]
    offset = int(model_path[pos+1:-3])
    data = widerface.load_data(dataset_dir)
    data = widerface.select(data[0] + data[1], blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    data = transform(data, sample_num, crop_size, image_size, resize_rate)
    train_generator = DataGenerator(data, image_size, feture_shape, batch_size)
    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.00005), loss=detection_loss, metrics=[confidence_loss, localization_loss, precision, recall])
    for i in range(1, 1111):
        epochs = 10
        model.fit(train_generator, epochs=epochs, workers=4, use_multiprocessing=True)
        offset += epochs
        model_file = f"{model_prefix}_{offset}.h5"
        model.save(model_file)
        print(f"{model_file} saved")

def image_draw_boxes(image, boxes, color, width=2):
    d = ImageDraw.Draw(image)
    for box in boxes:
        [x, y, w, h] = box[:4]
        d.rectangle([x,y,x+w,y+h], outline=color, width=width)
        d.point([(x+w/2-0.5,y+h/2-0.5), (x+w/2+0.5,y+h/2-0.5), (x+w/2-0.5,y+h/2+0.5), (x+w/2+0.5,y+h/2+0.5)], fill=color)

def test_codec():
    crop_size = (240, 180 )
    image_size = (160, 160)
    data = widerface.load_data()
    data = widerface.select(data[0], blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    data = transform(data, 100, crop_size, image_size, 0.5)
    for sample in data:
        image = image_crop(Image.open(sample["image"]), sample["crop"])
        boxes = np.array(sample["boxes"])
        print("image: " + sample["image"])
        print("croped boxes: " + str(boxes))
        if sample["resize"]:
            image, boxes = image_resize(image, image_size, boxes)
            print("resized boxes: " + str(boxes))
        if sample["flip"]:
            image, boxes = image_flip(image, boxes)
            print("fliped boxes: " + str(boxes))
        w, h = image.size
        # print("before encode: boxes=%s" % str(boxes))
        boxes = [[(b[0]+0.5*b[2])/w, (b[1]+0.5*b[3])/h, b[2]/w, b[3]/h] for b in boxes]
        features = encode(boxes)
        boxes=decode(features)
        boxes = [[(b[0]-0.5*b[2])*w, (b[1]-0.5*b[3])*h, b[2]*w, b[3]*h] for b in boxes]
        # print("after decode: boxes=%s" % str(boxes))
        image_draw_boxes(image, boxes, (0,255,0))
        plt.imshow(image)
        plt.show()

def test_model(model_path):
    crop_size = (240, 180)
    image_size = (160, 160)
    feture_shape = (3*(10*10+5*5+3*3+1), 6)
    model = models.load_model(model_path, compile=False)
    model.summary()
    data = widerface.load_data()
    data = widerface.select(data[1], blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    data = transform(data, 9, crop_size, image_size, 0.8)
    generator = DataGenerator(data, image_size, feture_shape, 9)
    batch_x, batch_y = generator[0]
    y_pred = model.predict(batch_x)
    for i in range(len(y_pred)):
        ax = plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        ax.set_title("Sample %d" % i)
        ax.axis('off')
        ax.imshow(np.array(batch_x[i]*255+127.5, dtype='uint8'))
        W, H = image_size
        # ground truth bounding boxes
        boxes = decode(batch_y[i])
        boxes = [[(b[0]-0.5*b[2])*W, (b[1]-0.5*b[3])*H, b[2]*W, b[3]*H] for b in boxes]
        for box in boxes:
            (x, y, w, h) = box[:4]
            print("Sample %d, true_box=(%d,%d,%d,%d)" % (i, x, y, w, h))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        # predicted bounding boxes
        boxes = decode(y_pred[i])
        boxes = [[(b[0]-0.5*b[2])*W, (b[1]-0.5*b[3])*H, b[2]*W, b[3]*H, b[4]] for b in boxes]
        for box in boxes:
            (x, y, w, h, c) = box
            print("Sample %d, predicted_box=(%d,%d,%d,%d) confidence: %f" % (i, x, y, w, h, c))
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("train", "test"):
        print(f"Usage: {sys.argv[0]} train <dataset-dir> <xxx_epoch.h5>")
        print(f"   Or: {sys.argv[0]} test <xxx_epoch.h5>")
        sys.exit(-1)
    if sys.argv[1] == "train" and len(sys.argv) > 3:
        train_model(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "test" and  Path(sys.argv[2]).is_file():
        test_model(sys.argv[2])
    else:
        print("Invalid input arguments!")