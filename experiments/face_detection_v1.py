import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import models, layers, utils, optimizers, callbacks, backend as K

sys.path.append(Path(__file__).absolute().parents[1].as_posix())
from datasets import widerface

g_scales = [0.3, 0.5, 0.7, 0.9]
g_sizes = [[10,10], [5,5], [3,3], [1,1]]
g_aspects = [0.5, 1.0, 1.5]

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
def encode(gboxes, scales=g_scales, sizes=g_sizes, aspects=g_aspects):
    dboxes = []
    for i in range(len(scales)):
        dboxes.append(generate_dboxes(scales[i], sizes[i], aspects))
    dboxes = np.concatenate(dboxes)
    features = np.zeros((dboxes.shape[0], 6), dtype='float32')
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

def decode(features, scales=g_scales, sizes=g_sizes, aspects=g_aspects):
    boxes = []
    for i in range(len(features)):
        c0, c1, x, y, w, h = features[i]
        max_c = max(c0, c1)
        c0, c1 = c0 - max_c, c1 - max_c
        c = np.exp(c0-max_c)/(np.exp(c0-max_c) + np.exp(c1-max_c))
        if c > 0.28:
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
    pos_num = K.sum(K.cast(y_true[:,:,0] > 0.5, dtype='int32'))
    epsilon = 1e-7
    pos_loss = -y_true[:,:,0]*K.log(c[:,:,0]+epsilon)
    neg_loss = -y_true[:,:,1]*K.log(c[:,:,1]+epsilon)
    neg_loss, _ = tf.math.top_k(tf.reshape(neg_loss, (-1,)), 3*pos_num+1)
    return (K.sum(pos_loss) + K.sum(neg_loss))/K.cast(pos_num+1, dtype="float32")

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
    def __init__(self, data, input_shape, output_shape, batch_size, batches, as_training=True):
        self.data = data
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.batches = batches
        self.as_training = as_training
        self.samples = self.generate_samples()
        self.batch_x = np.zeros((self.batch_size,) + self.input_shape, dtype='float32')
        self.batch_y = np.zeros((self.batch_size,) + self.output_shape, dtype='float32')
        self.epoch = 0
    def __len__(self):
        return self.batches
    def __getitem__(self, batch_index):
        if batch_index >= self.batches:
            raise IndexError()
        if batch_index == 0:
            if self.as_training and self.epoch > 0 and self.epoch % 10 == 0:
                self.samples = self.generate_samples()
            shuffle = np.random.permutation(np.arange(len(self.samples)))
            self.samples = [self.samples[i] for i in shuffle]
        self.epoch += 1
        w, h = (self.input_shape[1], self.input_shape[0])
        for i, sample in enumerate(self.samples[self.batch_size * batch_index: self.batch_size * (batch_index+1)]):
            self.batch_x[i] = np.array((np.array(sample["image"]) - 127.5)/255.0, dtype='float32')
            self.batch_y[i] = encode([[(b[0]+0.5*b[2])/w, (b[1]+0.5*b[3])/h, b[2]/w, b[3]/h] for b in sample["boxes"]])
        return self.batch_x, self.batch_y
    def generate_samples(self):
        crop_sizes = [(208, 160), (192, 160), (176, 160), (160, 176), (160, 192), (160, 208)]
        target_size = (self.input_shape[1], self.input_shape[0])
        crop_rate = 0.5
        flip_rate = 0.5
        index = -1
        result = []
        while (len(result) < self.batch_size * self.batches):
            index = index + 1 if index + 1 < len(self.data) else 0
            item = self.data[index]
            img = Image.open(item["image"])
            iw, ih = img.size
            for _ in range(11):
                # crop
                if np.random.rand() < crop_rate:
                    cw, ch = crop_sizes[np.random.randint(6)]
                else:
                    cw, ch = target_size
                if iw < cw  or ih < ch:
                    continue
                x = int((iw - cw)*np.random.rand())
                y = int((ih - ch)*np.random.rand())
                candidates = [b for b in item["boxes"] if x < b[0]+b[2]/2 and b[0]+b[2]/2 < x+cw and y < b[1]+b[3]/2 and b[1]+b[3]/2 < y+ch]
                boxes = [[b[0]-x, b[1]-y, b[2], b[3]] for b in candidates if b[0] > x and b[1] > y and b[0]+b[2] < x+cw and b[1]+b[3] < y+ch]
                if len(candidates) == 0 or len(candidates) != len(boxes):
                    continue
                img = img.crop([x, y, x+cw, y+ch])
                # resize
                if img.size != target_size:
                    sx, sy = target_size[0] / img.size[0], target_size[1] / img.size[1]
                    img = img.resize(target_size, Image.LINEAR)
                    boxes = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes]
                # flip
                if np.random.rand() < flip_rate:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    boxes = [[img.size[0]-b[0]-b[2], b[1], b[2], b[3]] for b in boxes]
                result.append({"image": img, "boxes": boxes})
                if len(result) % 100 == 0:
                    print("%6d samples generated" % len(result))
                break
        return result

def save_logs(save_path, epoch, logs):
    if not Path(save_path).exists():
        with open(save_path, "wt") as f:
            keys = logs.keys()
            title = ("%5s" + "%20s"*len(keys)) % (("epoch",) + tuple(keys))
            f.write(title)
    with open(save_path, "at") as f:
        values = logs.values()
        line = ("\n%5d" + "%20.6f"*len(values)) % ((epoch,) + tuple([v for v in values]))
        f.write(line)

class SaveModel(callbacks.Callback):
    def __init__(self, offset, prefix):
        super(SaveModel, self).__init__()
        self.offset = offset + 1 if offset > 0 else offset
        self.prefix = prefix
    def on_epoch_end(self, epoch, logs=None):
        log_file = f"{self.prefix}training.log"
        model_file = f"{self.prefix}{self.offset + epoch}.h5"
        save_logs(log_file, self.offset + epoch, logs)
        if epoch % 10 == 0:
            self.model.save(model_file)

def train_model(dataset_dir, model_path):
    if Path(model_path).is_file():
        model = models.load_model(model_path, compile=False)
        log_dir = Path(model_path).absolute().parent.as_posix()
        p = model_path.rfind('_')
        prefix = model_path[:p + 1]
        offset = int(model_path[p + 1: -3])
    else:
        model = build_modle()
        log_dir = model_path
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        prefix = log_dir + "/" + Path(model_path).name + "_"
        offset = 0
    model.summary()
    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.00005), loss=detection_loss,
                  metrics=[confidence_loss, localization_loss, precision, recall])
    # model.compile(optimizer="adam", loss=detection_loss, metrics=[confidence_loss, localization_loss, precision, recall])
    input_shape = model.input_shape[1:]
    output_shape = model.output_shape[1:]
    train_data, validate_data, _ = widerface.load_data(dataset_dir)
    train_data = widerface.select(train_data, blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    validate_data = widerface.select(validate_data, blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    train_generator = DataGenerator(train_data, input_shape, output_shape, batch_size=128, batches=200, as_training=True)
    validate_generator = DataGenerator(validate_data, input_shape, output_shape, batch_size=160, batches=10, as_training=False)
    model.fit(train_generator, validation_data=validate_generator, epochs=111, callbacks=[SaveModel(offset, prefix)], workers=4, use_multiprocessing=True)

def test_codec():
    data = widerface.load_data()
    data = widerface.select(data[1], blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    model = build_modle()
    model.summary()
    generator = DataGenerator(data,  model.input_shape[1:], model.output_shape[1:], 16, 16)
    for sample in generator.samples:
        image, boxes = sample["image"], sample["boxes"]
        print(f"image size: {image.size}")
        print(f"crop boxes: {boxes}")
        w, h = image.size
        print("before encode: boxes=%s" % str(boxes))
        boxes = [[(b[0]+0.5*b[2])/w, (b[1]+0.5*b[3])/h, b[2]/w, b[3]/h] for b in boxes]
        features = encode(boxes)
        boxes = decode(features)
        boxes = [[(b[0]-0.5*b[2])*w, (b[1]-0.5*b[3])*h, b[2]*w, b[3]*h] for b in boxes]
        print("after decode: boxes=%s" % str(boxes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for x, y, w, h in boxes:
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color=(0,255,0))
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        cv2.waitKey(30000)

def test_model(dataset_dir, model_path):
    from matplotlib import patches, pyplot as plt
    model = models.load_model(model_path, compile=False)
    model.summary()
    input_shape = model.input_shape[1:]
    output_shape = model.output_shape[1:]
    data = widerface.load_data(dataset_dir)
    data = widerface.select(data[1], blur="0", illumination="0", occlusion="0", pose="0", invalid="0", min_size=32)
    generator = DataGenerator(data, input_shape, output_shape, batch_size=9, batches=1, as_training=False)
    batch_x, batch_y = generator[0]
    y_pred = model.predict(batch_x)
    for i in range(len(y_pred)):
        ax = plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        ax.set_title("Sample %d" % i)
        ax.axis('off')
        ax.imshow(np.array(batch_x[i]*255+127.5, dtype='uint8'))
        W, H = input_shape[1], input_shape[0]
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
    if len(sys.argv) !=4 or sys.argv[1] not in ("train", "test"):
        print(f"Usage: {sys.argv[0]} train <dataset-dir> <xxx_epoch.h5>")
        print(f"   Or: {sys.argv[0]} test  <dataset-dir> <xxx_epoch.h5>")
        sys.exit(-1)
    if sys.argv[1] == "train":
        train_model(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "test":
        test_model(sys.argv[2], sys.argv[3])
    else:
        print("Invalid input arguments!")