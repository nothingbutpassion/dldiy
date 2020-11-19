import os
import sys
import getopt
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

g_scales = [0.3, 0.5, 0.7, 0.9]
g_sizes = [[10,10], [5,5], [3,3], [1,1]]
# NOTES: 
# v1 model: g_aspects = [0.5, 0.8, 1.0]
# v2 model: g_aspects = [0.5, 1.0, 1.5]
g_aspects = [0.5, 0.8, 1.0]

# NOTES:
# This function is tested on TF 2.0
# see https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_keras_model
def keras_to_tflite_v2(keras_file, tflite_file):
    model = keras.models.load_model(keras_file, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)

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

def decode(features, threshold=0.5, scales=g_scales, sizes=g_sizes, aspects=g_aspects):
    boxes = []
    for i in range(len(features)):
        c0, c1, x, y, w, h = features[i]
        max_c = max(c0, c1)
        c0, c1 = c0 - max_c, c1 - max_c
        c = np.exp(c0-max_c)/(np.exp(c0-max_c) + np.exp(c1-max_c))
        if c > threshold:
            dx, dy, dw, dh = get_dbox(i)
            gx, gy, gw, gh = x*dw+dx, y*dh+dy, np.exp(w)*dw, np.exp(h)*dh
            print("decode: index=%d, dbox=%s, gbox=%s" % (i, str([dx, dy, dw, dh]), str([gx, gy, gw, gh])))
            boxes.append([gx, gy, gw, gh, c])
    return boxes

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w = max(0, min(x1+w1/2, x2+w2/2) - max(x1-w1/2, x2-w2/2))
    h = max(0, min(y1+h1/2, y2+h2/2) - max(y1-h1/2, y2-h2/2))
    I = w*h
    U = w1*h1 + w2*h2 - I
    return I/U

def nms(boxes, threshold=0.01):
    candidates = boxes
    selected = []
    while len(candidates) > 0:
        scores = [b[4] for b in candidates]
        max_score_index = np.argmax(scores)
        b = candidates[max_score_index]
        selected.append(b)
        del candidates[max_score_index]
        i = 0
        while i < len(candidates):
            if iou(b[:4], candidates[i][:4]) > threshold:
                del candidates[i]
                i -= 1
            i += 1
    return selected

class Detector(object):
    def __init__(self, tflite_file):
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=tflite_file)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        self.interpreter = interpreter
        self.input_shape = input_details[0]['shape']
        self.input_index = input_details[0]['index']
        self.output_shape = output_details[0]['shape']
        self.output_index = output_details[0]['index']
        print("input shape: %s, index=%d" % (str(self.input_shape), self.input_index))
        print("output shape: %s, index=%d" % (str(self.output_shape), self.output_index))
    
    def detect(self, image):
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[2]))
        image = ((image-127.5)/255).astype('float32') 
        image = image.reshape((1,)+image.shape)
        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        features = self.interpreter.get_tensor(self.output_index)
        features = features.reshape(features.shape[1:])
        boxes = decode(features, 0.5)
        boxes = nms(boxes)
        boxes = [[(b[0]-0.5*b[2])*w, (b[1]-0.5*b[3])*h, b[2]*w, b[3]*h, b[4]] for b in boxes]
        return boxes


def test_detection(tflite_file):
    detector = Detector(tflite_file)
    c = cv2.VideoCapture(0)
    while True:
        ok, img = c.read()
        if not ok:
            break
        boxes = detector.detect(img)
        for box in boxes:
            (x, y, w, h, p) = box
            print("detected box: %s, confidence: %f " % (str(box), p))
            cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(20) == ord('q'):
            break

def parse_arguments():
    try:
	    opts, _ = getopt.getopt(sys.argv[1:], "hi:v:", ["help", "input=", "version="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(-1)
    model_version = "1"    
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/models/face_model_v1_2100.h5"
    for o, a in opts:
        if o in ("-h", "--help"):
            print("Usage: <this-app> -m <model-version> -i <keralh5-file|tflite-file>")
            sys.exit(0)
        elif o in ("-v", "--version"):
            model_version = a
        elif o in ("-i", "--input"):
            model_path = a
    return model_path, model_version

if __name__ == "__main__":
    model_path, model_version = parse_arguments()
    print("model file: " + model_path)
    print("model version: " + model_version)
    if model_version == "2":
        g_aspects = [0.5, 1.0, 1.5]
    tflite_file = model_path
    if model_path[-3:] == ".h5":
        tflite_file = model_path[-3:] + ".tflite"
        keras_to_tflite_v2(model_path, tflite_file)
    test_detection(tflite_file)
