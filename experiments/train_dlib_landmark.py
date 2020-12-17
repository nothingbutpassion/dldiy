import os
import sys
import cv2
import dlib
import numpy as np
import tensorflow as tf
from xml.etree import ElementTree as ET
from pathlib import Path

sys.path.append(Path(__file__).absolute().parents[1].as_posix())
import datasets.w300 as w300

g_scales = [0.3, 0.5, 0.7, 0.9]
g_sizes = [[10,10], [5,5], [3,3], [1,1]]
g_aspects = [0.5, 0.8, 1.0]

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
        interpreter = tf.lite.Interpreter(model_path=tflite_file)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        self.interpreter = interpreter
        self.input_shape = input_details[0]['shape']
        self.input_index = input_details[0]['index']
        self.output_shape = output_details[0]['shape']
        self.output_index = output_details[0]['index']
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

def landmark_box(landmarks):
    min_x, min_y, max_x, max_y = 111111, 111111, 0, 0
    for (x, y) in landmarks:
        min_x, min_y, max_x, max_y = min(x, min_x), min(y, min_y), max(x, max_x), max(y, max_y)
    return [min_x, min_y, max_x - min_x  , max_y - min_y ]

def landmark_center(landmarks):
    cx, cy = 0, 0
    for (x, y) in landmarks:
        cx += x
        cy += y
    cx /= len(landmarks)
    cy /= len(landmarks)
    return [cx, cy]

def generate_landmarks_data(dataset, detector):
    first_detected = 0
    second_detected = 0
    lossed = 0
    hitted_dataset = []
    for data in dataset:
        img = cv2.imread(data["image"])
        boxes = detector.detect(img)
        detected = False
        if len(boxes) >= 1:
            cx, cy = landmark_center(data["landmarks"])
            x, y, w, h = landmark_box(data["landmarks"])
            firsts = [ b for b in boxes if x < b[0]+b[2]/2 and b[0]+b[2]/2 < x+w and y < b[1]+b[3]/2 and b[1]+b[3]/2 < y+h]
            seconds = [ b for b in boxes if b[0] < cx and cx < b[0]+b[2] and b[1] < cy and cy < b[1]+b[3]]
            if len(firsts) == 1 or len(seconds) == 1:
                x, y, w, h = firsts[0][:4] if len(firsts) == 1 else seconds[0][:4]
                data["box"] = [int(x), int(y), int(w), int(h)]
                hitted_dataset.append(data)
                detected = True   
                first_detected += 1
        if not detected:
            x, y, w, h = landmark_box(data["landmarks"])
            ih, iw = img.shape[:2]
            for s in [1, 1/2, 1/4, 1/8]:
                x1, y1 = max(0, x-s*w), max(0, y-s*h)
                x2, y2 = min(iw, x+w+s*w), min(ih, y+h+s*h)
                cx, cy = landmark_center(data["landmarks"])
                boxes = detector.detect(img[int(y1):int(y2),int(x1):int(x2),:])
                if len(boxes) >= 1:
                    # NOTES:
                    # boxes is relative to (x1, y1)
                    boxes = [[x1+b[0],y1+b[1],b[2],b[3]] for b in boxes]
                    firsts = [ b for b in boxes if x < b[0]+b[2]/2 and b[0]+b[2]/2 < x+w and y < b[1]+b[3]/2 and b[1]+b[3]/2 < y+h]
                    seconds = [ b for b in boxes if b[0] < cx and cx < b[0]+b[2] and b[1] < cy and cy < b[1]+b[3]]
                    if len(firsts) == 1 or len(seconds) == 1:
                        x, y, w, h = firsts[0][:4] if len(firsts) == 1 else seconds[0][:4]
                        data["box"] = [int(x), int(y), int(w), int(h)]
                        hitted_dataset.append(data)
                        detected = True
                        second_detected += 1
                        break
        if not detected:
            x1, y1 = max(0, x-w/10), max(0, y-h/10)
            x2, y2 = min(iw, x+w+w/10), min(ih, y+h+h/10)
            data["box"] = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            lossed += 1
        print("first detected %d，second detected：%d, lossed: %d" % (first_detected, second_detected, lossed))
    return hitted_dataset

# xml formats:
# </dataset>
#     </images>
#         <image file='2009_004587.jpg'>
#             <box top='280' left='266' width='63' height='63'>
#                 <part name='01' x='299' y='329'/>
#                 ...
#                 <part name='67' x='299' y='329'/>
#             </box>
#             ...
#         </image>
#     </images>
# </dataset>
def generate_landmarks_trainning_file(data, landmarks_trainning_file):
    files = set([d["image"] for d in data])
    dataset = ET.Element("dataset")
    images = ET.SubElement(dataset, "images")
    for f in files:
        image = ET.SubElement(images, "image", attrib={"file": f})
        for b, parts in [ (d["box"], d["landmarks"]) for d in data if d["image"] == f]:
            box = ET.SubElement(image, "box", attrib={"left": str(b[0]), "top": str(b[1]), "width": str(b[2]), "height": str(b[3])})
            for i, (x, y) in enumerate(parts):
                ET.SubElement(box, "part", attrib={"name": "%02d" % i, "x": str(int(x)), "y": str(int(y))})
    tree = ET.ElementTree(dataset)
    tree.write(landmarks_trainning_file, encoding="utf8", xml_declaration=True)
    print("%s generated", landmarks_trainning_file)

def train_landmarks(landmarks_trainning_file, landmarks_model_file):
    options = dlib.shape_predictor_training_options()
    options.oversampling_amount = 20
    options.num_test_splits = 100
    options.feature_pool_size = 500
    options.feature_pool_region_padding = 0.05
    options.oversampling_translation_jitter = 0.1
    options.landmark_relative_padding_mode = True
    options.be_verbose = True
    dlib.train_shape_predictor(landmarks_trainning_file, landmarks_model_file, options)
    print("Training accuracy: {}".format(
        dlib.test_shape_predictor(landmarks_trainning_file, landmarks_model_file)))

def test_landmarks_predictor(detecor_model_file, landmarks_model_file):
    detector = Detector(detecor_model_file)
    predictor = dlib.shape_predictor(landmarks_model_file)
    camera = cv2.VideoCapture(0)
    while True:
        ok, img = camera.read()
        if not ok:
            break
        boxes = detector.detect(img)
        if len(boxes) >= 1:
            x, y, w, h = boxes[0][:4]
            cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), (0, 255, 0), 1)
            shape = predictor(img, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
            for p in shape.parts():
                cv2.circle(img, (p.x, p.y), 1, (0, 0, 255), 2)
        cv2.imshow("image", img)
        if cv2.waitKey(30) == ord('q'):
            break

if __name__ == "__main__":
    project_dir = Path(__file__).absolute().parents[1].as_posix()
    detector_model_file = os.path.join(project_dir, "models", "face_model_v1_2100.tflite")
    landmark_trainning_file = os.path.join(project_dir, "models", "face_model_v1_2100_w300_tranning_landmarks.xml")
    landmark_model_file = os.path.join(project_dir, "models", "face_model_v1_2100_w300_landmarks.dat")
    # generate landmarks trainning file
    detector = Detector(detector_model_file)
    data = w300.load_data()
    data = generate_landmarks_data(data[0]+data[1], detector)
    generate_landmarks_trainning_file(data, landmark_trainning_file)
    # train landmarks model
    train_landmarks(landmark_trainning_file, landmark_model_file)
    # test trained model
    test_landmarks_predictor(detector_model_file, landmark_trainning_file)






                


