import os
import sys
import dlib
import cv2

class CNNDetector(object):
    def __init__(self, filename):
        self.detector = dlib.cnn_face_detection_model_v1(filename)
    def detect(self, image):
        mmod_rects = self.detector([image],5)
        if len(mmod_rects) > 0:
            return [mr.rect for mr in  mmod_rects[0]]
        return []

class HOGDetector(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        return self.detector(image)

def test_detect_video(detector, videofile=0):
    c = cv2.VideoCapture(videofile)
    while True:
        ok, img = c.read()
        if not ok:
            break
        for r in detector.detect(img):
            x, y, w, h = r.left(), r.top(), r.width(), r.height()
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(3) == ord('q'):
            break

def test_detect_image(detector, imagedir):
    files = os.listdir(imagedir)
    paths = [f"{imagedir}/{f}" for f in files]
    imagefiles = [ p for p in paths if cv2.imread(p) is not None]
    for f in imagefiles:
        img = cv2.imread(f)
        for r in detector.detect(img):
            x, y, w, h = r.left(), r.top(), r.width(), r.height()
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(3000) == ord('q'):
            break

def test_cnn_detector():
    modelfile = os.path.dirname(os.path.abspath(__file__)) + "/models/mmod_human_face_detector.dat"
    detector = CNNDetector(modelfile)
    test_detect_video(detector)

def test_hog_detector():
    detector = HOGDetector()
    test_detect_video(detector)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(f"{sys.argv[0]} [img-dir]")
        sys.exit(-1)
    modelfile = os.path.dirname(os.path.abspath(__file__)) + "/models/mmod_human_face_detector.dat"
    if os.path.isfile(modelfile):
        detector = CNNDetector(modelfile)
    else:
        detector = HOGDetector()
    if len(sys.argv) == 2:
        test_detect_image(detector, sys.argv[1])
    else:
        test_detect_video(detector)