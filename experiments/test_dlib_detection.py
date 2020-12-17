import os
import sys
import dlib
import cv2
from pathlib import Path

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

def test_detect_video(hog_detector, cnn_detector=None, videofile=0):
    c = cv2.VideoCapture(videofile)
    while True:
        ok, img = c.read()
        if not ok:
            break
        for r in hog_detector.detect(img):
            x, y, w, h = r.left(), r.top(), r.width(), r.height()
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2, cv2.LINE_AA)
        if cnn_detector is not None:
            for r in cnn_detector.detect(img):
                x, y, w, h = r.left(), r.top(), r.width(), r.height()
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Image", img)
        if cv2.waitKey(3) == ord('q'):
            break

def test_detect_image(hog_detector, cnn_detector, imagedir):
    files = os.listdir(imagedir)
    paths = [f"{imagedir}/{f}" for f in files]
    imagefiles = [ p for p in paths if cv2.imread(p) is not None]
    for f in imagefiles:
        img = cv2.imread(f)
        for r in hog_detector.detect(img):
            x, y, w, h = r.left(), r.top(), r.width(), r.height()
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2, cv2.LINE_AA)
        if cnn_detector is not None:
            for r in cnn_detector.detect(img):
                x, y, w, h = r.left(), r.top(), r.width(), r.height()
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, cv2.LINE_AA)
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
    modelfile = Path(__file__).absolute().parents[1].as_posix() + "/models/mmod_human_face_detector.dat"
    hog_detector = HOGDetector()
    cnn_detector = CNNDetector(modelfile) if Path(modelfile).is_file() else None
    if len(sys.argv) == 2:
        test_detect_image(hog_detector, cnn_detector, sys.argv[1])
    else:
        test_detect_video(hog_detector, cnn_detector)