import os
import pickle
import cv2
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/fddb"
ellipse_file = dataset_dir + "/fddb_ellipse.pkl"
rectangle_file = dataset_dir + "/fddb_rectangle.pkl"

def _convert(data):
    result = []
    for sample in data:
        img = cv2.imread(sample["image"])
        boxes = sample["boxes"]
        rects = []
        for b in boxes:
            ra, rb, angle, cx, cy, score = b
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.ellipse(mask, (int(cx), int(cy)), (int(ra), int(rb)), 180*angle/3.1415926535, 0, 360, (255,255,255))
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # NOTES:
            # we only use first contour's bounding box
            r = cv2.boundingRect(contours[0])
            rects.append([r[0], r[1], r[2], r[3]])
        result.append({"image": sample["image"], "boxes": rects})
    return result

def _parse(fddb_folds, original_pics):
    """
    see http://vis-www.cs.umass.edu/fddb/fddb.pdf
    """
    result = []
    for i in range(10):
        label_file = fddb_folds + "/FDDB-fold-%02d-ellipseList.txt" % (i+1,)
        with open(label_file, "r") as f:
            lines = f.readlines()
        j = 0
        while j < len(lines):
            image_file = original_pics + "/" + lines[j].strip() + ".jpg"
            num_box = int(lines[j+1].strip())
            boxes = []
            for k in range(j+2, j+2+num_box):
                box = lines[k].strip().split()
                box = [float(b) for b in box]
                boxes.append(box)
            assert os.path.exists(image_file)
            assert len(boxes) > 0
            result.append({"image":image_file, "boxes":boxes})
            j = j+2+num_box
    return result
            
def _init_data(root):
    fddb_folds = root + "/FDDB-folds"
    original_pics = root + "/originalPics"
    assert (os.path.exists(fddb_folds) and os.path.exists(original_pics))
    dataset = _parse(fddb_folds, original_pics)
    print("creating ellipse pickle file ...")
    with open(ellipse_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("saved as " + ellipse_file)
    dataset = _convert(dataset)
    print("creating rectangle pickle file ...")
    with open(rectangle_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("saved as " + rectangle_file)

def load_data(root=dataset_dir, rectangle=True):
    """
    FDDB: Face Detection Data Set and Benchmark
    see http://vis-www.cs.umass.edu/fddb/index.html
    """
    assert os.path.exists(root)
    if rectangle:
        save_file = rectangle_file
    else:
        save_file = ellipse_file
    if not os.path.exists(save_file):
        _init_data(root)
    with open(save_file, "rb") as f:
        dataset = pickle.load(f)
    return dataset

# if __name__ == "__main__":
#     data = load_data()
#     for s in data:
#         img = cv2.imread(s["image"])
#         for b in s["boxes"]:
#             cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255, 0, 0), 2)
#         cv2.imshow("image", img)
#         if cv2.waitKey(3000) == ord('q'):
#             break


