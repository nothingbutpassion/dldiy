import os
import zipfile
import pickle
import numpy as np
import cv2

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "w300"
save_file = dataset_dir + os.path.sep + "300w.pkl"

# File content format
# version: 1
# n_points: 68
# {
# 324.884 298.025
# ...
# 530.151 452.817
# }
def _parse_data(dir, file_prefix):
    data = []
    for i in range(1, 301):
        pts_file = os.path.join(dir, file_prefix + ("%03d.pts" % i))
        png_file = os.path.join(dir, file_prefix + ("%03d.png" % i))
        item = {"image": png_file, "landmarks":[]}
        with open(pts_file, "r") as f:
            lines = f.readlines()
            fs = lines[1].strip().split()
            assert(fs[1] == "68")
            assert(len(lines) >= 72)
            for i in range(3, 71):
                fs = lines[i].strip().split()
                item["landmarks"].append([float(fs[0]), float(fs[1])])
        assert(os.path.exists(png_file))
        data.append(item)
    return data


def _parse_files(root):
    indoor_dir = os.path.join(root, "300W", "01_Indoor")
    outdoor_dir = os.path.join(root, "300W", "02_Outdoor")
    indoor_data = _parse_data(indoor_dir, "indoor_")
    outdoor_data = _parse_data(outdoor_dir, "outdoor_")
    return indoor_data, outdoor_data


def _extract_files(root):
    if not os.path.exists(root + os.path.sep + "300W"):
        zips = ["300w.zip.001", "300w.zip.002", "300w.zip.003", "300w.zip.004"]
        zip_name = root + os.path.sep + "300w.zip"
        with open(zip_name, 'wb') as zf:
            for f in zips:
                with open(root + os.path.sep + f, 'rb') as tmpf:
                    zf.write(tmpf.read())
        with zipfile.ZipFile(zip_name) as f:
            print("extracting %s ..." % zip_name)
            f.extractall(root)
            print("saved as %s" % root + os.path.sep + "300W")


def init_data(root):
    _extract_files(root)
    indoor_data, outdoor_data = _parse_files(root)
    dataset = (indoor_data, outdoor_data)
    print("creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("saved as " + save_file) 


def load_data(root=dataset_dir):
    """
    300 Faces In-the-Wild Challenge (300-W), ICCV 2013
    see https://ibug.doc.ic.ac.uk/resources/300-W/
    """
    assert(os.path.exists(root))
    if not os.path.exists(save_file):
        init_data(root)
    with open(save_file, "rb") as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == "__main__":
    data = load_data()
    for s in data[0]:
        img = cv2.imread(s["image"])
        for (x, y) in s["landmarks"]:
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 2)
        cv2.imshow("image", img)
        if cv2.waitKey(3000) == ord('q'):
            break