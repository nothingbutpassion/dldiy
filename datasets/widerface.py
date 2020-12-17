import os
import zipfile
import pickle
from pathlib import Path

def _extract_files(root):
    for name in ["wider_face_split", "WIDER_train", "WIDER_val", "WIDER_test"]:
        if not (Path(root)/name).is_dir():
            zip_file = root + "/" + name + ".zip"
            with zipfile.ZipFile(zip_file) as f:
                print("extracting %s ..." % zip_file)
                f.extractall(root)
                print("saved as %s" % root + "/" + name)

def _parse_bbx(root, image_dir, bbox_file):
    data = []
    with open(root + "/" + bbox_file, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            sample = {}
            sample["image"] = root + "/" + image_dir +"/" + lines[i].strip()
            sample["boxes"] = []
            boxes_num = int(lines[i+1])
            for j in range(i+2, i+2+boxes_num):
                box = lines[j].split()
                sample["boxes"].append(box)
            if len(sample["boxes"]) > 0:
                data.append(sample)
            i = i + 2 + boxes_num
    return data

def _parse_filelist(root, image_dir, list_file):
    data = []
    with open(root + "/" + list_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            path = root + "/" + image_dir + "/" + line.strip()
            data.append(path)
    return data

def _init_data(root):
    _extract_files(root)
    train_data = _parse_bbx(root, "WIDER_train/images","/wider_face_split/wider_face_train_bbx_gt.txt")
    val_data = _parse_bbx(root, "WIDER_val/images", "/wider_face_split/wider_face_val_bbx_gt.txt")
    test_data = _parse_filelist(root, "WIDER_test/images", "wider_face_split/wider_face_test_filelist.txt")
    return (train_data, val_data, test_data)


def select(data, blur=None, expression=None, illumination=None, invalid=None, occlusion=None, pose=None, min_size=12):
    """Attached the mappings between attribute names and label values.
    blur:
    clear->0
    normal blur->1
    heavy blur->2

    expression:
    typical expression->0
    exaggerate expression->1

    illumination:
    normal illumination->0
    extreme illumination->1

    occlusion:
    no occlusion->0
    partial occlusion->1
    heavy occlusion->2

    pose:
    typical pose->0
    atypical pose->1

    invalid:
    false->0(valid image)
    true->1(invalid image)

    The format of txt ground truth.
    File name
    Number of bounding box
    x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
    """
    result = []
    for sample in data:
        image = sample["image"]
        bboxes = []
        zeros = 0
        for box in sample["boxes"]:
            b = [float(s) for s in box[:4]]
            attributes = box[4:]
            requirements = [blur, expression, illumination, invalid, occlusion, pose]
            passed = True
            for i in range(len(attributes)):
                if requirements[i] and not (attributes[i] in requirements[i]):
                    passed = False
                    break
            if not passed:
                continue
            # NOTES:
            # some box' w, h is 0 (or too small), should exclude
            if b[2] < 1 or b[3] < 1:
                zeros += 1
            if b[2] >= min_size and b[3] >= min_size:
                bboxes.append(b)
        if len(bboxes) > 0 and len(bboxes) == len(sample["boxes"]) - zeros:
            result.append({"image": image, "boxes": bboxes})
    return result

def load_data(dataset_dir=None):
    """WIDERFace: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
    """
    root = (Path(__file__).parent/"widerface").as_posix()
    if dataset_dir is not None:
        root = Path(dataset_dir).as_posix()
    assert Path(root).is_dir()
    pickle_file = f"{root}/winderface.pkl"
    if not Path(pickle_file).is_file():
        dataset = _init_data(root)
        with open(pickle_file, "wb") as f:
            pickle.dump(dataset, f, -1)
            print("saved as {save_file}")
    else:
        with open(pickle_file, "rb") as f:
            dataset = pickle.load(f)
    return dataset

