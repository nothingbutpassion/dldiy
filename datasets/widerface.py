# coding: utf-8
import os
import zipfile
import pickle
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/widerface"
save_file = dataset_dir + "/winderface.pkl"

def _extract_files(root):
    for name in ["wider_face_split", "WIDER_train", "WIDER_val", "WIDER_test"]:
        if not os.path.exists(root + "/" + name):
            zip_name = root + "/" + name + ".zip"
            assert(os.path.exists(zip_name))
            with zipfile.ZipFile(zip_name) as f:
                print("extracting %s ..." % zip_name)
                f.extractall(root)
                print("saved as %s" % root + "/" + name)

def _parse_bbx(root, image_dir, bbox_file):
    data = []
    with open(root + "/" + bbox_file, "r") as f:
        lines = f.readlines()
        i = 0
        while(i < len(lines)):
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

def init_data(root):
    _extract_files(root)
    train_data = _parse_bbx(root, "WIDER_train/images","/wider_face_split/wider_face_train_bbx_gt.txt")
    val_data = _parse_bbx(root, "WIDER_train/images", "/wider_face_split/wider_face_val_bbx_gt.txt")
    test_data = _parse_filelist(root, "WIDER_test/images", "wider_face_split/wider_face_test_filelist.txt")
    dataset = (train_data, val_data, test_data)
    print("creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("saved as " + save_file) 

def select(data, blur=None, expression=None, illumination=None, invalid=None, occlusion=None, pose=None):
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
        for boxes in sample["boxes"]:
            box = [float(s) for s in boxes[:4]]
            attributes = boxes[4:]
            requirements = [blur, expression, illumination, invalid, occlusion, pose]
            passed = True
            for i in range(len(attributes)):
                if requirements[i] and attributes[i] != requirements[i]:
                    passed = False
                    break
            if passed:
                bboxes.append(box)
        if len(bboxes) > 0 and len(bboxes) == len(sample["boxes"]) :
            result.append({"image": image, "boxes": bboxes})
    return result

def load_data(root=dataset_dir):
    """WIDERFace: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
    """
    assert(os.path.exists(root))
    if not os.path.exists(save_file):
        init_data(root)
    with open(save_file, "rb") as f:
        dataset = pickle.load(f)
    return dataset

