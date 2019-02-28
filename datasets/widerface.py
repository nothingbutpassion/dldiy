# coding: utf-8
import os
import zipfile
import pickle
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/winderface"
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
            sample["bboxs"] = []
            bboxs_num = int(lines[i+1])
            for j in range(i+2, i+2+bboxs_num):
                bbox = lines[j].split()
                # NOTES:
                # exclude invalid bounding box
                # bounding box: x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
                # if bbox[7] != "1":
                sample["bboxs"].append(bbox)
            if len(sample["bboxs"]) > 0:
                data.append(sample)
            i = i + 2 + bboxs_num
    return data

def _parse_filelist(root, image_dir, list_file):
    data = []
    with open(root + "/" + list_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            path = root + "/" + image_dir + "/" + line.strip()
            data.append({"image": path})
    return data

def init_data(root):
    _extract_files(root)
    train_data = _parse_bbx(root, "WIDER_train/images","/wider_face_split/wider_face_train_bbx_gt.txt")
    val_data = _parse_bbx(root, "WIDER_train/images", "/wider_face_split/wider_face_val_bbx_gt.txt")
    test_data = _parse_filelist(root, "WIDER_test/images", "wider_face_split/wider_face_test_filelist.txt")
    return (train_data, val_data, test_data)



def load_data(root):
    """WIDERFace: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/"""
    assert(os.path.exists(root))
    return init_data(root)

def _test_data(root):
    from PIL import Image, ImageFont, ImageDraw
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    train_data, val_data, test_data = init_data(root)
    for sample in train_data:
        bboxs = sample["bboxs"]
        invalid = False
        for bbox in bboxs:
            if bbox[9] == "1":
                image_file = sample["image"]
                invalid = True
                break
        if invalid:
            break

    img = Image.open(image_file)
    # img = img.rotate(180)
    # img = img.resize((2048, 1536), Image.BILINEAR)
    # img = img.crop((0, 0, 1024, 768))
    data = np.asarray(img)
    fig, ax = plt.subplots(1)
    ax.imshow(data)
    for bbox in bboxs:
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show(block=True)


if __name__ == "__main__":
    _test_data("D:/share/wider")

