import pickle
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# <dataset>
#     ...
#     <images>
#         <image file='Data/CLS-LOC/test/ILSVRC2012_test_00000433.JPEG_RESAMPLED_34916508e9a348b0.png'>
#         </image>
#         ...
#         <image file='Data/CLS-LOC/test/ILSVRC2012_test_00008755.JPEG_RESAMPLED_259efb8a58c7cca6.png'>
#             <box top='98' left='255' width='117' height='121' ignore='1'/>
#             <box top='134' left='131' width='95' height='85'/>
#             ...
#         </image>
#     </images>
# </dataset>
def _parse_xml(xml_file):
    tree = ET.parse(xml_file)
    dataset = tree.getroot()
    images = dataset.find("images")
    result = {}
    for image in images:
        file = (Path(xml_file).parent/image.attrib["file"]).absolute().as_posix()
        boxes = []
        for box in image:
            ignore = 1 if "ignore" in box.attrib else 0
            box = [int(box.attrib["left"]), int(box.attrib["top"]), int(box.attrib["width"]), int(box.attrib["height"]), ignore]
            boxes.append(box)
        if len(boxes) > 0:
            result[file] = boxes
    return result

def _init_data(dataset_dir):
    xml_file = Path(dataset_dir)/"faces_2016_09_30.xml"
    return _parse_xml(xml_file.as_posix())

def load_data(dataset_dir):
    """
    dfdd(dlib face detection dataset): http://dlib.net/files/data/dlib_face_detection_dataset-2016-09-30.tar.gz
    """
    assert Path(dataset_dir).is_dir()
    pickle_file = Path(dataset_dir)/"dfdd.pkl"
    if pickle_file.is_file():
        with pickle_file.open("rb") as f:
            data = pickle.load(f)
    else:
        data = _init_data(dataset_dir)
        with pickle_file.open("wb") as f:
            pickle.dump(data, f)
    return data

def show_stat_info(data):
    total_boxes = []
    ignore = []
    normal = []
    for boxes in data.values():
        for b in boxes:
            total_boxes.append(b)
            if b[-1] == 1:
                ignore.append(b)
            else:
                normal.append(b)
    ignore = np.array(ignore)
    normal = np.array(normal)
    print(f"images: {len(data)}")
    print(f"all boxes: {len(total_boxes)}")
    print(f"ignore boxes: {len(ignore)}")
    print(f"ignore boxes: x_min={ignore[:, 0].min()}, x_max={ignore[:, 0].max()}, x_mean: {ignore[:, 0].mean()}")
    print(f"ignore boxes: y_min={ignore[:, 1].min()}, y_max={ignore[:, 1].max()}, y_mean: {ignore[:, 1].mean()}")
    print(f"ignore boxes: w_min={ignore[:, 2].min()}, w_max={ignore[:, 2].max()}, w_mean: {ignore[:, 2].mean()}")
    print(f"ignore boxes: h_min={ignore[:, 3].min()}, h_max={ignore[:, 3].max()}, h_mean: {ignore[:, 3].mean()}")
    print(f"normal boxes: {len(normal)}")
    print(f"normal boxes: x_min={normal[:, 0].min()}, x_max={normal[:, 0].max()}, x_mean: {normal[:, 0].mean()}")
    print(f"normal boxes: y_min={normal[:, 1].min()}, y_max={normal[:, 1].max()}, y_mean: {normal[:, 1].mean()}")
    print(f"normal boxes: w_min={normal[:, 2].min()}, w_max={normal[:, 2].max()}, w_mean: {normal[:, 2].mean()}")
    print(f"normal boxes: h_min={normal[:, 3].min()}, h_max={normal[:, 3].max()}, h_mean: {normal[:, 3].mean()}")

if __name__ == "__main__":
    import cv2
    import sys
    if len(sys.argv) != 2:
        print(f"Usage {sys.argv[0]} <dataset-dir>")
        sys.exit(-1)
    data = load_data(sys.argv[1])
    show_stat_info(data)
    for f, boxes in data.items():
        img = cv2.imread(f)
        print(f"{f}: {img.shape}")
        for (x, y, w, h, ignore) in boxes:
            color = [np.random.randint(255) for _ in range(3)] if ignore else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
            print(f"box: ({x},{y},{w},{h},{ignore})")
        cv2.imshow("image", img)
        if cv2.waitKey() == ord('q'):
            break