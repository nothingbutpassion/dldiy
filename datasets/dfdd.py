import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

# <dataset>
#     ...
#     <images>
#         <image file='Data/CLS-LOC/test/ILSVRC2012_test_00000433.JPEG_RESAMPLED_34916508e9a348b0.png'>
#         </image>
#
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

def init_data(dataset_dir):
    xml_file = Path(dataset_dir)/"faces_2016_09_30.xml"
    return _parse_xml(xml_file.as_posix())

def load_data(dataset_dir):
    pickle_file = Path(dataset_dir)/"dfdd.pkl"
    if pickle_file.is_file():
        with pickle_file.open("rb") as f:
            data = pickle.load(f)
    else:
        data = init_data(dataset_dir)
        with pickle_file.open("wb") as f:
            pickle.dump(data, f)
    return data

def show_stat_info(data):
    import numpy as np
    all_boxes = []
    ignore_boxes = []
    normal_boxes = []
    for boxes in data.values():
        for b in boxes:
            all_boxes.append(b)
            if b[-1] == 1:
                ignore_boxes.append(b)
            else:
                normal_boxes.append(b)
    ignore_boxes = np.array(ignore_boxes)
    normal_boxes = np.array(normal_boxes)
    print(f"images: {len(data)}")
    print(f"all boxes: {len(all_boxes)}")
    print(f"ignore boxes: {len(ignore_boxes)}")
    print(f"ignore boxes: x_min={ignore_boxes[:, 0].min()}, x_max={ignore_boxes[:, 0].max()}, x_mean: {ignore_boxes[:, 0].mean()}")
    print(f"ignore boxes: y_min={ignore_boxes[:, 1].min()}, y_max={ignore_boxes[:, 1].max()}, y_mean: {ignore_boxes[:, 1].mean()}")
    print(f"ignore boxes: w_min={ignore_boxes[:, 2].min()}, w_max={ignore_boxes[:, 2].max()}, w_mean: {ignore_boxes[:, 2].mean()}")
    print(f"ignore boxes: h_min={ignore_boxes[:, 3].min()}, h_max={ignore_boxes[:, 3].max()}, h_mean: {ignore_boxes[:, 3].mean()}")
    print(f"normal boxes: {len(normal_boxes)}")
    print(f"normal boxes: x_min={normal_boxes[:, 0].min()}, x_max={normal_boxes[:, 0].max()}, x_mean: {normal_boxes[:, 0].mean()}")
    print(f"normal boxes: y_min={normal_boxes[:, 1].min()}, y_max={normal_boxes[:, 1].max()}, y_mean: {normal_boxes[:, 1].mean()}")
    print(f"normal boxes: w_min={normal_boxes[:, 2].min()}, w_max={normal_boxes[:, 2].max()}, w_mean: {normal_boxes[:, 2].mean()}")
    print(f"normal boxes: h_min={normal_boxes[:, 3].min()}, h_max={normal_boxes[:, 3].max()}, h_mean: {normal_boxes[:, 3].mean()}")

if __name__ == "__main__":
    import cv2
    import sys
    import numpy as np

    if len(sys.argv) != 2:
        print(f"Usage {sys.argv[0]} <dataset-dir>")
        sys.exit(-1)
    data = load_data(sys.argv[1])
    show_stat_info(data)
    for f, boxes in data.items():
        img = cv2.imread(f)
        print(f"{f}: {img.shape}")
        for (x, y, w, h, ignore) in boxes:
            if ignore:
                b, g, r = np.random.randint(255, size=(3,))
                cv2.rectangle(img,(x, y), (x+w, y+h), (int(b), int(g), int(r)), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            print(f"box: ({x},{y},{w},{h},{ignore})")
        cv2.imshow("image", img)
        if cv2.waitKey() == ord('q'):
            break