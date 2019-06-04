import os
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET

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
def parse_file(trainning_file):
    result = []
    tree = ET.parse(trainning_file)
    images = tree.getroot()[0]
    for image in images:
        file = image.attrib["file"]
        for box in image:
            result.append({
                "file": file,
                "box": [int(box.attrib["left"]), int(box.attrib["top"]), int(box.attrib["width"]), int(box.attrib["height"])],
                "landmarks": [[int(p.attrib["x"]), int(p.attrib["y"])]  for p in box]
            })
    return result

class sample(object):
    def __init__(self, img, box, target_shape):
        self.img = img
        self.box = box
        self.target_shape = target_shape
        self.current_shape = None
        self.diff_shape = None

def norm_coord(box, pt):
    x, y, w, h = box
    if pt.ndim == 1:
        return (pt[0] - x)/w, (pt[1] - y)/h
    assert(pt.ndim == 2 and pt.shape[1] == 2)
    return np.float32([(pt[:,0] - x)/w, (pt[:,1] - y)/h]).T

def img_coord(box, pt):
    x, y, w, h = box
    if pt.ndim == 1:
        return x + pt[0]*w, y + pt[1]*h
    assert(pt.ndim == 2 and pt.shape[1] == 2)   
    return np.float32([x + pt[:,0]*w, y + pt[:,1]*h]).T

def generate_samples(metadata, oversampling_amount=1):
    samples = []
    for d in metadata:
        img = cv2.imread(d["file"])
        h, w = img.shape[:2]
        scale = 1.0
        while h*scale > 1024 and w*scale > 1024:
            scale *= 0.5
        if scale < 1.0:
            img = cv2.resize(img, (int(scale*w), int(scale*h)))
        s = sample(img, np.float32(d["box"])*scale, np.float32(d["landmarks"])*scale)
        s.target_shape = norm_coord(s.box, s.target_shape)
        samples.append(s)
    mean_shape = np.average([s.target_shape for s in samples], axis=0)
    result = []
    for s in samples:
        for i in range(oversampling_amount):
            r = sample(s.img, s.box, s.target_shape)
            if i % oversampling_amount == 0:
                r.current_shape = mean_shape
            else:
                index = int(np.random.rand()*len(samples))
                index = (index + 1)%len(samples) if s is samples[index] else index
                r.current_shape = samples[index].target_shape
            r.diff_shape = r.target_shape - r.current_shape
            result.append(r)
    return result, mean_shape


def randomly_sample_pixel_coords(mean_shape, cascade_depth=10, num_test_coords=400, padding=0.0):
    max_x, max_y = np.max(mean_shape, axis=0) + padding
    min_x, min_y = np.min(mean_shape, axis=0) - padding
    pixel_coords = np.zeros((cascade_depth, num_test_coords, 2))
    anchor_indices = np.zeros((cascade_depth, num_test_coords), dtype='uint8')
    deltas = np.zeros((cascade_depth, num_test_coords, 2))
    for i in range(cascade_depth):
        for j in range(num_test_coords):
            pixel_coords[i, j, :] = [np.random.rand()*(max_x-min_x)+min_x, np.random.rand()*(max_y-min_y)+min_y]
            anchor_indices[i,j] = np.argmin([dx*dx + dy*dy for (dx, dy) in mean_shape - pixel_coords[i, j, :]])
            deltas[i, j, :] = pixel_coords[i, j, :] - mean_shape[anchor_indices[i,j]]
    return pixel_coords, anchor_indices, deltas

def test_parse_file(trainning_file):
    metadata = parse_file(trainning_file)
    for d in metadata:
        img = cv2.imread(d["file"])
        h, w = img.shape[:2]
        scale = 1.0
        while h*scale > 1024 and w*scale > 1024:
            scale /= 2
        if scale < 1.0:
            img = cv2.resize(img, (int(scale*w), int(scale*h)))
        x, y, w, h = d["box"]
        cv2.rectangle(img, (int(scale*x),int(scale*y)), (int(scale*(x+w)),int(scale*(y+h))), (0, 255, 0), 1)
        for (x, y) in d["landmarks"]:
            cv2.circle(img, (int(scale*x), int(scale*y)), 1, (0, 255, 0), 2)
        cv2.imshow("sample", img)
        key = cv2.waitKey(3000)
        if key == ord('q'):
            break

def test_generate_samples(metadata):
    samples, _ = generate_samples(metadata)
    for s in samples:
        x, y, w, h = s.box 
        cv2.rectangle(s.img, (int(x),int(y)), (int(x+w),int(y+h)), (0, 255, 0), 1)
        for (x, y) in img_coord(s.box, s.target_shape):
            cv2.circle(s.img, (int(x), int(y)), 1, (0, 255, 0), 2)
        for (x, y) in img_coord(s.box, s.current_shape):
            cv2.circle(s.img, (int(x), int(y)), 1, (0, 0, 255), 2)
        cv2.imshow("sample", s.img)
        if cv2.waitKey(3000) == ord('q'):
            break
def test_randomly_sample_pixel_coords(mean_shape):
    pixel_coords, anchor_indices, deltas = randomly_sample_pixel_coords(mean_shape, 1, 500)
    box = [256, 256, 640, 640]
    img = np.zeros((1024, 1024, 3), dtype='uint8')
    x, y, w, h = box 
    cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), (0, 255, 0), 1)
    for (x, y) in img_coord(box, mean_shape):
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 2)
    for (x, y) in img_coord(box, pixel_coords[0]):
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), 2)
    for i, idx in enumerate(anchor_indices[0]):
        x1, y1 = img_coord(box, mean_shape[idx])
        x2, y2 = img_coord(box, mean_shape[idx] + deltas[0, i])
        cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow("image", img)
    cv2.waitKey()
        


if __name__ == "__main__":
    trainning_file = os.path.dirname(os.path.abspath(__file__)) + "/../datasets/w300/trainning_landmarks.xml"
    #test_parse_file(trainning_file)
    metadata = parse_file(trainning_file)
    #test_generate_samples(metadata)
    samples, mean_shape = generate_samples(metadata[:111])
    test_randomly_sample_pixel_coords(mean_shape)
    
    
