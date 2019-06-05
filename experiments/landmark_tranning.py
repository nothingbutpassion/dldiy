import os
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET


class sample(object):
    def __init__(self, img, box, target_shape):
        self.img = img
        self.box = box
        self.target_shape = target_shape
        self.current_shape = None
        self.diff_shape = None
        self.pixel_intensities = None

class split_feature(object):
    def __init__(self, idx1=0, idx2=0, thresh=0):
        self.idx1 = idx1
        self.idx2 = idx2
        self.thresh = thresh

class regression_tree(object):
    def __init__(self, depth, leaf_shape):
        self.depth = depth
        self.nodes = [split_feature() for i in range(np.power(2, depth)-1)]
        self.leaf_values = np.zeros((np.power(2, depth),) + leaf_shape)
    def left_child(self, idx):
        return 2*idx + 1
    def right_child(self, idx):
        return 2*idx + 2
    def leaf_value(self, pixel_intensities):
        i = 0
        while i < len(self.nodes):
            if pixel_intensities[self.nodes[i].idx1] - pixel_intensities[self.nodes[i].idx2] > self.nodes[i].thresh:
                i = self.left_child(i)
            else:
                i = self.right_child(i)
        i = i - len(self.nodes)
        return self.leaf_values[i]

# xml formats:
# </dataset>
#     </images>
#         <image file='2009_004587.jpg'>
#             <box top='280' left='266' width='63' height='63'>
#                 <part name='01' x='299' y='329'/>
#                 ...
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

def get_transform(mean_shap, current_shape):
    # A*X = B -> X = pinv(A)*B
    # A.shape: (n, 3)
    # X.shape: (3, 2)
    # B.shape: (n, 2)
    A = np.stack((mean_shap[:,0], mean_shap[:,1], np.ones(mean_shap.shape[0])), axis=-1)
    B = current_shape
    X = np.dot(np.linalg.pinv(A), B)
    return X
def transform(mean_coords, X):
    return np.dot(mean_coords, X[:2,:]) + X[2:,:]

def extract_pixel_intensities(samples, mean_shape, anchor_indices, deltas):
    for s in samples:
        X = get_transform(mean_shape, s.current_shape)
        shape = transform(mean_shape, X)
        s.pixel_intensities = np.zeros((len(anchor_indices),))
        for idx in anchor_indices:
            x, y = img_coord(s.box, shape[idx] + np.dot(deltas[idx], X[:2,:]))
            x = min(max(0, int(x)), s.img.shape[1]-1)
            y = min(max(0, int(y)), s.img.shape[0]-1)
            s.pixel_intensities[idx] = np.average(s.img[y,x])


def randomly_generate_splits(pixel_coords, num_test_coords, lamda_coefficient):
    split = split_feature()
    for i in range(num_test_coords*num_test_coords):
        idx1 = min(int(np.random.rand()*num_test_coords), num_test_coords-1)
        idx2 = min(int(np.random.rand()*num_test_coords), num_test_coords-1)
        while idx2 == idx1:
            idx2 = min(int(np.random.rand()*num_test_coords), num_test_coords-1) 
        dx, dy = pixel_coords[idx1] - pixel_coords[idx1]
        dist = np.sqrt(dx*dx + dy*dy)
        accept_prob = np.exp(-dist*lamda_coefficient)
        if accept_prob > np.random.rand():
            split.idx1 = idx1
            split.idx2 = idx2
            break
        split.thresh = (np.random.rand()*256 - 128)/2.0
    return split

def make_regression_tree(samples, pixel_corrds, tree_depth):
    pass

def test_transform(mean_shape, current_shape):
    X = get_transform(mean_shape, current_shape)
    shape = transform(mean_shape, X)
    average_error = np.average([np.sqrt(dx*dx + dy*dy) for (dx, dy) in current_shape - shape])
    print("average transform error: " + str(average_error))
    box = [256, 256, 640, 640]
    img = np.zeros((1024, 1024, 3), dtype='uint8')
    x, y, w, h = box 
    cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), (0, 255, 0), 1)
    for (x, y) in img_coord(box, mean_shape):
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 2)
    for (x, y) in img_coord(box, current_shape):
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)    
    for (x, y) in img_coord(box, shape):
        cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 2) 
    cv2.imshow("image", img)
    cv2.waitKey()

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
    box = [256, 256, 512, 640]
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
    
    # params
    oversampling_amount = 10
    cascade_depth = 10
    num_test_coords = 400
    num_trees_per_cascade = 500
    tree_depth = 4
    num_test_splits = 20

    #test_generate_samples(metadata)
    samples, mean_shape = generate_samples(metadata, oversampling_amount)

    #test_transform(mean_shape, samples[11].target_shape)
    #test_randomly_sample_pixel_coords(mean_shape)

    pixel_coords, anchor_indices, deltas = randomly_sample_pixel_coords(mean_shape, cascade_depth, num_test_coords)

    for cascade in range(len(cascade_depth)):
        extract_pixel_intensities(samples, mean_shape, anchor_indices[cascade], deltas[cascade])
        for i in range(len(num_trees_per_cascade)):
            make_regression_tree(samples, pixel_coords[cascade], tree_depth) 


    
    



    
    
