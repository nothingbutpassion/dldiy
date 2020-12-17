import os
import cv2
import pickle
import time
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

class progress_bar(object):
    def __init__(self, target):
        self.target = target
        self.start_time = time.time()
        self.last_time = self.start_time
    def _str(self, seconds):
        if seconds < 60:
            return "%.0f seconds" % seconds
        elif seconds < 60*60:
            return "%.2f minutes" % (seconds/60,)
        else:
            return "%.2f hours" % (seconds/(60*60),)
    def show_progress(self, current, user_str=""):
        current_time = time.time()
        if current > 0 and current_time - self.last_time > 0.1:
            elapsed = current_time - self.start_time
            remaining = elapsed*(self.target - current)/current
            print("Time elapsed: %s, remaining: %s%s" % (self._str(elapsed), self._str(remaining), user_str))
        self.last_time = current_time

class sample(object):
    def __init__(self, img, box, target_shape=None):
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
        self.nodes = [split_feature() for i in range(np.power(2, depth)-1)]
        self.leaf_values = np.zeros((np.power(2, depth),) + leaf_shape)
        self.num_nodes = len(self.nodes)
    def leaf_value(self, pixel_intensities):
        i = 0
        while i < len(self.nodes):
            if pixel_intensities[self.nodes[i].idx1] - pixel_intensities[self.nodes[i].idx2] > self.nodes[i].thresh:
                i = 2*i + 1
            else:
                i = 2*i + 2
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
    dir = os.path.dirname(trainning_file)
    tree = ET.parse(trainning_file)
    dataset = tree.getroot()
    images = [ e for e in dataset if e.tag == "images"][0]
    for image in images:
        file = image.attrib["file"]
        if not os.path.exists(file):
            file = os.path.join(dir, file)
        assert(os.path.exists(file))
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
                r.current_shape = np.copy(mean_shape)
            else:
                index = int(np.random.rand()*len(samples))
                index = (index + 1)%len(samples) if s is samples[index] else index
                r.current_shape = np.copy(samples[index].target_shape)
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
        for i, idx in enumerate(anchor_indices):
            x, y = img_coord(s.box, shape[idx] + np.dot(deltas[idx], X[:2,:]))
            if x > 0 and x < s.img.shape[1] and y > 0 and y < s.img.shape[0]:
                s.pixel_intensities[i] = np.average(s.img[int(y),int(x)])

def randomly_generate_split(pixel_coords, num_test_coords, lamda):
    split = split_feature()
    while True:
        split.idx1 = min(int(np.random.rand()*num_test_coords), num_test_coords-1)
        split.idx2 = min(int(np.random.rand()*num_test_coords), num_test_coords-1)
        while split.idx2 == split.idx1:
            split.idx2 = min(int(np.random.rand()*num_test_coords), num_test_coords-1) 
        dx, dy = pixel_coords[split.idx1] - pixel_coords[split.idx2]
        dist = np.sqrt(dx*dx + dy*dy)
        accept_prob = np.exp(-dist/lamda)
        if accept_prob > np.random.rand():
            break
    split.thresh = (np.random.rand()*256 - 128)/2.0
    return split

def generate_split(pixel_coords, num_test_coords, num_test_splits, lamda, samples, start, end, sum):
    best_split = None
    best_score = 0
    best_left_sum = None
    best_right_sum = None
    for i in range(num_test_splits):
        split = randomly_generate_split(pixel_coords, num_test_coords, lamda)
        left_sum = np.zeros_like(sum)
        left_count = 0
        for j in range(start, end):
            if samples[j].pixel_intensities[split.idx1] - samples[j].pixel_intensities[split.idx2] > split.thresh:
                left_sum += samples[j].diff_shape
                left_count += 1
        right_sum = sum - left_sum
        right_count = end - start - left_count
        score = 0
        if left_count > 0 and right_count > 0:
            score = np.sum(left_sum**2)/left_count + np.sum(right_sum**2)/right_count
        elif left_count > 0:
            score = np.sum(left_sum**2)/left_count
        elif right_count > 0:
            score = np.sum(right_sum**2)/right_count
        if best_split == None or best_score < score:
            best_split = split
            best_score = score
            best_left_sum = left_sum
            best_right_sum = right_sum
    return best_split, best_left_sum, best_right_sum

def partition_samples(split, samples, start, end):
    mid = start
    for j in range(start, end):
        if samples[j].pixel_intensities[split.idx1] - samples[j].pixel_intensities[split.idx2] > split.thresh:
            samples[mid], samples[j] = samples[j], samples[mid]
            mid += 1
    return mid

def make_regression_tree(samples, pixel_coords, tree_depth, num_test_coords, num_test_splits, lamda, nu):
    tree = regression_tree(tree_depth, samples[0].diff_shape.shape)
    sum = np.zeros(samples[0].diff_shape.shape)
    for s in samples:
        s.diff_shape = s.target_shape - s.current_shape
        sum += s.diff_shape
    # make tree nodes
    queue = [(0, len(samples), sum)]
    for i in range(tree.num_nodes):
        start, end, sum = queue[0]
        del queue[0]
        split, left_sum, right_sum = generate_split(pixel_coords, num_test_coords, num_test_splits, lamda, samples, start, end, sum)
        tree.nodes[i] = split
        mid = partition_samples(split, samples, start, end)
        queue.append((start, mid, left_sum))
        queue.append((mid, end, right_sum))
    # make tree leaves
    for i, (start, end, sum) in enumerate(queue):
        if start != end:
            tree.leaf_values[i] = nu*sum/(end - start)
        # update current shape
        for j in range(start, end):
            samples[j].current_shape += tree.leaf_values[i] 
    return tree

def train_error(samples):
    diff_shape = np.float32([s.target_shape - s.current_shape for s in samples])
    total_err = np.sqrt(diff_shape**2)
    average_err = np.average(total_err)
    return average_err

def train_model(trainning_file, model_file):
    # trainning params
    oversampling_amount = 10
    cascade_depth = 10
    num_test_coords = 400
    num_trees_per_cascade = 500
    tree_depth = 4
    num_test_splits = 100
    lamda = 0.1
    nu = 0.1

    # parse trainning file
    print("parse %s ..." % trainning_file)
    metadata = parse_file(trainning_file)

    # generate trainning samples
    print("generate trainning samples ...")
    bar = progress_bar(cascade_depth*num_trees_per_cascade)
    samples, mean_shape = generate_samples(metadata, oversampling_amount)
    
    # generate pixel coords
    print("randomly sample pixel coords ...")
    pixel_coords, anchor_indices, deltas = randomly_sample_pixel_coords(mean_shape, cascade_depth, num_test_coords)
    
    # generate regression forests
    print("make regression forests ...")
    forests = []
    for cascade in range(cascade_depth):
        extract_pixel_intensities(samples, mean_shape, anchor_indices[cascade], deltas[cascade])
        forest = []
        for i in range(num_trees_per_cascade):
            tree = make_regression_tree(samples, pixel_coords[cascade], tree_depth, num_test_coords, num_test_splits, lamda, nu)
            forest.append(tree)
            bar.show_progress(cascade*num_trees_per_cascade + i + 1, ", tranning error: %.6f" % train_error(samples))
        forests.append(forest)

    # save model
    print("save model to %s ... " % model_file)
    model_data = (forests, mean_shape, anchor_indices, deltas)
    with open(model_file, "wb") as f:
        pickle.dump(model_data, f, -1)
    print("model saved")

def load_model(model_file):
    assert(os.path.exists(model_file))
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)
    return model_data

def predict(model_data, img, box):
    forests, mean_shape, anchor_indices, deltas = model_data
    s = sample(img, box)
    s.current_shape = np.copy(mean_shape)
    for cascade, forest in enumerate(forests):
        extract_pixel_intensities([s], mean_shape, anchor_indices[cascade], deltas[cascade])
        for tree in forest:
            s.current_shape += tree.leaf_value(s.pixel_intensities)
    return img_coord(box, s.current_shape)

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
        if cv2.waitKey(3000) == ord('q'):
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

def test_landmarks_model(trainning_file, model_file):
    model_data = load_model(model_file)
    metadata = parse_file(trainning_file)
    for d in metadata:
        img = cv2.imread(d["file"])
        h, w = img.shape[:2]
        if w < 1024 or h < 1024:
            box = d["box"]
            for (x, y)  in d["landmarks"]:
                cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 2)
            landmarks = predict(model_data, img, box)
            for (x, y) in landmarks:
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)
            cv2.imshow("image", img)
            if cv2.waitKey(3000) == ord('q'):
                break

if __name__ == "__main__":
    project_dir = Path(__file__).absolute().parents[1].as_posix()
    trainning_file = project_dir + "/models/face_model_v1_2100_w300_tranning_landmarks.xml"
    out_model_file = project_dir + "/models/face_model_v1_2100_w300_landmarks.model"
    train_model(trainning_file, out_model_file)
    test_landmarks_model(trainning_file, out_model_file)

