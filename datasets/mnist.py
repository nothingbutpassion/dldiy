# coding: utf-8
import os
import gzip
import pickle
import numpy as np

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

urls = {
    "train_image": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_label": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_image": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_label": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

train_num = 60000
test_num = 10000
image_dim = (1, 28, 28)
image_size = 784

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/mnist"
save_file = dataset_dir + "/mnist.pkl"


def _load_image(file_name):
    file_path = dataset_dir + "/" + file_name
    print("converting  " + file_name + "to numpy array ...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, image_size)
    return data

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print("converting  " + file_name + " to numpy array ...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def _download():
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    for k,v in urls.items():
        local_file = dataset_dir + "/" + k + ".gz"
        if not os.path.exists(local_file):
            print("download " + k + " from " + v)
            with open(local_file,"wb") as f:
                f.write(urlopen(v).read())
                f.close()
            print("saved as " + local_file)


def _convert():
    dataset = {}
    dataset["train_image"] = _load_image("train_image.gz")
    dataset["train_label"] = _load_label("train_label.gz")
    dataset["test_image"] = _load_image("test_image.gz")
    dataset["test_label"] = _load_label("test_label.gz")
    print("creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("saved as " + save_file)

def _init_data():
    _download()
    _convert()

def load_data(normalize=True, flatten=True, one_hot_label=True):
    """MNIST: http://yann.lecun.com/exdb/mnist/"""
    
    if not os.path.exists(save_file):
        _init_data()
    
    with open(save_file, "rb") as f:
        dataset = pickle.load(f)
    
    if normalize:
        for k in ("train_image", "test_image"):
            dataset[k] = dataset[k].astype(np.float32)/255.0

    if one_hot_label:
        for k in ("train_label", "test_label"):
            T = np.zeros((dataset[k].shape[0], 10))
            for idx, row in enumerate(T):
                row[dataset[k][idx]] = 1
            dataset[k] = T

    if not flatten:
        for k in ("train_image", "test_image"):
            dataset[k] = dataset[k].reshape(-1, 1, 28, 28)

    return (dataset["train_image"], dataset["train_label"]), (dataset["test_image"], dataset["test_label"])
