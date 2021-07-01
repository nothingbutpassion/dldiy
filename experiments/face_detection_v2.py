import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from pathlib import Path
from collections import OrderedDict
sys.path.append(Path(__file__).absolute().parents[1].as_posix())
from datasets import dfdd

_aspects = [0.9, 1.0, 1.1]                 # anchor box aspects
_scales  = [0.2, 0.4, 0.6, 0.8]            # anchor box scales
_grids   = [[8,8], [5,5], [3,3], [1,1]]    # grids for each scale

def clean_data(data):
    result = []
    for f, boxes in data:
        valid = [[x, y, w, h] for x, y, w, h, ignore in boxes if not ignore]
        if len(valid) == 0:
            continue
        img = cv2.imread(f)
        if len(valid) < len(boxes):
            backup = []
            for x, y, w, h, ignore in boxes:
                x1, y1, x2, y2 = max(0, x), max(0, y), min(x + w, img.shape[1]), min(y + h, img.shape[0])
                if ignore:
                    if x1 < x2 and y1 < y2:
                        img[y1:y2, x1:x2] = img[y1, x1:x2]
                else:
                    backup.append((x1, y1, x2, y2, np.copy(img[y1:y2, x1:x2])))
            for x1, y1, x2, y2, b in backup:
                img[y1:y2, x1:x2] = b
        result += [(img, valid)]
        if len(result) % 100 == 0:
            print(f"{len(result):<5d} image prepared")
    print(f"{len(result):<5d} image prepared")
    return result

def random_translate(img, boxes):
    x, y, w, h  = boxes[0];
    x1, y1, x2, y2 = x, y, x+w, y+h
    for x, y, w, h in boxes[1:]:
        x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x2, x+w), max(y2, y+h)
    lx, ux = min(-x1, 0), max(img.shape[1] - x2, 0)
    ly, uy = min(-y1, 0), max(img.shape[0] - y2, 0)
    tx = np.random.randint(lx, ux+1) if ux > lx else 0
    ty = np.random.randint(ly, uy+1) if uy > ly else 0
    M = np.array([[1,0,tx],[0,1,ty]],dtype='f')
    img = cv2.warpAffine(img, M, img.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    boxes = [[b[0]+tx, b[1]+ty, b[2], b[3]] for b in boxes]
    return img, boxes

def random_crop(img, boxes):
    x, y, w, h  = boxes[0];
    x1, y1, x2, y2 = x, y, x+w, y+h
    for x, y, w, h in boxes[1:]:
        x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x2, x+w), max(y2, y+h)
    x1 = int(x1*(0.2 + 0.8*np.random.rand()))
    y1 = int(y1*(0.2 + 0.8*np.random.rand()))
    x2 = int(x2 + (img.shape[1]-x2)*(0.2 + 0.8*np.random.rand()))
    y2 = int(y2 + (img.shape[0]-y2)*(0.2 + 0.8*np.random.rand()))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    if x1 < x2 and y1 < y2:
        img = img[y1:y2,x1:x2]
        boxes = [[b[0]-x1, b[1]-y1, b[2], b[3]] for b in boxes]
    return img, boxes

def horizontal_flip(img, boxes):
    img = cv2.flip(img, 1)
    boxes = [[img.shape[1]-b[0]-b[2] ,b[1],b[2],b[3]] for b in boxes]
    return img, boxes

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w = max(0, min(x1+w1/2, x2+w2/2) - max(x1-w1/2, x2-w2/2))
    h = max(0, min(y1+h1/2, y2+h2/2) - max(y1-h1/2, y2-h2/2))
    I = w*h
    U = w1*h1 + w2*h2 - I
    return I/U

def generate_aboxes():
    aboxes = []
    for l, (scale, (rows, cols)) in enumerate(zip(_scales, _grids)):
        boxes = np.zeros((rows, cols, len(_aspects), 8))
        for i in range(rows):
            for j in range(cols):
                for k in range(len(_aspects)):
                    boxes[i,j,k,:] = [l, i, j, k, (j+0.5)/cols, (i+0.5)/rows, scale*_aspects[k], scale/_aspects[k]]
        aboxes += [boxes.reshape((-1, 8))]
    return np.concatenate(aboxes)

def encode_box(abox, gbox):
    l, i, j, k = map(int, abox[:4])
    x, y, w, h = abox[4:]
    gx, gy, gw, gh = gbox
    rows, cols = _grids[l]
    gx, gy = gx * cols, gy * rows
    assert np.floor(gy) == i and np.floor(gx) == j, print(f"invalid gbox: {gbox}, abox: {abox}")
    return [1, gx - j, gy - i, np.log(gw/w), np.log(gh/h)]

def matched_cells(gbox):
    indices = []
    gx, gy, gw, gh = gbox
    for rows, cols in _grids:
        gx, gy = gx * cols, gy * rows
        gi, gj = np.floor(gy), np.floor(gx)
        indices += [(gi, gj)]


def encode(aboxes, gboxes):
    features = np.zeros((len(aboxes), 5), dtype='float32')
    # for each ground-truth box find the anchor-box with max iou
    for gb in gboxes:
        aids = []
        for l, (rows, cols) in enumerate(_grids):
            gj, gi = np.floor(cols*gb[0]), np.floor(rows*gb[1])
            aids += [i for i, ab in enumerate(aboxes) if ab[0] == l and ab[1] == gi and ab[2] == gj]
        ious = [iou(gb, aboxes[i][4:]) for i in aids]
        if len(ious) > 0:
            max_idx = aids[np.argmax(ious)]
            features[max_idx,:] = encode_box(aboxes[max_idx], gb)
        else:
            print(f"invalid gbox: {gb}")
        # print(f"encode: index={max_idx}, box={gb}, feature={features[max_idx]}")
    # for each anchor box, ignore it if iou > 0.5 (but not the max iou)
    for i, ab in enumerate(aboxes):
        max_iou = np.max([iou(gb, ab[4:]) for gb in gboxes])
        if max_iou > 0.5 and features[i,0] < 0.5:
            features[i,0] = 0.5
    return features

def get_abox(index):
    aspect_nums = len(_aspects)
    feature_nums = [s[0]*s[1]*aspect_nums for s in _grids]
    l = 0
    while index >= feature_nums[l]:
        index -= feature_nums[l]
        l += 1
    rows, cols = _grids[l]
    i = index//(cols*aspect_nums)
    j = (index - i*cols*aspect_nums)//aspect_nums
    k = index - i*cols*aspect_nums - j*aspect_nums
    return [l, i, j, k, (j+0.5)/cols, (i+0.5)/rows, _scales[l]*_aspects[k], _scales[l]/_aspects[k]]

def decode(features, threshold=0.5):
    boxes = []
    for index, feature, in enumerate(features):
        c, x, y, w, h = feature[:5]
        w, h = np.exp([w, h])
        if c > threshold:
            l, i, j, k, dx, dy, dw, dh = get_abox(index)
            rows, cols = _grids[l]
            boxes += [[(j+x)/cols, (i+y)/rows, w*dw, h*dh]]
            # print(f"decode: index={index}, feature={feature}, box={[(j+x)/cols, (i+y)/rows, w*dw, h*dh]}")
    return boxes

class FaceDataset(Dataset):
    def __init__(self, data, image_size=(160, 160), training=True):
        super(Dataset, self).__init__()
        self.data = data
        self.training = training
        self.image_size = image_size
        self.aboxes = generate_aboxes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, boxes = self.data[index]
        if self.training:
            if np.random.rand() > 0.5:
                img, boxes = horizontal_flip(img, boxes)
        if np.random.rand() > 0.5:
            img, boxes = random_translate(img, boxes)
        else:
            img, boxes = random_crop(img, boxes)
        h, w = img.shape[:2]
        assert h > 0 and w > 0
        boxes = [[(b[0] + 0.5 * b[2]) / w, (b[1] + 0.5 * b[3]) / h, b[2] / w, b[3] / h] for b in boxes]
        img = cv2.resize(img, self.image_size)
        x = np.array(img.transpose(2, 0, 1)/255 - 0.5, dtype='float32')
        y = encode(self.aboxes, boxes)
        return x, y

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        self.dsc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=padding, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.dsc(x)

class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        mn_features = models.mobilenet_v2(pretrained=True).features[:17]
        self.backbone = nn.Sequential(*mn_features[:13])
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.f1 = nn.Sequential(mn_features[13])
        self.f2 = nn.Sequential(mn_features[14])
        self.f3 = nn.Sequential(mn_features[15])
        self.f4 = nn.Sequential(mn_features[16])
        self.b1 = nn.Sequential(
            SeparableConv2D(96, 96, 1, 1),
            nn.Conv2d(96, 15, 3)
        )
        self.b2 = nn.Sequential(
            SeparableConv2D(160, 96, 1, 1),
            nn.Conv2d(96, 15, 3, padding=1)
        )
        self.b3 = nn.Sequential(
            SeparableConv2D(160, 96, 1, 1),
            nn.Conv2d(96, 15, 3)
        )
        self.b4 = nn.Sequential(
            SeparableConv2D(160, 96, 1, 0),
            nn.Conv2d(96, 15, 3)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.f1(x)
        y1 = self.b1(x)     # 96, 10, 10 -> 32, 10, 10 -> 15, 8, 8
        x = self.f2(x)
        y2 = self.b2(x)     # 160, 5, 5  ->  64, 5, 5  -> 15, 5, 5
        x = self.f3(x)
        y3 = self.b3(x)     # 160, 5, 5  ->  64, 5, 5  -> 15, 3, 3
        x = self.f4(x)
        y4 = self.b4(x)     # 160, 5, 5  ->  64, 3, 3  -> 15, 1, 1
        # print(y1.shape, y2.shape, y3.shape, y4.shape)
        ys = [y.view(y.shape[0], y.shape[1], -1) for y in [y1,y2,y3,y4]]
        y = torch.cat(ys, dim=2)
        return y.view(y.shape[0], -1, 5)

    def predict(self, x):
        y = self.forward(x)
        y[...,:3] = torch.sigmoid(y[...,:3])
        return y

def obj_loss(y_pred, y_true):
    return F.binary_cross_entropy_with_logits(y_pred[...,0], y_true[...,0])

def noobj_loss(y_pred, y_true, noobj_nums):
    losses = F.binary_cross_entropy_with_logits(y_pred[..., 0], y_true[..., 0], reduction='none')
    topk_losses, _ = torch.topk(losses, noobj_nums)
    return topk_losses.mean()

def bbox_loss(y_pred, y_true):
    x_loss = F.mse_loss(torch.sigmoid(y_pred[...,0]), y_true[...,0])
    y_loss = F.mse_loss(torch.sigmoid(y_pred[...,1]), y_true[...,1])
    w_loss = F.mse_loss(y_pred[...,3], y_true[...,3])
    h_loss = F.mse_loss(y_pred[...,4], y_true[...,4])
    return x_loss + y_loss + w_loss + h_loss

def detect_loss(y_pred, y_true):
    obj_mask = (y_true[..., 0] > 0.5)
    noobj_mask = (y_true[..., 0] < 0.5)
    noobj_nums = 3*obj_mask.sum()
    return obj_loss(y_pred[obj_mask], y_true[obj_mask]) + \
           noobj_loss(y_pred[noobj_mask], y_true[noobj_mask], noobj_nums) + \
           bbox_loss(y_pred[obj_mask], y_true[obj_mask])

def conf_loss(y_pred, y_true):
    obj_mask = (y_true[..., 0] > 0.5)
    noobj_mask = (y_true[..., 0] < 0.5)
    noobj_nums = 3*obj_mask.sum()
    return obj_loss(y_pred[obj_mask], y_true[obj_mask]) + \
           noobj_loss(y_pred[noobj_mask], y_true[noobj_mask], noobj_nums)

def coord_loss(y_pred, y_true):
    obj_mask = (y_true[...,0] > 0.5)
    return bbox_loss(y_pred[obj_mask], y_true[obj_mask])

class Callback(object):
    def __init__(self, prefix, offset):
        self.prefix, self.offset = prefix, offset

    def on_epoch_end(self, model, epoch, logs=None):
        self.save_logs(f"{self.prefix}training.log", self.offset + epoch, logs)
        torch.save(model.state_dict(), f"{self.prefix}{self.offset + epoch}.pt")

    def save_logs(self, save_path, epoch, logs):
        if not Path(save_path).exists():
            with open(save_path, "wt") as f:
                keys = logs.keys()
                title = ("%5s" + "%16s" * len(keys)) % (("epoch",) + tuple(keys))
                f.write(title)
        with open(save_path, "at") as f:
            values = logs.values()
            line = ("\n%5d" + "%16.6f" * len(values)) % ((epoch,) + tuple([v for v in values]))
            f.write(line)

def fit(model, train_dataloader, val_dataloader, loss_func, optimizer, epochs, metrics=[], callback=None):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    history = OrderedDict()
    for epoch in range(epochs):
        logs = OrderedDict()
        logs["loss"] = []
        for metric in metrics:
            logs[metric.__name__] = []
        logs["val_loss"] = []
        for metric in metrics:
            logs["val_" + metric.__name__] = []
        model.train()
        for i, (bx, by) in enumerate(train_dataloader):
            x = bx.to(dev)
            y = by.to(dev)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            logs["loss"].append(loss.item())
            for metric in metrics:
                logs[metric.__name__].append(metric(y_pred, y).item())
        model.eval()
        with torch.no_grad():
            for bx, by in val_dataloader:
                x = bx.to(dev)
                y = by.to(dev)
                y_pred = model(x)
                loss = loss_func(y_pred, y)
                logs["val_loss"].append(loss.item())
                for metric in metrics:
                    logs["val_" + metric.__name__].append(metric(y_pred, y).item())
        logs = {k: np.array(v).mean() for k, v in logs.items()}
        log_str = f"Epoch {epoch:3d} " + " ".join([f"{k}: {v:.6f}" for k, v in logs.items()])
        print(log_str)
        callback.on_epoch_end(model, epoch, logs)
        for k in logs:
            if k in history:
                history[k] += [logs[k]]
            else:
                history[k] = [logs[k]]
    return history

def train(data_dir, model_path):
    model = FaceModel()
    if Path(model_path).is_file():
        model.load_state_dict(torch.load(model_path))
        p = model_path.rfind('_')
        prefix = model_path[:p + 1]
        offset = int(model_path[p + 1: -3]) + 1
    else:
        Path(model_path).mkdir(parents=True, exist_ok=True)
        prefix = model_path + "/" + Path(model_path).name + "_"
        offset = 0
    data = dfdd.load_data(data_dir)
    data = clean_data(data)
    split = (int(len(data)*0.9)//64)*64
    train_ds = FaceDataset(data[:split], training=True)
    val_ds = FaceDataset(data[split:], training=False)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00005)
    callback = Callback(prefix, offset)
    fit(model, train_dl, val_dl, loss_func=detect_loss, optimizer=optimizer,
        epochs=10000, metrics=[conf_loss, coord_loss], callback=callback)

def test_codec(data_dir):
    data = dfdd.load_data(data_dir)
    data = clean_data(data[:512])
    aboxes = generate_aboxes()
    for img, boxes in data:
        src = img
        img, boxes = horizontal_flip(img, boxes)
        img, boxes = random_translate(img, boxes)
        img, boxes = random_crop(img, boxes)
        h, w = img.shape[:2]
        boxes = [[(b[0] + 0.5 * b[2]) / w, (b[1] + 0.5 * b[3]) / h, b[2] / w, b[3] / h] for b in boxes]
        feature = encode(aboxes, boxes)
        boxes = decode(feature)
        img = cv2.resize(img, (160, 160))
        h, w = img.shape[:2]
        boxes = [[(b[0]-0.5*b[2])*w, (b[1]-0.5*b[3])*h, b[2]*w, b[3]*h] for b in boxes]
        for b in boxes:
            x, y, w, h = map(int, b)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0))
        cv2.imshow("src", src)
        cv2.imshow("dst", img)
        cv2.waitKey(300000)

def test_dataset(data_dir):
    data = dfdd.load_data(data_dir)
    data = clean_data(data[:512])
    dataset = FaceDataset(data)
    for x, y in dataset:
        img = np.array((x.transpose(1,2,0)+0.5)*255, dtype='uint8')
        h, w = img.shape[:2]
        boxes = decode(y)
        boxes = [[(b[0]-0.5*b[2])*w, (b[1]-0.5*b[3])*h, b[2]*w, b[3]*h] for b in boxes]
        for b in boxes:
            x, y, w, h = map(int, b)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0))
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("image", img)
        cv2.waitKey(300000)

def test_model(data_dir, model_path):
    model = FaceModel()
    model.load_state_dict(torch.load(model_path))
    model.to('cpu')
    data = dfdd.load_data(data_dir)
    data = clean_data(data[:256])
    for img, boxes in data:
        img, boxes = random_translate(img, boxes)
        img, boxes = random_crop(img, boxes)
        # ground-truth
        for b in boxes:
            x, y, w, h = map(int, b)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0))
        # prediction
        H, W = img.shape[:2]
        X = cv2.resize(img, (160, 160))
        X = np.array(X/255, dtype='float32')
        X = torch.from_numpy(X)
        X = torch.unsqueeze(X.permute(2,0,1), 0)
        Y = model.predict(X)[0]
        boxes = decode(Y, 0.6)
        boxes = [[(b[0] - 0.5 * b[2]) * W, (b[1] - 0.5 * b[3]) * H, b[2] * W, b[3] * H] for b in boxes]
        for b in boxes:
            x, y, w, h = map(int, b)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255))
        cv2.imshow("image", img)
        cv2.waitKey(300000)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage {sys.argv[0]} <dataset-dir> <model-path>")
        sys.exit(-1)
    dataset_dir, model_path = Path(sys.argv[1]).as_posix(), Path(sys.argv[2]).as_posix()
    print(f"dataset dir is {dataset_dir}, model path is {model_path}")
    # test_codec(dataset_dir)
    # test_dataset(dataset_dir)
    train(dataset_dir, model_path)