import numpy as np
from utils import im2col, col2im

class Layer(object):
    def __init__(self):
        self.name = "layer"
        self.input_shape = None
        self.output_shape = None

    def compute_output_shape(self, input_shape):
        return input_shape
        

class ReLU:
    def __init__(self):
        self.name = "relu"
        self.mask = None

    def forward(self, x):
        self.mask = x < 0
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, dy):
        dx = dy.copy()
        dx[self.mask] = 0
        return dx

class Sigmoid:
    def __init__(self):
        self.name = "sigmoid"
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        dx = dy*self.y*(1-self.y)
        return dx


class Affine:
    def __init__(self):
        self.name = "affine"
        self.W = None
        self.b = None
        self.x = None
        self.x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.x_shape = x.shape
        if x.ndim == 1:
            self.x = x.reshape(1, -1)          
        y = np.dot(self.x, self.W) + self.b
        if x.ndim == 1:
            y = y.reshape(-1)
        return y

    def backward(self, dy):
        if dy.ndim == 1:
            dy = dy.reshape(1, -1)
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        dx = np.dot(dy, self.W.T)
        dx = dx.reshape(*self.x_shape)
        return dx

class Softmax:
    def __init__(self):
        self.name = "softmax"
        self.y = None
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            exp_x = np.exp(x)
            y = exp_x/np.sum(exp_x, axis=0)
            self.y = y.T
        else:
            x = x - np.max(x)
            exp_x = np.exp(x)
            self.y = exp_x/np.sum(exp_x)
        return self.y
        
    def backward(self, dy):
        if dy.ndim == 2: 
            ds = np.sum((self.y * dy).T, axis=0)
            dx = self.y * (dy.T - ds).T
        else:
            ds = np.sum(self.y * dy)
            dx = self.y * (dy - ds)
        return dx

class Cov2D:
    def __init__(self, filter_w, filter_h, stride=1, pad=0, W=None, b=None):
        self.name = "cov2d"
        self.FW = filter_w
        self.FH = filter_h
        self.stride = stride
        self.pad = pad
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        self.x = None
        self.col = None

    def forward(self, x):
        N, C, H, W = x.shape
        OH = (H + 2*self.pad - self.FH)//self.stride + 1
        OW = (W + 2*self.pad - self.FW)//self.stride + 1
        col = im2col(x, self.FH, self.FW, self.stride, self.pad)
        
        # NOTES
        # W.shape: (C*FH*FW, FN)  wherein, each filter is a column
        # b.shape: (FN,)
        # col.shape: (N*OH*OW, C*FH*FW)
        # y.shape: (N*OH*OW, FN)
        # after reshape: (N, OH, OW, FN)
        # after transpose(N, FN, OH, OW)
        # FN (filter nums) == OC (output channels)
        y = np.dot(col, self.W) + self.b
        y = y.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        # save for backward
        self.x = x
        self.col = col

        return y

    def backward(self, dy):
        # dy.shape: (N, FN, OH, OW)
        # after transpose: (N, OH, OW, FN)
        # after reshape: (N*OH*OW, FN)
        N, FN, OH, OW = dy.shape
        dy=dy.transpose(0, 2, 3, 1).reshape(-1, FN)
        
        # y = x.w + b => dy/dx = w, dy/dw = x, dy/db = 1
        # dy.shape:                     (N*OH*OW, FN)
        # dcol.shape(same as col.shape):(N*OH*OW, C*FH*FW)
        # dw.shape(same as W.shape):    (C*FH*FW, FN)
        # db.shape(same as b.shape):    (FN,)
        # dx.shape(same as x.shape):    (N, C, H, W)
        self.dW = np.dot(self.col.T, dy)
        self.db = np.sum(dy, axis=0)
        dcol = np.dot(dy, self.W.T)
        dx = col2im(dcol, self.x.shape, self.FH, self.FW, self.stride, self.pad)
        return dx

class MaxPooling2D:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.name = 'maxpooling2d'
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.argmax = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h)//self.stride + 1
        out_w = (W - self.pool_w)//self.stride + 1

        # col.shape: (N*out_h*out_w, C*pool_h*pool_w)
        # after reshape: (N*out_h*out_w*C, pool_h*pool_w)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        # save for backward
        self.x = x
        self.argmax = np.argmax(col, axis=1)

        # col.shape: (N*out_h*out_w*C, pool_h*pool_w)
        # after np.max: (N*out_h*out_w*C,)
        # after reshape: (N, out_h, out_w, C)
        # after transpose: (N, C, out_h, out_W)
        # so y.shape: (N, C, out_h, out_w)
        col = np.max(col, axis=1)
        y = col.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, dy):
        N, C, out_h, out_w = dy.shape

        # dy.shape: (N, C, out_h, out_w)
        # after transpose: (N, out_h, out_w, C)
        dy = dy.transpose(0, 2, 3, 1)

        # col.shape: (N*out_h*out_w*C, pool_h*pool_w)
        # after reshape: (N*out_h*out_w, C*pool_h*pool_w)
        col = np.zeros((N*out_h*out_w*C, self.pool_h*self.pool_w))
        col[np.arange(N*out_h*out_w*C), self.argmax.flatten()] = dy.flatten()
        col = col.reshape(N*out_h*out_w, -1)

        # col.shape: (N*out_h*out_w, C*pool_h*pool_w) => dx.shape (N, C, H, W)
        dx = col2im(col, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

class Flatten:
    def __init__(self):
        self.name = 'flatten'
        self.x = None

    def forward(self, x):
        self.x = x
        if x.ndim == 1:
            return x
        y = x.reshape(x.shape[0], -1)
        return y

    def backward(self, dy):
        dx = dy.reshape(self.x.shape)
        return dx


