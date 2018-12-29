import numpy as np
from utils import im2col, col2im

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
    def __init__(self, W=None, b=None):
        self.name = "affine"
        self.W = W
        self.b = b
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
    def __init_(self, filter_w, filter_h, stride=1, pad=0, W=None, b=None):
        self.name = "cov2d"
        self.FW = filter_w
        self.FH = filter_h
        self.stride = stride
        self.pad = pad
        self.W = W
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        N, C, H, W = x.shape
        OH = (H + 2*self.pad - self.FH)//self.stride + 1
        OW = (W + 2*self.pad - self.FW)//self.stride + 1
        col = im2col(x, self.FH, self.FW, self.stride, self.pad)
        
        # NOTES:
        # W.shape: (C*FH*FW, FN)
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


        return y

    def backward(self, dy):

        
        # dy.shape: (N, FN, OH, OW)
        # after transpose: (N, OH, OW, FN)
        # after reshape: (N*OH*OW, FN)
        N, FN, OH, OW = dy.shape
        dy=dy.transpose(0, 2, 3, 1).reshape(-1, FN)


        #self.x shape: 
        #dx.shape: N, C, H, W
        #dw.shape: (C*FH*FW, FN)
        return dy














    def backward(self, dy):
        pass