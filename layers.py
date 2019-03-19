import numpy as np
from initializers import HeNormal
from utils import im2col, col2im

class Layer(object):
    def __init__(self, name=None, input_shape=None, output_shape=None):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

class ReLU(Layer):
    def __init__(self, **args):
        super(ReLU, self).__init__(**args)
        if not self.name:
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

class Sigmoid(Layer):
    def __init__(self, **args):
        super(Sigmoid, self).__init__(**args)
        if not self.name:
            self.name = "sigmoid"
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        return dy*self.y*(1-self.y)


class Linear(Layer):
    def __init__(self, units, initializer=HeNormal(), **args):
        super(Linear, self).__init__(**args)
        if not self.name: 
            self.name = "linear"
        self.units = units
        self.initializer = initializer
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.x = None

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        self.input_shape = input_shape
        self.W = self.initializer((input_shape[-1], self.units))
        self.b = self.initializer((self.units,))
        outputs = list(input_shape)
        outputs[-1] = self.units
        self.output_shape = tuple(outputs)

    def forward(self, x):
        self.x = x        
        return np.dot(self.x, self.W) + self.b

    def backward(self, dy):
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        return np.dot(dy, self.W.T)


class Softmax(Layer):
    def __init__(self, **args):
        super(Softmax, self).__init__(**args)
        if not self.name: 
            self.name = "softmax"
        self.y = None
        
    def forward(self, x):
        x = x.T
        x = x - np.max(x, axis=0)
        exp_x = np.exp(x)
        y = exp_x/np.sum(exp_x, axis=0)
        self.y = y.T
        return self.y
        
    def backward(self, dy):
        ds = np.sum((self.y * dy).T, axis=0)
        return self.y * (dy.T - ds).T

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, stride=1, pad=0, initializer=HeNormal(), **args):
        super(Conv2D, self).__init__(**args)
        if not self.name: 
            self.name = "conv2d"
        self.FN = filters
        self.FW = kernel_size[0]
        self.FH = kernel_size[1]
        self.stride = stride
        self.pad = pad
        self.initializer = initializer
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.x_shape = None
        self.col = None

    def build(self, input_shape):
        assert(len(input_shape) == 4)
        self.input_shape = input_shape
        N, C, H, W = input_shape
        OH = (H + 2*self.pad - self.FH)//self.stride + 1
        OW = (W + 2*self.pad - self.FW)//self.stride + 1
        self.W = self.initializer((C*self.FH*self.FW, self.FN))
        self.b = self.initializer((self.FN,))
        self.output_shape = (N, self.FN, OH, OW)

    def forward(self, x):
        N, C, H, W = x.shape
        OH = (H + 2*self.pad - self.FH)//self.stride + 1
        OW = (W + 2*self.pad - self.FW)//self.stride + 1
        col = im2col(x, self.FH, self.FW, self.stride, self.pad)
        
        # save for backward
        self.x_shape = x.shape
        self.col = col

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
        dx = col2im(dcol, self.x_shape, self.FH, self.FW, self.stride, self.pad)
        return dx

class MaxPool2D(Layer):
    def __init__(self, pool_size, stride=1, pad=0, **args):
        super(MaxPool2D, self).__init__(**args)
        if not self.name: 
            self.name = "maxpooling2d"
        self.pool_h = pool_size[0]
        self.pool_w = pool_size[1]
        self.stride = stride
        self.pad = pad
        self.x_shape = None
        self.argmax = None
    
    def build(self, input_shape):
        assert(len(input_shape) == 4)
        self.input_shape = input_shape
        N, C, H, W = input_shape
        out_h = (H + 2*self.pad - self.pool_h)//self.stride + 1
        out_w = (W + 2*self.pad- self.pool_w)//self.stride + 1
        self.output_shape = (N, C, out_h, out_w)

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - self.pool_h)//self.stride + 1
        out_w = (W + 2*self.pad- self.pool_w)//self.stride + 1

        # x.shape: (N, C, H, W)
        # col.shape: (N*out_h*out_w, C*pool_h*pool_w)
        # after reshape: (N*out_h*out_w*C, pool_h*pool_w)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        # save for backward
        self.x_shape = x.shape
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

        # col.shape: (N*out_h*out_w, C*pool_h*pool_w)
        # dx.shape (N, C, H, W)
        dx = col2im(col, self.x_shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

class Flatten(Layer):
    def __init__(self, **args):
        super(Flatten, self).__init__(**args)
        if not self.name: 
            self.name = "flatten"
        self.x_shape = None
    
    def build(self, input_shape):
        assert(len(input_shape) > 1)
        self.input_shape = input_shape
        features = 1
        for i in range(1, len(input_shape)):
            features *= input_shape[i]
        self.output_shape = (input_shape[0], features)

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(self.x_shape)

class Reshape(Layer):
    def __init__(self, output_shape, **args):
        super(Reshape, self).__init__(**args)
        if not self.name: 
            self.name = "reshape"
        self.output_shape = output_shape
    
    def build(self, input_shape):
        assert(len(input_shape) > 1)
        input_size = 1
        output_size = 1
        for i in range(1, len(input_shape)):
            input_size *= input_shape[i]
        for i in range(1, len(self.output_shape)):
            output_size *= self.output_shape[i]
        assert(input_size == output_size)
        self.input_shape = input_shape

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape((x.shape[0],) + self.output_shape[1:])

    def backward(self, dy):
        return dy.reshape(self.input_shape)


