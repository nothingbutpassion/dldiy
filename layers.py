import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x < 0;
        y = x.copy();
        y[self.mask] = 0;
        return y

    def backward(self, dy):
        dx = dy.copy()
        dx[self.mask] = 0
        return dx

class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        dx = dy*self.y*(1-self.y)
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
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

class SoftMax:
    def __init__(self)
        pass
        
    def forward(self, x):
        max_x = np.max(x)
        exp_x = np.exp(x-max_x)
        y = exp_x/np.sum(exp_x)
        return y
        
    def backward(self, dy):
        
        pass
