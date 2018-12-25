import numpy as np

class SGD:
    '''Stochastic Gradient Descent'''
    
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for k in params.keys():
            params[k] -= self.lr * grads[k]

class Momentum:
    ''' SGD with Momentum: https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/'''

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for k,v in params.items():
                self.v[k] = np.zeros_like(v)
                
        for k,v in params.items():
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]


class RMSProp:
    '''Root Mean Square Propogation: https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/

    It is recommended to leave the parameters of this optimizer at their default values (except the learning rate).
    This optimizer is usually a good choice for recurrent neural networks
    '''

    def __init__(self, lr=0.01, oth=0.9):
        self.lr = lr
        self.oth = oth


    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for k,v in params.items():
                self.v[k] = np.zeros_like(v)

        for k,v in params.items():
            self.v[k] = self.oth * self.v[k] + (1 - self.oth) * grads[k]*grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.v[k]) + 1e-7)