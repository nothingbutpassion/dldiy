import numpy as np

def mean_squared_error(y_true, y_pred):
    if y_pred.ndim == 1: 
        return 0.5*np.sum((y_true - y_pred)**2)

    return 0.5*np.sum((y_true - y_pred)**2)/y_pred.shape[0]

def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    if y_pred.ndim == 1:
        return -np.sum(y_true * np.log(y_pred + epsilon))
    return -np.sum(y_true * np.log(y_pred + epsilon))/y_pred.shape[0]


class MSE:
    def loss(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    def grad(self, y_true, y_pred ):
        return y_pred - y_true


class CrossEntropy:
    def loss(self, y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred)
    def grad(self, y_true, y_pred):
        epsilon = 1e-7
        return -y_true/(y_pred + epsilon)
        
    

