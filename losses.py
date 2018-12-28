import numpy as np

def _mean_squared_error(y_true, y_pred):
    if y_pred.ndim == 1: 
        return 0.5*np.sum((y_true - y_pred)**2)

    return 0.5*np.sum((y_true - y_pred)**2)/y_pred.shape[0]

def _categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    if y_pred.ndim == 1:
        return -np.sum(y_true * np.log(y_pred + epsilon))
    return -np.sum(y_true * np.log(y_pred + epsilon))/y_pred.shape[0]

def _categoriacl_accuracy(y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.sum(y_pred == y_true) / float(y_true.shape[0])

class MSE:
    def loss(self, y_true, y_pred):
        return _mean_squared_error(y_true, y_pred)
    def grad(self, y_true, y_pred ):
        return y_pred - y_true


class CrossEntropy:
    def loss(self, y_true, y_pred):
        return _categorical_crossentropy(y_true, y_pred)
    
    def accuracy(self, y_true, y_pred):
        return _categoriacl_accuracy(y_true, y_pred)

    def grad(self, y_true, y_pred):
        epsilon = 1e-7
        return -y_true/(y_pred + epsilon)
        
    

