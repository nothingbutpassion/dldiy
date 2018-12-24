import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.average((y_true - y_pred)**2)


def categorical_crossentropy(y_true, y_pred):
    delta = 1e-7
    if y_true.ndim == 1:
        return -np.sum(y_true * np.log(y_pred + delta))
    return -np.sum(y_true * np.log(y_pred + delta))/y_true.shape[0]

