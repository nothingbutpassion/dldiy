import numpy as np

def he_normal(shape):
    """ He kaiming normal initalizer

    A truncated normal distribution centered on 0 with "stddev = sqrt(2 / fan_in)"
    """
    fan_in = shape[0]
    return np.random.rand(*shape) * np.sqrt(2.0/fan_in)


def lecun_normal(shape):
    """LeCun normal initializer.

    A truncated normal distribution centered on 0 with "stddev = sqrt(1 / fan_in)"
    """
    fan_in = shape[0]
    return np.random.rand(*shape) * np.sqrt(1.0/fan_in)

        
    

