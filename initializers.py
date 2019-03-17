import numpy as np

def _he_normal(shape):
    """ He kaiming normal initalizer

    A truncated normal distribution centered on 0 with "stddev = sqrt(2 / fan_in)"
    """
    fan_in = shape[0]
    return np.random.randn(*shape) * np.sqrt(2.0/fan_in)


def _lecun_normal(shape):
    """LeCun normal initializer.

    A truncated normal distribution centered on 0 with "stddev = sqrt(1 / fan_in)"
    """
    fan_in = shape[0]
    return np.random.randn(*shape) * np.sqrt(1.0/fan_in)

class StdNormal:
    def __call__(self, shape):
        return np.random.randn(*shape)

class HeNormal:
    def __call__(self, shape):
        return _he_normal(shape)

class LeCunNormal:
    def __call__(self, shape):
        return _lecun_normal(shape)

