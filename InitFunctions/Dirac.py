import numpy as np

def dirac(value):
    def f(size):
        if hasattr(value, 'shape'):
            if len(value.shape) == 0:
                return value * np.ones(size)
            else:
                return value
        else:
            return value*np.ones(size)
    return f