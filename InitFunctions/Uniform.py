import numpy as np

def uniform(low,up):
    def f(size):
        return np.random.uniform(low,up,size)
    return f