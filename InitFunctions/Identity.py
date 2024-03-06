import numpy as np

def identity(x):
    def f(size):
        if size[0]!=size[1] :
            print('Not right')
        else :
            return x*np.eye(size[0])
    return f