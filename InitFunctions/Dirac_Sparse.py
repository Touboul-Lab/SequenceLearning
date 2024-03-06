import numpy as np
import random

def dirac_sparse(prob,value,identity = False, normalize = True):
    def f(size):
        if normalize :
            sol = np.zeros(size)
            for i in range(size[0]):
                select = random.sample(range(0, size[1]), int(size[1]*prob))
                sol[i,select] = value
            return sol
        else :
            if identity :
                return value * np.random.binomial(1, prob, size) * (1 - np.eye(size[0]))
            else :
                return value * np.random.binomial(1, prob, size)
    return f