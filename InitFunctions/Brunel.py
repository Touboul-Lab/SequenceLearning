import numpy as np

import random
def random_no_repeat(numbers, count):
    number_list = list(numbers)
    random.shuffle(number_list)
    return np.array(number_list[:count])

def brunel(r , J , g, epsilon):
    def f(size):
        sol = np.zeros(size)
        if size[0]!=size[1] :
            print('Not right')
        else :
            Ne = int(size[0]*r)
            Ni = size[0]-Ne
            Ce = int(epsilon*Ne)
            Ci = int(epsilon*Ni)
            for i in range(size[0]) :
                index_e = random_no_repeat(range(Ne), Ce)
                sol[i, index_e] = J
                if Ni>0 :
                    index_i = Ne+random_no_repeat(range(Ni), Ci)
                    sol[i, index_i] = -g*J
        return sol
    return f