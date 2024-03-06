import numpy as np

def gaussian(mean,var):
    def f(size):
        return np.random.normal(mean,var,size)
    return f

def gaussian_EI(r,J,sigma):
    def f(size):
        sol = np.zeros(size)
        N = size[0]
        if size[0]!=size[1] :
            print('Not right')
        else :
            Ne = int(size[0]*r)
            Ni = size[0]-Ne
            for i in range(Ne) :
                for j in range(Ne) :
                    sol[i, j] = np.random.normal(J[0],sigma[0])
                for j in np.arange(Ne,N) :
                    sol[i, j] = np.random.normal(J[1], sigma[1])
            for i in np.arange(Ne,N) :
                for j in range(Ne) :
                    sol[i, j] = np.random.normal(J[2],sigma[2])
                for j in np.arange(Ne,N) :
                    sol[i, j] = np.random.normal(J[3], sigma[3])
        return sol
    return f