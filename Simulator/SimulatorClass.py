import numpy as np
import copy

class simulatorClass:

    def __init__(self, networkInstance, dt=0.1, T=1.):
        self.network = networkInstance
        self.dt = dt
        self.T = T
        self.time = np.arange(0., self.T, self.dt)
        self.N = len(self.time)
        self.parameters = None

    def subclass(self, params, **kwargs):
        self.parameters = copy.deepcopy(params)
        for key in params.keys():
            if key in kwargs.keys():
                self.parameters[key] = kwargs[key]

    def simulate(self):
        for i in range(self.N):
            self.network.iterate(self.dt)
