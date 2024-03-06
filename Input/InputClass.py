import numpy as np
import copy

class inputClass:
    parameters = {}

    def __init__(self, neuronInstance, I_value, noise=None):
        self.neurons = neuronInstance
        self.noise = noise
        self.I_value = I_value
        self.current_I_input = I_value
        self.neurons.input.append(self)

    def subclass(self, params, **kwargs):
        self.parameters = copy.deepcopy(params)
        for key in params.keys():
            if key in kwargs.keys():
                self.parameters[key] = kwargs[key]

    def iterate(self):
        if self.noise is not None:
            self.current_I_input = self.I_value + np.random.normal(0., self.noise, np.shape(self.I))

    def update(self):
        pass
