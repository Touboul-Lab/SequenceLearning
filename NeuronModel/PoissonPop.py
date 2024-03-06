from NeuronModel.NeuronClass import neuronClass
import numpy as np

import matplotlib.pyplot as plt
hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))


class poissonPop(neuronClass):
    parameters = {'save': True, 'spike': True, 'b': 1., 'N': 1}

    def __init__(self, **kwargs):
        neuronClass.__init__(self, **kwargs)
        neuronClass.subclass(self, poissonPop.parameters, **kwargs)
        self.potential = [np.zeros(self.P)]
        self.current_potential = self.potential[-1]
        self.pattern = np.zeros(self.P)

    def iterate(self, dt, **kwargs):
        event = np.random.binomial(self.parameters['N'], self.parameters['b'] * np.ones_like(self.potential[0]) * dt)
        self.current_time = self.time[-1] + dt
        self.current_potential = event
        self.current_spike_count = event
