import numpy as np
import copy
from InitFunctions import gaussian
import matplotlib.pyplot as plt

hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

class weightClass:
    parameters = {'save': False}

    def __init__(self, neuronInstance1, neuronInstance2, synapticInstance, init_weight=gaussian(0., 1.),
                 connectivity=None, **kwargs):
        self.weight = [init_weight((neuronInstance2.P, neuronInstance1.P))]
        self.current_weight = self.weight
        self.mean_weight = [np.mean(self.current_weight)]
        self.var_weight = [np.var(self.current_weight)]

        self.neurons_input = neuronInstance1
        self.neurons_output = neuronInstance2
        self.neurons_output.input_weight.append(self)

        if connectivity is None:
            self.connectivity = False
        else:
            self.connectivity = True
            self.connectivity_matrix = connectivity((neuronInstance2.P, neuronInstance1.P))

        self.synaptic_current = synapticInstance(self, **kwargs)

        self.time = [0]
        self.current_time = self.time[0]

        self.save = self.parameters['save']

    def subclass(self, params, **kwargs):
        self.parameters = copy.deepcopy(params)
        for key in params.keys():
            if key in kwargs.keys():
                self.parameters[key] = kwargs[key]
        self.save = self.parameters['save']

    def iterate(self, dt, *args):
        self.current_time = self.time[-1] + dt
        self.synaptic_current.iterate(dt)

    def update(self):
        if self.save:
            self.time.append(self.current_time)
        else:
            self.time = [self.current_time]
        self.synaptic_current.update()

    def plot_history(self, ax, index=((0, 0),), linestyle='-'):
        if self.save:
            for j, u in enumerate(index):
                ax.plot(self.time, [self.weight[i][u] for i in range(len(self.weight))],
                        color=colors(index)[j], linestyle=linestyle)

    def plot_histogram(self, ax):
        ax.hist(self.weight[-1].flatten())

    def plot_mean(self, ax):
        ax.plot(self.time, self.mean_weight)
        ax.plot(self.time, self.var_weight)
