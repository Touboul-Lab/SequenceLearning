import numpy as np
import copy
from InitFunctions import gaussian
import matplotlib.pyplot as plt

hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

class neuronClass:
    parameters = {'save': True, 'spike': False}

    def __init__(self, P=1, init=gaussian(-60., 1.), **kwargs):
        self.P = P
        self.potential = [init((self.P,))]
        self.current_potential = self.potential[-1]
        self.time = [0.]
        self.current_time = self.time[-1]
        self.input = []
        self.input_weight = []

    def subclass(self, params, **kwargs):
        self.parameters = copy.deepcopy(params)
        for key in params.keys():
            if key in kwargs.keys():
                self.parameters[key] = kwargs[key]
        if self.parameters['spike']:
            self.spike_count = [np.zeros(self.P)]
            self.current_spike_count = self.spike_count[-1]

    def iterate(self, dt):
        self.current_time = self.time[-1] + dt
        self.current_potential = self.potential[-1]
        if self.parameters['spike']:
            self.spike_noise_count = [np.zeros(self.P)]

    def update(self):
        if self.parameters['save']:
            self.time.append(self.current_time)
            self.potential.append(self.current_potential)
            if self.parameters['spike']:
                self.spike_count.append(self.current_spike_count)
        else:
            self.time = [self.current_time]
            self.potential = [self.current_potential]
            if self.parameters['spike']:
                self.spike_count.append = [self.current_spike_count]

    def init_eq(self):
        self.current_potential = np.zeros(self.P)
        if self.parameters['spike']:
            self.current_spike_count = np.zeros(self.P)
        self.current_time = self.time[-1]
        self.update()

    def value(self, index=1):
        return self.potential[-index]

    def plot_trace(self, ax, index=None, label=''):
        ax.set_ylim(-100., 50.)
        ax.set_xlim(self.time[0], self.time[-1])
        for j, u in enumerate(index):
            ax.plot(self.time, [self.potential[i][u] for i in range(len(self.potential))],
                    label=label, color=colors(index)[j])
        ax.set_title('Potential')

    def plot_mean_trace(self, ax, index=None):
        if index is None:
            mean = [np.mean(self.potential[i]) for i in range(len(self.potential))]
        else:
            mean = [np.mean(self.potential[i][index]) for i in range(len(self.potential))]
        time = self.time
        ax.plot(time, mean)

    def plot_raster(self, ax):
        spikes = np.nonzero(np.array(self.spike_count))
        ax.scatter([self.time[u] for u in spikes[0]], spikes[1], color='k', s=1., marker='.')

    def plot_rate(self, ax):
        spikes = np.sum(np.array(self.spike_count), axis=1)
        ax.plot(self.time, spikes, '-')
