from NeuronModel.NeuronClass import neuronClass
import numpy as np
import matplotlib.pyplot as plt
hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

class stim(neuronClass):
    parameters = {'save': True, 'spike': True, 'noise': None}

    def __init__(self, **kwargs):
        neuronClass.__init__(self, **kwargs)
        neuronClass.subclass(self, stim.parameters, **kwargs)
        self.potential = [np.zeros(self.P)]
        self.current_potential = self.potential[-1]
        self.pattern = np.zeros(self.P)

        self.spike_noise_count = [np.zeros(self.P)]
        self.current_spike_noise_count = self.spike_noise_count[-1]

    def iterate(self, dt, **kwargs):
        event = self.pattern
        if self.parameters['noise'] is not None:
            spike_noise = np.random.binomial(1, dt * self.parameters['noise'], size=self.P)
            event = np.minimum(1., spike_noise + event)
        else:
            spike_noise = np.zeros_like(event)

        self.current_time = self.time[-1] + dt
        self.current_potential = np.zeros(self.P)
        self.current_spike_count = event
        self.current_spike_noise_count = spike_noise

    def update(self):
        neuronClass.update(self)
        if self.parameters['save']:
            self.spike_noise_count.append(self.current_spike_noise_count)
        else:
            self.spike_noise_count = [self.current_spike_noise_count]

    def init_eq(self):
        self.current_potential = np.zeros(self.P)
        self.current_spike_count = np.zeros(self.P)
        self.current_spike_noise_count = np.zeros(self.P)
        self.current_time = self.time[-1]
        self.update()

    def plot_trace(self, ax, index=None, label=''):
        ax.set_ylim(0, 2)
        ax.set_xlim(self.time[0], self.time[-1])
        for j, u in enumerate(index):
            ax.plot(self.time, [self.spike_count[i][u] for i in range(len(self.spike_count))],
                    label=label, color=colors(index)[j])
        ax.set_title('Potential')
