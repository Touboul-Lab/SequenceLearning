from NeuronModel.NeuronClass import neuronClass
import numpy as np

import matplotlib.pyplot as plt
hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

class integrate_and_fire(neuronClass):
    parameters = {'save': True, 'spike': True, 'tau': 10., 'R': 10., 'E_l': -65., 'V_th': -50., 'E_r': -70.,
                  'E_reset': 20., 'Delta_abs': 0., 'noise': None, 'scale_I': None}

    def __init__(self, **kwargs):
        neuronClass.__init__(self, **kwargs)
        neuronClass.subclass(self, integrate_and_fire.parameters, **kwargs)
        self.refractory = np.inf * np.ones(self.P)

    def iterate(self, dt, **kwargs):
        if self.parameters['scale_I'] is None:
            current_I = kwargs['I']
        else:
            current_I = kwargs['I'] * self.parameters['scale_I']
        potential = self.potential[-1]
        if self.parameters['noise'] is not None:
            next_potential = potential + dt / self.parameters['tau'] * \
                    (self.parameters['E_l'] - potential + self.parameters['R'] * current_I) \
                    + np.sqrt(dt/self.parameters['tau']) * np.random.normal(0., self.parameters['noise'])
        else:
            next_potential = potential + dt / self.parameters['tau'] * \
                    (self.parameters['E_l'] - potential + self.parameters['R'] * current_I)
        next_potential = np.where(self.refractory <= self.parameters['Delta_abs'], self.parameters['E_r'],
                                  next_potential)
        self.refractory += dt
        event = 1 * (next_potential > self.parameters['V_th'])
        self.refractory = np.where(event, 0., self.refractory)

        self.current_time = self.time[-1] + dt
        self.current_potential = np.where(event, self.parameters['V_th'], next_potential)
        self.current_spike_count = event

    def init_eq(self):
        self.current_potential = self.parameters['E_l'] * np.ones(self.P)
        self.current_spike_count = np.zeros(self.P)
        self.current_time = self.time[-1]
        self.refractory = np.inf*np.ones(self.P)
        self.update()

    def plot_trace(self, ax, index=None, label=''):
        ax.set_ylim(-100., 50.)
        ax.set_xlim(self.time[0], self.time[-1])
        for j, u in enumerate(index):
            ax.plot(self.time, [self.potential[i][u]+self.spike_count[i][u]
                                * (self.parameters['E_reset']-self.parameters['V_th'])
                                for i in range(len(self.potential))], label=label, color=colors(index)[j])
        ax.set_title('Potential')

    def plot_normalized(self, ax):
        ax.plot(self.time, (np.array(self.potential) - self.parameters['E_reset']) / (
                    self.parameters['V_th'] - self.parameters['E_r']), label='noise input')
        ax.set_ylim(0., 1.5)
        ax.set_xlim(self.time[0], self.time[-1])
