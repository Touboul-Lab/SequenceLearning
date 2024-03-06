from NeuronModel.Integrate_and_Fire import integrate_and_fire
import numpy as np

import matplotlib.pyplot as plt
hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

def R_func(x, params):
    return params['RI_slow'] + (params['RI_fast'] - params['RI_slow']) / \
            (1. + np.exp((x-params['V_RI'])/params['tau_V_RI']))

class integrate_and_fire_MSN(integrate_and_fire):
    parameters = {'save': True, 'spike': True, 'RI_slow': 80., 'RI_fast': 20., 'V_RI': 10., 'tau_V_RI': 10.,
                  'C': 10., 'E_l': -65., 'V_th': -50., 'E_r': -70., 'E_reset': 20., 'Delta_abs': 0.,
                  'noise': None, 'scale_I': None}

    def __init__(self, **kwargs):
        integrate_and_fire.__init__(self, **kwargs)
        integrate_and_fire.subclass(self, integrate_and_fire_MSN.parameters, **kwargs)
        self.refractory = np.inf*np.ones(self.P)
        potential = self.potential[-1]
        R = R_func(potential, self.parameters)
        self.parameters['tau'] = self.parameters['C'] * R

    def iterate(self, dt, **kwargs):
        if self.parameters['scale_I'] is None:
            current_I = kwargs['I']
        else:
            current_I = kwargs['I'] * self.parameters['scale_I']
        potential = self.potential[-1]
        R = R_func(potential, self.parameters)
        self.parameters['tau'] = self.parameters['C'] * R
        if self.parameters['noise'] is not None:
            next_potential = potential + dt/(self.parameters['C'] * R) * \
                            (self.parameters['E_l'] - potential + R * current_I) \
                            + np.sqrt(dt) * np.random.normal(0., self.parameters['noise'])
        else:
            next_potential = potential + dt/(self.parameters['C'] * R) * \
                             (self.parameters['E_l'] - potential + R * current_I)
        next_potential = np.where(self.refractory <= self.parameters['Delta_abs'], self.parameters['E_r'],
                                  next_potential)
        self.refractory += dt
        event = 1 * (next_potential > self.parameters['V_th'])
        self.refractory = np.where(event, 0., self.refractory)

        self.current_time = self.time[-1] + dt
        self.current_potential = np.where(event, self.parameters['V_th'], next_potential)
        self.current_spike_count = event