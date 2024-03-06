import numpy as np
from WeightModel.WeightClass import weightClass
from InitFunctions import gaussian

class STDP(weightClass):
    parameters = {'save': False, 'Aprepost': 1., 'Apostpre': -0.5, 'tprepost': 10., 'tpostpre': 10.,
                  'nearest': False, 'alpha': 0., 'w_min': 0., 'w_max': 1., 'homeostasy': 0., 'exp': True}

    def __init__(self, neuronInstance1, neuronInstance2, synapticClass, init_weight=gaussian(0., 1.), **kwargs):
        weightClass.__init__(self, neuronInstance1, neuronInstance2, synapticClass, init_weight=init_weight, **kwargs)
        weightClass.subclass(self, STDP.parameters, **kwargs)
        self.plasticity = True
        self.occlusion = 1.
        self.homeostasy_vector = np.ones(neuronInstance2.P)
        self.init_eq()

    def init_eq(self):
        if self.parameters['exp']:
            self.presynaptic_trace = [np.zeros(self.neurons_input.P)]
            self.postsynaptic_trace = [np.zeros(self.neurons_output.P)]
        else:
            self.presynaptic_trace = [np.inf * np.ones(self.neurons_input.P)]
            self.postsynaptic_trace = [np.inf * np.ones(self.neurons_output.P)]
        self.current_presynaptic_trace = self.presynaptic_trace[-1]
        self.current_postsynaptic_trace = self.postsynaptic_trace[-1]

    def iterate(self, dt, **kwargs):
        weightClass.iterate(self, dt)
        if self.plasticity:
            if self.parameters['exp']:
                if not(self.parameters['nearest']):
                    self.current_presynaptic_trace = self.presynaptic_trace[-1] * \
                        np.exp(-dt / self.parameters['tprepost']) \
                        + self.neurons_input.spike_count[-1]
                    self.current_postsynaptic_trace = self.postsynaptic_trace[-1] * \
                        np.exp(-dt/self.parameters['tpostpre']) \
                        + self.neurons_output.spike_count[-1]
                else:
                    self.current_presynaptic_trace = self.presynaptic_trace[-1] * \
                        np.exp(-dt/self.parameters['tprepost'])
                    self.current_presynaptic_trace = self.current_presynaptic_trace \
                        + (1 - self.current_presynaptic_trace) * \
                        self.neurons_input.spike_count[-1]
                    self.current_postsynaptic_trace = self.postsynaptic_trace[-1] * \
                        np.exp(-dt / self.parameters['tpostpre'])
                    self.current_postsynaptic_trace = self.current_postsynaptic_trace \
                        + (1 - self.current_postsynaptic_trace) * \
                        self.neurons_output.spike_count[-1]
                Delta_Wprepost = self.parameters['Aprepost'] * np.outer(self.neurons_output.spike_count[-1],
                                                                        self.presynaptic_trace[-1])
                Delta_Wpostpre = self.parameters['Apostpre'] * np.outer(self.postsynaptic_trace[-1],
                                                                        self.neurons_input.spike_count[-1])
            else:
                self.current_presynaptic_trace = self.presynaptic_trace[-1] + dt
                self.current_presynaptic_trace = np.where(self.neurons_input.spike_count[-1] < 0.5,
                                                          self.current_presynaptic_trace, 0.)
                self.current_postsynaptic_trace = self.postsynaptic_trace[-1] + dt
                self.current_postsynaptic_trace = np.where(self.neurons_output.spike_count[-1] < 0.5,
                                                           self.current_postsynaptic_trace, 0.)
                Delta_Wprepost = self.parameters['Aprepost'] * np.outer(
                    self.neurons_output.spike_count[-1], np.where((self.presynaptic_trace[-1]
                                                                   < self.parameters['tprepost']), 1., 0.))
                Delta_Wpostpre = self.parameters['Apostpre'] * np.outer(
                                    np.where((self.postsynaptic_trace[-1] < self.parameters['tpostpre']), 1., 0.),
                                    self.neurons_input.spike_count[-1])
            Delta_Wp = np.clip(Delta_Wprepost, 0., None) + np.clip(Delta_Wpostpre, 0., None)
            Delta_Wd = np.clip(Delta_Wprepost, None, 0.) + np.clip(Delta_Wpostpre, None, 0.)
            if self.parameters['alpha'] == 'clip':
                diff_W_STDP = Delta_Wp + Delta_Wd
                diff_W_homeostasy = self.parameters['homeostasy'] * np.outer(self.homeostasy_vector,
                                                                             self.neurons_input.spike_count[-1])
                self.current_weight = np.clip(self.weight[-1] + self.occlusion * (diff_W_STDP + diff_W_homeostasy),
                                              self.parameters['w_min'], self.parameters['w_max'])
            else:
                diff_W_STDP = (self.parameters['w_max'] - self.weight)**(self.parameters['alpha']) * Delta_Wp \
                              + (- self.parameters['w_min'] + self.weight)**(self.parameters['alpha']) * Delta_Wd
                diff_W_homeostasy = self.parameters['homeostasy'] * \
                    (self.parameters['w_max'] - self.weight)**(self.parameters['alpha']) * \
                    np.outer(self.homeostasy_vector, self.neurons_input.spike_count[-1])
                self.current_weight = self.weight[-1] + self.occlusion * (diff_W_STDP + diff_W_homeostasy)

    def update(self):
        weightClass.update(self)
        if self.save:
            self.presynaptic_trace.append(self.current_presynaptic_trace)
            self.postsynaptic_trace.append(self.current_postsynaptic_trace)
            self.mean_weight.append(np.mean(self.current_weight))
            self.var_weight.append(np.var(self.current_weight))
            self.weight.append(self.current_weight)
        else:
            self.presynaptic_trace = [self.current_presynaptic_trace]
            self.postsynaptic_trace = [self.current_postsynaptic_trace]
            self.mean_weight = [np.mean(self.current_weight)]
            self.var_weight = [np.var(self.current_weight)]
            self.weight = [self.current_weight]
