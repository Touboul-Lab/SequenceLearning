#TO BE UPDATED

import numpy as np
import math
from WeightModel.WeightClass import weightClass
from InitFunctions import gaussian

class graupner(weightClass) :
    parameters = {'save' : False, 'step' : None,
                  'Cpre': 1., 'Cpost' : 2., 'tCa' : 20., 'delay_C' : 0.,
                  'gammap' : 321.808, 'gammad' : 200.,'tau' : 150.*1000., 'sigma' : 0., 'thetap' : 1.3, 'thetad' : 1.}
    def __init__(self,neuronInstance1,neuronInstance2,synapticClass,init_weight=gaussian(0.,1.),**kwargs):
        weightClass.__init__(self,neuronInstance1,neuronInstance2,synapticClass,init_weight=init_weight ,**kwargs)
        weightClass.subclass(self,graupner.parameters,**kwargs)
        self.postsynaptic_calcium = np.zeros_like(self.weight, dtype=np.float32)
        if self.parameters['delay_C']<0. :
            self.refractory_input = np.inf * np.ones_like(self.weight, dtype=np.float32)
        elif self.parameters['delay_C']>0. :
            self.refractory_output = np.inf * np.ones_like(self.weight, dtype=np.float32)
        self.postsynaptic_calcium_history = [self.postsynaptic_calcium]

    def iterate(self,dt):
        weightClass.iterate(self,dt)
        if self.parameters['delay_C'] == 0. :
            spike_input_delay = self.neurons_input.spike_count[-1]
            spike_output_delay = self.neurons_output.spike_count[-1]
        elif self.parameters['delay_C']>0. :
            self.refractory_output = np.where(self.neurons_output.spike_count[-1], 0, self.refractory_output)
            spike_input_delay = self.neurons_input.spike_count[-1]
            spike_output_delay = 1.*(np.abs(self.refractory_output-self.parameters['delay_C']) < dt)
            self.refractory_output += dt
        elif self.parameters['delay_C']<0. :
            self.refractory_input = np.where(self.neurons_input.spike_count[-1], 0, self.refractory_input)
            spike_input_delay = 1.*(np.abs(self.refractory_input-self.parameters['delay_C']) < dt)
            spike_output_delay = self.neurons_output.spike_count[-1]
            self.refractory_input += dt
        self.postsynaptic_calcium += self.parameters['Cpre'] *  np.outer(np.ones(self.neurons_output.P),spike_input_delay) \
                                     + self.parameters['Cpost'] * np.outer(spike_output_delay,np.ones(self.neurons_input.P))
        if self.parameters['step'] is None :
            self.postsynaptic_calcium = self.postsynaptic_calcium*np.exp(-dt/self.parameters['tCa'])
        else :
            decrease = np.random.binomial(1, 1. / self.parameters['step'] * dt / self.parameters['tCa'] * self.postsynaptic_calcium)
            self.postsynaptic_calcium = np.where(decrease, np.maximum(0,self.postsynaptic_calcium - self.parameters['step']), self.postsynaptic_calcium)
        Delta_Wp = self.parameters['gammap'] * (self.postsynaptic_calcium > self.parameters['thetap'])
        Delta_Wd = self.parameters['gammad'] * (self.postsynaptic_calcium > self.parameters['thetad'])
        Noise = math.sqrt(dt/self.parameters['tau'])*self.parameters['sigma']*np.sqrt(1.*(self.postsynaptic_calcium > self.parameters['thetap'])+1.*(self.postsynaptic_calcium > self.parameters['thetad']))*np.random.normal(0.,1.,np.shape(self.weight))
        self.weight = self.weight+dt/self.parameters['tau']*(-self.weight*(self.weight-0.5)*(self.weight-1.)+(1.-self.weight)*Delta_Wp-self.weight*Delta_Wd) + 1./self.parameters['tau']*Noise
        self.mean_weight.append(np.mean(self.weight))
        self.var_weight.append(np.var(self.weight))
        if self.save :
            self.weight_history.append(self.weight)
            self.postsynaptic_calcium_history.append(self.postsynaptic_calcium)

    def plot_calcium(self,ax):
        ax.plot([self.postsynaptic_calcium_history[i][0] for i in range(len(self.postsynaptic_calcium_history))])
