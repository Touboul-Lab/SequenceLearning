#TO BE UPDATED

import numpy as np
from WeightModel.WeightClass import weightClass
from InitFunctions import gaussian

class interactiveSynapse(weightClass) :
    parameters = {'save' : False,
                  'Cpre': 1., 'Cpost' : 2., 'tCa' : 20., 'delay_C' : 0. ,
                  'thetap': 1.3, 'thetad': 1., 'eta' : 0.0001,
                  'tau': 150. * 10000.,  'kappa' : 0.0, 'gammap': 10.*321.808, 'gammad': 10.*200.,
                  'H': lambda x : x,
                  'alpha': 0, 'w_min' : 0., 'w_max' : 1.
                  }
    def __init__(self,neuronInstance1,neuronInstance2,synapticClass,init_weight=gaussian(0.,1.),**kwargs):
        weightClass.__init__(self,neuronInstance1,neuronInstance2,synapticClass,init_weight=init_weight ,**kwargs)
        weightClass.subclass(self,interactiveSynapse.parameters,**kwargs)
        self.postsynaptic_calcium = np.zeros_like(self.weight)
        self.omega_p = np.zeros_like(self.weight, dtype=np.float32)
        self.omega_d = np.zeros_like(self.weight, dtype=np.float32)
        if self.parameters['delay_C']<0. :
            self.refractory_input = np.inf * np.ones_like(self.weight, dtype=np.float32)
        elif self.parameters['delay_C']>0. :
            self.refractory_output = np.inf * np.ones_like(self.weight, dtype=np.float32)
        if self.save :
            self.postsynaptic_calcium_history = [self.postsynaptic_calcium]
            self.omega_d_history = [self.omega_d]
            self.omega_p_history = [self.omega_p]

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
        self.postsynaptic_calcium = self.postsynaptic_calcium*np.exp(-dt/self.parameters['tCa'])
        self.postsynaptic_calcium += self.parameters['Cpre'] *  np.outer(np.ones(self.neurons_output.P),spike_input_delay ) \
                                     + self.parameters['Cpost'] * np.outer(spike_output_delay,np.ones(self.neurons_input.P))
        Delta_Wp = self.parameters['gammap'] * (self.postsynaptic_calcium > self.parameters['thetap'])
        Delta_Wd = self.parameters['gammad'] * (self.postsynaptic_calcium > self.parameters['thetad'])
        self.omega_p = self.omega_p + dt * self.parameters['eta'] * (-self.omega_p + Delta_Wp)
        self.omega_d = self.omega_d + dt * self.parameters['eta'] * (-self.omega_d + Delta_Wd)
        if self.parameters['alpha'] == 'clip':
            self.weight = self.weight+dt/self.parameters['tau']*(-self.weight*self.parameters['kappa'] +  self.omega_p -  self.omega_d)
            self.weight = np.clip(self.weight, self.parameters['w_min'], self.parameters['w_max'])
        else:
            self.weight = self.weight + dt/self.parameters['tau']*(-self.weight*self.parameters['kappa'] + (self.parameters['w_max'] - self.weight) ** (
            self.parameters['alpha']) * self.omega_p - (-self.parameters['w_min'] + self.weight) ** (
                          self.parameters['alpha']) * self.omega_d)
        self.mean_weight.append(np.mean(self.weight))
        self.var_weight.append(np.var(self.weight))
        if self.save :
            self.weight_history.append(self.weight)
            self.postsynaptic_calcium_history.append(self.postsynaptic_calcium)
            self.omega_d_history.append(self.omega_d)
            self.omega_p_history.append(self.omega_p)

    def plot_calcium(self,ax):
        ax.plot(self.time,[self.postsynaptic_calcium_history[i][0] for i in range(len(self.postsynaptic_calcium_history))])

    def plot_omega(self,ax):
        ax.plot(self.time,[self.omega_d_history[i][0] for i in range(len(self.postsynaptic_calcium_history))], label = "omegad")
        ax.plot(self.time,[self.omega_p_history[i][0] for i in range(len(self.postsynaptic_calcium_history))], label = "omegap")
        ax.legend()
