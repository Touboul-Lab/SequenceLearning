#TO BE UPDATED

import numpy as np
import scipy
from WeightModel.WeightClass import weightClass
from InitFunctions import gaussian

class lebastard(weightClass) :
    parameters = {'save' : False,
                  'Cpre': 1., 'Cpost' : 2., 'tCa' : 20.,
                  'gammap' : 110., 'gammad' : 321.,'tau_rho' : 30.*1000., 'thetap' : 1.3, 'thetad' : 1.,
                  'tau_adapt' : 100., 'r_pre' : 0.3, 'r_post' : 0.3,
                  'rho_max' : 200. ,
                  's_attr' : 40., 't_stop' : 4., 't_0' : 1., 'sigma' : 1.,
                  'tau_w' : None,
                  }
    def __init__(self,neuronInstance1,neuronInstance2,synapticClass,init_weight=gaussian(0.,1.),**kwargs):
        weightClass.__init__(self,neuronInstance1,neuronInstance2,synapticClass,init_weight=init_weight ,**kwargs)
        weightClass.subclass(self,lebastard.parameters,**kwargs)
        self.postsynaptic_calcium = np.zeros_like(self.weight)
        self.rho = 40.*np.ones_like(self.weight)
        self.x_pre = np.zeros(neuronInstance1.P)
        self.x_post = np.zeros(neuronInstance2.P)
        self.postsynaptic_calcium_history = [self.postsynaptic_calcium]
        self.plasticity = True

    def iterate(self,dt):
        weightClass.iterate(self,dt)
        self.x_pre += dt/self.parameters['tau_adapt']*((1.-self.x_pre)-self.parameters['r_pre']*self.x_pre*self.neurons_input.spike_count[-1])
        self.x_post += dt / self.parameters['tau_adapt'] * (
                    (1. - self.x_post) - self.parameters['r_post'] * self.x_post * self.neurons_output.spike_count[-1])
        self.postsynaptic_calcium += self.parameters['Cpre'] * np.outer(np.ones(self.neurons_output.P),self.x_pre*self.neurons_input.spike_count[-1]) \
                                     + self.parameters['Cpost'] * np.outer(self.x_post*self.neurons_output.spike_count[-1],np.ones(self.neurons_input.P))
        self.postsynaptic_calcium = self.postsynaptic_calcium*np.exp(-dt/self.parameters['tCa'])
        if self.plasticity :
            Delta_Wp = self.parameters['gammap'] * (self.postsynaptic_calcium > self.parameters['thetap'])
            Delta_Wd = self.parameters['gammad'] * (self.postsynaptic_calcium > self.parameters['thetad'])
        else :
            Delta_Wp = 0.
            Delta_Wd = 0.
        self.rho = self.rho+dt/self.parameters['tau_rho']*(-self.rho*(self.rho-self.parameters['s_attr'])*(self.rho-self.parameters['rho_max'])+(self.parameters['rho_max']-self.rho)*Delta_Wp-self.rho*Delta_Wd)
        if self.parameters['tau_w'] is None :
            self.weight = 0.5*scipy.special.erfc((self.parameters['s_attr']-self.rho)/(2.*self.parameters['sigma']*(self.parameters['t_stop']-self.parameters['t_0'])**2))
        else :
            self.weight += -dt/self.parameters['tau_w']*(self.weight-0.5*scipy.special.erfc((self.parameters['s_attr']-self.rho)/(2.*self.parameters['sigma']*(self.parameters['t_stop']-self.parameters['t_0'])**2)))
        self.mean_weight.append(np.mean(self.weight))
        self.var_weight.append(np.var(self.weight))
        if self.save :
            self.weight_history.append(self.weight)
            self.postsynaptic_calcium_history.append(self.postsynaptic_calcium)

    def plot_calcium(self,ax):
        ax.plot([self.postsynaptic_calcium_history[i][0] for i in range(len(self.postsynaptic_calcium_history))])
