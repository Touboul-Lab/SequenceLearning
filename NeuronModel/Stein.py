#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np

class stein(neuronClass):
    parameters = {'save': True, 'spike':False,'tau':10.,'R':10.,'E_l':-65., 'a_e':0.1, 'a_i':0.1}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,stein.parameters,**kwargs)
    def iterate(self,dt,**kwargs):
        I_i=kwargs['I_i']
        I_e=kwargs['I_e']
        potential = self.potential[-1]
        next_potential = potential+dt/self.parameters['tau']*(self.parameters['E_l']-potential)+self.parameters['R']*(self.parameters['a_e']*np.random.binomial(1,I_e)-self.parameters['a_i']*np.random.binomial(1,I_i))
        self.time.append(self.time[-1] + dt)
        self.potential.append(next_potential)