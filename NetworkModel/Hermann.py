#TO BE UPDATED

import numpy as np
from NeuronModel import rate, fitzugh_nagumo
from NetworkModel import networkClass
from WeightModel import weightClass, rateBased, gapJuctions
from InitFunctions import gaussian_EI
from Input import inputClass
import pickle

class hermann(networkClass) :
    parameters = {'neuron' : 'rate', 'P' : 1000, 'I1' : 0., 'I2' : -3., 'J' : np.array([15.,-12.,16.,-5.]), 'sigma' : 0., 'rho' : (lambda x :(np.tanh(x)+1.)/2.)}
    def __init__(self,**kwargs):
        networkClass.subclass(self,hermann.parameters,**kwargs)
        if self.parameters['neuron'] == 'rate' :
            NEURON = rate(P=self.parameters['P'], R=1., tau = 1.)
            I = np.concatenate([self.parameters['I1']* np.ones(int(self.parameters['P'] / 2)), self.parameters['I2'] * np.ones(int(self.parameters['P'] / 2))])
            INPUT = inputClass(NEURON, I)
            WEIGHT = weightClass(NEURON, NEURON, rateBased,
                                 init_weight=gaussian_EI(0.5, 2. / self.parameters['P'] * self.parameters['J'],
                                                         self.parameters['sigma']* np.array([1., 1., 1., 1.]) / np.sqrt(self.parameters['P'] / 2.)), rho=self.parameters['rho'],
                                 save=False)
        elif self.parameters['neuron'] == 'FH' :
            NEURON = fitzugh_nagumo(P=self.parameters['P'], kappa = -2., a=0.4, b=2.)
            I = np.concatenate([self.parameters['I1']* np.ones(int(self.parameters['P'] / 2)), self.parameters['I2'] * np.ones(int(self.parameters['P'] / 2))])
            INPUT = inputClass(NEURON, I)
            WEIGHT = weightClass(NEURON, NEURON, gapJuctions,
                                 init_weight=gaussian_EI(0.5, 1. / self.parameters['P'] * self.parameters['J'],
                                                         self.parameters['sigma']* np.array([1., 1., 1., 1.]) / np.sqrt(self.parameters['P'])),
                                 save=False)
        self.populations = [NEURON]
        self.inputs = [INPUT]
        self.weights = [WEIGHT]

    def iterate(self,dt):
        for pop in self.populations :
            pop.iterate(dt, I = sum([u.I_input for u in pop.input])+sum([u.synaptic_current.I_output for u in pop.input_weight]))
        for weight in self.weights :
            weight.iterate(dt)
        for input in self.inputs :
            input.iterate()

    def save(self,path):
        with open(path+".network", "wb") as file:
            pickle.dump(self, file, -1)