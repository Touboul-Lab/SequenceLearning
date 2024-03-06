#TO BE UPDATED

import numpy as np
from NetworkModel import networkClass
from NeuronModel import fire
from WeightModel import weightClass, potentialBased
from InitFunctions import dirac_sparse, dirac

class CA3(networkClass) :
    parameters = {'n' : 1000, 'N' : 500, 'c' : np.zeros((2,2)), 'J' : np.zeros((2,2)), 'g' : 1., 'kappa' : 1.}
    def __init__(self,**kwargs):
        networkClass.subclass(self,CA3.parameters,**kwargs)

        self.EXCITATORY = fire(P=self.parameters['n'], spike = True, g = self.parameters['g'])
        self.INHIBITORY = fire(P=self.parameters['N']-self.parameters['n'], spike = False, kappa = self.parameters['kappa'])
        self.W_II = weightClass(self.INHIBITORY, self.INHIBITORY, potentialBased,
                                init_weight = dirac(self.parameters['J'][0,0]/self.parameters['n']),
                                connectivity = dirac_sparse(self.parameters['c'][0,0], 1., identity=True))
        self.W_IE = weightClass(self.EXCITATORY, self.INHIBITORY, potentialBased,
                                init_weight = dirac(self.parameters['J'][0,1]/self.parameters['n']),
                                connectivity = dirac_sparse(self.parameters['c'][0,1], 1., identity=False))
        self.W_EI = weightClass(self.INHIBITORY, self.EXCITATORY, potentialBased,
                                init_weight = dirac(self.parameters['J'][1,0]/self.parameters['n']),
                                connectivity = dirac_sparse(self.parameters['c'][1,0], 1., identity=False))
        self.W_EE = weightClass(self.EXCITATORY, self.EXCITATORY, potentialBased,
                                init_weight = dirac(self.parameters['J'][1,1]/self.parameters['n']),
                                connectivity=dirac_sparse(self.parameters['c'][1,1], 1., identity=True))
        self.populations = [self.EXCITATORY,self.INHIBITORY]
        self.inputs = []
        self.weights = [self.W_II,self.W_IE,self.W_EI,self.W_EE]

