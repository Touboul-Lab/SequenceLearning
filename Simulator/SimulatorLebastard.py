#TO BE UPDATED

import numpy as np
from NeuronModel import integrate_and_fire, poissonPop
from NetworkModel import networkClass
from WeightModel import weightClass, lebastard, spikeBased
from InitFunctions import identity, dirac, dirac_sparse
from Simulator.SimulatorClass import simulatorClass


class simulatorLebastard(simulatorClass):
    parameters = {'PE' : 320., 'PI': 80., 'g': 6., 'J': 0.1, 'c': 0.1}
    def __init__(self,dt=0.1,**kwargs):
        simulatorClass.subclass(self,simulatorLebastard.parameters,**kwargs)
        self.E = integrate_and_fire(P=self.parameters['PE'], tau=20., V_th=20., E_l=0, E_r=10., E_reset=40., init=dirac(10.), R=1.,Delta_abs=2.)
        self.I = integrate_and_fire(P=self.parameters['PI'], tau=20., V_th=20., E_l=0, E_r=10., E_reset=40., init=dirac(10.), R=1., Delta_abs=2.)
        self.INPUTE = poissonPop(P=self.parameters['PE'], b=b, N=0.1*self.parameters['PE'])
        self.INPUTI = poissonPop(P=self.parameters['PI'], b=b, N=0.1*self.parameters['PE'])
        W_inputE = weightClass(self.INPUTE,self.E, spikeBased, init_weight=identity(self.parameters['J']), delay = int(3/dt))
        W_inputI = weightClass(self.INPUTI, self.I, spikeBased, init_weight=identity(self.parameters['J']), delay = int(3/dt))
        self.WEE = lebastard(self.E,self.E, spikeBased, init_weight = dirac(0.5), connectivity=dirac_sparse(self.parameters['c'],2.*self.parameters['J'], normalize = True), delay = int(3/dt), save=True)
        self.WEI = weightClass(self.I,self.E, spikeBased, init_weight=dirac_sparse(self.parameters['c'],-self.parameters['J']*self.parameters['g'], normalize = True), delay = int(3/dt), save=True)
        self.WIE = lebastard(self.E,self.I, spikeBased, init_weight = dirac(0.5), connectivity=dirac_sparse(self.parameters['c'],2.*self.parameters['J'], normalize = True), delay = int(3/dt))
        self.WII = weightClass(self.I,self.I, spikeBased, init_weight=dirac_sparse(self.parameters['c'],-self.parameters['J']*self.parameters['g'], normalize = True), delay = int(3/dt))
        self.network = networkClass([self.INPUTE,self.INPUTI,self.E,self.I],[W_inputE,W_inputI,self.WEE,self.WEI,self.WIE,self.WII],[])
        self.dt = dt
        self.T = np.array([20.,20.,0.])
        self.time = [np.arange(0.,self.T[i],self.dt) for i in range(len(self.T))]
        self.N = [len(self.time[i]) for i in range(len(self.T))]

    def simulate(self):
        self.WEE.plasticity = False
        self.WIE.plasticity = False
        for i in range(self.N[0]):
            self.network.iterate(self.dt)
        self.WEE.plasticity = True
        self.WIE.plasticity = True
        for i in range(self.N[1]):
            self.network.iterate(self.dt)
        self.WEE.plasticity = False
        self.WIE.plasticity = False
        for i in range(self.N[2]):
            self.network.iterate(self.dt)