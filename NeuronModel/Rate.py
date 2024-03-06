#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
from InitFunctions import gaussian

class rate(neuronClass) :
    parameters = {'save': True, 'spike':False, 'tau': 10., 'R': 1.}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,rate.parameters,**kwargs)
        self.potential[0]=gaussian(0.,1.)(self.P)
    def iterate(self,dt,I=0.) :
        next_potential=self.potential[-1]+dt/self.parameters['tau']*(-self.potential[-1]+self.parameters['R']*I)
        self.time.append(self.time[-1] + dt)
        self.potential.append(next_potential)