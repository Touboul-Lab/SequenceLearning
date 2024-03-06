#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass

class integrate(neuronClass) :
    parameters = {'save': True, 'spike':False, 'tau': 10., 'R': 10., 'E_l': -65.}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,integrate.parameters,**kwargs)
    def iterate(self,dt,I=0.) :
        next_potential=self.potential[-1]+dt/self.parameters['tau']*(self.parameters['E_l']-self.potential[-1]+self.parameters['R']*I)
        self.time.append(self.time[-1] + dt)
        self.potential.append(next_potential)