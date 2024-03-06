#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass

class fire(neuronClass) :
    parameters = {'save': True, 'spike': True, 'g' : 0.1, 'kappa' : None}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,fire.parameters,**kwargs)

    def iterate(self,dt,**kwargs) :
        I=kwargs['I']
        if self.parameters['spike'] :
            next_potential = 1*(I>self.parameters['g'])
            self.spike_count.append(next_potential)
        else :
            next_potential = self.parameters['kappa']*I
        self.potential.append(next_potential)
        self.time.append(self.time[-1] + dt)

