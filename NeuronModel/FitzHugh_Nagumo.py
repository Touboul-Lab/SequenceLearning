#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np

class fitzugh_nagumo(neuronClass) :
    parameters = {'save': True,
                  'spike': False,
                  'tau' : 1.,
                  'a' : 0.8,
                  'b' : 0.7,
                  'kappa' : None
                  }
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,fitzugh_nagumo.parameters,**kwargs)
        self.potential[0]=np.zeros(self.P)
        self.w=[np.zeros(self.P)]

    def iterate(self,dt,I=0.) :
        potential = self.potential[-1]
        w = self.w[-1]
        if self.parameters['kappa'] is None :
            next_potential=potential+dt*(potential-potential**3/3-w+I)
            next_w=w+dt/self.parameters['tau']*(potential+self.parameters['a']-self.parameters['b']*w)
        else :
            next_potential=potential+dt*(potential*(1-potential)*(potential-self.parameters['kappa'])-w+I)
            next_w=w+dt*self.parameters['a']*(potential*self.parameters['b']-w)
        self.time.append(self.time[-1] + dt)
        self.potential.append(next_potential)
        self.w.append(next_w)

    def plot_phase(self,ax):
        ax.plot(self.w,self.potential)

    def plot_trace(self,ax,index=None,label=''):
        neuronClass.plot_trace(self,ax,index=index,label=label)
        ax.set_ylim(-3.,3.)