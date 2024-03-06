#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np

import matplotlib.pyplot as plt
hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

class poisson(neuronClass) :
    parameters = {'save': True, 'spike': True, 'tau': 10., 'R': 10., 'E_l': -65., 'E_r': -70., 'b': (lambda x :np.maximum(0.,x+50.)), 'E_reset' : 20.}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,poisson.parameters,**kwargs)
    def iterate(self,dt,**kwargs) :
        I=kwargs['I']
        potential = np.where(self.spike_count[-1],self.parameters['E_r'],self.potential[-1])
        next_potential = potential+dt/self.parameters['tau']*(self.parameters['E_l']-potential+self.parameters['R']*I)
        event = np.random.binomial(1,self.parameters['b'](potential)*dt)
        self.potential.append(next_potential)
        self.spike_count.append(event)
        self.time.append(self.time[-1] + dt)

    def plot_trace(self,ax,index=None,label=''):
        ax.set_ylim(-100.,50.)
        ax.set_xlim(self.time[0], self.time[-1])
        for j,u in enumerate(index):
            ax.plot(self.time,[self.potential[i][u]+self.spike_count[i][u]*(self.parameters['E_reset']-self.potential[i][u]) for i in range(len(self.potential))],label=label,color=colors(index)[j])
        ax.set_title('Potential')