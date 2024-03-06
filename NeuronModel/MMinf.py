#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np
import matplotlib.pyplot as plt

hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

class MMinf(neuronClass) :
    parameters = {'save': True, 'spike': True, 'step' : 1., 'b' : (lambda x :np.maximum(0.,x)), 'tau' : 1.}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,MMinf.parameters,**kwargs)

    def iterate(self,dt,**kwargs) :
        I=kwargs['I']
        potential = self.potential[-1]
        next_potential = potential+I*dt/self.parameters['tau']
        decrease = np.random.binomial(1, 1./self.parameters['step']*dt/self.parameters['tau']*next_potential)
        post_spike = np.random.binomial(1, dt*self.parameters['b'](next_potential))
        next_potential = np.where(decrease, np.maximum(0.,next_potential-self.parameters['step']),
                                  next_potential)
        next_potential = np.where(post_spike, 0.,
                                  next_potential)
        self.potential.append(next_potential)
        self.spike_count.append(post_spike)
        self.time.append(self.time[-1] + dt)

    def plot_trace(self,ax,index=None,label=''):
        ax.set_ylim(-100.,50.)
        ax.set_xlim(self.time[0], self.time[-1])
        for j,u in enumerate(index):
            ax.plot(self.time,[self.potential[i][u] for i in range(len(self.potential))],label=label,color=colors(index)[j])
        ax.set_title('Potential')
