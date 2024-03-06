#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np
import matplotlib.pyplot as plt

hsv = plt.get_cmap('hsv')
def colors(n):
    return hsv(np.linspace(0., 1.0, len(n)+1))

class MSN(neuronClass) :
    parameters = {'save': True, 'spike': True, 'tau': 10., 'R': lambda x:10., 'u_rest': -65., 'E_reset' :20.,
                  'u_r': lambda x : x, 'V_th' : -40., 'a':1., 'tau_w' : 1., 'b':0.}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,MSN.parameters,**kwargs)
        self.refractory = np.inf
        self.w = [np.zeros(self.P)]

    def g(self,potential):
        return self.parameters['a'] * (potential - self.parameters['u_rest'])

    def iterate(self,dt,**kwargs) :
        I=kwargs['I']
        w = self.w[-1]
        potential = np.where(self.refractory <= 0., self.parameters['u_r'](w)+self.parameters['u_rest'], self.potential[-1])
        self.refractory+=dt
        next_potential = potential+dt/self.parameters['tau']*(-(potential-self.parameters['u_rest'])+self.parameters['R'](w)*I)
        next_w = w + dt/self.parameters['tau_w']*(self.g(potential)-w)
        event = 1 * (next_potential > self.parameters['V_th'])
        next_w += self.parameters['b'] * event
        self.potential.append(np.where(event, self.parameters['V_th'], next_potential))
        self.w.append(next_w)
        self.spike_count.append(event)
        self.refractory = np.where(event, 0., self.refractory)
        self.time.append(self.time[-1] + dt)

    def plot_trace(self,ax,index=None,label=''):
        ax.set_ylim(-100.,50.)
        ax.set_xlim(self.time[0], self.time[-1])
        for j,u in enumerate(index):
            ax.plot(self.time,[self.potential[i][u]+self.spike_count[i][u]*(self.parameters['E_reset']-self.parameters['V_th']) for i in range(len(self.potential))],label=label,color=colors(index)[j])
        ax.set_title('Potential')

    def plot_trace_w(self,ax,index=None,label=''):
        ax.set_ylim(-100.,50.)
        ax.set_xlim(self.time[0], self.time[-1])
        for j in index :
            ax.plot(self.time,[self.w[i][j] for i in range(len(self.potential))],label=label,color=colors(index)[j])
        ax.set_title('w')