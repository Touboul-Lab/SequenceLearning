#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np

class quad_integrate_and_fire(neuronClass) :
    parameters = {'save': True, 'spike': True, 'tau': 10., 'R': 10., 'u_rest': -65., 'u_c': -50.,'u_r': -70., 'theta_reset' : 20., 'a_0': 0.1}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,quad_integrate_and_fire.parameters,**kwargs)
        self.refractory = np.inf*np.ones(self.P)

    def f(self,u,I):
        sol = self.parameters['a_0']*(self.parameters['u_rest'] - u)*(self.parameters['u_c']-u) + self.parameters['R'] * I
        return sol

    def iterate(self,dt,**kwargs) :
        I=kwargs['I']
        potential = np.where(self.refractory<=0,self.parameters['u_r'],self.potential[-1])
        self.refractory+=dt
        next_potential = potential+dt/self.parameters['tau']*(self.f(potential,I))
        event = 1*(next_potential>self.parameters['theta_reset'])
        self.potential.append(np.where(event,self.parameters['theta_reset'],next_potential))
        self.spike_count.append(event)
        self.refractory=np.where(event,0,self.refractory)
        self.time.append(self.time[-1]+dt)

    def plot_phase(self, ax, I=0.):
        potential = np.linspace(-100., 100., 100)
        ax.plot(potential, self.f(potential,I), label='noise input')
