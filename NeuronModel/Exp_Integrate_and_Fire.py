#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np

class exp_integrate_and_fire(neuronClass) :
    parameters = {'save': True, 'spike': True, 'tau': 10., 'R': 10., 'u_rest': -65., 'v_rh': -50.,'u_r': -70., 'Delta_T' : 1. , 'Delta_abs': 3., 'theta_reset' : 20.}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,exp_integrate_and_fire.parameters,**kwargs)
        self.refractory = np.inf

    def f(self,u,I):
        U_exp = self.parameters['Delta_T'] * np.exp((u - self.parameters['v_rh']) / self.parameters['Delta_T'])
        sol = self.parameters['u_rest'] - u + U_exp + self.parameters['R'] * I
        return sol

    def iterate(self,dt,**kwargs) :
        I=kwargs['I']
        potential = np.where(self.refractory<=self.parameters['Delta_abs'],self.parameters['u_r'],self.potential[-1])
        self.refractory+=dt
        next_potential = potential+dt/self.parameters['tau']*(self.f(potential,I))
        event = 1*(next_potential>self.parameters['theta_reset'])
        self.potential.append(np.where(event,self.parameters['theta_reset'],next_potential))
        self.spike_count.append(event)
        self.refractory=np.where(event,0.,self.refractory)
        self.time.append(self.time[-1]+dt)

    def plot_phase(self,ax,I=0.):
        potential = np.linspace(-100., 100., 100)
        ax.plot(potential, self.f(potential,I), label='noise input')
