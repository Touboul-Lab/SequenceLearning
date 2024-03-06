#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np

#normal
#parameters = {'spike': True, 'tau': 10, 'R': 10, 'u_rest': -65, 'v_rh': -50, 'u_r': -70, 'Delta_T' : 1. , 'Delta_abs': 3., 'theta_reset' : 0, 'a':1., 'tau_w' : 100, 'b':1.}
#tonic
#parameters = {'spike': True, 'tau': 20, 'R': 0.5, 'u_rest': -70, 'v_rh': -50, 'u_r': -60, 'Delta_T' : 2. , 'Delta_abs': 0., 'theta_reset' : 0, 'a':0., 'tau_w' : 30., 'b':60.}
#adapting
#parameters = {'spike': True, 'tau': 20, 'R': 0.5, 'u_rest': -70, 'v_rh': -50, 'u_r': -60, 'Delta_T' : 2. , 'Delta_abs': 0., 'theta_reset' : 0, 'a':0., 'tau_w' : 100., 'b':5.}
#initburst
#parameters = {'spike': True, 'tau': 5, 'R': 0.5, 'u_rest': -70, 'v_rh': -50, 'u_r': -51, 'Delta_T' : 2. , 'Delta_abs': 0., 'theta_reset' : 0, 'a':0.5, 'tau_w' : 100., 'b':7.}
#adapting
#parameters = {'spike': True, 'tau': 5, 'R': 0.5, 'u_rest': -70, 'v_rh': -50, 'u_r': -46, 'Delta_T' : 2. , 'Delta_abs': 0., 'theta_reset' : 0, 'a':-0.5, 'tau_w' : 100., 'b':7.}
#chaos
#parameters = {'spike': True, 'tau': 281/30, 'R':1/30,  'u_rest': -70.6, 'v_rh': -50.4, 'u_r': -47.2, 'Delta_T' : 2. , 'Delta_abs': 0., 'theta_reset' : 0, 'a':4, 'tau_w' : 40., 'b':80.}


class adex(neuronClass) :
    parameters = {'save': True, 'spike': True, 'tau': 10., 'R': 10., 'u_rest': -65., 'v_rh': -50., 'u_r': -70., 'Delta_T' : 1. , 'Delta_abs': 3., 'theta_reset' : 0., 'a':1., 'tau_w' : 100., 'b':1.}
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,adex.parameters,**kwargs)
        self.refractory = np.inf
        self.w = [np.zeros(self.P)]

    def f(self,u,I):
        U_exp = self.parameters['Delta_T'] * np.exp((u - self.parameters['v_rh']) / self.parameters['Delta_T'])
        sol = self.parameters['u_rest'] - u + U_exp + self.parameters['R'] * I
        return sol

    def nullcline_u(self,u,I):
        return self.f(u,I)/self.parameters['R']

    def nullcline_w(self,u):
        return self.parameters['a']*(u-self.parameters['u_rest'])

    def iterate(self,dt,**kwargs) :
        I=kwargs['I']
        potential = np.where(self.refractory<=0.,self.parameters['u_r'],self.potential[-1])
        w = self.w[-1]
        self.refractory+=dt
        next_potential = potential+dt/self.parameters['tau']*(self.f(potential,I)-self.parameters['R'] *w)
        next_w = w + dt/self.parameters['tau_w']*(self.parameters['a']*(potential-self.parameters['u_rest'])-w)
        event = 1 * (next_potential > self.parameters['theta_reset'])
        next_w += self.parameters['b'] * event
        self.w.append(next_w)
        self.potential.append(np.where(event, self.parameters['theta_reset'], next_potential))
        self.spike_count.append(event)
        self.refractory = np.where(event, 0., self.refractory)
        self.time.append(self.time[-1] + dt)

    def plot_phase(self,ax,index,I):
        u = np.linspace(-80.,-40.)
        ax.plot(u,self.nullcline_u(u,I))
        ax.plot(u, self.nullcline_w(u))
        for i in index :
            ax.plot(np.array(self.potential)[i],np.array(self.w)[i])
        ax.set_xlim(-80.,20.)
        ax.set_ylim(0.,4000.)
