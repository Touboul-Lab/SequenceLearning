#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np
from NeuronModel.Hodgkin_Huxley import hodgkin_huxley_parameters
from NeuronModel.Hodgkin_Huxley import mainer_parameters
from NeuronModel.Hodgkin_Huxley import neuronal_dynamics_parameters

def hodgkin_huxley(params = 'HH'):
    dict = {
        'HH': hodgkin_huxley_parameters,
        'ND': neuronal_dynamics_parameters,
        'mainer': mainer_parameters
    }
    class hodgkin_huxley_class(neuronClass) :
        parameters = dict[params].parameters
        x_n = dict[params].x_n
        x_m = dict[params].x_m
        x_h = dict[params].x_h
        tau_n = dict[params].tau_n
        tau_m = dict[params].tau_m
        tau_h = dict[params].tau_h
        def __init__(self,**kwargs):
            neuronClass.__init__(self,**kwargs)
            neuronClass.subclass(self,hodgkin_huxley_class.parameters,**kwargs)
            self.n = [hodgkin_huxley_class.x_n(self.potential[0])]
            self.m = [hodgkin_huxley_class.x_m(self.potential[0])]
            self.h = [hodgkin_huxley_class.x_h(self.potential[0])]
            self.I_Na = [np.zeros(self.P)]
            self.I_K = [np.zeros(self.P)]

        def iterate(self,dt,I=0.) :
            potential=self.potential[-1]
            m = self.m[-1]
            n = self.n[-1]
            h = self.h[-1]
            I_l= -self.parameters['g_l']*(potential-self.parameters['E_l'])
            I_Na = -self.parameters['g_Na']*m**3*h*(potential - self.parameters['E_Na'])
            I_K = -self.parameters['g_K'] * n**4 *(potential - self.parameters['E_K'])
            I_tot = I_l+I_Na+I_K+I
            next_potential=potential+dt/self.parameters['C']*I_tot
            next_m = m - dt / hodgkin_huxley_class.tau_m(potential) * (m - hodgkin_huxley_class.x_m(potential))
            next_n = n - dt / hodgkin_huxley_class.tau_n(potential) * (n - hodgkin_huxley_class.x_n(potential))
            next_h = h - dt / hodgkin_huxley_class.tau_h(potential) * (h - hodgkin_huxley_class.x_h(potential))
            self.time.append(self.time[-1] + dt)
            self.potential.append(next_potential)
            self.m.append(next_m)
            self.n.append(next_n)
            self.h.append(next_h)
            self.I_Na.append(I_Na)
            self.I_K.append(I_K)

        def plot_x(self,ax):
            potential = np.linspace(-100.,100.,100)
            ax.plot(potential, hodgkin_huxley_class.x_n(potential), label='n')
            ax.plot(potential, hodgkin_huxley_class.x_m(potential), label='m')
            ax.plot(potential, hodgkin_huxley_class.x_h(potential), label='h')
            ax.legend()

        def plot_I(self,ax):
            ax.plot(self.time, self.I_Na, label='I_Na')
            ax.plot(self.time, self.I_K, label='I_K')
            ax.legend()

        def plot_nmh(self,ax):
            ax.plot(self.time, self.n, label='n')
            ax.plot(self.time, self.m, label='m')
            ax.plot(self.time, self.h, label='h')
            ax.legend()


        def plot_tau(self,ax):
            potential = np.linspace(-100.,100.,100)
            ax.plot(potential, hodgkin_huxley_class.tau_n(potential), label='n')
            ax.plot(potential, hodgkin_huxley_class.tau_m(potential), label='m')
            ax.plot(potential, hodgkin_huxley_class.tau_h(potential), label='h')
            ax.legend()

        def spike_detection(self):
            self.spike = [0 for _ in range(len(self.time))]
            i=0
            k = 0
            inter_spike = []
            while i < len(self.time) :
                if self.potential[i]> 0 :
                    inter_spike.append(k)
                    k=0
                    self.spike[i]=1
                    i += 100
                    k += 100
                else :
                    i += 1
                    k += 1
            return inter_spike[1:]
    return hodgkin_huxley_class