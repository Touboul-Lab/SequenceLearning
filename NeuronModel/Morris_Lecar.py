#TO BE UPDATED

from NeuronModel.NeuronClass import neuronClass
import numpy as np

#https://web.njit.edu/~matveev/Courses/M430_635_F15/TEXT-Terman%20Ermentrout%20Computational%20Neuroscience%20Book%20-%20ML%20model.pdf

# Hopf 'T0' : 1/0.04, 'g_Ca' : 4.4, 'V3':2, 'V4':30
# SNLC 'T0' : 1/0.067, 'g_Ca' : 4., 'V3':12, 'V4':17.4
# Hopf 'T0' : 1/0.23, 'g_Ca' : 4., 'V3':12, 'V4':17.4

class morris_lecar(neuronClass) :
    parameters = {'save': True,
                  'spike': False, 'C': 20.,
                  'E_l': -60., 'g_l': 2.,
                  'E_Ca': 120., 'g_Ca': 4.,
                  'E_K': -84., 'g_K': 8.,
                  'V1' : -1.2 , 'V2':18.,
                  'V3' : 2. , 'V4' :30. ,
                  'T0': 1./0.04
                  }
    def __init__(self,**kwargs):
        neuronClass.__init__(self,**kwargs)
        neuronClass.subclass(self,morris_lecar.parameters,**kwargs)
        self.W=[self.W_ss(self.potential[-1])]

    def M_ss(self,x):
        return (1.+np.tanh((x-self.parameters['V1'])/self.parameters['V2']))/2.

    def W_ss(self,x):
        return (1.+np.tanh((x-self.parameters['V3'])/self.parameters['V4']))/2.

    def tau_W(self,x):
        return self.parameters['T0']*np.cosh((x-self.parameters['V3'])/(2.*self.parameters['V4']))

    def iterate(self,dt,I=0.) :
        potential = self.potential[-1]
        W = self.W[-1]
        I_Ca = -self.parameters['g_Ca']*self.M_ss(potential)*(potential-self.parameters['E_Ca'])
        I_K = -self.parameters['g_K'] * W * (potential - self.parameters['E_K'])
        I_l = -self.parameters['g_l'] * W * (potential - self.parameters['E_l'])
        I_tot = I + I_Ca + I_K + I_l
        next_potential=potential+dt/self.parameters['C']*I_tot
        next_W=W+dt/self.tau_W(potential)*(self.W_ss(potential)-W)
        self.time.append(self.time[-1] + dt)
        self.potential.append(next_potential)
        self.W.append(next_W)

    def plot_phase(self,ax):
        ax.plot(self.W,self.potential)