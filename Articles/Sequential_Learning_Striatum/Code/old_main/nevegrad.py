import numpy as np
import os
import nevergrad as ng
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

from Analysis.dataloader import dataloader_AP

experimentalist = 'Elodie'
neuron = 'MSN_In_Vitro'

path = '/users/gvignoud/Documents/numeric_networks/Data/'+experimentalist+'_'+neuron+'/'
file_list = [u for u in os.listdir(path) if u[-7:]=='_AP.mat']
print(file_list[2])
data_V = dataloader_AP(path+file_list[2])
length_AP = data_V.__len__()
duration_simu = 10000

V_AP = np.zeros([length_AP,duration_simu])
Spike_AP = np.zeros([length_AP,duration_simu])
for j in np.arange(length_AP):
    V_AP[j] = data_V.__getitem__(j)[:10000,1]
    Spike_AP[j,find_peaks(V_AP[j], prominence=0.035, distance=30)[0]] = 1.

Spike_AP_num = np.sum(Spike_AP, axis=1)
print(Spike_AP_num)

V_init = np.mean(V_AP[:,:1000])

dt = (data_V.__getitem__(0)[1,0]-data_V.__getitem__(0)[0,0])

def input_current_AP(step,min_I,lag, length_AP, duration_simu):
    I_AP = np.zeros([length_AP, duration_simu])
    for j in np.arange(length_AP):
        I_AP[j] = np.hstack([np.zeros(int(round(0.3/dt))),
                                   np.ones(int(round(0.5/dt))),
                                   np.zeros(int(round(0.2/dt)))]) * (min_I - lag + j*step)
    return I_AP

I_AP = input_current_AP(0.020,-0.300,0., length_AP, duration_simu)

def AP_to_FIT(R, tau, E_l, Delta_abs, V_th=-0.04,E_r=-0.08, E_reset=0.04, fit=True):
    V_AP_IaF = np.zeros_like(V_AP)
    Spike_AP_IaF = np.zeros_like(V_AP)
    V_AP_IaF[:,0] = V_init
    refractory = np.inf * np.ones_like(V_AP_IaF[:,0])
    box_kernel = Gaussian1DKernel(10)
    for i in np.arange(V_AP_IaF.shape[1]-1):
        V_AP_IaF[:,i+1] =  np.where(refractory <= Delta_abs, E_r, V_AP_IaF[:,i]+dt/tau * (E_l - V_AP_IaF[:,i] + R * I_AP[:,i]))
        refractory += dt
        event = V_AP_IaF[:, i + 1] > V_th
        V_AP_IaF[:, i + 1] = np.where(event, E_reset, V_AP_IaF[:, i + 1])
        Spike_AP_IaF[:,i + 1] = 1.*event
        refractory = np.where(event, 0., refractory)
    for j in np.arange(V_AP_IaF.shape[0]):
        Spike_AP_IaF[j, :] = np.convolve(Spike_AP_IaF[j, :], 10*np.sqrt(2 * np.pi)*box_kernel.array, mode = 'same')
    if fit:
        return 0.*np.sum(np.linalg.norm(V_AP_IaF - V_AP, axis=1)*(1.- 1.* (Spike_AP_num > 0.))) + 10.*np.sum(np.linalg.norm((Spike_AP_IaF - Spike_AP), axis=1)*(1.* (Spike_AP_num > 0.)))
    else :
        return V_AP_IaF, Spike_AP_IaF

variables = (ng.p.Scalar(),ng.p.Scalar(),ng.p.Scalar(),ng.p.Scalar())
instrum = ng.p.Instrumentation(*variables)
optimizer = ng.optimizers.DiscreteOnePlusOne(parametrization=instrum, budget=200, num_workers=1)
optimizer.parametrization.register_cheap_constraint(lambda x: x[0][0] >= 0.001 and x[0][0] <= 10.)
optimizer.parametrization.register_cheap_constraint(lambda x: x[0][1] >= 0.001 and x[0][1] <= 0.02)
optimizer.parametrization.register_cheap_constraint(lambda x: x[0][2] >= -0.1 and x[0][2] <= 0.)
optimizer.parametrization.register_cheap_constraint(lambda x: x[0][3] >= 0.  and x[0][3] <= 0.05)

recommendation = optimizer.minimize(AP_to_FIT)

print(recommendation.value[0])

plt.plot(V_AP.transpose(), 'b')
plt.plot(AP_to_FIT(*recommendation.value[0], fit=False)[1].transpose()[:,-1], 'r')
#plt.plot(AP_to_FIT(1.,0.01, -0.07, fit=False)[1].transpose()[:,-1], 'r')
#plt.plot(AP_to_FIT(1.,0.01, -0.07, fit=False)[0].transpose(), 'r')
plt.show()