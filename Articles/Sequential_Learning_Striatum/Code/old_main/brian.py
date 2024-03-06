import numpy as np
import os
from brian2 import *
from brian2modelfitting import *
from Analysis.dataloader import dataloader_AP

experimentalist = 'Elodie'
neuron = 'MSN_In_Vitro'

path = '/users/gvignoud/Documents/numeric_networks/Data/'+experimentalist+'_'+neuron+'/'
file_list = [u for u in os.listdir(path) if u[-7:]=='_AP.mat']
print(file_list[2])
data_V = dataloader_AP(path+file_list[2])
length_AP = data_V.__len__()
duration_simu = 10000

V_AP = zeros([length_AP,duration_simu])*volt
for j in np.arange(length_AP):
    V_AP[j] = data_V.__getitem__(j)[:10000,1] * volt

V_init = np.mean(V_AP[:,:1000])

dt = (data_V.__getitem__(0)[1,0]-data_V.__getitem__(0)[0,0]) * second
defaultclock.dt = dt
def input_current_AP(step,min_I,lag, length_AP, duration_simu):
    I_AP = zeros([length_AP, duration_simu]) * amp
    for j in np.arange(length_AP):
        I_AP[j] = np.hstack([np.zeros(int(round(300*ms/dt))),
                                   np.ones(int(round(500*ms/dt))),
                                   np.zeros(int(round(200*ms/dt)))]) * (min_I - lag + j*step) * pA
    return I_AP

I_AP = input_current_AP(20.,-300.,0., length_AP, duration_simu)

model = Equations('''
    dv/dt = ((E_l - v) + R * I) / tau : volt
    E_l : volt (constant)
    tau : second (constant)
    R : ohm (constant)
    ''')

n_opt = SkoptOptimizer('ET')
metric =  MSEMetric()
# pass parameters to the NeuronGroup
fitter = TraceFitter(model=model, dt=dt,
                     input_var='I', output_var='v',
                     input=I_AP[:20,:], output=V_AP[:20,:], n_samples=20, method='exponential_euler', param_init={'v': V_init})


res, error = fitter.fit(n_rounds=1,
                        optimizer=n_opt, metric=metric, R=[20.*Mohm,500.*Mohm], tau=[1.*ms,10.*ms], E_l=[V_init-2*mV,V_init+2*mV,])
print(res, error)

traces = fitter.generate_traces()

for j in np.arange(20):
    plt.plot(traces[j], 'r')
    plt.plot(V_AP[j], 'b')
plt.show()