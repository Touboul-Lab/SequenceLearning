import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

from Analysis.AP_Analysis import AP_protocol, add
from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import *

np.random.seed(0)

y_label = ['mV', 'MOhm', 'ms', 'nF']

path_export = '/users/gvignoud/Documents/numeric_networks/Articles/Sequential_Learning_Striatum/Models/'

experimentalist, neuron, num_neuron_tot, neuronModel, exclude_files, noise_value, delta_abs_value =\
    ('Elodie', 'MSN_In_Vitro', 1, 'IAF', ['20181002_2', '20190109_1'], 0.5, 10.)

path = '/users/gvignoud/Documents/numeric_networks/Data/' + experimentalist + '_' + neuron + '/'

file_list = [u for u in os.listdir(path) if u[-7:] == '_AP.mat' and u[:-7] not in exclude_files]

for path_ in [path, path_export]:
    if os.path.exists('{}Plot_{}.pdf'.format(path_, neuron)):
        os.remove('{}Plot_{}.pdf'.format(path_, neuron))
    if os.path.exists('{}Plot_Params{}.pdf'.format(path_, neuron)):
        os.remove('{}Plot_Params{}.pdf'.format(path_, neuron))
    if os.path.exists('{}Params_{}.xlsx'.format(path_, neuron)):
        os.remove('{}Params_{}.xlsx'.format(path_, neuron))

list_PARAMS_V = ['E_l', 'V_th', 'E_reset', 'E_r', 'E_rheobase', 'thrV']
list_PARAMS_R = ['R']
list_PARAMS_T = ['tau', 'Delta_abs']
list_PARAMS_C = ['C']

list_PARAMS = ['file'] + list_PARAMS_V + list_PARAMS_R + list_PARAMS_T + list_PARAMS_C

PARAMS_NEURONS = dict([(list_params_name, []) for list_params_name in list_PARAMS])

excel = pd.read_excel(path + 'AP_Protocol.xlsx', index_col=0)
num_neuron_all = len(file_list) * num_neuron_tot

fig, ax = plt.subplots(5, num_neuron_all, figsize=(10 * num_neuron_all, 24), sharex='row', sharey='row',
                       gridspec_kw=dict(hspace=0.2, wspace=0.1))
y_label_IAF = ['V AP (mV)', 'dV (mV)', 'Rate (Hz)', 'V IAF (mV)', '']
fig.subplots_adjust(top=1., bottom=0., left=0., right=1.)

count = 0
for k, label in enumerate(y_label_IAF):
    ax[k, 0].set_ylabel(label)
for file in file_list:
    for num_neuron in np.arange(num_neuron_tot):
        step_value = float(excel.loc[file[:-7]]['STEP_' + str(num_neuron + 1)])
        min_value = float(excel.loc[file[:-7]]['MIN_' + str(num_neuron + 1)])
        if experimentalist == 'Merie':
            lag = float(excel.loc[file[:-7]]['LAG_' + str(num_neuron + 1)])
        else:
            lag = 0.
        print(file, step_value, min_value, lag, num_neuron)
        duration = 0.5
        ANALYSIS = AP_protocol(path, file[:-4], duration, step_value, min_value, lag, experimentalist,
                               path_tot=path+file, num_neuron=num_neuron)
        ANALYSIS.iterate()
        ANALYSIS.plot_integrate_and_fire(ax[:, count], fig, count, neuronModel,
                                         noise_value=noise_value, delta_abs_value=delta_abs_value)
        add(PARAMS_NEURONS, ANALYSIS.params_integrate_and_fire)
        ANALYSIS.save()
        count = count + 1
plt.savefig('{}Plot_{}.pdf'.format(path, neuron), dpi=1000, bbox_inches='tight')
plt.savefig('{}Plot_{}.pdf'.format(path_export, neuron), dpi=1000, bbox_inches='tight')
plt.close()

data_array = pd.DataFrame(PARAMS_NEURONS, columns=list_PARAMS)
data_array.to_excel('{}Params_{}.xlsx'.format(path, neuron))
data_array.to_excel('{}Params_{}.xlsx'.format(path_export, neuron))

fig, ax = plt.subplots(1, 7, figsize=(8 + len(list_PARAMS), 3), gridspec_kw=dict(hspace=0., wspace=0.,
                       width_ratios=[len(list_PARAMS_V), 2, len(list_PARAMS_R), 2, len(list_PARAMS_T), 2,
                                     len(list_PARAMS_C)]))
for k, list_PARAMS_ in enumerate([list_PARAMS_V, list_PARAMS_R, list_PARAMS_T, list_PARAMS_C]):
    if k > 0:
        ax[2 * k - 1].remove()
    for i, key in enumerate(list_PARAMS_):
        ax[2*k].plot(i * np.ones(len(PARAMS_NEURONS[key])), PARAMS_NEURONS[key], '+', alpha=0.5,
                     color=colors[list(colors.keys())[i]])
        ax[2*k].errorbar(i, np.mean(PARAMS_NEURONS[key]), fmt='o', yerr=np.std(PARAMS_NEURONS[key])/2.,
                         alpha=1, color=colors[list(colors.keys())[i]])
    ax[2*k].set_xticks(np.arange(len(list_PARAMS_)))
    ax[2*k].set_xticklabels([key + '=' + str(np.round(np.mean(PARAMS_NEURONS[key]),
                                                      decimals=2)) for key in list_PARAMS_])
    ax[2*k].set_xlim(-0.5, len(list_PARAMS_)-0.5)
    ax[2*k].set_xticklabels(ax[2*k].get_xticklabels(), rotation=45, ha='right')
    ax[2*k].set_ylabel(y_label[k])
plt.savefig('{}Plot_Params{}.pdf'.format(path, neuron), dpi=1000, bbox_inches='tight')
plt.savefig('{}Plot_Params{}.pdf'.format(path_export, neuron), dpi=1000, bbox_inches='tight')
plt.close()
