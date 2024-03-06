import os
import pandas as pd
import numpy as np

experimentalist, neuron, num_neuron_tot, neuronModel, exclude_files, noise_value, delta_abs_value =\
    ('Elodie', 'MSN_In_Vitro', 1, 'IAF', ['20181002_2', '20190109_1'], 0.5, 10.)

path_export = '/users/gvignoud/Documents/numeric_networks/Articles/Sequential_Learning_Striatum/Models/'

data_array = pd.read_excel('{}Params_{}.xlsx'.format(path_export, neuron))

if os.path.exists('{}/params_IAF_{}_{}'.format(path_export, experimentalist, neuron)):
    os.remove('{}/params_IAF_{}_{}'.format(path_export, experimentalist, neuron))
if os.path.exists('{}/params_Izhi_{}_{}'.format(path_export, experimentalist, neuron)):
    os.remove('{}/params_Izhi_{}_{}'.format(path_export, experimentalist, neuron))

params_neuron_IAF = {
    'neuron_type': 'IAF',
    'init': np.mean(data_array['E_l'].values),
    'tau': np.mean(data_array['tau'].values),
    'R': np.mean(data_array['R'].values),
    'E_l': np.mean(data_array['E_l'].values),
    'V_th': np.mean(data_array['V_th'].values),
    'E_r': np.mean(data_array['E_r'].values),
    'noise': noise_value,
    'E_reset': np.mean(data_array['E_reset'].values),
    'Delta_abs': delta_abs_value,
    'scale_I': None,
    'Burst': None,
}
np.save('{}/params_IAF_{}_{}'.format(path_export, experimentalist, neuron), params_neuron_IAF)

params_neuron_Izhi = {
    'neuron_type': 'Izhi',
    'init': np.mean(data_array['E_l'].values),
    'v_rest': np.mean(data_array['E_l'].values),
    'C': 50. / 1000.,
    'c': np.mean(data_array['E_r'].values),
    'v_t': np.mean(data_array['thrV'].values),
    'noise': noise_value,
    'v_peak': np.mean(data_array['E_reset'].values),
    'k_input': 1. / 1000.,
    'a': 0.01,
    'b': -20. / 1000.,
    'd': 150. / 1000.,
    'scale_I': 100.,
    'Burst': None,
    'tau': np.mean(data_array['tau'].values),
    }

np.save('{}/params_Izhi_{}_{}'.format(path_export, experimentalist, neuron), params_neuron_Izhi)