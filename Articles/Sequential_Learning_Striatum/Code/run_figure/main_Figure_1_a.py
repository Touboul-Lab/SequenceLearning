import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg

experimentalist, neuron = ('Elodie', 'MSN_In_Vitro')
name_exp = '20190107_1_AP_1'


path_read = '../../Data/' + experimentalist + '_' + neuron + '/' + name_exp + '/'
path_save = 'Figures/article/'

data_dict = np.load(path_read + name_exp[:-2] + '_plot_IAF.npy', allow_pickle=True).item()

fig_y, fig_x = 16., 8.
top, bottom, left, right = 1., 1., 2., 1.
dict_margins = dict(top=1.-top/fig_x, bottom=bottom/fig_x, left=left/fig_y, right=1.-right/fig_y)
hspace = 1.
wspace = 1.5
width_average = (fig_y - left - right - wspace) / 2.
height_average = (fig_x - top - bottom - hspace) / 2.
dict_margins['width_ratios'] = [width_average, width_average]
dict_margins['height_ratios'] = [height_average, height_average]
dict_margins['hspace'] = hspace / height_average
dict_margins['wspace'] = wspace / width_average
fig, ax = plt.subplots(2, 2, figsize=(fig_y * cfg.cm, fig_x * cfg.cm),
                       gridspec_kw=dict(**dict_margins))

ax_AP_EXP = ax[0, 0]
ax_RI = ax[0, 1]
ax_IF = ax[1, 1]
ax_AP_IAF = ax[1, 0]
ax_AP_EXP.get_shared_x_axes().join(ax_AP_EXP, ax_AP_IAF)

spike = True
for x, y in zip(data_dict['AP_EXP']['time'], data_dict['AP_EXP']['V']):
    if spike and np.max(y) > 0.:
        ax_AP_EXP.plot(x, y, color=cfg.colors['black'], alpha=1., zorder=10)
        spike = False
    else:
        ax_AP_EXP.plot(x, y, color=cfg.colors['blue'], alpha=0.2)
ax_AP_EXP.set_xlim(data_dict['AP_EXP']['time'][0][0], data_dict['AP_EXP']['time'][0][-1])
ax_AP_EXP.set_ylabel(u'Membrane potential\nexperiments (mV)')

spike = True
for x, y in zip(data_dict['AP_IAF']['time'], data_dict['AP_IAF']['V']):
    if spike and np.max(y) > 0.:
        ax_AP_IAF.plot(x, y, color=cfg.colors['black'], alpha=1., zorder=10)
        spike = False
    else:
        ax_AP_IAF.plot(x, y, color=cfg.colors['dark green'], alpha=0.2)
ax_AP_IAF.set_xlim(data_dict['AP_IAF']['time'][0][0], data_dict['AP_IAF']['time'][0][-1])
ax_AP_IAF.set_xlabel('Time (ms)')
ax_AP_IAF.set_ylabel(u'Membrane potential\nsimulations (mV)')

ax_RI.plot(data_dict['RI']['I'], data_dict['RI']['diff_V'], '+-', color=cfg.colors['blue'])
ax_RI.plot(data_dict['RI']['I_RI'], data_dict['RI']['diff_V_linear'], color=cfg.colors['dark green'])
ax_RI.set_ylabel(u'Membrane potential\n' + r'{}V-E_l{} (mV)'.format(cfg.FigureSequential.latex,
                                                                    cfg.FigureSequential.latex))

ax_IF.plot(data_dict['IF']['I'], data_dict['IF']['Rate_SIMU'], '+', label='rate_SIMU', color=cfg.colors['dark green'])
ax_IF.plot(data_dict['IF']['I'], data_dict['IF']['Rate'], '+', label='rate_ISI', color=cfg.colors['blue'])
ax_IF.plot(data_dict['IF']['I'], data_dict['IF']['Rate_EXACT'],
           '-', label='rate_EXACT', color=cfg.colors['dark green'])
ax_IF.set_xlabel('I (pA)')
ax_IF.set_ylabel('Rate (Hz)')
ax_IF.set_ylim(0, 100)
ax_IF.set_yticks([0, 50, 100])

for ax_ in [ax_AP_EXP, ax_AP_IAF, ax_RI, ax_IF]:
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)

plt.savefig('{}Figure_1_a.png'.format(path_save), dpi=1000)
plt.close()
