import sys
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg
from NetworkModel.StriatumNeurons import params_MSN_Yim, params_MSN_Izhi
from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import set_blank_axis

dict_Izhi_to_IAF = {
    'E_l': 'v_rest',
    'E_r': 'c',
    'V_th': 'v_t',
    'C': 'C',
    'R': 'scale_I',
}

experimentalist, neuron = ('Elodie', 'MSN_In_Vitro')


y_label = ['$mV$', '$M\Omega$', '$ms$', '$nF$', '$UA$']
path_read = '../../Data/' + experimentalist + '_' + neuron + '/'
path_save = 'Figures/article/'

list_PARAMS_V = (['E_l', 'V_th', 'E_r'], [r'{}V_{{\rm eq}}'.format(cfg.FigureSequential.latex),
                                          r'{}V_{{\rm th}}'.format(cfg.FigureSequential.latex),
                                          r'{}V_{{\rm r}}'.format(cfg.FigureSequential.latex)])
list_PARAMS_R = (['R'], [r'{}R'.format(cfg.FigureSequential.latex)])
list_PARAMS_T = (['tau'], [r'{}\tau'.format(cfg.FigureSequential.latex)])
list_PARAMS_C = (['C'], [r'{}C'.format(cfg.FigureSequential.latex)])

list_PARAMS = ['file'] + list_PARAMS_V[0] + list_PARAMS_R[0] + list_PARAMS_T[0] + list_PARAMS_C[0]

PARAMS_NEURONS = pd.read_excel('{}Params_{}.xlsx'.format(path_read, neuron))

fig_y, fig_x = 16., 6.
top, bottom, left, right = 0.5, 0.5, 2., 1.
dict_margins = dict(top=1.-top/fig_x, bottom=bottom/fig_x, left=left/fig_y, right=1.-right/fig_y)
wspace = 1.5
hspace = 0.
width_average = (fig_y - left - right - 3. * wspace) / 4.
dict_margins['width_ratios'] = [len(list_PARAMS_V[0]), len(list_PARAMS_R[0]),
                                len(list_PARAMS_T[0]), len(list_PARAMS_C[0])]
dict_margins['height_ratios'] = [5, 2]
dict_margins['wspace'] = wspace / width_average
dict_margins['hspace'] = hspace
fig, ax = plt.subplots(2, 4, figsize=(fig_y * cfg.cm, fig_x * cfg.cm),
                       gridspec_kw=dict(**dict_margins))

dict_data = dict()

all_dict = pd.DataFrame([])

gs = ax[0, 0].get_gridspec()

for k, list_PARAMS_ in enumerate([list_PARAMS_V, list_PARAMS_R, list_PARAMS_T, list_PARAMS_C]):
    for i, key in enumerate(list_PARAMS_[0]):
        dict_data[key] = [np.mean(PARAMS_NEURONS[key])]
        ax[0, k].plot(i * np.ones(len(PARAMS_NEURONS[key])), PARAMS_NEURONS[key], '+', alpha=0.5,
                      color=cfg.colors['black'], mew=0.5)
        ax[0, k].errorbar(i, np.mean(PARAMS_NEURONS[key]), fmt='o',
                          yerr=np.std(PARAMS_NEURONS[key])/2., alpha=1, color=cfg.colors['dark green'])
        if key in dict_Izhi_to_IAF.keys():
            dict_data[key].append(params_MSN_Izhi[dict_Izhi_to_IAF[key]])
            ax[0, k].plot(i, params_MSN_Izhi[dict_Izhi_to_IAF[key]], '^', alpha=1, color=cfg.colors['brown'])
        else:
            dict_data[key].append(np.nan)
        if key in params_MSN_Yim.keys():
            dict_data[key].append(params_MSN_Yim[key])
            ax[0, k].plot(i, params_MSN_Yim[key], '^', alpha=1, color=cfg.colors['red'])
        else:
            dict_data[key].append(np.nan)
    ax[0, k].set_xticks(np.arange(len(list_PARAMS_[0])))
    ax[0, k].set_xticklabels(['{}{}'.format(legend, cfg.FigureSequential.latex)
                           for (key, legend) in zip(list_PARAMS_[0], list_PARAMS_[1])])
    ax[0, k].set_xlim(-0.5, len(list_PARAMS_[0]) - 0.5)
    ax[0, k].set_xticklabels(ax[0, k].get_xticklabels())
    ax[0, k].set_ylabel(y_label[k])

table_params = pd.DataFrame(dict_data, index=['MSN_IAF_EXP', 'MSN_YIM', 'MSN_IZHI']).transpose()
table_params.to_excel('Figures/article/Table.xlsx')

for ax_ in ax[0, :]:
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)

for ax_ in ax[1, :]:
    ax_.remove()

ax_legend = fig.add_subplot(gs[1, :])
set_blank_axis(ax_legend)
ax_legend.set_zorder(-10)

legend = [Line2D([0], [0], color=cfg.colors['black'], linestyle='', marker='+',
                 label='Single experiments', alpha=0.5),
          Line2D([0], [0], color=cfg.colors['dark green'], linestyle='', marker='o',
                 label='Mean over experiments (M1)'),
          Line2D([0], [0], color=cfg.colors['red'], linestyle='', marker='^',
                 label='Parameters from Yim (2011)'),
          Line2D([0], [0], color=cfg.colors['brown'], linestyle='', marker='^',
                 label='Parameters from Izhikevich, 2007 (M2)')
          ]

legend = ax_legend.legend(handles=legend, ncol=2, loc=8, fontsize=6, handlelength=1., frameon=False)
legend.get_frame().set_linewidth(0.5)

plt.savefig('{}Figure_1_b.png'.format(path_save), dpi=1000)
plt.close()
