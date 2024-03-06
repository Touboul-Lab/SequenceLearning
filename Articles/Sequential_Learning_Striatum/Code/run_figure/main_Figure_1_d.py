import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import set_blank_axis
from Simulator.SimulatorPattern import simulatorPatternExample
from Simulator.Pattern import patternConstant
import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg

params_simu = {
        'dt': 0.1,
        'save': True,
    }

params_network = {
            'P': 2,
            'homeostasy': 0.5,
            'Apostpre': 1.0,
            'Aprepost': -1.0,
            'tprepost': 20.,
            'tpostpre': 20.,
            'nearest': False,
            'exp': True,
            'noise_stim': None,
            'noise_input': None,
            'epsilon': 0.05,
            'save': True,
        }

params_pattern = {
                'n_pattern': 1,
                'duration': 80.,
            }

color_INPUT = ['green', 'blue']
list_timing_INPUT = ([0, 1, 0], [10., 30., 70.])
list_timing_MSN = [50.]

SIMULATOR_list = []

for current_reward in [1, 0]:
    np.random.seed(0)
    SIMULATOR = simulatorPatternExample(params_simu=params_simu,
                                        params_pattern=params_pattern, params_network=params_network)
    SIMULATOR.pattern_list = [patternConstant(SIMULATOR, *list_timing_INPUT, current_reward)]
    SIMULATOR.pattern_list_MSN = [[int(current_timing / params_simu['dt']) for current_timing in list_timing_MSN]]

    SIMULATOR.run('Figure_{:d}'.format(current_reward))
    SIMULATOR_list.append(SIMULATOR)


fig_y, fig_x = 10., 6.
top, bottom, left, right = 0., 0.5, 3., 1.
dict_margins = dict(top=1.-top/fig_x, bottom=bottom/fig_x, left=left/fig_y, right=1.-right/fig_y)
hspace = 0.5
height_average = (fig_x - top - bottom - hspace) / 5.
dict_margins['height_ratios'] = [1.5 * height_average, 1.5 * height_average,
                                 3. * height_average, 3. * height_average, height_average]
dict_margins['hspace'] = hspace / height_average
fig, ax = plt.subplots(5, 1, figsize=(fig_y * cfg.cm, fig_x * cfg.cm),
                       gridspec_kw=dict(**dict_margins))

ax[0].set_ylabel(u'$2$ presynaptic\nneurons', rotation=0, labelpad=40).set_y(0.)
ax[1].set_ylabel(u'$1$ postsynaptic\nneuron', rotation=0, labelpad=40).set_y(0.)
ax[2].set_ylabel(r'$W_1$, $W_2$' + '\n' + r'$A_{\rm reward}{=}0$', rotation=0, labelpad=40).set_y(0.)
ax[3].set_ylabel(u'$W_1$, $W_2$' + '\n' + r'$A_{\rm reward}{>}0$', rotation=0, labelpad=40).set_y(0.)

x_0 = 0.
x_1 = 1. * params_pattern['duration']

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlim(x_0, x_1)
ax[0].set_ylim(0., 1.5)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_color(cfg.colors['black'])
ax[0].spines['bottom'].set_linewidth(0.5)
for current_input in np.arange(params_network['P']):
    for time_step in np.arange(len(SIMULATOR_list[0].network.STIM.time)):
        if int(SIMULATOR_list[0].network.STIM.spike_count[time_step][current_input]) == 1:
            ax[0].plot([SIMULATOR_list[0].network.STIM.time[time_step],
                        SIMULATOR_list[0].network.STIM.time[time_step]], [0., 1.],
                       color=cfg.colors[color_INPUT[current_input]])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlim(x_0, x_1)
ax[1].set_ylim(0., 1.5)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['bottom'].set_color(cfg.colors['black'])
ax[1].spines['bottom'].set_linewidth(0.5)
for time_step in np.arange(len(SIMULATOR_list[0].network.NEURON.time)):
    if int(SIMULATOR_list[0].network.NEURON.spike_count[time_step][0]) == 1:
        ax[1].plot([SIMULATOR_list[0].network.NEURON.time[time_step],
                    SIMULATOR_list[0].network.NEURON.time[time_step]], [0., 1.],
                   color=cfg.colors['brown'])

for ax_ in [ax[2], ax[3]]:
    ax_.set_xticks([])
    ax_.set_yticks([])
    ax_.set_xlim(x_0, x_1)
    ax_.set_ylim(-0.03, 0.07)
    ax_.spines['right'].set_visible(False)
    ax_.spines['top'].set_visible(False)
    ax_.spines['bottom'].set_visible(False)

index_weight = [(0, u) for u in range(params_network['P'])]
for current_input, current_weight in enumerate(index_weight):
    ax[2].plot(SIMULATOR_list[0].network.WEIGHT.time, [SIMULATOR_list[0].network.WEIGHT.weight[i][current_weight]
                                                       - SIMULATOR_list[0].network.WEIGHT.weight[0][current_weight]
               for i in range(len(SIMULATOR_list[0].network.WEIGHT.weight))],
               color=cfg.colors[color_INPUT[current_input]], linestyle='-')

for current_input, current_weight in enumerate(index_weight):
    ax[3].plot(SIMULATOR_list[1].network.WEIGHT.time, [SIMULATOR_list[1].network.WEIGHT.weight[i][current_weight]
                                                       - SIMULATOR_list[1].network.WEIGHT.weight[0][current_weight]
               for i in range(len(SIMULATOR_list[1].network.WEIGHT.weight))],
               color=cfg.colors[color_INPUT[current_input]], linestyle='-')

legend_elements = [Line2D([0], [0], color=cfg.colors['green'], lw=1., linestyle='-',
                          label=r'Neuron 1, {}W_1{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex)),
                   Line2D([0], [0], color=cfg.colors['blue'], lw=1., linestyle='-',
                          label=r'Neuron 2, {}W_1{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex)),
                   Line2D([0], [0], color=cfg.colors['brown'], lw=1., linestyle='-',
                          label=r'MSN'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex))
                   ]

set_blank_axis(ax[4])
ax[4].legend(handles=legend_elements, loc=8, fontsize=6, ncol=3,
             frameon=False, handlelength=1.)

ax[0].set_zorder(100)

ax[0].annotate('', xy=(10., -0.4),  xycoords='data',
               xytext=(50., -0.4), textcoords='data',
               arrowprops=dict(arrowstyle="<->", linewidth=0.5), annotation_clip=False
               )
ax[0].text(30., -0.9, '{}t_1{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex),
           fontsize=6, va='center', ha='center')

ax[0].annotate('', xy=(30., -1.4),  xycoords='data',
               xytext=(50., -1.4), textcoords='data',
               arrowprops=dict(arrowstyle="<->", linewidth=0.5), annotation_clip=False
               )
ax[0].text(40., -1.9, '{}t_2{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex),
           fontsize=6, va='center', ha='center')

ax[0].annotate('', xy=(70., -0.4),  xycoords='data',
               xytext=(50., -0.4), textcoords='data',
               arrowprops=dict(arrowstyle="<->", linewidth=0.5), annotation_clip=False
               )
ax[0].text(60., -0.9, '{}t_3{}'.format(cfg.FigureSequential.latex, cfg.FigureSequential.latex),
           fontsize=6, va='center', ha='center')

ax[2].set_zorder(10)
ax[2].annotate(r'{}\Delta W_1{{=}}A_{{\rm pre-post}}e^{{-t_1/\tau_s}}{}'.format(cfg.FigureSequential.latex,
                                                                                cfg.FigureSequential.latex) + '\n' +
               r'{}\Delta W_2{{=}}A_{{\rm pre-post}}e^{{-t_2/\tau_s}}{}'.format(cfg.FigureSequential.latex,
                                                                                cfg.FigureSequential.latex),
               xy=(48., -0.01),  xycoords='data',
               xytext=(2, -0.05), textcoords='data',
               arrowprops=dict(arrowstyle="->", linewidth=0.5),
               horizontalalignment='left', verticalalignment='center',
               fontsize=6
               )
ax[2].annotate(r'{}\Delta W_1{{=}}A_{{\rm post-pre}}e^{{-t_3/\tau_s}}{}'.format(cfg.FigureSequential.latex,
                                                                                cfg.FigureSequential.latex),
               xy=(68., 0.005),  xycoords='data',
               xytext=(2, 0.05), textcoords='data',
               arrowprops=dict(arrowstyle="->", linewidth=0.5),
               horizontalalignment='left', verticalalignment='center',
               fontsize=6
               )

fig.savefig('Figures/article/Figure_1_d.png', dpi=1000)
