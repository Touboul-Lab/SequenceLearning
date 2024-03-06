import sys
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

from Simulator.SimulatorPattern import simulatorPattern, simulatorPatternDual
import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg
from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import *

num_training_figure = 4
params_simu = {
    'dt': 0.1,
    'num_training': 50,
    'num_simu': 1,
    'stop_learning': 'None',
    'test_iteration': None,
    'test_iteration_detail': None,
    'save': False,
    'plot': False,
    'num_success_params': 0,
    'num_spike_output': None,
    'reset_training': False,
    'accuracy_subpattern': False,
}

params_network = {
    'P': 4,
    'neuronClass': 'MSN_IAF_EXP',
    'homeostasy': 0.95,
    'Apostpre': 0.,
    'Aprepost': -1.,
    'tprepost': 20.,
    'tpostpre': 20.,
    'nearest': False,
    'exp': True,
    'noise_input': 0.004,
    'noise_stim': 0.001,
    'epsilon': 0.2,
    'init_weight': ['uniform', (0.05, 0.1)],
    'clip_weight': (0., 2.),
    'save': True,
}

N_STIM = 3

SETS = [list(itertools.combinations(np.arange(params_network['P']), n))
        for n in np.arange(1, min(N_STIM + 1, params_network['P'] + 1))]

params_pattern = {
        'type': 'list_pattern',
        'n_pattern': 2,
        'stim_by_pattern': 3,
        'delay': 1.,
        'random_time': None,
        'duration': 50.,
        'offset': 20.,
        'p_reward': 0.5,
        'sets': SETS,
        'sample_pattern': False,
        'repartition': 'uniform_stim',
    }

np.random.seed(0)
SIMULATOR = simulatorPattern(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network)
np.random.seed(0)
SIMULATOR.init_pattern()
SIMULATOR.run(name='Test')

output = SIMULATOR.params_output['output_train']

x_0 = 0.
x_1 = params_pattern['n_pattern'] * params_pattern['duration']
x_2 = x_1 + num_training_figure * params_pattern['duration']
x_2_ = x_2 + params_pattern['duration']
x_3 = x_1 + params_simu['num_training'] * params_pattern['duration']
x_4 = x_1 + params_simu['num_training'] * params_pattern['duration'] + \
            params_pattern['n_pattern'] * params_pattern['duration']

fig_y, fig_x = 12., 10.
top, bottom, left, right = 1., 1., 2.5, 0.5
dict_margins = dict(top=1.-top/fig_x, bottom=bottom/fig_x, left=left/fig_y, right=1.-right/fig_y)
dict_margins['width_ratios'] = [params_pattern['n_pattern'], num_training_figure, 1., params_pattern['n_pattern']]
dict_margins['height_ratios'] = [0.5, 0.5, 1., 1., 1., 1., 1., 1.]
dict_margins['hspace'] = 0.
dict_margins['wspace'] = 0.
fig, ax = plt.subplots(8, 4, figsize=(fig_y * cfg.cm, fig_x * cfg.cm),
                       gridspec_kw=dict(**dict_margins))

gs = ax[0, 0].get_gridspec()

for current_l in [0, 1, 2, 3]:
    for current_k in np.arange(8):
        ax[current_k, current_l].set_zorder(100 - current_k)

ax_legend_spike = set_subplot_legend(fig, gs[2:6, 0], u'$P{=}4$ cortical\ninput neurons', set_y_value=0.5,
                                     fontsize=6, rotation=0, labelpad=30, va='center')
ax_legend_output = set_subplot_legend(fig, gs[7:, 0], u'MSN (M1)', set_y_value=0.5,
                                      fontsize=6, rotation=0, labelpad=30, va='center')
ax_legend_phase = set_subplot_legend(fig, gs[:2, 0], u'Phases\n$N_p{=}2$ patterns', set_y_value=0.5,
                                     fontsize=6, rotation=0, labelpad=30, va='center')
ax_legend_spike.set_zorder(-10)

ax[6, 0].set_ylabel(u'Random input', fontsize=6, rotation=0, labelpad=30, va='center').set_y(0.5)

ax_legend_training = fig.add_subplot(gs[0, 1:3])
ax_legend_training.set_xticks([])
ax_legend_training.set_yticks([])

ax_legend_training.set_zorder(200)

ax[0, 0].set_ylim(0., 2.)
ax[0, 0].add_patch(patches.Rectangle((x_0, 0.), x_1 - x_0, 2., facecolor=colors['white'], edgecolor=colors['black'],
                                     linewidth=0.5))
ax_legend_training.set_ylim(0., 2.)
ax_legend_training.add_patch(patches.Rectangle((x_1, 0.), x_2_ - x_1, 2., facecolor=colors['pale red'],
                                               edgecolor=colors['black'], linewidth=0.5))
ax[0, 3].set_ylim(0., 2.)
ax[0, 3].add_patch(patches.Rectangle((x_3, 0.), x_4 - x_3, 2., facecolor=colors['white'], edgecolor=colors['black'],
                                     linewidth=0.5))

ax[0, 0].text((x_0 + x_1) / 2., 1., u'Test Before', horizontalalignment='center', verticalalignment='center',
              fontsize=8)
ax_legend_training.text((x_1 + x_2_) / 2., 1., u'Training', horizontalalignment='center', verticalalignment='center',
                        fontsize=8)
ax[0, 3].text((x_3 + x_4) / 2., 1., u'Test After', horizontalalignment='center', verticalalignment='center',
              fontsize=8)

ax[0, 1].remove()
ax[0, 2].remove()

for current_l in [0, 1, 2, 3]:
    for current_k in np.arange(8):
        if current_k > 1:
            ax[current_k, current_l].spines['right'].set_visible(False)
            ax[current_k, current_l].spines['top'].set_visible(False)
        ax[current_k, current_l].spines['left'].set_zorder(10)
        ax[current_k, current_l].set_xticks([])
        ax[current_k, current_l].set_yticks([])
        if current_l == 0:
            ax[current_k, current_l].set_xlim(x_0, x_1)
        elif current_l == 1:
            if current_k > 0:
                ax[current_k, current_l].set_xlim(x_1, x_2)
            else:
                ax_legend_training.set_xlim(x_1, x_2_)
        elif current_l == 2:
            if current_k > 0:
                ax[current_k, current_l].set_xlim(x_2, x_2_)
        elif current_l == 3:
            ax[current_k, current_l].set_xlim(x_3, x_4)
    ax[1, current_l].set_ylim(0., 2.)
    for current_k in [2, 3, 4, 5]:
        ax[current_k, current_l].set_ylim(0., 2.)
        if current_l == 2:
            ax[current_k, current_l].spines['bottom'].set_color(colors['green'])
            ax[current_k, current_l].spines['bottom'].set_linestyle((0, (5, 5)))
            ax[current_k, current_l].spines['left'].set_visible(False)
        else:
            ax[current_k, current_l].spines['bottom'].set_color(colors['green'])
    ax[6, current_l].set_ylim(0., 2.)
    if current_l == 2:
        ax[6, current_l].spines['bottom'].set_color(colors['yellow'])
        ax[6, current_l].spines['bottom'].set_linestyle((0, (5, 5)))
        ax[6, current_l].spines['left'].set_visible(False)
    else:
        ax[6, current_l].spines['bottom'].set_color(colors['yellow'])
    ax[7, current_l].set_ylim(-150., 70.)
    if current_l == 2:
        ax[7, current_l].spines['bottom'].set_color(colors['brown'])
        ax[7, current_l].spines['bottom'].set_linestyle((0, (5, 5)))
        ax[7, current_l].spines['left'].set_visible(False)
    else:
        ax[7, current_l].spines['bottom'].set_color(colors['brown'])

for i in np.arange(num_training_figure):
    if SIMULATOR.pattern_list[output[i]['pattern']].reward > 0:
        reward_name = '(+)'
        patch = patches.Rectangle((x_1 + i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                  facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
    else:
        reward_name = '(-)'
        patch = patches.Rectangle((x_1 + i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                  facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
    if output[i]['pattern'] == 0:
        pattern_name = 'A '
    else:
        pattern_name = 'B '
    ax[1, 1].add_patch(patch)
    ax[1, 1].text(x_1 + (i + 0.5) * params_pattern['duration'], 1., pattern_name + reward_name,
                  horizontalalignment='center', verticalalignment='center', fontsize=8)

pattern_name = '...'
ax[1, 2].add_patch(
    patches.Rectangle((x_2, 0.), params_pattern['duration'], 2.,
                      facecolor=colors['white'], edgecolor=colors['black'],
                      linewidth=0.5))
ax[1, 2].text(x_2 + 0.5 * params_pattern['duration'], 1., pattern_name,
              horizontalalignment='center', verticalalignment='center', fontsize=8)

for current_l in [0, 3]:
    if current_l == 0:
        x_init = x_0
    elif current_l == 3:
        x_init = x_3
    for i in np.arange(params_pattern['n_pattern']):
        if SIMULATOR.pattern_list[i].reward > 0:
            reward_name = '(+)'
            patch = patches.Rectangle((x_init + i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                      facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
        else:
            reward_name = '(-)'
            patch = patches.Rectangle((x_init + i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                      facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
        if i == 0:
            pattern_name = 'A '
        else:
            pattern_name = 'B '
        ax[1, current_l].add_patch(patch)
        ax[1, current_l].text(x_init + (i + 0.5) * params_pattern['duration'], 1., pattern_name + reward_name,
                              horizontalalignment='center', verticalalignment='center', fontsize=8)

for current_k in [2, 3, 4, 5, 6, 7]:
    for i in range(num_training_figure):
        if output[i]['stim'] == 1:
            if SIMULATOR.pattern_list[output[i]['pattern']].reward > 0:
                ax[current_k, 1].axvspan(x_1 + i * params_pattern['duration'],
                                         x_1 + (i + 1) * params_pattern['duration'], facecolor=colors['pale red'],
                                         edgecolor=colors['pale red'], linewidth=0.5)

for current_l in [0, 1, 3]:
    for current_k in [2, 3, 4, 5]:
        for time_step in np.arange(len(SIMULATOR.network.STIM.time)):
            if int(SIMULATOR.network.STIM.spike_count[time_step][current_k - 2]) == 1:
                if int(SIMULATOR.network.STIM.spike_noise_count[time_step][current_k - 2]) == 1:
                    ax[current_k, current_l].plot([SIMULATOR.network.STIM.time[time_step],
                                                   SIMULATOR.network.STIM.time[time_step]],
                               [0., 1.], color=colors['grey'])
                else:
                    ax[current_k, current_l].plot([SIMULATOR.network.STIM.time[time_step],
                                                   SIMULATOR.network.STIM.time[time_step]],
                                                   [0., 1.], color=colors['green'])

    for time_step in np.arange(len(SIMULATOR.network.STIM.time)):
        if int(SIMULATOR.network.RANDOM_INPUT.spike_count[time_step][0]) == 1:
            ax[6, current_l].plot([SIMULATOR.network.STIM.time[time_step],
                                   SIMULATOR.network.STIM.time[time_step]],
                                   [0., 1.], color=colors['yellow'])

    ax[7, current_l].plot(SIMULATOR.network.NEURON.time,
                          [SIMULATOR.network.NEURON.potential[i][0] + SIMULATOR.network.NEURON.spike_count[i][0] * (
                                SIMULATOR.network.NEURON.parameters['E_reset'] -
                                SIMULATOR.network.NEURON.parameters['V_th'])
                           for i in range(len(SIMULATOR.network.NEURON.potential))], color=colors['brown'])

for current_count, (current_l, current_x_init) in enumerate(zip([0, 3], [x_0, x_3])):
    for j in np.arange(params_pattern['n_pattern']):
        if SIMULATOR.params_output['output_test'][current_count]['list'][j]['accuracy'] > 0:
            ax[7, current_l].axvspan(current_x_init + j * params_pattern['duration'],
                                     current_x_init + (j + 1.) * params_pattern['duration'],
                                     ymin=0., ymax=0.1, facecolor=colors['green'])
        else:
            ax[7, current_l].axvspan(current_x_init + j * params_pattern['duration'],
                                     current_x_init + (j + 1) * params_pattern['duration'],
                                     ymin=0., ymax=0.1, facecolor=colors['red'])

ax[7, 1].set_xlabel('Iterations', labelpad=5)
fig.savefig('Figures/article/Figure_2_a.png', dpi=1000)
