import sys
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from Simulator.SimulatorDMS import simulatorPatternDMS
from Articles.DMS_DLS.Code.FiguresDMS import *

Apostpre_list = np.array([-1., 0.0, 1.])

def f_STDP(x, A):
    y = np.zeros(len(x))
    for counter in np.arange(len(x)):
        if x[counter] > 0:
            y[counter] = -1. * np.exp(-x[counter])
        else:
            y[counter] = A * np.exp(x[counter])
    return y


c_list_figure = [c_list[1], c_list[3], c_list[5]]

gs_kw = dict(hspace=0., width_ratios=[1.5, 0.5], top=0.9, bottom=0.35, left=0.2, right=0.9)
fig, ax = plt.subplots(figsize=(2., 1.5), ncols=2, nrows=1, gridspec_kw=gs_kw)

set_blank_axis(ax[1])

ax[0].set_zorder(10)

ax[0].plot([-5., 5.], [0., 0.], 'k--')
ax[0].plot([0., 0.], [-1., 1.5], 'k--')
x_1 = np.linspace(0., 5., 1000)[1:]
x_2 = np.linspace(-5., 0., 1000)[:-1]
ax[0].plot(x_1, f_STDP(x_1, 0.), color=colors['black'])
for k, Apostpre in enumerate(Apostpre_list):
    ax[0].plot(x_2, f_STDP(x_2, Apostpre), color=colors[c_list_figure[k]])
ax[0].set_xlim(-5., 5.)
ax[0].patch.set_facecolor(colors['white'])
ax[0].set_yticks([-1., 0., 1.])
ax[0].set_ylim(-1., 1.5)
ax[0].set_xticks([])
ax[0].set_xlabel(r'$\Delta t=t_{\rm post}-t_{\rm pre}$')
ax[0].set_ylabel(r'$\Phi$', labelpad=-2)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

legend_elements = [Line2D([0], [0], color=colors[c_list_figure[k]], lw=0.5, label=str(Apostpre))
                   for k, Apostpre in enumerate(Apostpre_list)]

ax[1].legend(handles=legend_elements, loc='center', fontsize=6., title=r'$A_{\rm post-pre}$',
             frameon=False, handlelength=1.)
ax[0].set_title(r'$\Phi(\Delta t) = A_{\rm post-pre}\exp(\Delta t/\tau_s),\,' +
                u'\Delta t < 0$\n' +
                r'$\Phi(\Delta t) = A_{\rm pre-post}\exp(-\Delta t/\tau_s),\, \Delta t > 0$',
                fontsize=6, x=0.7, y=-0.6)
fig.savefig('Figures/article/STDP.svg')
fig.savefig('Figures/phd/STDP.svg')
plt.close(fig)

params_simu = {
    'dt': 0.2,
    'num_training_initial': 4,
    'num_training_learning': 4,
    'num_training_maintenance': 4,
    'num_training_recall': 4,
    'ratio_noise_learning_maintenance': 5.,
    'num_simu': 1,
    'new_set': 0,
    'stim_recall': 0.5,
    'stop_learning': 'None',
    'test_iteration': None,
    'test_iteration_detail': None,
    'save': False,
    'num_success_params': 0,
    'num_spike_output': None,
    'reset_training': False,
    'accuracy_subpattern': False,
}

params_network = {
    'neuronClass': 'MSN_Yim',
    'P': 4,
    'homeostasy': 0.95,
    'Apostpre': 1.,
    'Aprepost': -1.,
    'tprepost': 2.,
    'tpostpre': 2.,
    'nearest': False,
    'exp': True,
    'noise_input': 0.005,
    'noise_stim': 0.005 / 4.,
    'epsilon': 0.04,
    'save': True,
    'homeostasy_post': 0.,
    'init_weight': ['uniform', (0., 0.05)],
    'clip_weight': (0., 2.),
}

SETS = list(itertools.combinations(np.arange(4), 3))
params_pattern = {
        'type': 'list_pattern_DMS',
        'n_pattern': 2,
        'stim_by_pattern': 3,
        'random_time': None,
        'duration': 50.,
        'offset': 20.,
        'p_reward': 0.5,
        'sets': SETS,
        'noise_pattern': 1.,
        'sample_pattern': True,
    }

np.random.seed(3)

SIMULATOR = simulatorPatternDMS(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network)

np.random.seed(2)

SIMULATOR.init_pattern()

np.random.seed(23)

SIMULATOR.run(name='Test')
output = SIMULATOR.params_output['output_train']

x_0 = 0.
x_1 = params_simu['num_training_initial'] * params_pattern['duration']
x_2 = x_1 + params_simu['num_training_learning'] * params_pattern['duration']
x_3 = x_2 + params_simu['num_training_maintenance'] * params_pattern['duration']
x_4 = x_3 + params_simu['num_training_recall'] * params_pattern['duration']
x_5 = x_4 + params_pattern['n_pattern'] * params_pattern['duration']

gs_kw = dict(hspace=0., wspace=0., height_ratios=[0.5, 0.5, 1., 1., 1., 1., 1., 1.])
fig, ax = plt.subplots(figsize=(5.2, 3.5*7./6.5), ncols=1, nrows=8, gridspec_kw=gs_kw)
fig.subplots_adjust(top=1., bottom=0., left=0.075, right=1.)

gs = ax[0].get_gridspec()
ax_legend_spike = fig.add_subplot(gs[2:6])

ax_legend_spike.spines['right'].set_visible(False)
ax_legend_spike.spines['top'].set_visible(False)
ax_legend_spike.spines['left'].set_visible(False)
ax_legend_spike.spines['bottom'].set_visible(False)
ax_legend_spike.set_xticks([])
ax_legend_spike.set_yticks([])
ax_legend_spike.set_ylabel(u'$P$ cortical input neurons\n', labelpad=5, fontsize=8)
ax[7].set_ylabel(u'MSN\n(V)', labelpad=5, fontsize=8)

for k in np.arange(8):
    if k > 0:
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['top'].set_visible(False)
    ax[k].spines['left'].set_zorder(10)
    ax[k].set_xticks([])
    ax[k].set_yticks([])
    ax[k].set_xlim(0, x_5)

ax[0].set_ylim(0., 2.)
ax[0].add_patch(patches.Rectangle((x_0, 0.), x_1-x_0, 2.,
                                  facecolor=colors['pale grey'], edgecolor=colors['black'], linewidth=0.5))
ax[0].add_patch(patches.Rectangle((x_1, 0.), x_2-x_1, 2.,
                                  facecolor=colors['pale red'], edgecolor=colors['black'], linewidth=0.5))
ax[0].add_patch(patches.Rectangle((x_2, 0.), x_3-x_2, 2.,
                                  facecolor=colors['pale grey'], edgecolor=colors['black'], linewidth=0.5))
ax[0].add_patch(patches.Rectangle((x_3, 0.), x_4-x_3, 2.,
                                  facecolor=colors['pale red'], edgecolor=colors['black'], linewidth=0.5))
ax[0].add_patch(patches.Rectangle((x_4, 0.), x_5-x_4, 2.,
                                  facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5))

ax[0].text((x_0+x_1)/2., 1., u'Initial\nreward OFF, $\eta=\eta_m$',
           horizontalalignment='center', verticalalignment='center', fontsize=8)
ax[0].text((x_1+x_2)/2., 1., u'Learning\nreward ON, $\eta=1$',
           horizontalalignment='center', verticalalignment='center', fontsize=8)
ax[0].text((x_2+x_3)/2., 1., u'Maintenance\nreward OFF, $\eta=\eta_m$',
           horizontalalignment='center', verticalalignment='center', fontsize=8)
ax[0].text((x_3+x_4)/2., 1.,  u'Relearning\nreward ON, $\eta=1$',
           horizontalalignment='center', verticalalignment='center', fontsize=8)
ax[0].text((x_4+x_5)/2., 1.,  u'Test',
           horizontalalignment='center', verticalalignment='center', fontsize=8)
ax[0].set_ylabel(u'Phases\n', labelpad=5, fontsize=8)

ax[1].set_ylim(0., 2.)
for i in np.arange(len(output)):
    if output[i]['stim'] == 1:
        if SIMULATOR.pattern_list[output[i]['pattern']].reward > 0:
            reward_name = '(+)'
            patch = patches.Rectangle((i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                      facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
        else:
            reward_name = '(-)'
            patch = patches.Rectangle((i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                      facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
        if output[i]['pattern'] == 0:
            pattern_name = 'A '
        else:
            pattern_name = 'B '
        ax[1].add_patch(patch)
        ax[1].text((i + 0.5) * params_pattern['duration'], 1., pattern_name + reward_name,
                   horizontalalignment='center', verticalalignment='center', fontsize=8)
    else:
        ax[1].add_patch(
            patches.Rectangle((i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                              facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5))
        ax[1].text((i + 0.5) * params_pattern['duration'], 1., r'$\varnothing$',
                   horizontalalignment='center', verticalalignment='center', fontsize=8)

for i in np.arange(params_pattern['n_pattern']):
    if SIMULATOR.pattern_list[i].reward > 0:
        reward_name = '(+)'
        patch = patches.Rectangle((x_4 + i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                  facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
    else:
        reward_name = '(-)'
        patch = patches.Rectangle((x_4 + i * params_pattern['duration'], 0.), params_pattern['duration'], 2.,
                                  facecolor=colors['white'], edgecolor=colors['black'], linewidth=0.5)
    if i == 0:
        pattern_name = 'A '
    else:
        pattern_name = 'B '
    ax[1].add_patch(patch)
    ax[1].text(x_4 + (i + 0.5) * params_pattern['duration'], 1., pattern_name + reward_name,
               horizontalalignment='center', verticalalignment='center', fontsize=8)

for k in [2, 3, 4, 5, 6, 7]:
    for i in range(len(output)):
        if output[i]['stim'] == 1:
            if SIMULATOR.pattern_list[output[i]['pattern']].reward > 0:
                if not (i * params_pattern['duration'] < x_1 or (x_2 <= i * params_pattern['duration'] < x_3)):
                    ax[k].axvspan(i * params_pattern['duration'],
                                  (i + 1) * params_pattern['duration'], facecolor=colors['pale red'],
                                  edgecolor=colors['pale red'], linewidth=0.5)

for k in [2, 3, 4, 5]:
    ax[k].set_ylim(0., 2.)
    ax[k].spines['bottom'].set_color(colors['green'])
    for time_step in np.arange(len(SIMULATOR.network.STIM.time)):
        if int(SIMULATOR.network.STIM.spike_count[time_step][k-2]) == 1:
            if int(SIMULATOR.network.STIM.spike_noise_count[time_step][k-2]) == 1:
                ax[k].plot([SIMULATOR.network.STIM.time[time_step], SIMULATOR.network.STIM.time[time_step]],
                           [0., 1.], color=colors['grey'])
            else:
                ax[k].plot([SIMULATOR.network.STIM.time[time_step], SIMULATOR.network.STIM.time[time_step]],
                           [0., 1.], color=colors['green'])

ax[6].set_ylim(0., 2.)
ax[6].spines['bottom'].set_color(colors['yellow'])
for time_step in np.arange(len(SIMULATOR.network.RANDOM_INPUT.time)):
    if int(SIMULATOR.network.RANDOM_INPUT.spike_count[time_step][0]) == 1:
        ax[6].plot([SIMULATOR.network.RANDOM_INPUT.time[time_step], SIMULATOR.network.RANDOM_INPUT.time[time_step]],
                   [0., 1.], color=colors['yellow'])
ax[6].set_ylabel(u'Random\ninput', labelpad=5, fontsize=8)

ax[7].set_ylim(-100., 70.)
ax[7].spines['bottom'].set_color(colors['brown'])
ax[7].plot(SIMULATOR.network.NEURON.time,
           [SIMULATOR.network.NEURON.potential[i][0] + SIMULATOR.network.NEURON.spike_count[i][0] * (
                   SIMULATOR.network.NEURON.parameters['E_reset']
                   - SIMULATOR.network.NEURON.parameters['V_th'])
            for i in range(len(SIMULATOR.network.NEURON.potential))], color=colors['brown'])

for k in np.arange(2, 7):
    for x_ in [x_1, x_2, x_3]:
        ax[k].plot([x_, x_], [0., 2.], '--', color=colors['black'])
    for x_ in [x_4]:
        ax[k].plot([x_, x_], [0., 2.], '-', color=colors['black'])

for x_ in [x_1, x_2, x_3]:
    ax[7].plot([x_, x_], [-100., 70.], '--', color=colors['black'])
for x_ in [x_4]:
    ax[7].plot([x_, x_], [-100., 70.], '-', color=colors['black'])

for k in np.arange(8):
    ax[k].set_zorder(100 - k)

ax[7].set_xlabel('Iterations', labelpad=5)
fig.savefig('Figures/article/Setup.svg')
fig.savefig('Figures/phd/Setup.svg')
plt.close(fig)
