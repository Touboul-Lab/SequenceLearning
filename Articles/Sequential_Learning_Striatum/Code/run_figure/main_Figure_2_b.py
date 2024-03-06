import sys
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools

import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg
from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import set_blank_axis

path = ['Simu/article_example/Figure_2/single_10_MSN_IAF_EXP',
        '0.5_0.001_0.1_0.0_0.0_None_0_0.1_2000_50.0_20.0_True_False_None_example',
        '0.5_0.001_0.0_0.0_0.0_None_0_0.1_2000_50.0_20.0_True_False_None_example']

A_post_pre_list = [('-1.0', '-1.0'), ('-1.0', '1.0'), ('1.0', '-1.0'), ('1.0', '1.0')]
colors_list = ['blue', 'green0', 'brown0', 'red']
label_list = ['Symmetric LTD', 'Hebbian', 'Anti-Hebbian', 'Symmetric LTP']

legend = [Line2D([0], [0], color=cfg.colors[current_color], lw=0.5, linestyle='-',
                 label=current_label) for current_color, current_label in zip(colors_list, label_list)] + \
         [Line2D([0], [0], color=cfg.colors['black'], lw=0.5, linestyle='-',
                 label='Low synaptic weights'),
          Line2D([0], [0], color=cfg.colors['black'], lw=0.5, linestyle='--',
                 label='High synaptic weights')]

fig_y, fig_x = 16., 13.
top, bottom, left, right = 1., 0., 2.5, 1.
dict_margins = dict(top=1.-top/fig_x, bottom=bottom/fig_x, left=left/fig_y, right=1.-right/fig_y)
hspace = 0.5
wspace = 1.5
width_average = (fig_y - left - right - wspace) / 2.
height_average = (fig_x - top - bottom - hspace) / 5.
dict_margins['width_ratios'] = [width_average, width_average]
dict_margins['height_ratios'] = [4. * height_average, 3. * height_average, 4. * height_average,
                                 4. * height_average, 4. * height_average, 3.5 * height_average]
dict_margins['hspace'] = hspace / height_average
dict_margins['wspace'] = wspace / width_average
fig, ax = plt.subplots(6, 2, figsize=(fig_y * cfg.cm, fig_x * cfg.cm),
                       gridspec_kw=dict(**dict_margins))

gs = ax[0, 0].get_gridspec()

ax[5, 0].remove()
ax[5, 1].remove()
ax[1, 0].remove()
ax[1, 1].remove()

ax_title_A = fig.add_subplot(gs[0, :])
ax_title_B = fig.add_subplot(gs[2, :])
set_blank_axis(ax_title_A)
set_blank_axis(ax_title_B)
ax_title_A.set_zorder(-10)
ax_title_B.set_zorder(-10)

ax_legend = fig.add_subplot(gs[5, :])
set_blank_axis(ax_legend)
ax_legend.set_zorder(-10)

ax_title_A.set_title('Response to random activity')
ax_title_B.set_title('Learning one spatio-temporal pattern')

ax_example_A_no_reward = ax[0, 0]
ax_example_A_no_reward.set_ylim(0., 2.)
ax_example_A_no_reward.set_xticks([])
ax_example_A_no_reward.set_ylabel('Norm of\nsynaptic weights')
ax_example_A_reward = ax[0, 1]
ax_example_A_reward.set_ylim(0., 2.)
ax_example_A_reward.set_xticks([])

ax_example_B_no_reward = ax[2, 0]
ax_example_B_no_reward.set_ylim(0., 1.1)
ax_example_B_no_reward.set_xticks([])
ax_example_B_no_reward.set_ylabel('No spike')
ax_example_B_reward = ax[2, 1]
ax_example_B_reward.set_ylim(0., 1.1)
ax_example_B_reward.set_xticks([])

ax_example_B_diff_no_reward = ax[3, 0]
ax_example_B_diff_no_reward.set_ylim(0., 1.1)
ax_example_B_diff_no_reward.set_xticks([])
ax_example_B_diff_no_reward.set_ylabel('Relative time\nfirst spike')
ax_example_B_diff_reward = ax[3, 1]
ax_example_B_diff_reward.set_ylim(0., 1.1)
ax_example_B_diff_reward.set_xticks([])

ax_example_B_accuracy_no_reward = ax[4, 0]
ax_example_B_accuracy_no_reward.set_ylim(0., 1.1)
ax_example_B_accuracy_no_reward.set_ylabel('Accuracy')
ax_example_B_accuracy_reward = ax[4, 1]
ax_example_B_accuracy_reward.set_ylim(0., 1.1)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[1], 1, 500, 'A', synaptic_weight_init), allow_pickle=True).item()

        ax_example_A_no_reward.plot(data['output_test']['iteration'][0],
                                    np.nanmean(data['output_test']['weight_norm'], axis=0),
                                    color=cfg.colors[current_color], linestyle=linestyle)
    ax_example_A_no_reward.text(0.8, 0.8, 'no reward', ha='center', va='center',
                                transform=ax_example_A_no_reward.transAxes)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[1], 0, 500, 'A', synaptic_weight_init), allow_pickle=True).item()

        ax_example_A_reward.plot(data['output_test']['iteration'][0],
                                 np.nanmean(data['output_test']['weight_norm'], axis=0),
                                 color=cfg.colors[current_color], linestyle=linestyle)
    ax_example_A_reward.text(0.8, 0.8, 'reward', ha='center', va='center',
                             transform=ax_example_A_reward.transAxes)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[2], 1, 500, 'B', synaptic_weight_init), allow_pickle=True).item()

        ax_example_B_no_reward.plot(data['output_test']['iteration'][0],
                                    np.nanmean(data['output_test']['timing_first_spike'], axis=0),
                                    color=cfg.colors[current_color], linestyle=linestyle)
    ax_example_B_no_reward.text(0.8, 0.8, 'no reward', ha='center', va='center',
                                transform=ax_example_B_no_reward.transAxes)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[2], 0, 500, 'B', synaptic_weight_init), allow_pickle=True).item()

        ax_example_B_reward.plot(data['output_test']['iteration'][0],
                                 np.nanmean(data['output_test']['timing_first_spike'], axis=0),
                                 color=cfg.colors[current_color], linestyle=linestyle)
    ax_example_B_reward.text(0.8, 0.8, 'reward', ha='center', va='center',
                             transform=ax_example_B_reward.transAxes)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[2], 1, 500, 'B', synaptic_weight_init), allow_pickle=True).item()

        ax_example_B_diff_no_reward.plot(data['output_test']['iteration'][0],
                                         np.nanmean(data['output_test']['timing_first_spike_diff'], axis=0),
                                         color=cfg.colors[current_color], linestyle=linestyle)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[2], 0, 500, 'B', synaptic_weight_init), allow_pickle=True).item()

        ax_example_B_diff_reward.plot(data['output_test']['iteration'][0],
                                      np.nanmean(data['output_test']['timing_first_spike_diff'], axis=0),
                                      color=cfg.colors[current_color], linestyle=linestyle)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[2], 1, 500, 'B', synaptic_weight_init), allow_pickle=True).item()

        ax_example_B_accuracy_no_reward.plot(data['output_test']['iteration'][0],
                                             np.nanmean(data['output_test']['accuracy'], axis=0),
                                             color=cfg.colors[current_color], linestyle=linestyle)

for synaptic_weight_init, linestyle in zip(['low', 'high'], ['-', '--']):
    for (A_postpre, A_prepost), current_color in zip(A_post_pre_list, colors_list):
        data = np.load('{}_{}_{}_{}_{:d}_{:d}_{}_{}/Plot_data_1.npy'.format(
            path[0], A_postpre, A_prepost, path[2], 0, 500, 'B', synaptic_weight_init), allow_pickle=True).item()

        ax_example_B_accuracy_reward.plot(data['output_test']['iteration'][0],
                                          np.nanmean(data['output_test']['accuracy'], axis=0),
                                          color=cfg.colors[current_color], linestyle=linestyle)

for ax_ in [ax_example_A_no_reward, ax_example_A_reward, ax_example_B_accuracy_no_reward, ax_example_B_accuracy_reward]:
    ax_.set_xticks([0, 500, 1000, 1500, 2000])
    ax_.set_xlabel('Iterations')
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)

for ax_ in [ax_example_B_diff_no_reward, ax_example_B_diff_reward,
            ax_example_B_no_reward, ax_example_B_reward]:
    ax_.set_xticks([0, 500, 1000, 1500, 2000])
    ax_.set_xticklabels(['', '', '', '', ''])
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)

legend = ax_legend.legend(handles=legend, ncol=3, loc=8, fontsize=6, handlelength=1.)
legend.get_frame().set_linewidth(0.5)

fig.savefig('Figures/article/Figure_2_b.png', dpi=1000)
