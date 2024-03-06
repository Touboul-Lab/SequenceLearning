import sys
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from Articles.DMS_DLS.Code.FiguresDMS import *

ratio = 0.4

name = 'article'
name_dir = 'main'

P_stim_by_pattern_list_noise = [[10, [3, 5], [10, 15], [0.005], [0., 0.01, 0.02, 0.1, 0.2, 1.0]],
                                [20, [3], [30], [0.005], [0., 0.01, 0.02, 0.1, 0.2, 1.0]]]
Apostpre_list = [-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5, 2.0]
Aprepost = -1.0
epsilon = 0.02
homeostasy = 0.95
noise_pattern = 0.2
stop_learning = 'None'
num_success_params = 0
p_reward = 0.5
stim_duration = 100.
stim_offset = 50.
homeostasy_post = 0.
dt = 0.2
num_training_initial = 500
num_training_learning = 500
num_training_maintenance = 500
num_training_recall = 500
num_simu = 200
new_set = 0
save = 'True'
plot = 'False'
random_seed = None

params_simu = {
    'dt': dt,
    'num_training_initial': num_training_initial,
    'num_training_learning': num_training_learning,
    'num_training_maintenance': num_training_maintenance,
    'num_training_recall': num_training_recall,
    'num_simu': num_simu,
    'num_stats': 20,
    'test_iteration': 50,
    'range_simu_fit': 2,
    'ylim_accuracy': (0.4, 1.1),
    'ratio_recall': 0.6,
    'which_Apostpre': [1, 3, 5],
    'which_Apostpre_example_fit': [3],
    'ref': 1,
    'new_set': new_set
}

params_control = [[Apostpre_list[k], 0., 0.95] for k in params_simu['which_Apostpre']]

control_lines = [{'linestyle': 'dotted', 'lw': 0.5, 'color': colors[c_list[k]]}
                 for k in params_simu['which_Apostpre']]

save_directory = '../Figures/simu/' + name + '/' + name_dir
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
else:
    shutil.rmtree(save_directory)
    os.makedirs(save_directory)

for P, stim_by_pattern_list, n_pattern_list, noise_stim_list, stim_recall_list in P_stim_by_pattern_list_noise:
    for stim_by_pattern in stim_by_pattern_list:
        for n_pattern in n_pattern_list:
            for noise_stim in noise_stim_list:
                plot_data_control = []
                for Apostpre_control, Aprepost_control, homeostasy_control in params_control:
                    plot_data_control.append(np.load(
                       path_name(name + '/control', n_pattern, P, stim_by_pattern, Apostpre_control,
                                 Aprepost_control,
                                 homeostasy_control, epsilon, noise_pattern, noise_stim, stop_learning,
                                 num_success_params,
                                 p_reward, stim_duration, stim_offset, homeostasy_post, 1., dt,
                                 num_training_initial,
                                 num_training_learning, num_training_maintenance, num_training_recall, num_simu,
                                 new_set, save, plot, random_seed), allow_pickle=True).item())
                value_analysis = {
                    'value_list_tau': np.zeros((len(Apostpre_list), len(stim_recall_list),
                                                int(params_simu['num_simu']/params_simu['num_stats']))),
                    'value_list_accuracy': np.zeros((4, len(Apostpre_list), len(stim_recall_list),
                                                     int(params_simu['num_simu']/params_simu['num_stats']))),
                    'value_list_tau_recall': np.zeros((len(Apostpre_list), len(stim_recall_list),
                                                       int(params_simu['num_simu'] / params_simu['num_stats']))),
                    'value_weight': np.zeros((2, len(Apostpre_list), len(stim_recall_list),
                                              int(params_simu['num_simu']))),
                }
                for current_l, stim_recall in enumerate(stim_recall_list):
                    gs_kw_1 = dict(hspace=0., wspace=0., height_ratios=[2., 7., 1.5, 3., 1.],
                                   width_ratios=[1.5, 15., 1.5],
                                   top=1., bottom=0., left=0., right=1.)
                    fig1, ax1_ = plt.subplots(figsize=(ratio * 18., ratio * 14.5),
                                              ncols=3, nrows=5, gridspec_kw=gs_kw_1)

                    gs_1 = ax1_[0, 0].get_gridspec()
                    for ax_ in list(ax1_[:, 0]) + list(ax1_[:, 1]) + list(ax1_[:, 2]):
                        ax_.remove()

                    gs_kw_1_subplot_1 = dict(hspace=0., wspace=0., height_ratios=[0.5, 3., 1.5, 0.5, 1.5])
                    gs_1_subplot_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_1[1, 1],
                                                                                 **gs_kw_1_subplot_1)
                    ax_1_iter = [fig1.add_subplot(gs_1_subplot_1[0, 0]), fig1.add_subplot(gs_1_subplot_1[1, 0]),
                                 fig1.add_subplot(gs_1_subplot_1[3, 0]), fig1.add_subplot(gs_1_subplot_1[4, 0])]

                    legend_different_phases(ax_1_iter[0], params_simu, stim_recall)
                    ax_1_iter[0].set_xticks([])
                    ax_1_iter[0].set_yticks([])
                    figure_different_phases(ax_1_iter[1], params_simu, ylim=params_simu['ylim_accuracy'], xticks=True)

                    ax_1_iter[1].set_ylabel(u'Accuracy\nTemporal course', labelpad=5)
                    ax_1_iter[1].set_yticks([0.4, 0.6, 0.8, 1.])
                    ax_1_iter[1].set_xlabel('Iterations')

                    legend_different_phases(ax_1_iter[2], params_simu, stim_recall)
                    ax_1_iter[2].set_xticks([])
                    ax_1_iter[2].set_yticks([])

                    figure_different_phases(ax_1_iter[3], params_simu, ylim=params_simu['ylim_accuracy'])
                    ax_1_iter[3].set_ylabel('Accuracy\nEnd of phase', labelpad=5)
                    ax_1_iter[3].set_yticks([0.5, 1.])

                    gs_kw_1_subplot_2 = dict(hspace=0., wspace=0., height_ratios=[0.5, 3., 1.5,  0.5, 1.5],
                                             width_ratios=[params_simu['num_training_initial'],
                                                           params_simu['num_training_learning'],
                                                           params_simu['num_training_maintenance'],
                                                           params_simu['num_training_recall']])

                    gs_1_subplot_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=gs_1[1, 1],
                                                                                 **gs_kw_1_subplot_2)
                    ax_1_accuracy = [fig1.add_subplot(gs_1_subplot_2[4, 0]), fig1.add_subplot(gs_1_subplot_2[4, 1]),
                                     fig1.add_subplot(gs_1_subplot_2[4, 2]), fig1.add_subplot(gs_1_subplot_2[4, 3])]

                    gs_kw_1_subplot_3 = dict(hspace=0., wspace=0., width_ratios=[3., 1.5, 3., 1.5, 3., 1.5, 3.])

                    gs_1_subplot_3 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs_1[3, 1],
                                                                                 **gs_kw_1_subplot_3)
                    ax_1_weight = [fig1.add_subplot(gs_1_subplot_3[0, 0]), fig1.add_subplot(gs_1_subplot_3[0, 2]),
                                   fig1.add_subplot(gs_1_subplot_3[0, 4]), fig1.add_subplot(gs_1_subplot_3[0, 6])]

                    figure_different_phases_maintenance(ax_1_weight[0], params_simu)
                    ax_1_weight[0].set_ylabel('Accuracy', labelpad=5)
                    ax_1_weight[0].set_yticks([0.4, 0.6, 0.8])
                    ax_1_weight[0].set_xlabel('Iterations')
                    ax_1_weight[0].set_xlim(1000, 1100)
                    ax_1_weight[0].set_xticks([1000, 1100])

                    legend_elements_fit = [Line2D([0], [0], color=colors['black'], linestyle='None', lw=1., marker='o',
                                                  mfc='none', markersize=3, markeredgewidth=0.5,
                                                  label='Average of {:d} sim.'.format(
                                                    params_simu['num_stats']))] +\
                                          [Line2D([0], [0], color=colors['black'], lw=0.5, label='Associated fit')] + \
                                          [Line2D([0], [0], color=colors['black'], lw=0.5, linestyle='dotted',
                                                  label='Tangent', alpha=0.5)] + \
                                          [Line2D([0], [0], color=colors['black'], lw=0.5, linestyle='dashed',
                                                  label='Final accuracy', alpha=0.5)] + \
                                          [Line2D([0], [0], color=colors['black'], linestyle='None', lw=1., marker='x',
                                                  mfc='none', markersize=2, markeredgewidth=0.5,
                                                  label=r'$T_{{\rm maintenance}}$ distribution')]

                    ax_1_weight[0].legend(handles=legend_elements_fit, loc=2, ncol=1,
                                          frameon=False, labelspacing=0.2, fontsize=6)

                    ax_1_weight[1].set_xticks([-1., 0., 1., 2.])
                    ax_1_weight[1].set_ylabel(r'$T_{\rm maintenance}$', labelpad=5)
                    ax_1_weight[1].set_xlabel(r'$A_{\rm post-pre}$', labelpad=5)

                    ax_1_weight[2].set_ylabel('Weight similarity measures', labelpad=5)
                    ax_1_weight[2].set_xlabel(r'$A_{\rm post-pre}$', labelpad=5)
                    ax_1_weight[2].set_ylim(-0.5, 1.5)

                    legend_elements_weight = [Line2D([0], [0], color=colors['black'], lw=0.5, marker='x', markersize=2,
                                                     markeredgewidth=0.2, linestyle='dotted',
                                                     label=u'Norm $d_2(W)$')] + \
                                             [Line2D([0], [0], color=colors['black'], marker='o', markersize=2,
                                                     markeredgewidth=0.2, lw=0.5, linestyle='--',
                                                     label=u'Cosine $sp(W)$')]

                    leg_weight = ax_1_weight[2].legend(handles=legend_elements_weight, loc=2, ncol=1,
                                                       frameon=False, labelspacing=0.2)

                    ax_1_weight[3].set_xticks([-1., 0., 1., 2.])
                    ax_1_weight[3].set_ylabel(r'$T_{\rm relearning}$', labelpad=5)
                    ax_1_weight[3].set_xlabel(r'$A_{\rm post-pre}$', labelpad=5)

                    for current_i in np.arange(len(params_control)):
                        plot_results(plot_data_control[current_i], ax_1_iter[1], None, params_simu,
                                     control=True, control_lines=control_lines[current_i])
                    for current_k, Apostpre in enumerate(Apostpre_list):
                        plot_data = np.load(
                            path_name(name + '/' + name_dir, n_pattern, P, stim_by_pattern, Apostpre, Aprepost,
                                      homeostasy, epsilon, noise_pattern, noise_stim, stop_learning, num_success_params,
                                      p_reward, stim_duration, stim_offset, homeostasy_post, stim_recall, dt,
                                      num_training_initial, num_training_learning, num_training_maintenance,
                                      num_training_recall, num_simu,
                                      new_set, save, plot, random_seed), allow_pickle=True).item()
                        if current_k in params_simu['which_Apostpre']:
                            if current_k in params_simu['which_Apostpre_example_fit']:
                                plot_results(plot_data, ax_1_iter[1], ax_1_weight[0], params_simu,
                                             value_analysis=value_analysis, current_k=current_k, current_l=current_l)
                            else:
                                plot_results(plot_data, ax_1_iter[1], None, params_simu,
                                             value_analysis=value_analysis, current_k=current_k, current_l=current_l)
                        else:
                            plot_results(plot_data, None, None, params_simu,
                                         value_analysis=value_analysis, current_k=current_k,
                                         current_l=current_l)
                        plot_results_weight(plot_data, None, params_simu,
                                            value_analysis=value_analysis, current_k=current_k,
                                            current_l=current_l)

                    c_list_figure = [c_list[i] for i in params_simu['which_Apostpre']]
                    legend_elements = [Line2D([0], [0], color=colors[c_list_figure[k]],
                                              linewidth=0.5, label=str(Apostpre)) for k, Apostpre in
                                       enumerate([-1., 0.0, 1.])] + [Line2D([0], [0], color='w')] + \
                                      [Line2D([0], [0], color=colors['black'], lw=0.5, label='Simulations')] + \
                                      [Line2D([0], [0], color=colors['black'], lw=0.5, linestyle='dotted',
                                              label=r'Control ($A_{\rm pre-post}=0$)')] + \
                                      [Line2D([0], [0], color='w')]

                    leg = ax_1_iter[1].legend(handles=legend_elements, loc=(0.01, 0.2), ncol=1,
                                              title=r'$A_{\rm post-pre}$',
                                              frameon=False, labelspacing=0.2)
                    leg._legend_box.align = 'left'

                    for current_ax_number, current_ax_1 in enumerate(ax_1_accuracy):
                        figure_stats(current_ax_1, np.array(Apostpre_list),
                                     value_analysis['value_list_accuracy'][current_ax_number, :, current_l],
                                     color=colors['black'], ref=params_simu['ref'])
                        current_ax_1.set_ylim(*params_simu['ylim_accuracy'])
                        current_ax_1.set_yticks([])
                        current_ax_1.set_xlabel(r'$A_{\rm post-pre}$')

                    figure_stats(ax_1_weight[1], np.array(Apostpre_list),
                                 value_analysis['value_list_tau'][:, current_l],
                                 color=colors['black'], ref=params_simu['ref'])

                    figure_stats(ax_1_weight[2], np.array(Apostpre_list),
                                 value_analysis['value_weight'][0, :, current_l],
                                 color=colors['black'], linestyle='-', ref=params_simu['ref'])
                    figure_stats(ax_1_weight[2], np.array(Apostpre_list),
                                 value_analysis['value_weight'][1, :, current_l],
                                 color=colors['grey'], linestyle='-', stats_text_up=False, ref=params_simu['ref'])

                    figure_stats(ax_1_weight[3], np.array(Apostpre_list),
                                 value_analysis['value_list_tau_recall'][:, current_l],
                                 color=colors['black'], ref=params_simu['ref'])

                    ax_1_weight[0].set_ylim(params_simu['ylim_accuracy'][0], 1.)
                    for ax_ in [ax_1_weight[1], ax_1_weight[3]]:
                        ax_.set_ylim(top=ax_.get_ylim()[1] + 0.4 * (ax_.get_ylim()[1] - ax_.get_ylim()[0]))

                    title = '{:d}_{:d}_{}_{}_{:d}'.format(P, stim_by_pattern, str(stim_recall),
                                                          str(noise_stim), n_pattern)

                    ax_1_iter[0].set_title(
                        r'Set-up: $P={:d}$ input neurons, $N_{{\rm stim}}={:d}$ stimulations by pattern,'
                        u'\n$N_p={:d}$ number of patterns and proportion of pattern presentation $\eta_m={}$, '
                        r'$\lambda_{{\rm MSN}}={}\,Hz$'.format(P, stim_by_pattern, n_pattern,
                                                               str(stim_recall), str(1000 * noise_stim)), pad=20)
                    fig1.savefig(save_directory + '/' + title + '.svg')
                    plt.close(fig1)
                gs_kw_2 = dict(hspace=0., wspace=0., width_ratios=[1.5, 4., 1.5, 4., 1.5, 4., 1.5],
                               height_ratios=[2., 3., 1.],
                               top=1., bottom=0., left=0., right=1.)
                fig2, ax2 = plt.subplots(figsize=(ratio * 18., ratio * 6.), ncols=7, nrows=3,
                                         gridspec_kw=gs_kw_2, sharex=False)

                gs_2 = ax2[0, 0].get_gridspec()

                for ax_ in [ax2[1, 0], ax2[1, 2], ax2[1, 4], ax2[1, 6]] + list(ax2[0]) + list(ax2[2]):
                    ax_.remove()

                ax_2_title = fig2.add_subplot(gs_2[1, :])
                set_blank_axis(ax_2_title)
                ax_2_title.set_zorder(-10)

                ax2[1, 1].set_ylabel(r'$T_{\rm maintenance}$', labelpad=5,
                                     color=colors['black'])

                ax2[1, 3].set_ylabel('Accuracy, end of maintenance', labelpad=5, color=colors['black'])

                ax2[1, 5].set_ylabel(r'$T_{\rm relearning}$', labelpad=5,
                                     color=colors['black'])

                legend_elements_delay = [Line2D([0], [0], color=colors[c_list_figure[k]], linestyle='-',
                                                marker='+', lw=0.5,
                                                label=str(Apostpre), markersize=4,
                                                markeredgewidth=0.5) for k, Apostpre in
                                         enumerate([-1.0, 0., 1.])]
                for counter_ax in [1, 3, 5]:
                    if counter_ax == 1:
                        ax2[1, counter_ax].legend(title=r'$A_{\rm post-pre}$', handles=legend_elements_delay,
                                                  loc=1, frameon=False)
                    ax2[1, counter_ax].spines['top'].set_visible(False)
                    ax2[1, counter_ax].spines['right'].set_visible(False)
                    ax2[1, counter_ax].set_xlim(-0.1, 1.1)
                    ax2[1, counter_ax].set_xlabel('Proportion of pattern presentation $\eta_m$', labelpad=5)

                for current_k in params_simu['which_Apostpre']:
                    ax2[1, 1].plot(np.array(stim_recall_list),
                                   np.mean(value_analysis['value_list_tau'][current_k, :], axis=1),
                                   '-+', color=colors[c_list[current_k]], label=str(Apostpre_list[current_k]),
                                   markersize=4, markeredgewidth=0.5)
                    ax2[1, 3].plot(np.array(stim_recall_list),
                                   np.mean(value_analysis['value_list_accuracy'][2, current_k, :], axis=1),
                                   '--+', color=colors[c_list[current_k]], label=str(Apostpre_list[current_k]),
                                   markersize=4, markeredgewidth=0.5)
                    ax2[1, 5].plot(np.array(stim_recall_list),
                                   np.mean(value_analysis['value_list_tau_recall'][current_k, :], axis=1),
                                   '-+', color=colors[c_list[current_k]], label=str(Apostpre_list[current_k]),
                                   markersize=4, markeredgewidth=0.5)

                ax_2_title.set_title(
                    'Set-up: $P=' + str(P) + r'$ input neurons, $N_{\rm stim}=' + str(stim_by_pattern) +
                    '$ stimulations by pattern and $N_p=' + str(n_pattern) + '$ number of patterns, ' +
                    r'$\lambda_{{\rm MSN}}={}\,Hz$'.format(str(1000 * noise_stim)), pad=20)

                fig2.savefig(
                    save_directory + '/' + str(P) + '_' + str(stim_by_pattern) + '_'
                    + str(n_pattern) + '.svg')
                plt.close(fig2)
