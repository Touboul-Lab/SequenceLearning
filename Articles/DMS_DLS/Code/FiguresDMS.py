import numpy as np
from scipy.optimize import curve_fit
import matplotlib.patches as patches
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 7.
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth'] = 0.5
# set tick width
matplotlib.rcParams['xtick.major.size'] = 2
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['xtick.minor.size'] = 1
matplotlib.rcParams['xtick.minor.width'] = 0.5
matplotlib.rcParams['ytick.major.size'] = 2
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['ytick.minor.size'] = 1
matplotlib.rcParams['ytick.minor.width'] = 0.5

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'

path_simu = '../'


def RGB(R, G, B):
    return R/255., G/255., B/255.


colors = {
        'red': RGB(249, 102, 94),
        'pale red': RGB(252, 216, 226),
        'blue': RGB(68, 119, 178),
        'green': RGB(3, 192, 60),
        'yellow': RGB(241, 219, 5),
        'black': RGB(0, 0, 0),
        'brown': RGB(206, 156, 111),
        'white': RGB(255, 255, 255),
        'grey': RGB(145, 143, 144),
        'pale grey': RGB(222, 221, 222),
        'orange': RGB(255, 152, 6),
        'dark purple': RGB(92, 29, 100),
        'light purple': RGB(202, 111, 214),
        'dark green': 	RGB(128, 128, 0),
        'light green': RGB(204, 255, 144)
        }

c_list = ['white', 'blue', 'yellow', 'brown', 'dark purple', 'orange', 'red', 'black']
style = ['o', '+', 's', '.', 'x']
line_style = ['+-', 'o-', 'x-']


def set_blank_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def path_name(name, n_pattern, *args, title=False):
    path = 'Simu/'+name+'/' + ''.join([str(key) + '_' for key in args])[:-1]
    if title:
        return ''.join([str(key) + '_' for key in args])[:-1]
    else:
        return path_simu+path+'/Plot_data_'+str(n_pattern)+'.npy'


def burr_funct(t, init_accuracy, end_accuracy):
    def func(x, d, g):
        return end_accuracy + (init_accuracy - end_accuracy)*(1.+((x-t)/g))**(-d)
    return func


def burr_tau(popt_):
    tau = np.maximum(0., popt_[1] / popt_[0])
    return tau


def stats(ax, x_axis, list_values, color=colors['black'], stats_text_up=True, ref=0):
    for i in np.arange(len(x_axis)):
        if i == ref:
            if not np.isnan(np.mean(list_values[ref])):
                if stats_text_up:
                    ax.text(x_axis[ref], np.mean(list_values[ref]) + np.std(list_values[ref]), u'\u2193',
                            horizontalalignment='center', verticalalignment='bottom', color=color,
                            fontsize=8).set_clip_on(False)
                else:
                    ax.text(x_axis[ref], np.mean(list_values[ref]) - np.std(list_values[ref]), u'\u2191',
                            horizontalalignment='center', verticalalignment='top', color=color, fontsize=8).set_clip_on(
                        False)
        else:
            statistic, pvalue = scipy.stats.ttest_ind(list_values[i], list_values[ref])
            text = ''
            p = .05
            while pvalue < p and p > 0.00005:
                text += '*'
                p /= 10.
            if len(text) == 0:
                text = ''
            if not np.isnan(np.mean(list_values[i])):
                if stats_text_up:
                    ax.text(x_axis[i], np.mean(list_values[i]) + 0.5 * np.std(list_values[i]), text,
                            horizontalalignment='center', verticalalignment='center',
                            color=color, fontsize=8).set_clip_on(False)
                else:
                    ax.text(x_axis[i], np.mean(list_values[i]) - 0.7 * np.std(list_values[i]), text,
                            horizontalalignment='center', verticalalignment='center',
                            color=color, fontsize=8).set_clip_on(False)

def figure_stats(ax, xx, yy, color=None, label=None, linestyle='-', stats_text_up=True, ref=0):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if label is not None:
        ax.errorbar(xx, np.mean(yy, axis=1),
                    yerr=np.std(yy, axis=1) / 2., color=color, label=label, linestyle=linestyle)
    else:
        ax.errorbar(xx, np.mean(yy, axis=1),
                    yerr=np.std(yy, axis=1) / 2., color=color, linestyle=linestyle)
    stats(ax, xx, yy, color=color, stats_text_up=stats_text_up, ref=ref)
    ax.tick_params(axis='y', which='both', color=color)

def plot_results(plot_data, ax, ax_fit, params_simu, value_analysis=None, current_k=0, current_l=0,
                 control=False, control_lines=None):
    if ax is not None:
        if not control:
            ax.plot(plot_data['output_test_iteration'],
                    np.mean(plot_data['output_test_accuracy'], axis=0), '-',
                    color=colors[c_list[current_k]])
        else:
            ax.plot(plot_data['output_test_iteration'],
                    np.mean(plot_data['output_test_accuracy'], axis=0), **control_lines)
    if not control:
        x_0 = 0
        x_1 = x_0 + params_simu['num_training_initial']
        x_2 = x_1 + params_simu['num_training_learning']
        x_3 = x_2 + params_simu['num_training_maintenance']
        x_4 = x_3 + params_simu['num_training_recall']
        bounds_list = [[x_0, x_1], [x_1, x_2], [x_2, x_3], [x_3, x_4]]

        for n in np.arange(int(params_simu['num_simu']/params_simu['num_stats'])):
            x_iteration = plot_data['output_test_iteration']
            current_simu = params_simu['num_stats'] * n, params_simu['num_stats'] * (n+1)
            x_scaled = np.mean(plot_data['output_test_accuracy'][current_simu[0]:current_simu[1]], axis=0)
            for u, bounds in enumerate(bounds_list):
                index_low, index_high = (np.where(x_iteration == bounds[0])[0][0],
                                         np.where(x_iteration == bounds[1])[0][0])
                value_analysis['value_list_accuracy'][u, current_k, current_l, n] = \
                    x_scaled[index_high]
                if u == 2:
                    last_accuracy = value_analysis['value_list_accuracy'][1, current_k, current_l, n]
                    base_accuracy = value_analysis['value_list_accuracy'][0, current_k, current_l, n]
                    xx_2 = np.linspace(bounds[0], bounds[1], 1000)
                    popt, pcov = curve_fit(burr_funct(bounds[0], last_accuracy, base_accuracy),
                                           x_iteration[index_low:index_high],
                                           x_scaled[index_low:index_high], p0=(1., 0.1),
                                           bounds=([0., 0.], [np.inf, np.inf]), maxfev=10000000)
                    tau = burr_tau(popt)
                    value_analysis['value_list_tau'][current_k, current_l, n] = np.minimum(tau, 2 * (bounds[1] - bounds[0]))
                    yy_2 = burr_funct(bounds[0], last_accuracy, base_accuracy)(xx_2, *popt)
                    yy_2_last = base_accuracy
                    yy_2_linear = last_accuracy - (xx_2 - bounds[0]) * (last_accuracy - base_accuracy) / tau
                    if ax_fit is not None:
                        if n < params_simu['range_simu_fit']:
                            ax_fit.plot(x_iteration, x_scaled, 'o', mfc='none',
                                        color=colors[c_list[current_k]], markersize=3, markeredgewidth=0.5)
                            ax_fit.plot(xx_2, yy_2, color=colors[c_list[current_k]])
                            ax_fit.plot([bounds[0], bounds[1]], [yy_2_last, yy_2_last], linestyle='dashed',
                                        color=colors[c_list[current_k]], alpha=0.5)
                            ax_fit.plot(xx_2[yy_2_linear >= 0.], yy_2_linear[yy_2_linear >= 0.], linestyle='dotted',
                                        color=colors[c_list[current_k]], alpha=0.5)
                            ax_fit.plot([bounds[0] + tau, bounds[0] + tau], [params_simu['ylim_accuracy'][0],
                                        popt[0]], '-', color=colors[c_list[current_k]], alpha=0.2)

                elif u == 3:
                    accuracy_base = value_analysis['value_list_accuracy'][0, current_k, current_l, n]
                    accuracy_learning = value_analysis['value_list_accuracy'][1, current_k, current_l, n]
                    tau_recall_where = (x_scaled[index_low:index_high] - accuracy_base) - params_simu['ratio_recall']\
                        * (accuracy_learning - accuracy_base) > 0.
                    if any(tau_recall_where):
                        tau_recall = x_iteration[index_low:index_high][tau_recall_where][0] - x_3
                    else:
                        tau_recall = x_4 - x_3
                    value_analysis['value_list_tau_recall'][current_k, current_l, n] = tau_recall

        if ax_fit is not None:
            where_tau = value_analysis['value_list_tau'][current_k, current_l] < x_3 - x_2
            ax_fit.plot(x_2 + value_analysis['value_list_tau'][current_k, current_l][where_tau],
                        params_simu['ylim_accuracy'][0] * np.ones_like(
                            value_analysis['value_list_tau'][current_k, current_l][where_tau]), 'x',
                        color=colors[c_list[current_k]], clip_on=False, ms=2., markeredgewidth=0.5, zorder=10)


def plot_results_weight(plot_data, ax, params_simu, value_analysis=None, current_k=0, current_l=0):
    x_0 = 0
    x_1 = x_0 + params_simu['num_training_initial']
    x_2 = x_1 + params_simu['num_training_learning']
    x_3 = x_2 + params_simu['num_training_maintenance']
    index_low, index_high = (np.where(plot_data['output_test_iteration'] == x_2)[0][0],
                             np.where(plot_data['output_test_iteration'] == x_3)[0][0])
    if ax is not None:
        ax.plot(plot_data['output_test_iteration'][index_low: index_high],
                np.mean(plot_data['norm_weight_diff_standard'], axis=0)[index_low: index_high],
                linestyle='dotted', color=colors[c_list[current_k]], markersize=4, markeredgewidth=0.5)
        ax.plot(plot_data['output_test_iteration'][index_low: index_high],
                np.mean(plot_data['cosine_sim_center'], axis=0)[index_low: index_high],
                linestyle='--', color=colors[c_list[current_k]], markersize=4, markeredgewidth=0.5)
    for n in np.arange(int(params_simu['num_simu'])):
        value_analysis['value_weight'][0, current_k, current_l, n] = \
            plot_data['norm_weight_diff_standard'][n][index_high - 1]
        value_analysis['value_weight'][1, current_k, current_l, n] = plot_data['cosine_sim_center'][n][index_high - 1]


def figure_different_phases(ax, params_simu, ylim, xticks=False):
    x_0 = 0
    x_1 = x_0 + params_simu['num_training_initial']
    x_2 = x_1 + params_simu['num_training_learning']
    x_3 = x_2 + params_simu['num_training_maintenance']
    x_4 = x_3 + params_simu['num_training_recall']
    ax.set_xlim(x_0, x_4)
    if xticks:
        ax.set_xticks([x_0, x_1, x_2, x_3, x_4])
    else:
        ax.set_xticks([])
    for x_ in [x_1, x_2, x_3]:
        ax.plot([x_, x_], [*ylim], '--', color=colors['black'])
    ax.set_ylim(*ylim)


def figure_different_phases_maintenance(ax, params_simu):
    x_0 = 0
    x_1 = x_0 + params_simu['num_training_initial']
    x_2 = x_1 + params_simu['num_training_learning']
    x_3 = x_2 + params_simu['num_training_maintenance']
    ax.set_xlim(x_2, x_3)
    ax.set_xticks([x_2, x_3])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def legend_different_phases(ax, params_simu, eta):
    x_0 = 0
    x_1 = x_0 + params_simu['num_training_initial']
    x_2 = x_1 + params_simu['num_training_learning']
    x_3 = x_2 + params_simu['num_training_maintenance']
    x_4 = x_3 + params_simu['num_training_recall']

    ax.set_xlim(x_0, x_4)
    ax.set_ylim(0., 2.)
    ax.add_patch(patches.Rectangle((x_0, 0.), x_1 - x_0, 2.,
                                   facecolor=colors['pale grey'], edgecolor=colors['black'], linewidth=0.5))
    ax.add_patch(patches.Rectangle((x_1, 0.), x_2 - x_1, 2.,
                                   facecolor=colors['pale red'], edgecolor=colors['black'], linewidth=0.5))
    ax.add_patch(patches.Rectangle((x_2, 0.), x_3 - x_2, 2.,
                                   facecolor=colors['pale grey'], edgecolor=colors['black'], linewidth=0.5))
    ax.add_patch(patches.Rectangle((x_3, 0.), x_4 - x_3, 2.,
                                   facecolor=colors['pale red'], edgecolor=colors['black'], linewidth=0.5))

    ax.text((x_0 + x_1) / 2., 1., u'Initial\nreward OFF, $\eta=' + str(eta) + '$',
            horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax.text((x_1 + x_2) / 2., 1., u'Learning\nreward ON, $\eta=1$',
            horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax.text((x_2 + x_3) / 2., 1., u'Maintenance\nreward OFF, $\eta=' + str(eta) + '$',
            horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax.text((x_3 + x_4) / 2., 1., u'Relearning\nreward ON, $\eta=1$',
            horizontalalignment='center', verticalalignment='center', fontsize=6)
