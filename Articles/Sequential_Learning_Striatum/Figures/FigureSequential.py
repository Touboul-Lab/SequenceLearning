import numpy as np
import scipy.stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from Articles.Sequential_Learning_Striatum.Figures.cfg_color import colors

latex = '$'

def stats_text(p_value):
    text = ''
    p = 0.05
    while p_value < p and p > 0.00005:
        text += '*'
        p /= 10.
    if len(text) == 0:
        text = 'n.s'
    return text

def accuracy_max(data):
    def func_(n_1, n_2):
        results = np.zeros_like(data)
        if 2 * n_1 < data.shape[1]:
            for j in np.arange(data.shape[1]):
                if n_1 - 1 < j < data.shape[1] - n_1:
                    current_accuracy = data[:, j - n_1: j + n_1 + 1]
                    current_accuracy_sort = np.sort(current_accuracy, axis=1)
                    results[:, j] = np.mean(current_accuracy_sort[:, - n_2:], axis=1)
                else:
                    results[:, j] = np.nan * np.ones_like(data[:, j])
        else:
            results = np.nan * np.ones_like(data)
        return results
    return func_

def set_subplot_legend(fig, gs, name, set_y_value=None, **kwargs):
    ax_legend = fig.add_subplot(gs)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    if set_y_value is None:
        ax_legend.set_ylabel(name, **kwargs)
    else:
        ax_legend.set_ylabel(name, **kwargs).set_y(set_y_value)
    return ax_legend

def set_blank_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_results(plot_data, plot_data_control, ax, which_measure,
                 params_others, params_legend_handles, compute_max=True,
                 value_analysis=None, value_analysis_control=None,
                 current_k=0, current_l=0):
    if compute_max:
        n_1, n_2 = 10, 1
        results = accuracy_max(plot_data['output_test_' + which_measure])(n_1, n_2)
    else:
        n_1, n_2 = 0, None
        results = plot_data['output_test_' + which_measure]

    if ax is not None:
        ax.plot(plot_data['output_test_iteration'],
                np.nanmean(results, axis=0), linestyle=params_legend_handles['linestyle'],
                color=colors[params_legend_handles['color']], label=params_legend_handles['label'])

    if plot_data_control is not None:
        if compute_max:
            results_control = accuracy_max(plot_data_control['output_test_' + which_measure])(n_1, n_2)
        else:
            results_control = plot_data_control['output_test_' + which_measure]

    if value_analysis is not None or value_analysis_control is not None:
        value_analysis['value_list'][current_k, current_l] = np.nanmean(results, axis=0)[- 1 - n_1]
        value_analysis['value_list_std'][current_k, current_l] = np.nanstd(results, axis=0)[- 1 - n_1]
        if plot_data_control is not None:
            value_analysis_control['value_list'][current_k, current_l] = np.nanmean(results_control, axis=0)[-1 - n_1]
            value_analysis_control['value_list_std'][current_k, current_l] = \
                np.nanstd(results_control, axis=0)[-1 - n_1]

            statistic_control, p_value_control = scipy.stats.ttest_ind(results[:, - 1 - n_1],
                                                                       results_control[:, - 1 - n_1],
                                                                       nan_policy='omit')
            text_control = stats_text(p_value_control)

            value_analysis_control['stats'][current_k, current_l] = text_control

        if 'WiRef' in params_others['stats']:
            value_analysis['ref_stats'][current_l][params_others['stats'][2:]] = results[:, - 1 - n_1]
        value_analysis['data_stats'][current_k, current_l] = results[:, - 1 - n_1]

def compute_stat(params_stats, value_analysis=None, current_k=0, current_l=0, vertical=True):
    text_stats = ''
    if value_analysis['data_stats'][current_k, current_l] is not None:
        if 'WiRef' in params_stats:
            text_stats = u'\u2193' if vertical else u'\u2190'
        else:
            if params_stats[:2] == 'To':
                if params_stats[2:] in value_analysis['ref_stats'][current_l].keys():
                    statistic_stats, p_value_stats = \
                        scipy.stats.ttest_ind(value_analysis['data_stats'][current_k, current_l],
                                              value_analysis['ref_stats'][current_l][params_stats[2:]],
                                              nan_policy='omit')
                elif len(params_stats) > 4 and params_stats[:5] == 'ToRef':
                    return ''
                else:
                    statistic_stats, p_value_stats = \
                        scipy.stats.ttest_1samp(value_analysis['data_stats'][current_k, current_l],
                                                float(params_stats[2:]), nan_policy='omit')
                text_stats = stats_text(p_value_stats)
    return text_stats
