import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from Articles.Sequential_Learning_Striatum.Figures.cfg_color import colors
from Articles.Sequential_Learning_Striatum.Figures.FigureSequential \
    import plot_results, compute_stat, set_blank_axis

latex = '$'

class legend_handles:
    def __init__(self, item_legend):
        self.item_legend = item_legend
        self.handles = self.item_legend['handles']
        if self.handles == 'line' or self.handles == '':
            self.function = Line2D
        elif self.handles == 'patch':
            self.function = Patch

    def args(self):
        if self.handles == 'line':
            args = ([0.], [0.])
            kwargs = {
                'linestyle': self.item_legend['linestyle'],
                'color': colors[dict_key_legend_handles['color'].dtype(self.item_legend['color'])],
                'lw': 1.,
                'label': self.item_legend['label'],
                'marker': self.item_legend['marker'],
            }
        elif self.handles == 'patch':
            args = ()
            kwargs = {
                'fill': True,
                'color': colors[dict_key_legend_handles['color'].dtype(self.item_legend['color'])],
                'label': self.item_legend['label'],
            }
        elif self.handles == '':
            args = ([0.], [0.])
            kwargs = {
                'linestyle': '',
                'color': colors['white'],
                'lw': 0.,
                'label': '',
                'marker': '',
            }
        return args, kwargs

    def output(self):
        args, kwargs = self.args()
        return self.function(*args, **kwargs)

class key:
    def __init__(self, dtype, label, title=None):
        self.dtype = dtype
        self.label = label
        if title is None:
            self.title = self.label
        else:
            self.title = title

    def label_value(self, label):
        if label == 'accuracy':
            return 'accuracy'.format(latex)
        elif label == 'accuracy_max':
            return r'accuracy max'
        elif label == 'success':
            return 'success rate'
        elif label == 'list_pattern':
            return 'Task 1 (sequences of cortical spikes)'
        elif label == 'poisson':
            return 'Task 4 (Poisson patterns)'
        elif label == 'jitter':
            return 'Task 3 (sequences of jittered cortical spikes)'
        elif label == 'succession':
            return 'Task 2 (all possibles ordered sequences) with {0}P=N_{{p}}{0}'.format(latex)
        elif label == 'None':
            return 'Strategy 1'
        elif label == 'number_success':
            return 'Strategy 2'
        elif label == 'MSN_IAF_EXP':
            return 'linear IAF (M1)'
        elif label == 'MSN_IAF_MSN_EXP':
            return 'IAF-MSN Model from experimental data'
        elif label == 'MSN_Burst_EXP':
            return 'Non-linear IAF with bursting'
        elif label == 'MSN_Izhi_EXP':
            return 'Non-linear IAF with latency'
        elif label == 'MSN_Izhi':
            return 'non-linear IAF (M2)'
        elif label == 'MSN_Hump_1':
            return 'Non-linear IAF from Humphrey et al. (2009)a'
        elif label == 'MSN_Hump_2':
            return 'Non-linear IAF from Humphrey et al. (2009)b'
        else:
            return str(label)

def bool_str(str_value):
    return str_value == 'True'

def color_white(str_value):
    if str_value == '':
        return 'white'
    else:
        return str_value

def float_or_None(item):
    if item == 'None':
        return None
    else:
        return item


dict_key = {
    'network':  key(str, 'type of network'),
    'P': key(int, 'Number of input neurons {0}P{0}'.format(latex), title='{0}P={0}'.format(latex)),
    'neuronClass': key(str, 'Neuronal models', title=''),
    'Apostpre': key(float, r'amplitude post-pre {0}A_{{\rm post-pre}}{0}'.format(latex)),
    'Aprepost': key(float, r'amplitude pre-post {0}A_{{\rm pre-post}}{0}'.format(latex)),
    'homeostasy': key(float, r'non-associative LTP {0}A_{{\rm reward}}{0}'.format(latex)),
    'epsilon': key(float, '{0}\epsilon{0}'.format(latex)),
    'noise_stim': key(float, r'{0}\lambda_{{\rm stim}}{0} (KHz)'.format(latex),
                      title=r'{0}\lambda_{{\rm stim}}={0}'.format(latex)),
    'noise_input': key(float, r'{0}\lambda_{{\rm ext}}{0} (KHz)'.format(latex)),
    'noise_pattern': key(float, r'{0}\tau_{{\rm pattern}}{0} (ms)'.format(latex)),
    'stop_learning': key(str, '{0}S_{{end}}{0}'.format(latex)),
    'num_success_params': key(int, 'num_success_params'),
    'dt': key(float, '{0}dt{0}'),
    'num_training': key(int, 'number of training iterations'),
    'stim_duration': key(float, 'pattern duration'),
    'stim_offset': key(float, 'pattern offset'),
    'save': key(bool_str, 'save'),
    'plot': key(bool_str, 'plot'),
    'random_seed': key(int, 'random_seed'),
    'J_matrix': key(int, 'type of inhibition'),
    'J_value': key(float, 'value of inhibition {0}J{0}'.format(latex)),
    'J_reward': key(str, 'J_reward'),
    'pattern': key(str, 'type of pattern'),
    'stim_by_pattern': key(int, r'Maximum number of stimulations by pattern {0}N_{{\rm stim}}{0}'.format(latex),
                           title=r'{0}N_{{\rm stim}}={0}'.format(latex)),
    'repartition': key(str, 'repartition pattern'),
    'p_reward': key(float, 'probability of reward'),
    'stim_delay': key(float, r'{0}T_{{\rm delay}}{0}'),
    'num_simu': key(int, 'number of simulations'),
    'duration_poisson': key(float, r'{0}T_{{\rm poisson}}{0}'.format(latex),
                            title=r'{0}T_{{\rm poisson}}={0}'.format(latex)),
    'noise_poisson': key(float, r'{0}\lambda_{{\rm poisson}}{0}'.format(latex),
                         title=r'{0}\lambda_{{\rm poisson}}={0}'.format(latex)),
}

dict_key_others = {
    'name': key(str, 'name'),
    'dir': key(str, 'dir_simu'),
    'which_list': key(str, 'parameter'),
    'stats': key(str, 'stats'),
    'color_stats': key(str, 'color_stats'),
    'fontsize': key(float, 'fontsize'),
    'n_pattern': key(int, 'number of patterns {0}N_{{p}}{0}'.format(latex),
                     title='{0}N_{{p}}={0}'.format(latex)),
}

dict_key_legend_handles = {
    'handles': key(str, 'handles'),
    'color': key(color_white, 'color'),
    'linestyle': key(str, 'linestyle'),
    'label': key(str, 'label'),
    'marker': key(str, 'marker'),
}

dict_key_legend = {
    'ncol': key(int, 'ncol'),
    'loc': key(int, 'loc'),
    'facecolor': key(str, 'facecolor'),
    'framealpha': key(float, 'framealpha'),
    'fontsize': key(int, 'fontsize'),
}

dict_key_ax = {
    'ylim0': key(float, 'ylim0'),
    'ylim1': key(float, 'ylim1'),
    'ylabel': key(float_or_None, 'ylabel'),
    'size_legend': key(int, 'size_legend'),
}

def read_legend(path_legend, dict_legend_item):
    if path_legend == 'Figures/legend_txt/None':
        dict_legend_item['args'] = None
        dict_legend_item['value'] = None
    else:
        f_legend = open(path_legend + '.txt', "r")
        current_counter = 0
        for f_legend_line in f_legend:
            if current_counter == 0:
                dict_legend_item['args'] = f_legend_line[:-1][1:-1].split("' '")
                dict_legend_item['value'] = []
                current_counter = 1
            elif current_counter == 1:
                dict_legend_item['value'].append(f_legend_line[:-1][1:-1].split("' '"))

def read_conf_args(dict_parameters, args):
    dict_parameters['range_for_name'] = []
    dict_parameters['range_for_values'] = []

    dict_for = dict(zip(args.for_args_name[1:-1].split("' '"), args.for_args_value[1:-1].split("' '")))
    for current_key in dict_for.keys():
        if current_key in dict_key.keys():
            current_dict_key = dict_key
        elif current_key in dict_key_others.keys():
            current_dict_key = dict_key_others
        value_list = dict_for[current_key].split()
        if len(value_list) > 1:
            dict_parameters['range_for_name'].append(current_key)
            dict_for[current_key] = None
            dict_parameters['range_for_values'].append(
                [current_dict_key[current_key].dtype(value) for value in value_list])
        else:
            dict_for[current_key] = current_dict_key[current_key].dtype(dict_for[current_key])

    dict_parameters['params'].update(dict_for)

    list_dict_section = []
    current_counter = -1
    inside_plot = False
    inside_legend = False
    f_plot = open(args.args_plot, "r")
    for f_plot_line in f_plot:
        if f_plot_line[:6] == '$$plot':
            inside_plot = True
            list_f_plot_line = f_plot_line[:-1].split('$$')
            len_for_which = len(list_f_plot_line[5].split(" '"))
            if len_for_which == 3:
                dict_figure = dict(for_which=list_f_plot_line[5].split(" '")[1][:-1],
                                   list_for_which=list_f_plot_line[5].split(" '")[2].strip()[:-1],
                                   fig_size=list(map(float, list_f_plot_line[3].split(" '")[1][:-2].split(","))),
                                   margin_size=list(map(float, list_f_plot_line[4].split(" '")[1][:-2].split(","))))
            else:
                dict_figure = dict(for_which=[list_f_plot_line[5].split(" '")[k][:-1]
                                              for k in np.arange(1, len_for_which, 2)],
                                   list_for_which=[list_f_plot_line[5].split(" '")[k].strip()[:-1]
                                                   for k in np.arange(2, len_for_which, 2)],
                                   fig_size=list(map(float, list_f_plot_line[3].split(" '")[1][:-2].split(","))),
                                   margin_size=list(map(float, list_f_plot_line[4].split(" '")[1][:-2].split(","))))
            list_f_options = list_f_plot_line[2].strip().split(' ')
            dict_figure['type_figure'] = list_f_options[0]
            dict_figure['type_figure_options'] = dict()
            for current_options in list_f_options[1:]:
                dict_figure['type_figure_options'][current_options] = True
        elif f_plot_line[:-1] == '$$endplot':
            inside_plot = False
            current_counter = -1
        elif f_plot_line[:8] == '$$legend':
            inside_legend = True
            list_f_plot_line = f_plot_line[:-1].split('$$')
            dict_legend = dict(title=list(map(lambda x: x.strip("'"), list_f_plot_line[3][6:-1].split(" '"))),
                               type_legend=(list_f_plot_line[2].strip()),
                               item=dict())
        elif f_plot_line[:-1] == '$$endlegend':
            inside_legend = False
            current_counter = -1
        elif inside_plot:
            if f_plot_line[:3] == '$$$':
                current_counter = 0
                current_dict_section = {}
            elif current_counter == 0:
                current_dict_section['args'] = f_plot_line[:-1][1:-1].split("' '")
                current_dict_section['value'] = []
                current_counter = 1
            elif len(f_plot_line[:-1]) == 0:
                list_dict_section.append(current_dict_section)
                current_counter = -1
            elif current_counter == 1:
                current_dict_section['value'].append(f_plot_line[:-1][1:-1].split("' '"))
        elif inside_legend:
            if current_counter == -1:
                current_counter = 0
                dict_legend['args'] = f_plot_line[:-1][1:-1].split("' '")
            elif current_counter == 0:
                dict_legend['value'] = f_plot_line[:-1][1:-1].split("' '")
                current_counter = 1
                read_legend('Figures/legend_txt/' + dict_legend['type_legend'], dict_legend['item'])
    list_dict_section.append(current_dict_section)
    dict_figure['list_dict_section'] = list_dict_section
    f_plot.close()

    for current_dict_section in list_dict_section:
        current_dict_section['list_dict'] = []
        for current_item in current_dict_section['value']:
            current_dict_item = dict(params={}, legend_handles={}, others={})
            for current_key, current_value in zip(current_dict_section['args'], current_item):
                if current_key in dict_key.keys():
                    current_dict_item['params'][current_key] = dict_key[current_key].dtype(current_value)
                elif current_key in dict_key_legend_handles.keys():
                    current_dict_item['legend_handles'][current_key] = \
                        dict_key_legend_handles[current_key].dtype(current_value)
                elif current_key in dict_key_others.keys():
                    current_dict_item['others'][current_key] = \
                        dict_key_others[current_key].dtype(current_value)
            current_dict_section['list_dict'].append(current_dict_item)
    if isinstance(dict_figure['for_which'], str):
        if dict_figure['for_which'] in dict_key.keys():
            dict_figure['list_for_which'] = [dict_key[dict_figure['for_which']].dtype(value) for value in
                                             dict_figure['list_for_which'].split()]
        elif dict_figure['for_which'] in dict_key_others.keys():
            dict_figure['list_for_which'] = [dict_key_others[dict_figure['for_which']].dtype(value) for value in
                                             dict_figure['list_for_which'].split()]
    else:
        new_list_for_which = []
        for current_for_which, current_for_which_values in zip(dict_figure['for_which'], dict_figure['list_for_which']):
            if current_for_which in dict_key.keys():
                new_list_for_which.append([dict_key[current_for_which].dtype(value) for value in
                                           current_for_which_values.split()])
            elif current_for_which in dict_key_others.keys():
                new_list_for_which.append([dict_key_others[current_for_which].dtype(value) for value in
                                           current_for_which_values.split()])
        dict_figure['list_for_which'] = new_list_for_which

    dict_legend['options_legend'] = {}
    dict_legend['options_ax'] = {}
    for current_key, current_value in zip(dict_legend['args'], dict_legend['value']):
        if current_key in dict_key_legend:
            dict_legend['options_legend'][current_key] = dict_key_legend[current_key].dtype(current_value)
        elif current_key in dict_key_ax:
            dict_legend['options_ax'][current_key] = dict_key_ax[current_key].dtype(current_value)
    if dict_legend['item']['args'] is not None:
        dict_legend['item_list'] = [dict(zip(dict_legend['item']['args'], item))
                                    for item in dict_legend['item']['value']]
        for current_item in dict_legend['item_list']:
            current_item['label'] = current_item['label'].format(latex)
        dict_legend['handles_list'] = [legend_handles(current_item).output()
                                       for current_item in dict_legend['item_list']]
    else:
        dict_legend['item'] = None
    return dict_figure, dict_legend

def figure_bar(origin_dir, dict_parameters, dict_figure, dict_legend, dict_parameters_range, cfg):
    fig_y, fig_x = dict_figure['fig_size']
    top, bottom, left, right = dict_figure['margin_size']
    dict_margins = dict(top=1. - top / fig_x, bottom=bottom / fig_x, left=left / fig_y, right=1. - right / fig_y)
    hspace = 1.
    wspace = 1.
    width_average = (fig_y - left - right - wspace) / 2.
    if dict_legend['item'] is None:
        dict_margins['width_ratios'] = [2. * width_average, 5. * width_average]
        dict_margins['wspace'] = wspace / width_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=2, nrows=1, **dict_margins)

        ax_title = fig.add_subplot(spec[:])
        ax_example = fig.add_subplot(spec[0])
        ax_bar = fig.add_subplot(spec[1])

        set_blank_axis(ax_title)
    else:
        height_average = (fig_x - top - bottom - hspace) / 2.
        dict_margins['width_ratios'] = [2. * width_average, 5. * width_average]
        dict_margins['height_ratios'] = [4. * height_average, dict_legend['options_ax']['size_legend'] * height_average]
        dict_margins['hspace'] = hspace / height_average
        dict_margins['wspace'] = wspace / width_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=2, nrows=2, **dict_margins)

        ax_legend = fig.add_subplot(spec[1, :])
        ax_title = fig.add_subplot(spec[0, :])
        ax_example = fig.add_subplot(spec[0, 0])
        ax_bar = fig.add_subplot(spec[0, 1])

        set_blank_axis(ax_legend)
        set_blank_axis(ax_title)

    handles_maxaccuracy = [Line2D([0], [0], color=cfg.colors['black'], lw=0.5, linestyle='-',
                                  label='MaxAccuracy'),
                           Line2D([0], [0], color=cfg.colors['black'], lw=0.5, linestyle='--',
                                  label='Accuracy')]

    len_which_list = len(dict_figure['list_for_which'])
    len_item_list = sum([len(current_dict_section['list_dict']) for current_dict_section
                        in dict_figure['list_dict_section']])
    value_analysis = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_control = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_perceptron = dict(list_score_set=np.nan * np.ones((len_item_list, len_which_list)))

    title = dict_key['pattern'].label_value(dict_parameters['params']['pattern']) + u':\n'
    name_plot = dict_parameters['params']['pattern'] + '_'
    for current_key in dict_legend['title']:
        if not current_key == '':
            current_key_label = dict_key[current_key]
            if dict_parameters['params'][current_key] is None:
                current_key_value = dict_parameters_range[current_key]
            else:
                current_key_value = dict_parameters['params'][current_key]
            title += current_key_label.title + \
                current_key_label.label_value(current_key_value) + ', '
            name_plot += str(current_key_value) + '_'

    title = title[:-2]
    name_plot = name_plot + dict_figure['for_which']

    which_measure = 'accuracy'

    for count_for_which, current_for_which in enumerate(dict_figure['list_for_which']):
        count_list_dict_section = 0
        for current_dict_section in dict_figure['list_dict_section']:
            for _, current_dict in enumerate(current_dict_section['list_dict']):
                current_dict_plot = dict(dict_parameters['params'])
                current_dict_plot.update(dict(current_dict['params']))
                current_dict_plot[dict_figure['for_which']] = current_for_which
                current_dict_plot.update(dict_parameters_range)

                name_dir = ''.join([str(current_dict_plot[current_key]) + '_' for current_key in dict_key.keys()
                                    if (current_key in current_dict_plot.keys())])[:-1]
                name_dir_control = ''.join([('0.0_' if current_key == 'homeostasy' else str(
                                                current_dict_plot[current_key]) + '_')
                                            for current_key in dict_key.keys()
                                            if (current_key in current_dict_plot.keys())])[:-1]

                name_project = current_dict['others']['dir']
                name_subproject = current_dict['others']['name']

                simu_dir = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir)
                simu_dir_control = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir_control)

                plot_data = np.load('{}/Plot_data_{}.npy'.format(simu_dir, current_dict_plot['n_pattern']),
                                    allow_pickle=True).item()
                plot_data_control = np.load('{}/Plot_data_{}.npy'.format(simu_dir_control,
                                                                         current_dict_plot['n_pattern']),
                                            allow_pickle=True).item()
                if count_for_which == 0:
                    plot_results(plot_data, plot_data_control, ax_example,
                                 which_measure=which_measure,
                                 params_others=current_dict['others'],
                                 params_legend_handles=current_dict['legend_handles'],
                                 compute_max=True,
                                 value_analysis=value_analysis, value_analysis_control=value_analysis_control,
                                 current_k=count_list_dict_section, current_l=count_for_which
                                 )
                    current_dict_accuracy = dict(current_dict['legend_handles'])
                    current_dict_accuracy['linestyle'] = '--'
                    plot_results(plot_data, plot_data_control, ax_example,
                                 which_measure=which_measure,
                                 params_others=current_dict['others'],
                                 params_legend_handles=current_dict_accuracy,
                                 compute_max=False,
                                 value_analysis=None, value_analysis_control=None,
                                 current_k=count_list_dict_section, current_l=count_for_which
                                 )
                else:
                    plot_results(plot_data, plot_data_control, None,
                                 params_others=current_dict['others'],
                                 which_measure=which_measure,
                                 params_legend_handles=current_dict['legend_handles'],
                                 compute_max=True,
                                 value_analysis=value_analysis, value_analysis_control=value_analysis_control,
                                 current_k=count_list_dict_section, current_l=count_for_which
                                 )
                value_analysis_perceptron['list_score_set'][count_list_dict_section, count_for_which] = \
                    np.nanmean(plot_data['score_set'])
                count_list_dict_section += 1

    step_P = 1. / float(len_item_list + 1.)
    for count_for_which, current_for_which in enumerate(dict_figure['list_for_which']):
        ax_bar.plot([count_for_which - (len_item_list + 0.5) * step_P/2.,
                     count_for_which + (len_item_list + 0.5) * step_P/2.],
                    [np.nanmean(value_analysis_perceptron['list_score_set'][:, count_for_which]),
                     np.nanmean(value_analysis_perceptron['list_score_set'][:, count_for_which])],
                    color=colors['black'], linestyle='dashdot')
    count_list_dict_section = 0
    for current_dict_section in dict_figure['list_dict_section']:
        for _, current_dict in enumerate(current_dict_section['list_dict']):
            ax_bar.bar(np.arange(len_which_list) + (1. + 2. * count_list_dict_section - len_item_list) * step_P/2.,
                       value_analysis['value_list'][count_list_dict_section],
                       color=colors[current_dict['legend_handles']['color']], width=step_P,
                       label=current_dict['legend_handles']['label'])
            if not (current_dict['others']['stats'] == 'None'):
                for count_for_which, _ in enumerate(dict_figure['list_for_which']):
                    text_stats = compute_stat(current_dict['others']['stats'],
                                              value_analysis=value_analysis,
                                              current_k=count_list_dict_section, current_l=count_for_which)
                    ax_bar.text(count_for_which +
                                (1. + 2. * count_list_dict_section - len_item_list) * step_P/2.,
                                (dict_legend['options_ax']['ylim1'] - 1.)/2. + 1.,
                                text_stats, horizontalalignment='center', verticalalignment='center',
                                color=colors[current_dict['others']['color_stats']], fontsize=7).set_clip_on(True)
                    ax_bar.plot([count_for_which + (2. * count_list_dict_section - len_item_list) * step_P/2.,
                                count_for_which + (2. + 2. * count_list_dict_section - len_item_list) * step_P/2.],
                                [value_analysis_control['value_list'][count_list_dict_section, count_for_which],
                                value_analysis_control['value_list'][count_list_dict_section, count_for_which]],
                                color=colors['black'], linestyle='-')
                    ax_bar.text(count_for_which + (1. + 2. * count_list_dict_section - len_item_list) * step_P/2., 0.05,
                                value_analysis_control['stats'][count_list_dict_section, count_for_which],
                                horizontalalignment='center', verticalalignment='center',
                                color=colors['black'], fontsize=7).set_clip_on(True)
                ax_bar.errorbar(np.arange(len_which_list) +
                                (1. + 2. * count_list_dict_section - len_item_list) * step_P / 2.,
                                value_analysis['value_list'][count_list_dict_section],
                                yerr=value_analysis['value_list_std'][count_list_dict_section] / 2.,
                                color=colors['black'], linestyle='',
                                fmt=current_dict['legend_handles']['marker'],
                                markersize=5., markeredgewidth=1., capsize=5.)
            else:
                current_ax_bar.plot(np.arange(len_which_list) +
                                    (1. + 2. * count_list_dict_section - len_item_list) * step_P / 2.,
                                    current_value_analysis['value_list'][count_list_dict_section],
                                    color=colors['black'], linestyle='',
                                    marker=current_dict['legend_handles']['marker'],
                                    markersize=5., markeredgewidth=1.)
            count_list_dict_section += 1

    if dict_legend['item'] is not None:
        legend = ax_legend.legend(handles=dict_legend['handles_list'], frameon=False, **dict_legend['options_legend'])
        legend.get_frame().set_linewidth(0.5)
    ax_bar.set_xlim(0. - len_item_list * step_P/2., len_which_list - 1 + len_item_list * step_P/2.)
    ax_bar.set_xticks(np.arange(len_which_list))
    ax_bar.set_xticklabels([str(u) for u in dict_figure['list_for_which']])
    ax_bar.set_xlabel('{}'.format(dict_key_others['n_pattern'].label[0].upper() +
                                  dict_key_others['n_pattern'].label[1:], labelpad=5))
    ax_bar.set_yticks([0., 0.5, 1.])
    ax_bar.set_ylim(dict_legend['options_ax']['ylim0'], dict_legend['options_ax']['ylim1'])

    legend = ax_example.legend(handles=handles_maxaccuracy, loc=8, ncol=1, fontsize=6)
    legend.get_frame().set_linewidth(0.5)

    ax_example.set_xlim(0., dict_parameters['params']['num_training'] * dict_parameters['params']['P'])
    ax_example.set_xticks([0, 250, 500])
    ax_example.set_xlabel('Iterations, {0}N_p={1}{0}'.format(latex, dict_figure['list_for_which'][0]),
                          labelpad=5)
    ax_example.set_ylabel(dict_legend['options_ax']['ylabel'])
    ax_example.set_yticks([0., 0.5, 1.])
    ax_example.set_ylim(dict_legend['options_ax']['ylim0'], dict_legend['options_ax']['ylim1'])

    for ax_ in [ax_bar, ax_example]:
        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)

    ax_title.set_title(title, pad=10)
    return fig, name_plot

def figure_bar_subset(origin_dir, dict_parameters, dict_figure, dict_legend, dict_parameters_range, cfg):
    fig_y, fig_x = dict_figure['fig_size']
    top, bottom, left, right = dict_figure['margin_size']
    dict_margins = dict(top=1. - top / fig_x, bottom=bottom / fig_x, left=left / fig_y, right=1. - right / fig_y)
    hspace = 1.
    wspace = 1.
    width_average = (fig_y - left - right - wspace) / 2.
    height_average = (fig_x - top - bottom - hspace) / 3.
    dict_margins['width_ratios'] = [2. * width_average, 5. * width_average]
    dict_margins['height_ratios'] = [4. * height_average, 4. * height_average,
                                     dict_legend['options_ax']['size_legend'] * height_average]
    dict_margins['hspace'] = hspace / height_average
    dict_margins['wspace'] = wspace / width_average

    fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
    spec = fig.add_gridspec(ncols=2, nrows=3, **dict_margins)

    ax_legend = fig.add_subplot(spec[2, :])
    ax_title = fig.add_subplot(spec[0, :])
    ax_example = fig.add_subplot(spec[0, 0])
    ax_bar = fig.add_subplot(spec[0, 1])
    ax_example_subset = fig.add_subplot(spec[1, 0])
    ax_bar_subset = fig.add_subplot(spec[1, 1])

    set_blank_axis(ax_legend)
    set_blank_axis(ax_title)

    len_which_list = len(dict_figure['list_for_which'])
    len_item_list = sum([len(current_dict_section['list_dict']) for current_dict_section
                        in dict_figure['list_dict_section']])

    handles_maxaccuracy = [Line2D([0], [0], color=cfg.colors['black'], lw=0.5, linestyle='-',
                                  label='MaxAccuracy'),
                           Line2D([0], [0], color=cfg.colors['black'], lw=0.5, linestyle='--',
                                  label='Accuracy')]

    value_analysis = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_control = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_perceptron = dict(list_score_set=np.nan * np.ones((len_item_list, len_which_list)))

    value_analysis_subset = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_control_subset = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_perceptron_subset = dict(list_score_set=np.nan * np.ones((len_item_list, len_which_list)))

    title = dict_key['pattern'].label_value(dict_parameters['params']['pattern']) + u':\n'
    name_plot = dict_parameters['params']['pattern'] + '_'
    for current_key in dict_legend['title']:
        if not current_key == '':
            current_key_label = dict_key[current_key]
            if dict_parameters['params'][current_key] is None:
                current_key_value = dict_parameters_range[current_key]
            else:
                current_key_value = dict_parameters['params'][current_key]
            title += current_key_label.title + \
                current_key_label.label_value(current_key_value) + ', '
            name_plot += str(current_key_value) + '_'

    title = title[:-2]
    name_plot = name_plot + dict_figure['for_which']

    for current_ax_bar, current_ax_example, which_measure, ylabel_value, current_value_analysis, \
        current_value_analysis_control,\
        current_value_analysis_perceptron in zip([ax_bar, ax_bar_subset], [ax_example, ax_example_subset],
                                                 ['', '_subset'], ['MaxAccuracy', 'MaxAccuracy\non subsets'],
                                                 [value_analysis, value_analysis_subset],
                                                 [value_analysis_control, value_analysis_control_subset],
                                                 [value_analysis_perceptron, value_analysis_perceptron_subset]):

        for count_for_which, current_for_which in enumerate(dict_figure['list_for_which']):
            count_list_dict_section = 0
            for current_dict_section in dict_figure['list_dict_section']:
                for _, current_dict in enumerate(current_dict_section['list_dict']):
                    current_dict_plot = dict(dict_parameters['params'])
                    current_dict_plot.update(dict(current_dict['params']))
                    current_dict_plot[dict_figure['for_which']] = current_for_which
                    current_dict_plot.update(dict_parameters_range)

                    name_dir = ''.join([str(current_dict_plot[current_key]) + '_' for current_key in dict_key.keys()
                                        if (current_key in current_dict_plot.keys())])[:-1]
                    name_dir_control = ''.join([('0.0_' if current_key == 'homeostasy' else str(
                                                    current_dict_plot[current_key]) + '_')
                                                for current_key in dict_key.keys()
                                                if (current_key in current_dict_plot.keys())])[:-1]

                    name_project = current_dict['others']['dir']
                    name_subproject = current_dict['others']['name']

                    simu_dir = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir)
                    simu_dir_control = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir_control)

                    plot_data = np.load('{}/Plot_data_{}.npy'.format(simu_dir, current_dict_plot['n_pattern']),
                                        allow_pickle=True).item()
                    plot_data_control = np.load('{}/Plot_data_{}.npy'.format(simu_dir_control,
                                                                             current_dict_plot['n_pattern']),
                                                allow_pickle=True).item()
                    if count_for_which == 0:
                        plot_results(plot_data, plot_data_control, current_ax_example,
                                     which_measure='accuracy' + which_measure,
                                     params_others=current_dict['others'],
                                     params_legend_handles=current_dict['legend_handles'],
                                     compute_max=True,
                                     value_analysis=current_value_analysis,
                                     value_analysis_control=current_value_analysis_control,
                                     current_k=count_list_dict_section, current_l=count_for_which
                                     )
                        current_dict_accuracy = dict(current_dict['legend_handles'])
                        current_dict_accuracy['linestyle'] = '--'
                        plot_results(plot_data, plot_data_control, current_ax_example,
                                     which_measure='accuracy' + which_measure,
                                     params_others=current_dict['others'],
                                     params_legend_handles=current_dict_accuracy,
                                     compute_max=False,
                                     value_analysis=None, value_analysis_control=None,
                                     current_k=count_list_dict_section, current_l=count_for_which
                                     )
                    else:
                        plot_results(plot_data, plot_data_control, None,
                                     params_others=current_dict['others'],
                                     which_measure='accuracy' + which_measure,
                                     params_legend_handles=current_dict['legend_handles'],
                                     compute_max=True,
                                     value_analysis=current_value_analysis,
                                     value_analysis_control=current_value_analysis_control,
                                     current_k=count_list_dict_section, current_l=count_for_which
                                     )
                    current_value_analysis_perceptron['list_score_set'][count_list_dict_section, count_for_which] = \
                        np.nanmean(plot_data['score_set' + which_measure])
                    count_list_dict_section += 1

        step_P = 1. / float(len_item_list + 1.)
        for count_for_which, current_for_which in enumerate(dict_figure['list_for_which']):
            current_ax_bar.plot([count_for_which - (len_item_list + 0.5) * step_P / 2.,
                                count_for_which + (len_item_list + 0.5) * step_P / 2.],
                                [np.nanmean(current_value_analysis_perceptron['list_score_set'][:, count_for_which]),
                                 np.nanmean(current_value_analysis_perceptron['list_score_set'][:, count_for_which])],
                                color=colors['black'], linestyle='dashdot')
        count_list_dict_section = 0
        for current_dict_section in dict_figure['list_dict_section']:
            for _, current_dict in enumerate(current_dict_section['list_dict']):
                current_ax_bar.bar(
                    np.arange(len_which_list) + (1. + 2. * count_list_dict_section - len_item_list) * step_P/2.,
                    current_value_analysis['value_list'][count_list_dict_section],
                    color=colors[current_dict['legend_handles']['color']], width=step_P,
                    label=current_dict['legend_handles']['label'])
                if not (current_dict['others']['stats'] == 'None'):
                    for count_for_which, _ in enumerate(dict_figure['list_for_which']):
                        text_stats = compute_stat(current_dict['others']['stats'],
                                                  value_analysis=current_value_analysis,
                                                  current_k=count_list_dict_section, current_l=count_for_which)
                        current_ax_bar.text(count_for_which +
                                            (1. + 2. * count_list_dict_section - len_item_list) * step_P/2.,
                                            (dict_legend['options_ax']['ylim1'] - 1.)/2. + 1.,
                                            text_stats, horizontalalignment='center', verticalalignment='center',
                                            color=colors[current_dict['others']['color_stats']],
                                            fontsize=7).set_clip_on(True)
                        if which_measure == '':
                            current_ax_bar.plot(
                                [count_for_which + (2. * count_list_dict_section - len_item_list) * step_P/2.,
                                 count_for_which + (2. + 2. * count_list_dict_section - len_item_list) * step_P/2.],
                                [current_value_analysis_control['value_list'][count_list_dict_section, count_for_which],
                                 current_value_analysis_control['value_list'][count_list_dict_section,
                                                                              count_for_which]],
                                color=colors['black'], linestyle='-')
                            current_ax_bar.text(
                                count_for_which + (1. + 2. * count_list_dict_section - len_item_list) * step_P/2., 0.05,
                                current_value_analysis_control['stats'][count_list_dict_section, count_for_which],
                                horizontalalignment='center', verticalalignment='center',
                                color=colors['black'], fontsize=7).set_clip_on(True)
                    current_ax_bar.errorbar(np.arange(len_which_list) +
                                            (1. + 2. * count_list_dict_section - len_item_list) * step_P / 2.,
                                            current_value_analysis['value_list'][count_list_dict_section],
                                            yerr=current_value_analysis['value_list_std'][count_list_dict_section] / 2.,
                                            color=colors['black'], linestyle='',
                                            fmt=current_dict['legend_handles']['marker'],
                                            markersize=5., markeredgewidth=1., capsize=5.)
                else:
                    current_ax_bar.plot(np.arange(len_which_list) +
                                        (1. + 2. * count_list_dict_section - len_item_list) * step_P / 2.,
                                        current_value_analysis['value_list'][count_list_dict_section],
                                        color=colors['black'], linestyle='',
                                        marker=current_dict['legend_handles']['marker'],
                                        markersize=5., markeredgewidth=1.)
                count_list_dict_section += 1
        legend = ax_legend.legend(handles=dict_legend['handles_list'], frameon=False, **dict_legend['options_legend'])
        legend.get_frame().set_linewidth(0.5)
        current_ax_bar.set_xlim(0. - len_item_list * step_P / 2., len_which_list - 1 + len_item_list * step_P/2.)
        current_ax_bar.set_xticks(np.arange(len_which_list))
        current_ax_bar.set_xticklabels([str(u) for u in dict_figure['list_for_which']])
        current_ax_bar.set_yticks([0., 0.5, 1.])
        current_ax_bar.set_ylim(dict_legend['options_ax']['ylim0'], dict_legend['options_ax']['ylim1'])

        current_ax_example.set_xlim(0., dict_parameters['params']['num_training'] * dict_parameters['params']['P'])
        current_ax_example.set_xticks([0, 250, 500])
        current_ax_example.set_ylabel(ylabel_value)
        current_ax_example.set_yticks([0., 0.5, 1.])
        current_ax_example.set_ylim(dict_legend['options_ax']['ylim0'], dict_legend['options_ax']['ylim1'])
        if which_measure is '_subset':
            legend = current_ax_example.legend(handles=handles_maxaccuracy, loc=8, ncol=1, fontsize=6)
            legend.get_frame().set_linewidth(0.5)
    ax_bar_subset.set_xlabel('{}'.format(dict_key_others['n_pattern'].label[0].upper() +
                                         dict_key_others['n_pattern'].label[1:], labelpad=5))
    ax_example_subset.set_xlabel('Iterations, {0}N_p={1}{0}'.format(latex, dict_figure['list_for_which'][0]),
                                 labelpad=5)

    for ax_ in [ax_bar, ax_example, ax_bar_subset, ax_example_subset]:
        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)

    ax_title.set_title(title, pad=10)
    return fig, name_plot

def figure_bar_comp(origin_dir, dict_parameters, dict_figure, dict_legend, dict_parameters_range, cfg):
    if not isinstance(dict_figure['for_which'], list):
        dict_figure['for_which'] = [dict_figure['for_which']]
        dict_figure['list_for_which'] = [dict_figure['list_for_which']]

    fig_y, fig_x = dict_figure['fig_size']
    top, bottom, left, right = dict_figure['margin_size']
    dict_margins = dict(top=1. - top / fig_x, bottom=bottom / fig_x, left=left / fig_y, right=1. - right / fig_y)
    hspace = 1.
    if dict_legend['item'] is None:
        height_average = (fig_x - top - bottom - hspace)
        dict_margins['height_ratios'] = [4. * height_average]
        dict_margins['hspace'] = hspace / height_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=1, nrows=1, **dict_margins)

        ax_bar = fig.add_subplot(spec[0])

    else:
        height_average = (fig_x - top - bottom - hspace) / 2.
        dict_margins['height_ratios'] = [4. * height_average,
                                         dict_legend['options_ax']['size_legend'] * height_average]
        dict_margins['hspace'] = hspace / height_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=1, nrows=2, **dict_margins)

        ax_legend = fig.add_subplot(spec[1])
        ax_bar = fig.add_subplot(spec[0])

        set_blank_axis(ax_legend)

    len_which_list = len(dict_figure['list_for_which'][0])
    len_item_list = sum([len(current_dict_section['list_dict']) for current_dict_section
                        in dict_figure['list_dict_section']])
    value_analysis = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_control = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_perceptron = dict(list_score_set=np.nan * np.ones((len_item_list, len_which_list)))

    title = dict_key['pattern'].label_value(dict_parameters['params']['pattern']) + u':\n'
    name_plot = dict_parameters['params']['pattern'] + '_'
    for current_key in dict_legend['title']:
        if not current_key == '':
            if current_key in dict_key.keys():
                current_key_label = dict_key[current_key]
            elif current_key in dict_key_others.keys():
                current_key_label = dict_key_others[current_key]
            if dict_parameters['params'][current_key] is None:
                current_key_value = dict_parameters_range[current_key]
            else:
                current_key_value = dict_parameters['params'][current_key]
            title += current_key_label.title + \
                current_key_label.label_value(current_key_value) + ', '
            name_plot += str(current_key_value) + '_'
    title = title[:-2]
    name_plot = name_plot + dict_figure['for_which'][0]

    which_measure = 'accuracy'
    for count_for_which, current_for_which in enumerate(zip(*dict_figure['list_for_which'])):
        count_list_dict_section = 0
        for current_dict_section in dict_figure['list_dict_section']:
            for _, current_dict in enumerate(current_dict_section['list_dict']):
                current_dict_plot = dict(dict_parameters['params'])
                current_dict_plot.update(dict(current_dict['params']))
                for which_for_which, which_for_which_value in zip(dict_figure['for_which'], current_for_which):
                    if which_for_which in dict_key.keys():
                        current_dict_plot[which_for_which] = which_for_which_value
                    elif which_for_which == 'name':
                        current_dict['others'][which_for_which] = which_for_which_value
                    elif which_for_which == 'n_pattern':
                        current_dict_plot[which_for_which] = which_for_which_value

                current_dict_plot.update(dict_parameters_range)

                name_dir = ''.join([str(current_dict_plot[current_key]) + '_' for current_key in dict_key.keys()
                                    if (current_key in current_dict_plot.keys())])[:-1]
                name_dir_control = ''.join([('0.0_' if current_key == 'homeostasy' else str(
                                                current_dict_plot[current_key]) + '_')
                                            for current_key in dict_key.keys()
                                            if (current_key in current_dict_plot.keys())])[:-1]

                name_project = current_dict['others']['dir']
                name_subproject = current_dict['others']['name']

                simu_dir = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir)
                simu_dir_control = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir_control)
                plot_data = np.load('{}/Plot_data_{}.npy'.format(simu_dir, current_dict_plot['n_pattern']),
                                    allow_pickle=True).item()
                plot_data_control = np.load('{}/Plot_data_{}.npy'.format(simu_dir_control,
                                                                         current_dict_plot['n_pattern']),
                                            allow_pickle=True).item()
                plot_results(plot_data, plot_data_control, None,
                             params_others=current_dict['others'],
                             which_measure=which_measure,
                             params_legend_handles=current_dict['legend_handles'],
                             compute_max=True,
                             value_analysis=value_analysis, value_analysis_control=value_analysis_control,
                             current_k=count_list_dict_section, current_l=count_for_which
                             )
                value_analysis_perceptron['list_score_set'][count_list_dict_section, count_for_which] = \
                    np.nanmean(plot_data['score_set'])
                count_list_dict_section += 1

    step_P = 1. / float(len_item_list + 1.)
    for count_for_which, current_for_which in enumerate(zip(*dict_figure['list_for_which'])):
        ax_bar.plot([count_for_which - (len_item_list + 0.5) * step_P / 2.,
                     count_for_which + (len_item_list + 0.5) * step_P / 2.],
                    [np.nanmean(value_analysis_perceptron['list_score_set'][:, count_for_which]),
                     np.nanmean(value_analysis_perceptron['list_score_set'][:, count_for_which])],
                    color=colors['black'], linestyle='dashdot')
    count_list_dict_section = 0
    for current_dict_section in dict_figure['list_dict_section']:
        for _, current_dict in enumerate(current_dict_section['list_dict']):
            ax_bar.bar(np.arange(len_which_list) + (1. + 2. * count_list_dict_section - len_item_list) * step_P/2.,
                       value_analysis['value_list'][count_list_dict_section],
                       color=colors[current_dict['legend_handles']['color']], width=step_P,
                       label=current_dict['legend_handles']['label'])
            if not (current_dict['others']['stats'] == 'None'):
                for count_for_which, _ in enumerate(zip(*dict_figure['list_for_which'])):
                    text_stats = compute_stat(current_dict['others']['stats'],
                                              value_analysis=value_analysis,
                                              current_k=count_list_dict_section, current_l=count_for_which)
                    ax_bar.text(count_for_which +
                                (1. + 2. * count_list_dict_section - len_item_list) * step_P/2.,
                                (dict_legend['options_ax']['ylim1'] - 1.)/2. + 1.,
                                text_stats, horizontalalignment='center', verticalalignment='center',
                                color=colors[current_dict['others']['color_stats']], fontsize=7).set_clip_on(True)
                    ax_bar.plot([count_for_which + (2. * count_list_dict_section - len_item_list) * step_P/2.,
                                count_for_which + (2. + 2. * count_list_dict_section - len_item_list) * step_P/2.],
                                [value_analysis_control['value_list'][count_list_dict_section, count_for_which],
                                value_analysis_control['value_list'][count_list_dict_section, count_for_which]],
                                color=colors['black'], linestyle='-')
                    ax_bar.text(count_for_which + (1. + 2. * count_list_dict_section - len_item_list) * step_P/2., 0.05,
                                value_analysis_control['stats'][count_list_dict_section, count_for_which],
                                horizontalalignment='center', verticalalignment='center',
                                color=colors['black'], fontsize=7).set_clip_on(True)
                ax_bar.errorbar(np.arange(len_which_list) +
                                (1. + 2. * count_list_dict_section - len_item_list) * step_P / 2.,
                                value_analysis['value_list'][count_list_dict_section],
                                yerr=value_analysis['value_list_std'][count_list_dict_section] / 2.,
                                color=colors['black'], linestyle='',
                                fmt=current_dict['legend_handles']['marker'],
                                markersize=5., markeredgewidth=1., capsize=5.)
            else:
                ax_bar.plot(np.arange(len_which_list) +
                            (1. + 2. * count_list_dict_section - len_item_list) * step_P / 2.,
                            value_analysis['value_list'][count_list_dict_section],
                            color=colors['black'], linestyle='',
                            marker=current_dict['legend_handles']['marker'],
                            markersize=5., markeredgewidth=1.)
            count_list_dict_section += 1
    if dict_legend['item'] is not None:
        legend = ax_legend.legend(handles=dict_legend['handles_list'], frameon=False, **dict_legend['options_legend'])
        legend.get_frame().set_linewidth(0.5)
    ax_bar.set_xlim(0. - len_item_list * step_P/2., len_which_list - 1 + len_item_list * step_P/2.)
    ax_bar.set_xticks(np.arange(len_which_list))
    if dict_figure['for_which'][0] in dict_key.keys():
        current_key_label = dict_key[dict_figure['for_which'][0]]
    else:
        current_key_label = dict_key_others[dict_figure['for_which'][0]]
    ax_bar.set_xticklabels([current_key_label.label_value(current_key_value)
                            for current_key_value in dict_figure['list_for_which'][0]])
    if dict_figure['for_which'][0] in dict_key.keys():
        ax_bar.set_xlabel('{}'.format(dict_key[dict_figure['for_which'][0]].label[0].upper() +
                                      dict_key[dict_figure['for_which'][0]].label[1:], labelpad=5))
    elif dict_figure['for_which'][0] in dict_key_others.keys():
        ax_bar.set_xlabel('{}'.format(dict_key_others[dict_figure['for_which'][0]].label[0].upper() +
                                      dict_key_others[dict_figure['for_which'][0]].label[1:], labelpad=5))
    ax_bar.set_yticks([0., 0.5, 1.])
    ax_bar.set_ylim(dict_legend['options_ax']['ylim0'], dict_legend['options_ax']['ylim1'])
    ax_bar.set_ylabel(dict_legend['options_ax']['ylabel'])
    for ax_ in [ax_bar]:
        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)

    ax_bar.set_title(title, pad=10)
    return fig, name_plot

def figure_line_comp(origin_dir, dict_parameters, dict_figure, dict_legend, dict_parameters_range, cfg):
    if not isinstance(dict_figure['for_which'], list):
        dict_figure['for_which'] = [dict_figure['for_which']]
        dict_figure['list_for_which'] = [dict_figure['list_for_which']]

    fig_y, fig_x = dict_figure['fig_size']
    top, bottom, left, right = dict_figure['margin_size']
    dict_margins = dict(top=1. - top / fig_x, bottom=bottom / fig_x, left=left / fig_y, right=1. - right / fig_y)
    hspace = 1.
    if dict_legend['item'] is None:
        height_average = (fig_x - top - bottom - hspace)
        dict_margins['height_ratios'] = [4. * height_average]
        dict_margins['hspace'] = hspace / height_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=1, nrows=1, **dict_margins)

        ax_bar = fig.add_subplot(spec[0])

    else:
        height_average = (fig_x - top - bottom - hspace) / 2.
        dict_margins['height_ratios'] = [4. * height_average,
                                         dict_legend['options_ax']['size_legend'] * height_average]
        dict_margins['hspace'] = hspace / height_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=1, nrows=2, **dict_margins)

        ax_legend = fig.add_subplot(spec[1])
        ax_bar = fig.add_subplot(spec[0])

        set_blank_axis(ax_legend)

    len_which_list = len(dict_figure['list_for_which'][0])
    len_item_list = sum([len(current_dict_section['list_dict']) for current_dict_section
                        in dict_figure['list_dict_section']])
    value_analysis = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_perceptron = dict(list_score_set=np.nan * np.ones((len_item_list, len_which_list)))

    title = dict_key['pattern'].label_value(dict_parameters['params']['pattern']) + u':\n'
    name_plot = dict_parameters['params']['pattern'] + '_'
    for current_key in dict_legend['title']:
        if not current_key == '':
            if current_key in dict_key.keys():
                current_key_label = dict_key[current_key]
            elif current_key in dict_key_others.keys():
                current_key_label = dict_key_others[current_key]
            if dict_parameters['params'][current_key] is None:
                current_key_value = dict_parameters_range[current_key]
            else:
                current_key_value = dict_parameters['params'][current_key]
            title += current_key_label.title + \
                current_key_label.label_value(current_key_value) + ', '
            name_plot += str(current_key_value) + '_'
    title = title[:-2]
    name_plot = name_plot + dict_figure['for_which'][0]

    which_measure = 'accuracy'
    for count_for_which, current_for_which in enumerate(zip(*dict_figure['list_for_which'])):
        count_list_dict_section = 0
        for current_dict_section in dict_figure['list_dict_section']:
            for _, current_dict in enumerate(current_dict_section['list_dict']):
                current_dict_plot = dict(dict_parameters['params'])
                current_dict_plot.update(dict(current_dict['params']))
                for which_for_which, which_for_which_value in zip(dict_figure['for_which'], current_for_which):
                    if which_for_which in dict_key.keys():
                        current_dict_plot[which_for_which] = which_for_which_value
                    elif which_for_which == 'name':
                        current_dict['others'][which_for_which] = which_for_which_value
                    elif which_for_which == 'n_pattern':
                        current_dict_plot[which_for_which] = which_for_which_value

                current_dict_plot.update(dict_parameters_range)

                name_dir = ''.join([str(current_dict_plot[current_key]) + '_' for current_key in dict_key.keys()
                                    if (current_key in current_dict_plot.keys())])[:-1]

                name_project = current_dict['others']['dir']
                name_subproject = current_dict['others']['name']

                simu_dir = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir)
                plot_data = np.load('{}/Plot_data_{}.npy'.format(simu_dir, current_dict_plot['n_pattern']),
                                    allow_pickle=True).item()
                plot_results(plot_data, None, None,
                             params_others=current_dict['others'],
                             which_measure=which_measure,
                             params_legend_handles=current_dict['legend_handles'],
                             compute_max=True,
                             value_analysis=value_analysis, value_analysis_control=None,
                             current_k=count_list_dict_section, current_l=count_for_which
                             )
                value_analysis_perceptron['list_score_set'][count_list_dict_section, count_for_which] = \
                    np.nanmean(plot_data['score_set'])
                count_list_dict_section += 1

    count_list_dict_section = 0
    for current_dict_section in dict_figure['list_dict_section']:
        for _, current_dict in enumerate(current_dict_section['list_dict']):
            if not (current_dict['others']['stats'] == 'None'):
                ax_bar.errorbar(dict_figure['list_for_which'][0],
                                value_analysis['value_list'][count_list_dict_section],
                                yerr=value_analysis['value_list_std'][count_list_dict_section] / 2.,
                                color=colors[current_dict['legend_handles']['color']], linestyle='-',
                                capsize=5.)
            else:
                ax_bar.plot(dict_figure['list_for_which'][0],
                            value_analysis['value_list'][count_list_dict_section],
                            color=colors[current_dict['legend_handles']['color']], linestyle='')
            count_list_dict_section += 1
    ax_bar.set_xticks(dict_figure['list_for_which'][0])
    ax_bar.set_xticklabels(
        [current_key_label.label_value(current_key_value) for current_key_value in dict_figure['list_for_which'][0]])
    if dict_legend['item'] is not None:
        legend = ax_legend.legend(handles=dict_legend['handles_list'], frameon=False, **dict_legend['options_legend'])
        legend.get_frame().set_linewidth(0.5)
    if dict_figure['for_which'][0] in dict_key.keys():
        ax_bar.set_xlabel('{}'.format(dict_key[dict_figure['for_which'][0]].label[0].upper() +
                                      dict_key[dict_figure['for_which'][0]].label[1:], labelpad=5))
    elif dict_figure['for_which'][0] in dict_key_others.keys():
        ax_bar.set_xlabel('{}'.format(dict_key_others[dict_figure['for_which'][0]].label[0].upper() +
                                      dict_key_others[dict_figure['for_which'][0]].label[1:], labelpad=5))
    ax_bar.set_yticks([0., 0.5, 1.])
    ax_bar.set_ylim(dict_legend['options_ax']['ylim0'], dict_legend['options_ax']['ylim1'])
    ax_bar.set_ylabel(dict_legend['options_ax']['ylabel'])

    for ax_ in [ax_bar]:
        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)

    ax_bar.set_title(title, pad=10)
    return fig, name_plot

def figure_line_comp_stats(origin_dir, dict_parameters, dict_figure, dict_legend, dict_parameters_range, cfg):
    if not isinstance(dict_figure['for_which'], list):
        dict_figure['for_which'] = [dict_figure['for_which']]
        dict_figure['list_for_which'] = [dict_figure['list_for_which']]

    fig_y, fig_x = dict_figure['fig_size']
    top, bottom, left, right = dict_figure['margin_size']
    dict_margins = dict(top=1. - top / fig_x, bottom=bottom / fig_x, left=left / fig_y, right=1. - right / fig_y)
    hspace = 1.
    if dict_legend['item'] is None:
        height_average = (fig_x - top - bottom - hspace)
        dict_margins['height_ratios'] = [4. * height_average]
        dict_margins['hspace'] = hspace / height_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=1, nrows=1, **dict_margins)

        ax_bar = fig.add_subplot(spec[0])

    else:
        height_average = (fig_x - top - bottom - hspace) / 2.
        dict_margins['height_ratios'] = [4. * height_average,
                                         dict_legend['options_ax']['size_legend'] * height_average]
        dict_margins['hspace'] = hspace / height_average

        fig = plt.figure(figsize=(fig_y * cfg.cm, fig_x * cfg.cm))
        spec = fig.add_gridspec(ncols=1, nrows=2, **dict_margins)

        ax_legend = fig.add_subplot(spec[1])
        ax_bar = fig.add_subplot(spec[0])

        set_blank_axis(ax_legend)

    len_which_list = 1
    len_item_list = sum([len(current_dict_section['list_dict']) for current_dict_section
                        in dict_figure['list_dict_section']])
    value_analysis = {
        'value_list': np.nan * np.ones((len_item_list, len_which_list)),
        'value_list_std': np.nan * np.ones((len_item_list, len_which_list)),
        'stats': np.zeros((len_item_list, len_which_list), dtype=np.dtype('U4')),
        'data_stats': np.zeros((len_item_list, len_which_list), dtype=object),
        'ref_stats': [{} for _ in np.arange(len_which_list)]
    }

    value_analysis_perceptron = dict(list_score_set=np.nan * np.ones((len_item_list, len_which_list)))

    title = dict_key['pattern'].label_value(dict_parameters['params']['pattern']) + u':\n'
    name_plot = dict_parameters['params']['pattern'] + '_'
    for current_key in dict_legend['title']:
        if not current_key == '':
            if current_key in dict_key.keys():
                current_key_label = dict_key[current_key]
            elif current_key in dict_key_others.keys():
                current_key_label = dict_key_others[current_key]
            if dict_parameters['params'][current_key] is None:
                current_key_value = dict_parameters_range[current_key]
            else:
                current_key_value = dict_parameters['params'][current_key]
            title += current_key_label.title + \
                current_key_label.label_value(current_key_value) + ', '
            name_plot += str(current_key_value) + '_'
    title = title[:-2]
    name_plot = name_plot + dict_figure['for_which'][0]

    count_for_which = 0
    which_measure = 'accuracy'
    count_list_dict_section = 0
    for current_dict_section in dict_figure['list_dict_section']:
        for _, current_dict in enumerate(current_dict_section['list_dict']):
            current_dict_plot = dict(dict_parameters['params'])
            current_dict_plot.update(dict(current_dict['params']))

            current_dict_plot.update(dict_parameters_range)

            name_dir = ''.join([str(current_dict_plot[current_key]) + '_' for current_key in dict_key.keys()
                                if (current_key in current_dict_plot.keys())])[:-1]

            name_project = current_dict['others']['dir']
            name_subproject = current_dict['others']['name']

            simu_dir = '{}{}/{}/{}'.format(origin_dir, name_project, name_subproject, name_dir)
            plot_data = np.load('{}/Plot_data_{}.npy'.format(simu_dir, current_dict_plot['n_pattern']),
                                allow_pickle=True).item()
            plot_results(plot_data, None, None,
                         params_others=current_dict['others'],
                         which_measure=which_measure,
                         params_legend_handles=current_dict['legend_handles'],
                         compute_max=True,
                         value_analysis=value_analysis, value_analysis_control=None,
                         current_k=count_list_dict_section, current_l=count_for_which
                         )
            value_analysis_perceptron['list_score_set'][count_list_dict_section, count_for_which] = \
                np.nanmean(plot_data['score_set'])
            count_list_dict_section += 1

    ax_bar.errorbar(dict_figure['list_for_which'][0],
                    value_analysis['value_list'][:, 0],
                    yerr=value_analysis['value_list_std'][:, 0] / 2.,
                    color=colors[current_dict['legend_handles']['color']], linestyle='-',
                    capsize=5.)
    count_list_dict_section = 0
    for current_dict_section in dict_figure['list_dict_section']:
        for _, current_dict in enumerate(current_dict_section['list_dict']):
            if not (current_dict['others']['stats'] == 'None'):
                text_stats = compute_stat(current_dict['others']['stats'],
                                          value_analysis=value_analysis,
                                          current_k=count_list_dict_section, current_l=count_for_which)
                ax_bar.text(dict_figure['list_for_which'][0][count_list_dict_section],
                            (dict_legend['options_ax']['ylim1'] - 1.)/2. + 1.,
                            text_stats, horizontalalignment='center', verticalalignment='center',
                            color=colors[current_dict['others']['color_stats']], fontsize=7).set_clip_on(True)
            count_list_dict_section += 1
    ax_bar.set_xticks(dict_figure['list_for_which'][0])
    ax_bar.set_xticklabels(
        [current_key_label.label_value(current_key_value) for current_key_value in dict_figure['list_for_which'][0]])
    if dict_legend['item'] is not None:
        legend = ax_legend.legend(handles=dict_legend['handles_list'], frameon=False, **dict_legend['options_legend'])
        legend.get_frame().set_linewidth(0.5)
    if dict_figure['for_which'][0] in dict_key.keys():
        ax_bar.set_xlabel('{}'.format(dict_key[dict_figure['for_which'][0]].label[0].upper() +
                                      dict_key[dict_figure['for_which'][0]].label[1:], labelpad=5))
    elif dict_figure['for_which'][0] in dict_key_others.keys():
        ax_bar.set_xlabel('{}'.format(dict_key_others[dict_figure['for_which'][0]].label[0].upper() +
                                      dict_key_others[dict_figure['for_which'][0]].label[1:], labelpad=5))
    ax_bar.set_yticks([0., 0.5, 1.])
    ax_bar.set_ylim(dict_legend['options_ax']['ylim0'], dict_legend['options_ax']['ylim1'])
    ax_bar.set_ylabel(dict_legend['options_ax']['ylabel'])

    for ax_ in [ax_bar]:
        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)

    ax_bar.set_title(title, pad=10)
    return fig, name_plot


dict_figure_functions = dict(figure_bar=figure_bar, figure_bar_subset=figure_bar_subset,
                             figure_bar_comp=figure_bar_comp, figure_line_comp=figure_line_comp,
                             figure_line_comp_stats=figure_line_comp_stats)
