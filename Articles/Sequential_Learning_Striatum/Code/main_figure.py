import matplotlib.pyplot as plt
import itertools
import argparse
import sys
import os
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import *
from Articles.Sequential_Learning_Striatum.Figures.FigurePlot import read_conf_args, dict_figure_functions
import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg

main_parser = argparse.ArgumentParser()
main_parser.add_argument('--name_project', type=str, help='name_project')
main_parser.add_argument('--name_figure', type=str, help='name_figure')
main_parser.add_argument('--for_args_name', type=str, help='for_args_name')
main_parser.add_argument('--for_args_value', type=str, help='for_args_value')
main_parser.add_argument('--args_plot', type=str, help='args_plot')

args = main_parser.parse_args()

gs_kw = dict(width_ratios=[10, 2], height_ratios=[1], hspace=0., wspace=0.)

name_project = args.name_project
origin_dir = 'Simu/'
save_dir = 'Figures/simu/' + name_project + '/' + args.name_figure + '/'

dict_parameters = dict(params=dict(save=True, random_seed='None', plot='False'))

dict_figure, dict_legend = read_conf_args(dict_parameters, args)

for xs in itertools.product(*dict_parameters['range_for_values']):
    current_dict_parameters_range = {}
    for current_key, current_value in zip(dict_parameters['range_for_name'], xs):
        current_dict_parameters_range[current_key] = current_value
    fig, name_plot = dict_figure_functions[dict_figure['type_figure']](
        origin_dir, dict_parameters, dict_figure, dict_legend, current_dict_parameters_range, cfg,
        **dict_figure['type_figure_options'])
    plt.savefig('{}/{}.png'.format(save_dir, name_plot),
                dpi=1000)
    plt.close()
