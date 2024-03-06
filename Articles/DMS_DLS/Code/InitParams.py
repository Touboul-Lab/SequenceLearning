import numpy as np
import itertools

from Simulator.SimulatorDMS import simulatorPatternDMS
from Articles.DMS_DLS.Code.ParserPattern import create_parser_main_DMS
from NetworkModel.StriatumNeurons import find_params_neuron

class paramsGestDMS:
    def __init__(self):
        self.parent_parser = create_parser_main_DMS()
        self.args = self.parent_parser.parse_args()

    def init_main_pattern(self):
        if self.args.random_seed is not None:
            np.random.seed(self.args.random_seed)

        _, neuronParams = find_params_neuron('MSN_Yim')

        self.params_simu = {
            'dt': self.args.dt,
            'num_training_initial': self.args.num_training_initial,
            'num_training_learning': self.args.num_training_learning,
            'num_training_maintenance': self.args.num_training_maintenance,
            'num_training_recall': self.args.num_training_recall,
            'ratio_noise_learning_maintenance': 4.,
            'num_simu': self.args.num_simu,
            'new_set': self.args.new_set,
            'stim_recall': self.args.stim_recall,
            'stop_learning': self.args.stop_learning,
            'test_iteration': 50,
            'test_iteration_detail': 5,
            'save': self.args.save,
            'num_success_params': self.args.num_success_params,
            'plot': self.args.plot,
            'num_spike_output': 1,
            'reset_training': False,
            'accuracy_subpattern': False,
        }

        self.params_network = {
            'P': self.args.P,
            'neuronClass': 'MSN_Yim',
            'homeostasy': self.args.homeostasy,
            'Apostpre': self.args.Apostpre / float(
                self.params_simu['num_spike_output']) if neuronParams['Burst'] is not None else self.args.Apostpre,
            'Aprepost': self.args.Aprepost / float(
                self.params_simu['num_spike_output']) if neuronParams['Burst'] is not None else self.args.Aprepost,
            'tprepost': 20.,
            'tpostpre': 20.,
            'nearest': False,
            'exp': True,
            'noise_input': self.args.noise_input,
            'noise_stim': self.args.noise_input / float(self.args.P),
            'epsilon': self.args.epsilon,
            'init_weight': ['uniform', (0., 0.05)],
            'clip_weight': (0., 2.),
            'save': (not self.args.save) or (not self.args.plot),
            'homeostasy_post': self.args.homeostasy_post,
        }

        self.simulator = simulatorPatternDMS

        SETS = list(itertools.combinations(np.arange(self.args.P), self.args.stim_by_pattern))

        self.params_pattern = {
            'type': 'list_pattern_DMS',
            'n_pattern': None,
            'stim_by_pattern': self.args.stim_by_pattern,
            'random_time': None,
            'duration': self.args.stim_duration,
            'offset': self.args.stim_offset,
            'p_reward': self.args.p_reward,
            'sets': SETS,
            'noise_pattern': self.args.noise_pattern,
            'sample_pattern': True,
        }

        if self.params_pattern['sample_pattern']:
            self.params_pattern['noise_pattern'] = self.args.noise_pattern
        else:
            self.params_pattern['noise_pattern'] = None

        if self.args.P == 10:
            n_pattern_list = [10, 15]
        elif self.args.P == 20:
            n_pattern_list = [30]
        else:
            n_pattern_list = np.arange(self.args.P // 2, 2 * self.args.P, self.args.P // 2, dtype=np.int)

        self.params_simu['n_pattern_list'] = n_pattern_list

    def update_main_pattern_n_pattern(self, n_pattern):
        self.params_pattern['n_pattern'] = n_pattern
        if self.args.random_seed is not None:
            np.random.seed(self.args.random_seed)
            self.random_seed = np.random.choice(100000, size=(self.params_simu['num_simu']), replace=False)

    def update_main_pattern_n_simu(self, n_simu):
        pass
