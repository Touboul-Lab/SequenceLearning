import numpy as np
import itertools

from Simulator.SimulatorPattern import simulatorPattern, simulatorPatternDual
from Articles.Sequential_Learning_Striatum.Code.ParserPattern import create_parser_main_pattern
from NetworkModel.StriatumNeurons import find_params_neuron

class paramsGestPattern:
    def __init__(self):
        self.parent_parser = create_parser_main_pattern()
        self.args = self.parent_parser.parse_args()

    def init_main_pattern(self):
        if self.args.random_seed is not None:
            np.random.seed(self.args.random_seed)

        _, neuronParams = find_params_neuron(self.args.neuronClass)

        self.params_simu = {
            'dt': self.args.dt,
            'num_training': None,
            'num_simu': None,
            'stop_learning': self.args.stop_learning,
            'test_iteration': 5,
            'save': self.args.save,
            'plot': self.args.plot,
            'num_success_params': self.args.num_success_params,
            'num_spike_output': neuronParams['Burst'] if neuronParams['Burst'] is not None else None,
            'reset_training': None,
            'accuracy_subpattern': True if self.args.pattern == 'list_pattern' else False,
        }

        self.params_network = {
            'P': None,
            'neuronClass': self.args.neuronClass,
            'homeostasy': self.args.homeostasy,
            'Apostpre': self.args.Apostpre / float(
                self.params_simu['num_spike_output']) if neuronParams['Burst'] is not None else self.args.Apostpre,
            'Aprepost': self.args.Aprepost / float(
                self.params_simu['num_spike_output']) if neuronParams['Burst'] is not None else self.args.Aprepost,
            'tprepost': 20.,
            'tpostpre': 20.,
            'nearest': False,
            'exp': True,
            'noise_stim': self.args.noise_stim,
            'noise_input': self.args.noise_input,
            'epsilon': self.args.epsilon,
            'init_weight': ['uniform', (0., 0.05)],
            'clip_weight': (0., 2.),
            'save': (not self.args.save) or (not self.args.plot),
        }

        if self.args.noise_input == 0.:
            self.params_network['noise_input'] = None

        if self.args.noise_stim == 0.:
            self.params_network['noise_stim'] = None

        if self.args.network == 'dual':
            if self.args.J_matrix == 0:
                J_matrix = np.zeros((2, 2))
            elif self.args.J_matrix == 1:
                J_matrix = np.array([[0., -1.], [-1., 0.]])
            elif self.args.J_matrix == 2:
                J_matrix = np.array([[0., -1.], [0., 0.]])
            elif self.args.J_matrix == 3:
                J_matrix = np.array([[0., 0.], [-1., 0.]])
            else:
                raise NameError('Not the right value for J_matrix')
            J_value = 'random' if self.args.J_value == 'random' else float(self.args.J_value)
            self.params_network['J_matrix'] = J_matrix
            self.params_network['J_value'] = J_value
            self.simulator = simulatorPatternDual

        else:
            self.simulator = simulatorPattern

        if self.args.pattern == 'list_pattern' or self.args.pattern == 'jitter':
            self.params_simu['num_training'] = np.minimum(2000, self.args.num_training * self.args.P)
            self.params_simu['num_simu'] = self.args.num_simu
            self.params_simu['reset_training'] = False
            self.params_network['P'] = self.args.P
            if self.args.repartition == 'uniform':
                SETS = list(itertools.chain.from_iterable(
                    sum(map(lambda u: [list(p) for p in itertools.permutations(u)],
                        itertools.combinations(np.arange(self.args.P), n)), [])
                    for n in np.arange(1, min(self.args.stim_by_pattern + 1, self.args.P + 1))))
            elif self.args.repartition == 'uniform_stim':
                SETS = [sum(map(lambda u: [list(p) for p in itertools.permutations(u)],
                            itertools.combinations(np.arange(self.args.P), n)), [])
                        for n in np.arange(1, min(self.args.stim_by_pattern + 1, self.args.P + 1))]
            self.params_pattern = {
                'type': self.args.pattern,
                'n_pattern': None,
                'stim_by_pattern': self.args.stim_by_pattern,
                'delay': self.args.stim_delay,
                'random_time': None,
                'duration': self.args.stim_duration,
                'offset': self.args.stim_offset,
                'p_reward': self.args.p_reward,
                'sets': SETS,
                'sample_pattern': False if self.args.noise_pattern == 0. else True,
                'repartition': self.args.repartition,
            }
            if self.args.network == 'dual':
                self.params_pattern['J_reward'] = self.args.J_reward
            if self.args.P == 10:
                n_pattern_list = np.array([5, 10, 15, 20], dtype=np.int)
            elif self.args.P > 10:
                n_pattern_list = np.array([self.args.P], dtype=np.int)
            else:
                n_pattern_list = np.arange(2, len(SETS), dtype=np.int)

        elif self.args.pattern == 'succession':
            self.params_simu['num_training'] = self.args.num_training
            self.params_simu['reset_training'] = False
            self.params_pattern = {
                'type': 'succession',
                'n_pattern': None,
                'delay': self.args.stim_delay,
                'random_time': None,
                'duration': self.args.stim_duration,
                'offset': self.args.stim_offset,
                'sample_pattern': False if self.args.noise_pattern == 0. else True
            }
            if self.args.network == 'dual':
                self.params_pattern['J_reward'] = self.args.J_reward
            if self.params_pattern['sample_pattern']:
                self.params_pattern['noise_pattern'] = self.args.noise_pattern
            n_pattern_list = np.arange(2, self.args.P, dtype=np.int)

        elif self.args.pattern == 'poisson':
            self.params_simu['num_training'] = np.minimum(2000, self.args.num_training * self.args.P)
            self.params_simu['num_simu'] = self.args.num_simu
            self.params_simu['reset_training'] = False
            self.params_network['P'] = self.args.P
            self.params_pattern = {
                'type': 'poisson',
                'n_pattern': None,
                'duration': self.args.stim_duration,
                'offset': self.args.stim_offset,
                'duration_poisson': self.args.duration_poisson,
                'noise_poisson': self.args.noise_poisson,
                'p_reward': self.args.p_reward,
                'sample_pattern': False if self.args.noise_pattern == 0. else True
            }
            if self.args.network == 'dual':
                self.params_pattern['J_reward'] = self.args.J_reward
            if self.args.P == 10:
                n_pattern_list = np.array([5, 10, 15, 20], dtype=np.int)
            elif self.args.P == 20:
                n_pattern_list = np.array([10, 20, 30], dtype=np.int)
            else:
                n_pattern_list = np.arange(2, len(SETS), dtype=np.int)

        elif self.args.pattern == 'example':
            self.params_simu['num_simu'] = self.args.num_simu
            self.params_simu['num_training'] = self.args.num_training
            self.params_simu['reset_training'] = True
            self.params_network['P'] = self.args.P
            if self.args.start_weight == 'high':
                self.params_network['init_weight'] = ['uniform', (0.05, 0.5)]
            self.params_pattern = {
                'type': self.args.pattern + '_' + self.args.pattern_example,
                'n_pattern': None,
                'duration': self.args.stim_duration,
                'offset': self.args.stim_offset,
                'p_reward': None,
                'no_reward': self.args.no_reward,
                'sample_pattern': False if self.args.noise_pattern == 0. else True,
            }
            if self.args.network == 'dual':
                self.params_pattern['J_reward'] = self.args.J_reward
            n_pattern_list = np.array([1])

        self.params_simu['n_pattern_list'] = n_pattern_list
        if self.params_pattern['sample_pattern']:
            self.params_pattern['noise_pattern'] = self.args.noise_pattern
        else:
            self.params_pattern['noise_pattern'] = None

    def update_main_pattern_n_pattern(self, n_pattern):
        if self.args.pattern == 'list_pattern' or self.args.pattern == 'poisson' or self.args.pattern == 'jitter':
            self.params_pattern['n_pattern'] = n_pattern
            self.params_simu['test_iteration'] = np.minimum(50, n_pattern)
        elif self.args.pattern == 'succession':
            self.params_simu['num_simu'] = 2 ** n_pattern
            self.params_network['P'] = n_pattern
            self.params_pattern['n_pattern'] = n_pattern
        elif self.args.pattern == 'example':
            self.params_pattern['n_pattern'] = n_pattern
            self.params_simu['num_training'] = self.args.num_training
            self.params_simu['test_iteration'] = 5
        if self.args.random_seed is not None:
            np.random.seed(self.args.random_seed)
            self.random_seed = np.random.choice(100000, size=(self.params_simu['num_simu']), replace=False)

    def update_main_pattern_n_simu(self, n_simu):
        if self.args.pattern == 'succession':
            self.params_pattern['no_reward'] = n_simu
