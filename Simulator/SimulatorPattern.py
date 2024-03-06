import numpy as np
import matplotlib.pyplot as plt

from NetworkModel.NetworkPatternRecognition import patternRecognition, patternRecognitionDual, patternRecognitionExample
from lmfit import Minimizer, Parameters

from Simulator.Pattern import patternClass, pattern_Succession, \
                                pattern_Poisson, pattern_None, pattern_jitter, get_subpattern
from datetime import datetime

def linear_p(pars, x):
    x = np.concatenate([x for _ in range(int((pars['N']+1)/x.shape[1])+1)], axis=1)
    array = np.expand_dims(np.array([pars['A_{:d}'.format(i)] for i in np.arange(int(pars['N']))]), axis=0)
    sol = 1. / (1. + np.exp(-np.matmul(array, x) + pars['B']))
    return sol

def log_ref(pars, x, data):
    data = np.concatenate([data for _ in range(int((pars['N']+1)/x.shape[1])+1)])
    sol = - data * np.log(linear_p(pars, x)) - (1.-data) * np.log(1.-linear_p(pars, x))
    return sol

def create_lmfit(x, y, N):
    pfit = Parameters()
    pfit.add(name='N', value=N, vary=False)
    pfit.add(name='B', value=1., min=0.)
    for i in np.arange(N):
        pfit.add(name='A_{}'.format(i), value=0.1, min=0., max=5.)
    mini = Minimizer(log_ref, pfit, fcn_args=(x, y))
    out = mini.leastsq(max_nfev=10000000)
    return out

class simulatorPattern:

    def __init__(self, params_simu=None, params_pattern=None, params_network=None):
        self.params_simu = params_simu
        self.params_pattern = params_pattern
        self.params_network = params_network

        self.params_output = {
            'output_train': [],
            'output_test': [],
            'weight_ref': []
        }

        self.iteration = 0
        self.error = 0

        self.network = None
        self.init_network()

        self.reward_trace = None
        self.pattern_list = None
        if self.params_simu['accuracy_subpattern']:
            self.subpattern_list = None

    def init_network(self):
        self.network = patternRecognition(**self.params_network)
        weight_ref = dict(iteration=self.iteration, weight=self.network.WEIGHT.weight[-1])
        self.params_output['weight_ref'].append(weight_ref)

    def init_pattern(self):
        self.reward_trace = np.zeros(self.params_pattern['n_pattern'], dtype=np.int)
        if self.params_pattern['type'] == 'list_pattern' or self.params_pattern['type'] == 'jitter':
            type_pattern = patternClass if self.params_pattern['type'] == 'list_pattern' else pattern_jitter
            if self.params_pattern['repartition'] == 'uniform':
                index_pattern = np.random.choice(len(self.params_pattern['sets']), self.params_pattern['n_pattern'],
                                                 replace=False)
                self.pattern_list = [type_pattern(self, self.params_pattern['sets'][i]) for i in index_pattern]
            elif self.params_pattern['repartition'] == 'uniform_stim':
                ok = False
                while not ok:
                    stim_pattern = np.sort(np.random.choice(len(self.params_pattern['sets']),
                                                            self.params_pattern['n_pattern'],
                                                            replace=True))
                    self.pattern_list = []
                    for j in np.arange(len(self.params_pattern['sets'])):
                        num_j = np.sum(stim_pattern == j)
                        if num_j <= len(self.params_pattern['sets'][j]):
                            index_pattern_j = np.random.choice(len(self.params_pattern['sets'][j]), num_j,
                                                               replace=False)
                            self.pattern_list = self.pattern_list + [type_pattern(self,
                                                                     self.params_pattern['sets'][j][i])
                                                                     for i in index_pattern_j]
                    if len(self.pattern_list) == self.params_pattern['n_pattern']:
                        ok = True
            if self.params_simu['accuracy_subpattern']:
                self.subpattern_list = sum([get_subpattern(current_pattern, self)
                                            for current_pattern in self.pattern_list], [])
        elif self.params_pattern['type'] == 'succession':
            self.pattern_list = [pattern_Succession(self, num=i) for i in range(self.params_pattern['n_pattern'])]
        elif self.params_pattern['type'] == 'poisson':
            self.pattern_list = [pattern_Poisson(self) for _ in range(self.params_pattern['n_pattern'])]
        elif self.params_pattern['type'] == 'example_A':
            self.pattern_list = [pattern_None(self) for _ in range(self.params_pattern['n_pattern'])]
        elif self.params_pattern['type'] == 'example_B':
            self.params_pattern['noise_poisson'] = 1.
            self.params_pattern['duration_poisson'] = 5.
            self.pattern_list = [pattern_Poisson(self) for _ in range(self.params_pattern['n_pattern'])]

    def run(self, name):
        self.params_output['name'] = name + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        if self.pattern_list[0].input is None:
            score_set = None
        else:
            if self.params_simu['accuracy_subpattern']:
                (score_set, score_set_subset) = \
                    self.run_LP(self.pattern_list, test=self.subpattern_list)
                self.params_output['score_set_subset'] = score_set_subset
            else:
                score_set = self.run_LP(self.pattern_list)
        self.params_output['score_set'] = score_set
        if self.params_simu['test_iteration'] is None:
            self.simulate_test(sample_pattern=self.params_pattern['sample_pattern'])
        self.simulate_training(LTP=self.params_network['homeostasy']*self.params_network['epsilon'],
                               num_training=self.params_simu['num_training'],
                               noise_input=self.params_network['noise_input'],
                               noise_stim=self.params_network['noise_stim'],
                               stim_recall=None, sample_pattern=self.params_pattern['sample_pattern'])
        self.simulate_test(sample_pattern=self.params_pattern['sample_pattern'])

    def evaluate(self, output, pattern, test=False):
        if self.params_simu['num_spike_output'] is None:
            if pattern.reward < 0.:
                if output['spike_after_pattern'][0] is None:
                    if test:
                        return 1.
                    else:
                        self.reward_trace[output['pattern']] += 1
                        return 1.
                else:
                    if test:
                        return 0.
                    else:
                        self.reward_trace[output['pattern']] = 0
                        self.error += 1.
                        return 0.
            elif pattern.reward > 0.:
                if output['spike_after_pattern'][0] is None or output['spike_after_pattern'][0] < 0:
                    if test:
                        return 0.
                    else:
                        self.reward_trace[output['pattern']] = 0
                        self.error += 1.
                        return 0.
                else:
                    if test:
                        return 1.
                    else:
                        self.reward_trace[output['pattern']] += 1.
                        return 1.
        else:
            if 1 * (output['num_spike'] >= self.params_simu['num_spike_output'])[0] == int(pattern.reward_vector[0]):
                if test:
                    return 1.
                else:
                    self.reward_trace[output['pattern']] += 1
                    return 1.
            else:
                if test:
                    return 0.
                else:
                    self.reward_trace[output['pattern']] = 0
                    self.error += 1.
                    return 0.

    def test_iteration(self, i):
        if self.params_simu['test_iteration'] is None:
            pass
        elif self.params_simu['test_iteration'] is not None:
            if i % self.params_simu['test_iteration'] == 0:
                self.simulate_test(sample_pattern=self.params_pattern['sample_pattern'])

    def start_training(self, **kwargs):
        self.network.STIM.parameters['noise'] = kwargs['noise_stim']
        if self.network.parameters['noise_input'] is not None:
            self.network.RANDOM_INPUT.parameters['b'] = kwargs['noise_input']

    def start_pattern(self, selected_pattern, LTP):
        if self.params_simu['stop_learning'] == 'number_success':
            if self.reward_trace[selected_pattern] > self.params_simu['num_success_params']:
                self.network.WEIGHT.occlusion = 0.
            else:
                self.network.WEIGHT.occlusion = 1.
        self.network.WEIGHT.parameters['homeostasy'] = LTP
        self.network.WEIGHT.homeostasy_vector = self.pattern_list[selected_pattern].reward_vector

    def end_training(self):
        self.error *= np.exp(-self.params_simu['dt'])
        if self.params_simu['stop_learning'] == 'exponential_trace':
            self.network.WEIGHT.occlusion = self.error
        self.network.WEIGHT.parameters['homeostasy'] = 0.
        self.network.WEIGHT.homeostasy_vector = np.ones(self.network.NEURON.P)

    def start_test(self):
        previous = [self.network.STIM.parameters['noise']]
        self.network.STIM.parameters['noise'] = None
        previous.append(self.network.NEURON.parameters['noise'])
        self.network.NEURON.parameters['noise'] = None
        if self.network.parameters['noise_input'] is not None:
            previous.append(self.network.RANDOM_INPUT.parameters['b'])
            self.network.RANDOM_INPUT.parameters['b'] = 0.
        else:
            previous.append(None)
        previous.append(self.network.WEIGHT.occlusion)
        self.network.WEIGHT.occlusion = 0.
        return previous

    def end_test(self, previous):
        self.network.STIM.parameters['noise'] = previous[0]
        self.network.NEURON.parameters['noise'] = previous[1]
        if self.network.parameters['noise_input'] is not None:
            self.network.RANDOM_INPUT.parameters['b'] = previous[2]
        self.network.WEIGHT.occlusion = previous[3]

    def run_LP(self, pattern_list, test=None):
        X_set = 1. - np.array([1. * (u.input < 0.) for u in pattern_list]).transpose()
        Y = np.array([(u.reward + 1.) / 2. for u in pattern_list], dtype=np.int).transpose()
        try:
            logreg_set = create_lmfit(X_set, Y, self.network.STIM.P)
            score_set = np.mean((linear_p(logreg_set.params, X_set)[0, :X_set.shape[1]] > 0.5) == Y)
            if test is not None:
                X_set_subset = 1. - np.array([1. * (u.input < 0.) for u in test]).transpose()
                Y_subset = np.array([(u.reward + 1.) / 2. for u in test], dtype=np.int).transpose()
                if len(X_set_subset) > 0:
                    score_set_subset = np.mean((linear_p(logreg_set.params, X_set_subset)[0, :X_set_subset.shape[1]]
                                                > 0.5) == Y_subset)
                else:
                    score_set_subset = np.nan
                return score_set, score_set_subset
            else:
                return score_set
        except ValueError:
            score_set, score_set_subset = np.nan, np.nan
            if test is not None:
                return score_set, score_set_subset
            else:
                return score_set

    def simulate_training(self, **kwargs):
        self.start_training(**kwargs)
        for i in range(kwargs['num_training']):
            self.test_iteration(i)
            if self.params_simu['reset_training']:
                self.network.NEURON.init_eq()
                self.network.WEIGHT.init_eq()
            selected_pattern = np.random.randint(0, len(self.pattern_list))
            output_train = dict()
            output_train['iteration'] = self.iteration
            output_train['iteration_train'] = i
            output_train['pattern'] = selected_pattern
            output_train['pattern_index'] = self.pattern_list[selected_pattern].index
            output_train['pattern_timing'] = self.pattern_list[selected_pattern].timing
            output_train['stim_recall'] = kwargs['stim_recall']
            output_train['sample_pattern'] = self.params_pattern['sample_pattern']
            if self.params_pattern['sample_pattern']:
                self.pattern_list[selected_pattern].sample(self.params_pattern['noise_pattern'])
            self.start_pattern(selected_pattern, LTP=kwargs['LTP'])
            self.network.iterate_pattern(self.params_simu['dt'], self.pattern_list[selected_pattern], output_train)
            if output_train['stim'] == 1:
                output_train['accuracy'] = self.evaluate(output_train, self.pattern_list[selected_pattern])
            else:
                output_train['accuracy'] = np.nan
            self.end_training()
            self.params_output['output_train'].append(output_train)
            self.iteration += 1

    def simulate_test(self, plot=False, **kwargs):
        previous = self.start_test()
        output_test = dict()
        output_test['list'] = []
        output_test['list_subpattern'] = []
        output_test['iteration'] = self.iteration
        output_test['weight'] = self.network.WEIGHT.weight[-1]
        for i in range(len(self.pattern_list)):
            self.network.NEURON.init_eq()
            self.network.WEIGHT.init_eq()
            selected_pattern = i
            output_test_pattern = dict()
            output_test_pattern['iteration'] = self.iteration
            output_test_pattern['iteration_test'] = i
            output_test_pattern['pattern'] = selected_pattern
            output_test_pattern['pattern_index'] = self.pattern_list[selected_pattern].index
            output_test_pattern['stim_recall'] = None
            if kwargs['sample_pattern']:
                output_test_pattern['sample_pattern'] = self.params_pattern['sample_pattern']
                self.pattern_list[selected_pattern].sample(None)
            self.network.iterate_pattern(self.params_simu['dt'], self.pattern_list[selected_pattern],
                                         output_test_pattern)
            output_test_pattern['accuracy'] = self.evaluate(output_test_pattern, self.pattern_list[selected_pattern],
                                                            test=True)
            output_test['list'].append(output_test_pattern)
        if self.params_simu['accuracy_subpattern']:
            for i in np.random.choice(len(self.subpattern_list), min(self.params_pattern['n_pattern'],
                                      len(self.subpattern_list)), replace=False):
                self.network.NEURON.init_eq()
                self.network.WEIGHT.init_eq()
                selected_subpattern = i
                output_test_subpattern = dict()
                output_test_subpattern['iteration'] = self.iteration
                output_test_subpattern['iteration_test'] = i
                output_test_subpattern['pattern'] = selected_subpattern
                output_test_subpattern['pattern_index'] = self.subpattern_list[selected_subpattern].index
                output_test_subpattern['stim_recall'] = None
                if kwargs['sample_pattern']:
                    output_test_subpattern['sample_pattern'] = self.params_pattern['sample_pattern']
                    self.subpattern_list[selected_subpattern].sample(0.)
                self.network.iterate_pattern(self.params_simu['dt'], self.subpattern_list[selected_subpattern],
                                             output_test_subpattern)
                output_test_subpattern['accuracy'] = self.evaluate(output_test_subpattern,
                                                                   self.subpattern_list[selected_subpattern], test=True)
                output_test['list_subpattern'].append(output_test_subpattern)
        self.end_test(previous)
        self.params_output['output_test'].append(output_test)
        if plot:
            self.plot_test()

    def plot_accuracy_train(self, ax, convolve=20):
        accuracy_iteration = np.array([output_train['iteration'] for output_train in self.params_output['output_train']
                                       if output_train['stim'] == 1])
        accuracy = np.array([output_train['accuracy'] for output_train in self.params_output['output_train']
                             if output_train['stim'] == 1])
        before = accuracy[:int(convolve)][::-1]
        after = accuracy[-int(convolve):][::-1]
        x_padded = np.concatenate([before, accuracy, after])
        x_filtered = np.zeros(len(accuracy))
        for i in range(len(accuracy)):
            x_filtered[i] = np.mean(x_padded[i:i+2*convolve+1])
        ax.plot(accuracy_iteration, x_filtered)

    def plot(self):
        fig = plt.figure(figsize=(18, 10))
        ax1 = fig.add_subplot(611)
        ax2 = fig.add_subplot(612, sharex=ax1)
        ax3 = fig.add_subplot(613, sharex=ax1)
        ax4 = fig.add_subplot(614)
        ax5 = fig.add_subplot(615)
        ax6 = fig.add_subplot(616)

        self.network.STIM.plot_trace(ax1, index=range(self.params_network['P']))
        self.network.NEURON.plot_trace(ax2, index=range(self.network.NEURON.P))
        self.network.WEIGHT.plot_history(ax3, index=[(0, u) for u in range(self.params_network['P'])])
        if self.network.NEURON.P == 2:
            self.network.WEIGHT.plot_history(ax3, index=[(1, u) for u in range(self.params_network['P'])],
                                             linestyle='--')
        self.plot_accuracy_train(ax4)
        ax4.set_ylim(0., 1.)
        accuracy_iteration = np.array([output_test['iteration'] for output_test in self.params_output['output_test']])
        accuracy = np.array([np.mean([item['accuracy'] for item in output_test['list']])
                             for output_test in self.params_output['output_test']])
        ax5.plot(accuracy_iteration, accuracy, "-+")
        ax5.set_ylim(0., 1.)
        ax6.plot(np.arange(len(self.pattern_list)), self.reward_trace)
        ax6.set_xticks(np.arange(len(self.pattern_list)))
        ax6.set_xticklabels([str(self.pattern_list[j].index) + '_' + str(self.pattern_list[j].reward)
                             for j in range(len(self.pattern_list))])
        plt.tight_layout()
        plt.show()

    def plot_test(self):
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        self.network.STIM.plot_trace(ax2, index=range(self.params_network['P']))
        self.network.NEURON.plot_trace(ax1, index=range(self.network.NEURON.P))
        ax1.set_xlim(self.network.NEURON.time[-1]-len(self.pattern_list)*self.params_pattern['duration'],
                     self.network.NEURON.time[-1])
        ax1.set_ylim(-80., 20.)
        ax2.set_xticks([self.network.NEURON.time[-1]-(len(self.pattern_list)-i-0.5)*self.params_pattern['duration']
                        for i in range(len(self.pattern_list))])
        ax2.set_xticklabels([str(self.pattern_list[j].index) + '_' + str(self.pattern_list[j].reward)
                             for j in range(len(self.pattern_list))])
        plt.tight_layout()
        if not(self.params_simu['save']):
            plt.savefig(self.params_simu['dir'] + '/Plot_' + str(len(self.pattern_list)) + '_pattern_'
                        + str(self.params_pattern['no_reward']) + '.pdf')
            plt.close()
        else:
            plt.show()

class simulatorPatternDual(simulatorPattern):
    def init_network(self):
        self.network = patternRecognitionDual(**self.params_network)
        weight_ref = dict(iteration=self.iteration, weight=self.network.WEIGHT.weight[-1])
        self.params_output['weight_ref'].append(weight_ref)

class simulatorPatternExample:

    def __init__(self, params_simu=None, params_pattern=None, params_network=None):
        self.params_simu = params_simu
        self.params_pattern = params_pattern
        self.params_network = params_network

        self.params_output = {
            'output_train': [],
            'output_test': [],
            'weight_ref': []
        }

        self.iteration = 0

        self.network = None
        self.init_network()

    def init_network(self):
        self.network = patternRecognitionExample(**self.params_network)

    def init_pattern(self):
        self.pattern_list = []
        self.pattern_list_MSN = []

    def run(self, name):
        self.simulate_training(LTP=self.params_network['homeostasy']*self.params_network['epsilon'],
                               num_training=self.params_pattern['n_pattern'])

    def start_training(self, **kwargs):
        pass

    def start_pattern(self, selected_pattern, LTP):
        self.network.WEIGHT.parameters['homeostasy'] = LTP
        self.network.WEIGHT.homeostasy_vector = self.pattern_list[selected_pattern].reward_vector

    def end_training(self):
        self.network.WEIGHT.parameters['homeostasy'] = 0.
        self.network.WEIGHT.homeostasy_vector = np.ones(self.network.NEURON.P)

    def simulate_training(self, **kwargs):
        self.start_training(**kwargs)
        for i in range(kwargs['num_training']):
            selected_pattern = i
            output_train = dict()
            output_train['iteration'] = self.iteration
            output_train['iteration_train'] = i
            output_train['pattern'] = selected_pattern
            output_train['pattern_index'] = self.pattern_list[selected_pattern].index
            output_train['pattern_timing'] = self.pattern_list[selected_pattern].timing
            output_train['stim_recall'] = None
            output_train['pattern_MSN'] = self.pattern_list_MSN[i]
            self.start_pattern(selected_pattern, LTP=kwargs['LTP'])
            self.network.iterate_pattern(self.params_simu['dt'], self.pattern_list[selected_pattern], output_train)
            self.end_training()
            self.params_output['output_train'].append(output_train)
            self.iteration += 1

    def plot(self):
        pass
