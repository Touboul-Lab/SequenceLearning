import matplotlib.pyplot as plt

from Simulator.Pattern import pattern_DMS
from Simulator.SimulatorPattern import simulatorPattern
from Articles.DMS_DLS.Code.FiguresDMS import *

from datetime import datetime


class simulatorPatternDMS(simulatorPattern):

    def init_pattern(self):
        self.reward_trace = np.zeros(self.params_pattern['n_pattern'], dtype=np.int)
        index_pattern = np.random.choice(len(self.params_pattern['sets']), self.params_pattern['n_pattern'],
                                         replace=False)
        self.pattern_list = [pattern_DMS(self, self.params_pattern['sets'][i]) for i in index_pattern]

    def test_iteration(self, i):
        if self.params_simu['test_iteration'] is None:
            pass
        elif self.params_simu['test_iteration'] is not None:
            if i < self.params_simu['test_iteration']:
                if i % self.params_simu['test_iteration_detail'] == 0:
                    self.simulate_test(sample_pattern=self.params_pattern['sample_pattern'])
            else:
                if i % self.params_simu['test_iteration'] == 0:
                    self.simulate_test(sample_pattern=self.params_pattern['sample_pattern'])

    def run(self, name):
        self.params_output['name'] = name + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.simulate_training(LTP=self.params_network['homeostasy_post']*self.params_network['epsilon'],
                               num_training=self.params_simu['num_training_initial'],
                               noise_input=self.params_simu[
                                'ratio_noise_learning_maintenance'] * self.params_network['noise_input'],
                               noise_stim=self.params_simu[
                                'ratio_noise_learning_maintenance'] * self.params_network['noise_stim'],
                               stim_recall=self.params_simu['stim_recall'],
                               sample_pattern=self.params_pattern['sample_pattern'])
        weight_ref = dict(iteration=self.iteration, weight=self.network.WEIGHT.weight[-1])
        self.params_output['weight_ref'].append(weight_ref)
        self.simulate_training(LTP=self.params_network['homeostasy']*self.params_network['epsilon'],
                               num_training=self.params_simu['num_training_learning'],
                               noise_input=self.params_network['noise_input'],
                               noise_stim=self.params_network['noise_stim'],
                               stim_recall=None, sample_pattern=self.params_pattern['sample_pattern'])
        weight_ref = dict(iteration=self.iteration, weight=self.network.WEIGHT.weight[-1])
        self.params_output['weight_ref'].append(weight_ref)
        self.simulate_training(LTP=self.params_network['homeostasy_post']*self.params_network['epsilon'],
                               num_training=self.params_simu['num_training_maintenance'],
                               noise_input=self.params_simu[
                                'ratio_noise_learning_maintenance'] * self.params_network['noise_input'],
                               noise_stim=self.params_simu[
                                'ratio_noise_learning_maintenance'] * self.params_network['noise_stim'],
                               stim_recall=self.params_simu['stim_recall'],
                               sample_pattern=self.params_pattern['sample_pattern'])
        if self.params_simu['new_set'] == 1:
            self.init_pattern()
        weight_ref = dict(iteration=self.iteration, weight=self.network.WEIGHT.weight[-1])
        self.params_output['weight_ref'].append(weight_ref)
        self.simulate_training(LTP=self.params_network['homeostasy']*self.params_network['epsilon'],
                               num_training=self.params_simu['num_training_recall'],
                               noise_input=self.params_network['noise_input'],
                               noise_stim=self.params_network['noise_stim'],
                               stim_recall=None,
                               sample_pattern=self.params_pattern['sample_pattern'])
        self.simulate_test(sample_pattern=self.params_pattern['sample_pattern'])

    def plot(self):
        fig = plt.figure(figsize=(18, 10))
        ax1 = fig.add_subplot(611)
        ax2 = fig.add_subplot(612, sharex=ax1)
        ax3 = fig.add_subplot(613, sharex=ax1)
        ax4 = fig.add_subplot(614)
        ax5 = fig.add_subplot(615)
        ax6 = fig.add_subplot(616)

        self.network.STIM.plot_trace(ax1, index=range(self.params_network['P']))
        self.network.NEURON.plot_trace(ax2, index=[0])
        self.network.WEIGHT.plot_history(ax3, index=[(0, u) for u in range(self.params_network['P'])])
        self.plot_accuracy_train(ax4)
        accuracy_iteration = np.array([output_test['iteration'] for output_test in self.params_output['output_test']])
        accuracy = np.array([np.mean([item['accuracy'] for item in output_test['list']]) for output_test in
                             self.params_output['output_test']])
        ax5.plot(accuracy_iteration, accuracy, "-+")
        ax5.set_ylim(0., 1.)
        ax6.plot(np.arange(len(self.pattern_list)), self.reward_trace)
        ax6.set_xticks(np.arange(len(self.pattern_list)))
        ax6.set_xticklabels([str(self.pattern_list[j].index) + '_' +
                             str(self.pattern_list[j].reward) for j in range(len(self.pattern_list))])
        for ax in [ax4, ax5]:
            figure_different_phases(ax, self.params_simu, [0., 1.])
        plt.tight_layout()
        plt.show()
