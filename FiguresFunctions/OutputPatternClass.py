import os
import errno
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

from FiguresFunctions.FiguresPattern import *
from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import accuracy_max
from Articles.DMS_DLS.Code.FiguresDMS import figure_different_phases

def save_only(instance, current_output, keys_save):
    dict_save = dict(params_simu=instance.params_simu, params_network=instance.params_network,
                     params_pattern=instance.params_pattern)
    for key in keys_save:
        if isinstance(key, list):
            if len(key) == 3:
                dict_save['_'.join(key[:2])] = current_output[key[0]][key[1]][key[2]]
            else:
                dict_save['_'.join(key)] = current_output[key[0]][key[1]]
        else:
            try:
                dict_save[key] = current_output[key]
            except KeyError:
                dict_save[key] = instance.output[key][-1]
    return dict_save

class outputPatternClass:
    def __init__(self, args, params_simu, params_pattern, params_network, light=True):
        self.light = light
        self.params_simu = params_simu
        self.params_pattern = params_pattern
        self.params_network = params_network
        dict_args = vars(args)
        self.name = ''.join(
            [str(dict_args[key]) + '_' for key in dict_args.keys() if not (key == 'name' or key == 'save_dir')])[:-1]
        name_project, name_subproject = args.name.split('/')
        if args.save:
            save_dir = args.save_dir + '/' + name_project + '/'
        else:
            save_dir = '../Simu/' + name_project + '/'

        try:
            os.mkdir(save_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        path = save_dir + name_subproject + '/'
        try:
            os.mkdir(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        self.path = path + '/' + self.name
        try:
            os.mkdir(self.path)
            textfile = open(self.path + '/args.txt', 'w')
            textfile.write(args.__str__()[10:-1])
            textfile.close()
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        self.output = {}
        self.keys_input = ['name', 'score_set', 'output_test']

        if self.params_simu['accuracy_subpattern']:
            self.keys_input += ['score_set_subset']

        if self.light:
            self.keys_input += ['weight_ref']
            self.output['weight_ref'] = outputItem(self, 'iteration', 'weight')

        self.output['name'] = outputItemList(self)
        self.output['score_set'] = outputItemList(self)
        if self.params_simu['accuracy_subpattern']:
            self.output['score_set_subset'] = outputItemList(self)

        self.output['output_test'] = outputItemTest(self, 'iteration', 'weight', 'list', 'list_subpattern')
        self.output['output_test'].add_transform_keys('accuracy', 'accuracy_list', 'pattern_list', 'success')
        if self.params_simu['accuracy_subpattern']:
            self.output['output_test'].add_transform_keys('accuracy_subset', 'accuracy_list_subset',
                                                          'pattern_list_subset', 'success_subset')
        if self.light:
            self.keys_input.append('output_train')
            self.output['output_train'] = outputItemTrain(self, 'iteration', 'iteration_train', 'pattern',
                                                          'stim_recall', 'stim', 'sample_pattern', 'accuracy')
            self.output['output_train'].add_transform_keys('accuracy_train_filtered', 'accuracy_train_interp')

        self.keys_transform = ['n_pattern_list', 'accuracy_mean', 'accuracy_std', 'score_set_mean', 'score_set_std',
                               'success', 'accuracy_max']

        if self.params_simu['accuracy_subpattern']:
            self.keys_transform += ['accuracy_subset_mean', 'score_set_subset_mean', 'score_set_subset_std']

        self.keys_save = ['name', ['output_test', 'iteration', 0], ['output_test', 'accuracy'],
                          ['output_test', 'success'], 'score_set']
        if self.params_simu['accuracy_subpattern']:
            self.keys_save += [['output_test', 'accuracy_subset'], 'score_set_subset']

        for key in self.keys_transform:
            self.output[key] = []

    def create_n_pattern(self, n_pattern):
        self.now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        if os.path.exists(self.path + '/Plot_data_' + str(n_pattern) + '.npy'):
            self.rerun = True
        else:
            self.rerun = False
            self.current_n_pattern = n_pattern
            self.output['n_pattern_list'].append(self.current_n_pattern)
            self.current_output = {}
            for key in self.keys_input:
                self.output[key].create_n_pattern(n_pattern)
                self.current_output[key] = self.output[key].current_output
        print('#'*40)
        print('{:2d} patterns, {}'.format(n_pattern, self.now))
        print('#'*40)

    def add_output(self, output):
        for key in self.keys_input:
            self.output[key].add_to_output_pattern(output[key])

    def end_n_pattern(self):
        for key in self.keys_input:
            self.output[key].end_n_pattern()
        self.transform()
        self.figure_end_pattern()
        print('#'*40)
        for n_pattern, accuracy, current_accuracy_max in zip(self.output['n_pattern_list'],
                                                             self.output['accuracy_mean'],
                                                             self.output['accuracy_max']):
            print('{:2d} patterns --- Accuracy: {:.3f}, Accuracy Max: {:.3f}'.format(n_pattern,
                  accuracy, np.nanmean(current_accuracy_max, axis=0)[-1]))

    def figure_end_pattern(self):
        iteration_test = self.output['output_test'].current_output['iteration'][0]
        if self.light:
            iteration_train = self.output['output_train'].current_output['iteration'][0]
            fig, ax = plt.subplots(3, 1, figsize=(10, 8))
            ax1 = ax[0]
            ax2 = ax[1]
            ax3 = ax[2]
            ax1.plot(iteration_train,
                     np.mean(self.output['output_train'].current_output['accuracy_train_interp'], axis=0),
                     '+-', color='blue', label='accuracy_train')
            ax1.set_xlim(iteration_train[0], iteration_train[-1])
            ax1.set_ylim(0., 1.1)
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            ax2 = ax[0]
            ax3 = ax[1]
        ax2.plot(iteration_test, np.mean(self.output['output_test'].current_output['accuracy'], axis=0),
                 '+-', color='blue', label='accuracy_test')
        ax2.plot(iteration_test, np.nanmean(self.output['accuracy_max'][-1], axis=0),
                 'o-', color='blue', label='accuracy_test_max')
        if self.params_simu['accuracy_subpattern']:
            ax2.plot(iteration_test, np.nanmean(self.output['output_test'].current_output['accuracy_subset'], axis=0),
                     '+-', color='green', label='accuracy_subset_test')
        ax2.plot([iteration_test[0], iteration_test[-1]],
                 [np.nanmean(self.current_output['score_set']), np.nanmean(self.current_output['score_set'])],
                 '--', color='blue', label='logistic_regression_set')
        if self.params_simu['accuracy_subpattern']:
            ax2.plot([iteration_test[0], iteration_test[-1]],
                     [np.nanmean(self.current_output['score_set_subset']),
                      np.nanmean(self.current_output['score_set_subset'])],
                     '--', color='green', label='logistic_regression_subset_set')
        legend_ax = ax2.get_legend_handles_labels()
        ax3.axis('off')
        ax3.legend(*legend_ax, loc='center', title='Test')
        ax2.set_xlim(iteration_test[0], iteration_test[-1])
        ax2.set_ylim(0., 1.1)
        plt.suptitle(self.name + '_' + str(self.current_n_pattern))
        if self.params_simu['save']:
            dict_subset = save_only(self, self.current_output, self.keys_save)
            np.save(self.path + '/Plot_data_' + str(self.current_n_pattern) + '.npy', dict_subset)
        else:
            plt.savefig(self.path + '/Plot_' + str(self.current_n_pattern) + '.pdf')
            plt.show()

    def transform(self):
        self.output['accuracy_mean'].append(np.mean(self.current_output['output_test']['accuracy'][:, -1]))
        self.output['accuracy_std'].append(np.std(self.current_output['output_test']['accuracy'][:, -1]))
        self.output['score_set_mean'].append(np.mean(self.current_output['score_set']))
        self.output['score_set_std'].append(np.std(self.current_output['score_set']))
        self.output['success'].append(np.mean(self.current_output['output_test']['success'][:, -1]))

        self.output['accuracy_max'].append(accuracy_max(self.current_output['output_test']['accuracy'])(10, 1))

        if self.params_simu['accuracy_subpattern']:
            self.output['accuracy_subset_mean'].append(
                np.nanmean(self.current_output['output_test']['accuracy_subset'][:, -1]))
            self.output['score_set_subset_mean'].append(np.nanmean(self.current_output['score_set']))
            self.output['score_set_subset_std'].append(np.nanstd(self.current_output['score_set']))

    def end(self):
        for key in self.keys_transform:
            try:
                self.output[key] = np.array(self.output[key])
            except (ValueError, np.VisibleDeprecationWarning):
                self.output[key] = self.output[key]
        self.figure_end()

    def figure_end(self):
        plt.plot(self.output['n_pattern_list'], self.output['accuracy_mean'],
                 label='accuracy', linestyle='-', marker='+')
        plt.plot(self.output['n_pattern_list'], self.output['score_set_mean'],
                 label='score_set', linestyle='-', marker='+')
        plt.plot(self.output['n_pattern_list'], self.output['success'], label='success', linestyle='-', marker='+')
        plt.title(self.name)
        plt.legend()
        if self.params_simu['save']:
            plt.savefig(self.path + '/Plot.pdf')
            plt.close()
        else:
            plt.show()

class outputPatternDMS(outputPatternClass):
    def __init__(self, args, params_simu, params_pattern, params_network, light=False):
        self.light = light
        self.params_simu = params_simu
        self.params_pattern = params_pattern
        self.params_network = params_network
        dict_args = vars(args)
        self.name = ''.join(
            [str(dict_args[key]) + '_' for key in dict_args.keys() if not (key == 'name' or key == 'save_dir')])[:-1]
        name_project, name_subproject = args.name.split('/')
        if args.save:
            save_dir = args.save_dir + '/' + name_project + '/'
        else:
            save_dir = '../Simu/' + name_project + '/'

        try:
            os.mkdir(save_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        path = save_dir + name_subproject + '/'
        try:
            os.mkdir(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        self.path = path + '/' + self.name
        try:
            os.mkdir(self.path)
            textfile = open(self.path + '/args.txt', 'w')
            textfile.write(args.__str__()[10:-1])
            textfile.close()
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        self.output = {}
        self.keys_input = ['name', 'output_test']
        self.keys_input += ['weight_ref']
        self.output['weight_ref'] = outputItem(self, 'iteration', 'weight')

        self.output['name'] = outputItemList(self)

        self.output['output_test'] = outputItemTest(self, 'iteration', 'weight', 'list', 'list_subpattern')
        self.output['output_test'].add_transform_keys('accuracy', 'accuracy_list', 'pattern_list', 'success')
        self.keys_input.append('output_train')
        self.output['output_train'] = outputItemTrain(self, 'iteration', 'iteration_train', 'pattern', 'stim_recall',
                                                      'stim', 'sample_pattern', 'accuracy')
        self.output['output_train'].add_transform_keys('accuracy_train_filtered', 'accuracy_train_interp')

        self.keys_transform = ['n_pattern_list', 'accuracy_mean', 'accuracy_std', 'success']

        self.keys_save = ['name', ['output_test', 'iteration', 0], ['output_test', 'accuracy'],
                          ['output_test', 'success'], 'norm_weight_diff', 'norm_weight_diff_standard',
                          'scalar_prod_center', 'cosine_sim_center']

        for key in self.keys_transform:
            self.output[key] = []

    def create_n_pattern(self, n_pattern):
        self.rerun = False
        self.now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        print('#'*40)
        print('{:2d} patterns, {}'.format(n_pattern, self.now))
        print('#'*40)
        self.current_n_pattern = n_pattern
        self.output['n_pattern_list'].append(self.current_n_pattern)
        self.current_output = {}
        for key in self.keys_input:
            self.output[key].create_n_pattern(n_pattern)
            self.current_output[key] = self.output[key].current_output

    def end_n_pattern(self):
        for key in self.keys_input:
            self.output[key].end_n_pattern()
        self.transform()
        self.figure_end_pattern()
        print('#'*40)
        for n_pattern, accuracy in zip(self.output['n_pattern_list'], self.output['accuracy_mean']):
            print('{:2d} patterns --- Accuracy: {:.3f}'.format(n_pattern,
                  accuracy))

    def figure_end_pattern(self):
        iteration_test = self.output['output_test'].current_output['iteration'][0]
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        ax2 = ax[0]
        ax3 = ax[-1]
        ax4 = ax[1]
        ax5 = ax[2]
        ax2.plot(iteration_test, np.mean(self.output['output_test'].current_output['accuracy'], axis=0),
                 '+-', color='blue', label='accuracy_test')
        ax4.plot(iteration_test, np.mean(self.current_output['cosine_sim_center'], axis=0),
                 '+-', color='blue', label='cosine_sim_center')
        ax5.plot(iteration_test, np.mean(self.current_output['norm_weight_diff_standard'], axis=0),
                 '+-', color='blue', label='norm_weight_diff_standard')
        legend_ax = ax2.get_legend_handles_labels()
        ax3.axis('off')
        ax3.legend(*legend_ax, loc='center', title='Test')
        figure_different_phases(ax2, self.params_simu, [0., 1.])
        plt.suptitle(self.name + '_' + str(self.current_n_pattern))
        if self.params_simu['save']:
            dict_subset = save_only(self, self.current_output, self.keys_save)
            np.save(self.path + '/Plot_data_' + str(self.current_n_pattern) + '.npy', dict_subset)
        else:
            plt.savefig(self.path + '/Plot_' + str(self.current_n_pattern) + '.pdf')
            plt.show()

    def transform(self):
        weight_ref = np.expand_dims(self.current_output['weight_ref']['weight'][:, 2], axis=1)
        weight_ref_norm = np.linalg.norm(weight_ref, axis=(2, 3))
        norm_weight_diff = np.linalg.norm(self.current_output['output_test']['weight'] - weight_ref, axis=(2, 3))
        self.current_output['norm_weight_diff'] = norm_weight_diff
        self.current_output['norm_weight_diff_standard'] = 1. / (1. + norm_weight_diff/weight_ref_norm)

        weight_center = self.current_output['output_test']['weight'] - \
            np.mean(self.current_output['output_test']['weight'], axis=(2, 3), keepdims=True)
        weight_ref_center = weight_ref - np.mean(weight_ref, axis=(2, 3), keepdims=True)

        scalar_prod_center = np.sum(np.multiply(weight_ref_center, weight_center), axis=(2, 3))
        self.current_output['scalar_prod_center'] = scalar_prod_center
        cosine_sim_center = np.where(np.linalg.norm(weight_center, axis=(2, 3)) == 0., 0., scalar_prod_center / (
                    np.linalg.norm(weight_ref_center, axis=(2, 3)) * np.linalg.norm(weight_center, axis=(2, 3))))
        self.current_output['cosine_sim_center'] = cosine_sim_center

        self.output['accuracy_mean'].append(np.mean(self.current_output['output_test']['accuracy'][:, -1]))
        self.output['accuracy_std'].append(np.std(self.current_output['output_test']['accuracy'][:, -1]))
        self.output['success'].append(np.mean(self.current_output['output_test']['success'][:, -1]))

    def end(self):
        for key in self.keys_transform:
            try:
                self.output[key] = np.array(self.output[key])
            except (ValueError, np.VisibleDeprecationWarning):
                self.output[key] = self.output[key]
        self.figure_end()

    def figure_end(self):
        plt.plot(self.output['n_pattern_list'], self.output['accuracy_mean'],
                 label='accuracy', linestyle='-', marker='+')
        plt.plot(self.output['n_pattern_list'], self.output['success'], label='success', linestyle='-', marker='+')
        plt.title(self.name)
        plt.legend()
        if self.params_simu['save']:
            plt.savefig(self.path + '/Plot.pdf')
            plt.close()
        else:
            plt.show()

class outputPatternExamples(outputPatternClass):
    def __init__(self, args, params_simu, params_pattern, params_network):
        self.params_simu = params_simu
        self.params_pattern = params_pattern
        self.params_network = params_network
        dict_args = vars(args)
        self.name = ''.join(
            [str(dict_args[key]) + '_' for key in dict_args.keys() if not (key == 'name' or key == 'save_dir')])[
                    :-1]
        name_project, name_subproject = args.name.split('/')
        if args.save:
            save_dir = args.save_dir + '/' + name_project + '/'
        else:
            save_dir = '../Simu/' + name_project + '/'

        try:
            os.mkdir(save_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        path = save_dir + name_subproject + '/'
        try:
            os.mkdir(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        self.path = path + '/' + self.name
        try:
            os.mkdir(self.path)
            textfile = open(self.path + '/args.txt', 'w')
            textfile.write(args.__str__()[10:-1])
            textfile.close()
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        self.output = {}
        self.keys_input = ['name', 'output_test']

        self.output['name'] = outputItemList(self)

        self.output['output_test'] = outputItemTest(self, 'iteration', 'weight', 'list', 'list_subpattern')
        self.output['output_test'].add_transform_keys('accuracy', 'accuracy_list', 'pattern_list', 'success')

        self.keys_input.append('output_train')
        self.output['output_train'] = outputItemTrain(self, 'iteration', 'iteration_train', 'pattern',
                                                      'stim_recall', 'stim', 'sample_pattern', 'accuracy')
        self.output['output_train'].add_transform_keys('accuracy_train_filtered', 'accuracy_train_interp')

        self.keys_transform = ['n_pattern_list', 'accuracy_mean', 'accuracy_std', 'success', 'accuracy_max']

        if self.params_pattern['type'] in ['example_B']:
            self.output['output_test'].add_transform_keys('timing_first_spike_diff', 'timing_first_spike')
        self.output['output_test'].weight_transform = True
        self.output['output_test'].add_transform_keys('weight_mean', 'weight_mean_stim', 'weight_mean_no_stim',
                                                      'weight_norm', 'weight_scalar_stim')

        for key in self.keys_transform:
            self.output[key] = []

    def transform(self):
        self.output['accuracy_mean'].append(np.mean(self.current_output['output_test']['accuracy'][:, -1]))
        self.output['accuracy_std'].append(np.std(self.current_output['output_test']['accuracy'][:, -1]))
        self.output['success'].append(np.mean(self.current_output['output_test']['success'][:, -1]))

        self.output['accuracy_max'].append(accuracy_max(self.current_output['output_test']['accuracy'])(10, 1))

    def figure_end_pattern(self):
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        iteration = self.output['output_train'].current_output['iteration'][0]
        for values in ['accuracy_train_interp']:
            values_iteration = np.nanmean(self.output['output_train'].current_output[values], axis=0)
            ax1.plot(iteration, values_iteration, label=values)
        ax1.legend(loc=2)

        if self.params_pattern['type'] in ['example_B']:
            list_values = ['accuracy', 'weight_norm', 'timing_first_spike_diff', 'timing_first_spike']
        else:
            list_values = ['accuracy', 'weight_norm']
        iteration = self.output['output_test'].current_output['iteration'][0]
        for values in list_values:
            values_iteration = np.nanmean(self.output['output_test'].current_output[values], axis=0)
            if values != 'accuracy':
                values_iteration = values_iteration / np.abs(values_iteration[0])
            ax2.plot(iteration, values_iteration, label=values)
        accuracy_max_current = np.nanmean(accuracy_max(self.output['output_test'].current_output['accuracy'])(10, 1),
                                          axis=0)
        ax2.plot(iteration, accuracy_max_current, label='accuracy max')
        ax2.legend(loc=2)
        for ax in [ax1, ax2]:
            ax.axvspan(0, self.params_simu['num_training'], alpha=0.5, color=colors['green'])
            ax.set_xlim(0, self.params_simu['num_training'])
        plt.suptitle(self.name + '_' + str(self.current_n_pattern))
        if self.params_simu['save']:
            with open(self.path + '/' + str(self.current_n_pattern) + '_' + self.now + '.experiment', 'wb') as handle:
                pickle.dump([self.params_simu, self.params_pattern, self.params_network], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            np.save(self.path + '/Plot_data_' + str(self.current_n_pattern) + '.npy', self.current_output)
            plt.savefig(self.path + '/Plot_' + str(self.current_n_pattern) + '.pdf')
            plt.close()
        else:
            plt.savefig(self.path + '/Plot_' + str(self.current_n_pattern) + '.pdf')
            plt.show()

    def figure_end(self):
        pass

class outputItem:
    def __init__(self, parent, *keys):
        self.parent = parent
        self.output = {}
        self.keys_input = list(keys)
        self.keys = list(keys)
        self.keys_transform = []

    def add_input_keys(self, *keys):
        self.keys_input = self.keys_input + list(keys)
        self.keys = self.keys + list(keys)

    def add_transform_keys(self, *keys):
        self.keys_transform = self.keys_transform + list(keys)
        self.keys = self.keys + list(keys)

    def create_n_pattern(self, n_pattern):
        self.output[n_pattern] = {key: [] for key in self.keys}
        self.current_output = self.output[n_pattern]

    def add_to_output_pattern(self, output_iteration):
        for key in self.keys_input:
            self.current_output[key].append([])
        for output in output_iteration:
            for key in self.keys_input:
                self.current_output[key][-1].append(output[key])
        for key in self.keys_input:
            if hasattr(self.current_output[key][-1][0], '__len__'):
                if all(len(item) > 0 for item in self.current_output[key][-1]):
                    if hasattr(self.current_output[key][-1][0], 'shape'):
                        if all(self.current_output[key][-1][0].shape == item.shape
                                for item in self.current_output[key][-1]):
                            self.current_output[key][-1] = np.array(self.current_output[key][-1])
            else:
                self.current_output[key][-1] = np.array(self.current_output[key][-1])
        self.transform()

    def end_n_pattern(self):
        for key in self.keys:
            if hasattr(self.current_output[key][0], 'shape'):
                if all(self.current_output[key][0].shape == item.shape for item in self.current_output[key]):
                    self.current_output[key] = np.array(self.current_output[key])

    def transform(self):
        pass

class outputItemList(outputItem):
    def __init__(self, parent):
        self.parent = parent
        self.output = {}
        self.keys_input = None
        self.keys = None

    def add_transform_keys(self, *keys):
        pass

    def create_n_pattern(self, n_pattern):
        self.output[n_pattern] = []
        self.current_output = self.output[n_pattern]

    def add_to_output_pattern(self, output_iteration):
        self.current_output.append(output_iteration)
        self.transform()

    def end_n_pattern(self):
        self.current_output = np.array(self.current_output)

    def transform(self):
        pass

class outputItemTest(outputItem):
    def __init__(self, *keys):
        outputItem.__init__(self, *keys)
        self.weight_transform = False

    def transform(self):
        output = dict((key, []) for key in self.keys_transform)

        for output_test_list in self.current_output['list'][-1]:
            output['pattern_list'].append([output_test['pattern'] for output_test in output_test_list])
            output['accuracy_list'].append(np.array([output_test['accuracy'] for output_test in output_test_list]))
            output['accuracy'].append(np.mean([output_test['accuracy'] for output_test in output_test_list]))
            output['success'].append(float(int(np.sum(output['accuracy_list'][-1])) == len(output['pattern_list'][-1])))

        if self.parent.params_simu['accuracy_subpattern']:
            for output_test_list in self.current_output['list_subpattern'][-1]:
                if len(output_test_list) > 0:
                    output['pattern_list_subset'].append([output_test['pattern'] for output_test in output_test_list])
                    output['accuracy_list_subset'].append(np.array([output_test['accuracy']
                                                                    for output_test in output_test_list]))
                    output['accuracy_subset'].append(np.mean([output_test['accuracy']
                                                              for output_test in output_test_list]))
                    output['success_subset'].append(float(int(
                        np.sum(output['accuracy_list'][-1])) == len(output['pattern_list'][-1])))
                else:
                    output['pattern_list_subset'].append(np.nan)
                    output['accuracy_list_subset'].append(np.nan)
                    output['accuracy_subset'].append(np.nan)
                    output['success_subset'].append(np.nan)

        if self.parent.params_pattern['type'] in ['example_B']:
            for output_test_list in self.current_output['list'][-1]:
                current_first_spike_diff = []
                current_first_spike = []
                for output_test in output_test_list:
                    if len(output_test['timing_spike'][0]) == 0:
                        first_spike = np.nan
                    else:
                        first_spike = output_test['timing_spike'][0][0]
                    current_first_spike_input = np.nan
                    current_last_spike_input = np.nan
                    for spike_input in output_test['timing_spike_input']:
                        if len(spike_input) > 0:
                            current_first_spike_input = np.nanmin([current_first_spike_input, spike_input[0]])
                            current_last_spike_input = np.nanmax([current_last_spike_input, spike_input[-1]])
                    duration = current_last_spike_input - current_first_spike_input
                    if np.isnan(first_spike):
                        current_first_spike_diff.append(np.nan)
                        current_first_spike.append(1.)
                    else:
                        current_first_spike_diff.append((first_spike - self.parent.params_simu['dt'] -
                                                         current_first_spike_input) / duration)
                        current_first_spike.append(0.)
                output['timing_first_spike_diff'].append(np.mean(current_first_spike_diff))
                output['timing_first_spike'].append(np.mean(current_first_spike))

        if self.weight_transform:
            for weight, output_test_list in zip(self.current_output['weight'][-1], self.current_output['list'][-1]):
                pattern_list = [list(output_test['pattern_index']) for output_test in output_test_list]
                output['weight_mean'].append(np.mean(weight[0]))
                pattern_stim = set(sum(pattern_list, []))
                if len(pattern_stim) > 0:
                    weight_stim = np.array(list(pattern_stim))
                    output['weight_mean_stim'].append(np.mean(weight[0][weight_stim]))
                else:
                    output['weight_mean_stim'].append(np.nan)
                pattern_no_stim = set(np.arange(self.parent.params_network['P']))-pattern_stim
                if len(pattern_no_stim) > 0:
                    weight_no_stim = np.array(list(pattern_no_stim))
                    output['weight_mean_no_stim'].append(np.mean(weight[0][weight_no_stim]))
                else:
                    output['weight_mean_no_stim'].append(np.nan)
                output['weight_norm'].append(np.linalg.norm(weight[0]))
                weight_ref = np.ones(self.parent.params_network['P'], dtype=np.float)
                if len(pattern_no_stim) > 0:
                    weight_ref[np.array(list(pattern_no_stim))] = -1.
                scalar_prod = np.sum(np.multiply(weight_ref, weight[0]))
                output['weight_scalar_stim'].append(scalar_prod)

        for key in output.keys():
            self.current_output[key].append(np.array(output[key]))

class outputItemTrain(outputItem):
    def __init__(self, *keys):
        outputItem.__init__(self, *keys)
        self.params = {
            'convolve': 20
        }

    def transform(self):
        accuracy_iteration = self.current_output['iteration'][-1][self.current_output['stim'][-1] == 1]
        accuracy = self.current_output['accuracy'][-1][self.current_output['stim'][-1] == 1]
        before = accuracy[:int(self.params['convolve'])][::-1]
        after = accuracy[-int(self.params['convolve']):][::-1]
        accuracy_padded = np.concatenate([before, accuracy, after])
        accuracy_filtered = np.zeros(len(accuracy))
        for i in range(len(accuracy)):
            accuracy_filtered[i] = np.mean(accuracy_padded[i:i + 2 * int(self.params['convolve']) + 1])
        self.current_output['accuracy_train_filtered'].append(np.array([accuracy_iteration, accuracy]))
        accuracy_interp = np.interp(self.current_output['iteration'][-1], accuracy_iteration, accuracy_filtered)
        self.current_output['accuracy_train_interp'].append(accuracy_interp)
