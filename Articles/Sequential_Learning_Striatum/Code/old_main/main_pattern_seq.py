import sys,os,gc
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import pickle
from Simulator.SimulatorPattern import simulatorPatternSeq
from FiguresFunctions.FiguresPattern import *
import matplotlib.pyplot as plt
import argparse
import itertools
from datetime import datetime

parser = argparse.ArgumentParser(description='Compute the accuracy of the network')
parser.add_argument('--name', type=str,
                    help='name', default='pattern_seq')
parser.add_argument('--P', type=int,
                    help='Number of neurons', default=10)
parser.add_argument('--stim_by_pattern', type=int,
                    help='stim_by_pattern', default=3)
parser.add_argument('--Apostpre', type=float,
                    help='Apostpre', default= -1.0)
parser.add_argument('--Aprepost', type=float,
                    help='Aprepost', default=-1.0)
parser.add_argument('--homeostasy', type=float,
                    help='homeostasy', default= 0.5)
parser.add_argument('--epsilon', type=float,
                    help='epsilon', default=0.1)
parser.add_argument('--noise_pattern', type=float,
                    help='noise_pattern', default = 1.)
parser.add_argument('--noise_stim', type=float,
                    help='noise_stim', default = 0.001)
parser.add_argument('--homeostasy_post', type=float,
                    help='homeostasy_post', default= 0.)
parser.add_argument('--stim_recall', type=float,
                    help='stim_recall', default= 0.)
parser.add_argument('--new_set', type=int,
                    help='new_set', default= 0)
parser.add_argument('--stop_learning', type=str,
                    help='stop_learning', default = 'None')
parser.add_argument('--save', type=bool,
                    help='save', default=False)
parser.add_argument('--random_seed', type=int,
                    help='random_seed', default=None)

args = parser.parse_args()

if args.random_seed is not None :
    np.random.seed(args.random_seed)

name_dir=args.__str__()[10:-1].replace(', ','_').replace('save=True_','')

if args.save :
    save_dir = '/scratch/gvignoud/results_seq/'+args.name+'/'+name_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

else :
    save_dir = '../Test'

params_simu = {
    'dt': 0.1,
    'num_training_learning': 100,
    'num_training_maintenance': 10,
    'num_phases' : 5,
    'num_simu': 50,
    'new_set': args.new_set,
    'stim_recall': args.stim_recall,
    'stop_learning': args.stop_learning,
    'test_iteration': 50,
    'test_iteration_detail' : 5,
    'save' : args.save,
    'num_success_params': 0,
}

params_network = {
    'P': args.P,
    'homeostasy' : args.homeostasy,
    'Apostpre' : args.Apostpre,
    'Aprepost' : args.Aprepost,
    'noise_input': args.P*args.noise_stim,
    'noise_stim': args.noise_stim,
    'epsilon' : args.epsilon,
    'save' : not(args.save),
    'homeostasy_post' : args.homeostasy_post,
}

SETS = list(itertools.combinations(np.arange(args.P),args.stim_by_pattern))

if args.P > 5 :
    n_pattern_list = np.arange(args.P/2, 2*args.P, args.P/2, dtype=np.int)
else :
    n_pattern_list = np.arange(2, len(SETS), dtype=np.int)

for n_pattern in n_pattern_list :
    accuracy_list = []
    accuracy_iteration_list = []
    reward_list = []
    reward_iteration_list = []
    x_filtered_list = []
    score_set_list = []
    params_pattern = {
        'type': 'list_pattern',
        'n_pattern': n_pattern,
        'stim_by_pattern': args.stim_by_pattern,
        'delay': 1.,
        'random_time': None,
        'duration': 100.,
        'p_reward': 0.5,
        'sets': SETS,
        'noise_pattern': args.noise_pattern,
        'sample_pattern': False,
    }
    raw_list = [params_simu, params_pattern, params_network]
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    for i in range(params_simu['num_simu']):
        SIMULATOR = simulatorPatternSeq(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network)
        SIMULATOR.run(name=str(n_pattern)+'_'+str(i)+'_'+dt_string)
        if not(args.save):
            SIMULATOR.plot()
        raw_list.append(SIMULATOR.output)
        reward_list.append(SIMULATOR.output['reward'])
        reward_iteration_list.append(SIMULATOR.output['reward_iteration'])
        accuracy_list.append(SIMULATOR.output['accuracy'])
        accuracy_iteration_list.append(SIMULATOR.output['accuracy_iteration'])
        score_set_list.append(SIMULATOR.output['score_set'])
        del SIMULATOR
        gc.collect()
    plot_data={
        'reward_list' : np.array(reward_list),
        'reward_iteration_list' : np.array(reward_iteration_list),
        'accuracy_list' : np.array(accuracy_list),
        'accuracy_iteration_list' : np.array(accuracy_iteration_list),
        }
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i in range(params_simu['num_simu']):
        before = plot_data['reward_list'][i][:int(20)][::-1]
        after = plot_data['reward_list'][i][-int(20):][::-1]
        x_padded = np.concatenate([before,plot_data['reward_list'][i],after])
        x_filtered = np.zeros(len(plot_data['reward_list'][i]))
        for j in range(len(plot_data['reward_list'][i])):
            x_filtered[j]=np.mean(x_padded[j:j+2*20+1])
        x_filtered_list.append(np.interp(range(params_simu['num_phases'] * (
                params_simu['num_training_learning'] + params_simu[
            'num_training_maintenance']) + params_simu[
                        'num_training_learning']),
                                         plot_data['reward_iteration_list'][i], x_filtered))
    ax1.plot(np.mean(np.array(x_filtered_list),axis=0))
    ax2.plot(plot_data['accuracy_iteration_list'][0],np.mean(plot_data['accuracy_list'],axis=0),'+-')
    for ax in [ax1, ax2]:
        for i in range(params_simu['num_phases']):
            ax.axvspan(i * (params_simu['num_training_learning'] + params_simu['num_training_maintenance']),
                       i * (params_simu['num_training_learning'] + params_simu['num_training_maintenance']) +
                       params_simu['num_training_learning'], alpha=0.5, color=colors['green'])
            ax.axvspan(i * (params_simu['num_training_learning'] + params_simu['num_training_maintenance']) +
                       params_simu['num_training_learning'],
                       (i + 1) * (params_simu['num_training_learning'] + params_simu[
                           'num_training_maintenance']),
                       alpha=0.5, color=colors['yellow'])
        if params_simu['new_set'] == 0:
            ax.axvspan(params_simu['num_phases'] * (params_simu['num_training_learning'] + params_simu[
                'num_training_maintenance']),
                       params_simu['num_phases'] * (
                               params_simu['num_training_learning'] + params_simu[
                           'num_training_maintenance']) + params_simu[
                           'num_training_learning'],
                       alpha=0.5, color=colors['green'])
        elif params_simu['new_set'] == 1:
            ax.axvspan(params_simu['num_phases'] * (params_simu['num_training_learning'] + params_simu[
                'num_training_maintenance']),
                       params_simu['num_phases'] * (
                               params_simu['num_training_learning'] + params_simu[
                           'num_training_maintenance']) + params_simu[
                           'num_training_learning'],
                       alpha=0.5, color=colors['red'])
        ax.set_xlim(0, params_simu['num_phases'] * (
                params_simu['num_training_learning'] + params_simu[
            'num_training_maintenance']) + params_simu[
                        'num_training_learning'])
        ax.set_ylim(0., 1.)
    plt.suptitle(name_dir + '_' + str(n_pattern))
    if args.save :
        with open(save_dir + '/' + str(n_pattern) + '_' + dt_string + '.experiment', 'wb') as handle:
            pickle.dump(raw_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(save_dir + '/Plot_data_' + str(n_pattern) + '.npy', plot_data)
        plt.savefig(save_dir + '/Plot_' + str(n_pattern) + '.pdf')
        plt.close()
    else:
        plt.savefig(save_dir + '/Plot_' + str(n_pattern) + '.pdf')
        plt.show()