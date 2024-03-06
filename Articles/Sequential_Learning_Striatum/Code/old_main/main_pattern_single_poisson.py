import sys,os,gc
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import pickle
from Simulator.SimulatorPattern import simulatorPattern
from FiguresFunctions.FiguresPattern import *
import matplotlib.pyplot as plt
import argparse
import itertools
from datetime import datetime

parser = argparse.ArgumentParser(description='Compute the accuracy of the network')
parser.add_argument('--name', type=str,
                    help='name', default='pattern_single_poisson')
parser.add_argument('--P', type=int,
                    help='Number of neurons', default=10)
parser.add_argument('--Apostpre', type=float,
                    help='Apostpre', default=-1.)
parser.add_argument('--Aprepost', type=float,
                    help='Aprepost', default=-1.)
parser.add_argument('--homeostasy', type=float,
                    help='homeostasy', default=0.95)
parser.add_argument('--epsilon', type=float,
                    help='epsilon', default=0.1)
parser.add_argument('--noise_stim', type=float,
                    help='noise_stim', default = None)
parser.add_argument('--noise_input', type=float,
                    help='noise_input', default = None)
parser.add_argument('--stop_learning', type=str,
                    help='stop_learning', default = 'number_success')
parser.add_argument('--num_success_params', type=int,
                    help='num_success_params', default = 0)
parser.add_argument('--p_reward', type=float,
                    help='p_reward', default = 0.5)
parser.add_argument('--stim_duration', type=float,
                    help='stim_duration', default = 100.)
parser.add_argument('--duration_poisson', type=float,
                    help='duration_poisson', default = 10.)
parser.add_argument('--noise_poisson', type=float,
                    help='noise_poisson', default = 0.1)
parser.add_argument('--dt', type=float,
                    help='dt', default = 0.1)
parser.add_argument('--num_training', type=int,
                    help='num_training', default = 1000)
parser.add_argument('--num_simu', type=int,
                    help='num_simu', default = 10)
parser.add_argument('--save', type=bool,
                    help='save', default=False)
parser.add_argument('--random_seed', type=int,
                    help='random_seed', default=0)

args = parser.parse_args()

if args.random_seed is not None :
    np.random.seed(args.random_seed)

dict_args=vars(args)

name_dir=''.join([str(dict_args[key])+'_' for key in dict_args.keys() if not(key=='name')])[:-1]

if args.save :
    name_project, name_subproject = args.name.split('/')
    save_dir = '/scratch/gvignoud/results_single_poisson/'+name_project+'/'+name_subproject+'/'+name_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

else :
    save_dir = '../../../out'

textfile = open(save_dir+'/args.txt', 'w')
textfile.write(args.__str__()[10:-1])
textfile.close()

params_simu = {
    'dt': args.dt,
    'num_training': args.num_training,
    'num_simu': args.num_simu,
    'stop_learning' : args.stop_learning,
    'test_iteration': 50,
    'save' : args.save,
    'num_success_params': args.num_success_params
}

plot_pattern = {
    'accuracy_mean' : [],
    'score_set_mean' : [],
    'score_timing_mean' : [],
    'accuracy_std' : [],
    'score_set_std' : [],
    'score_timing_std' : [],
    'pattern' : [],
    'success' : []
}


params_network = {
    'P': args.P,
    'homeostasy' : args.homeostasy,
    'Apostpre' : args.Apostpre,
    'Aprepost' : args.Aprepost,
    'noise_stim' : args.noise_stim,
    'noise_input': args.noise_input,
    'epsilon' : args.epsilon,
    'save' : not(args.save),
}

if args.noise_input == 0.:
    params_network['noise_input'] = None

if args.noise_stim == 0.:
    params_network['noise_input'] = None

n_pattern_list = np.arange(args.P/2, 5*args.P, args.P/2, dtype=np.int)

for n_pattern in n_pattern_list :
    accuracy_list = []
    accuracy_iteration_list = []
    reward_list = []
    reward_iteration_list = []
    x_filtered_list=[]
    score_set_list = []
    score_timing_list = []
    success_list = []
    params_pattern = {
        'type': 'poisson',
        'n_pattern': n_pattern,
        'duration': args.stim_duration,
        'duration_poisson': args.duration_poisson,
        'noise_poisson': args.noise_poisson,
        'p_reward': args.p_reward,
        'sample_pattern': False
    }
    raw_list = [params_simu, params_pattern, params_network]
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    for i in range(params_simu['num_simu']):
        SIMULATOR = simulatorPattern(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network)
        SIMULATOR.run(name=str(n_pattern)+'_'+str(i)+'_'+dt_string)
        if not(args.save):
            SIMULATOR.plot()
        raw_list.append(SIMULATOR.output)
        reward_list.append(SIMULATOR.output['reward'])
        reward_iteration_list.append(SIMULATOR.output['reward_iteration'])
        accuracy_list.append(SIMULATOR.output['accuracy'])
        accuracy_iteration_list.append(SIMULATOR.output['accuracy_iteration'])
        score_set_list.append(SIMULATOR.output['score_set'])
        score_timing_list.append(SIMULATOR.output['score_timing'])
        success_list.append(SIMULATOR.output['success'])
        del SIMULATOR
        gc.collect()
    plot_data={
        'reward_list' : np.array(reward_list),
        'reward_iteration_list' : np.array(reward_iteration_list),
        'accuracy_list' : np.array(accuracy_list),
        'accuracy_iteration_list' : np.array(accuracy_iteration_list),
        'score_set_list': np.array(score_set_list),
        'score_timing_list': np.array(score_timing_list),
        'success_list': np.array(success_list)
        }
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i in range(params_simu['num_simu']):
        before = plot_data['reward_list'][i][:int(20)][::-1]
        after = plot_data['reward_list'][i][-int(20):][::-1]
        x_padded = np.concatenate([before, plot_data['reward_list'][i], after])
        x_filtered = np.zeros(len(plot_data['reward_list'][i]))
        for j in range(len(plot_data['reward_list'][i])):
            x_filtered[j] = np.mean(x_padded[j:j + 2 * 20 + 1])
        x_filtered_list.append(np.interp(range(
            params_simu['num_training']) , plot_data['reward_iteration_list'][i], x_filtered))
    ax1.plot(np.mean(np.array(x_filtered_list), axis=0))
    ax2.plot(plot_data['accuracy_iteration_list'][0], np.mean(plot_data['accuracy_list'], axis=0), '+-')
    for ax in [ax1,ax2]:
        ax.axvspan(0, params_simu['num_training'], alpha=0.5, color=colors['green'])
        ax.set_xlim(0, params_simu['num_training'])
        ax.set_ylim(0, 1)
    plt.suptitle(name_dir + '_' + str(n_pattern))
    if args.save:
        with open(save_dir + '/' + str(n_pattern) + '_' + dt_string + '.experiment', 'wb') as handle:
            pickle.dump(raw_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(save_dir + '/Plot_data_' + str(n_pattern) + '.npy', plot_data)
        plt.savefig(save_dir + '/Plot_' + str(n_pattern) + '.pdf')
        plt.close()
    else:
        plt.savefig(save_dir + '/Plot_' + str(n_pattern) + '.pdf')
        plt.show()
    plot_pattern['pattern'].append(n_pattern)
    plot_pattern['accuracy_mean'].append(np.mean(plot_data['accuracy_list'][:,-1]))
    plot_pattern['accuracy_std'].append(np.std(plot_data['accuracy_list'][:,-1]))
    plot_pattern['score_set_mean'].append(np.mean(plot_data['score_set_list']))
    plot_pattern['score_set_std'].append(np.std(plot_data['score_set_list']))
    plot_pattern['score_timing_mean'].append(np.mean(plot_data['score_timing_list']))
    plot_pattern['score_timing_std'].append(np.std(plot_data['score_timing_list']))
    plot_pattern['success'].append(np.mean(plot_data['success_list'][:,-1]))
    if not (args.save):
        print('Reward {} : {}'.format(plot_pattern['pattern'],plot_pattern['success']))
plt.plot(plot_pattern['pattern'], plot_pattern['accuracy_mean'],label='accuracy',linestyle='-', marker = '+')
plt.plot(plot_pattern['pattern'], plot_pattern['score_set_mean'], label='score_set',linestyle='-', marker = '+')
plt.plot(plot_pattern['pattern'], plot_pattern['score_timing_mean'], label='score_timing',linestyle='-', marker = '+')
plt.plot(plot_pattern['pattern'], plot_pattern['success'], label='success',linestyle='-', marker = '+')
plt.title(name_dir)
plt.legend()
if args.save :
    plt.savefig(save_dir+'/Plot.pdf')
    plt.close()
    np.save(save_dir + '/Plot_pattern.npy', plot_pattern)
else :
    plt.show()