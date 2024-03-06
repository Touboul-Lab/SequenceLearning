import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Simulator.SimulatorPatternDual import simulatorPatternDual
from FiguresFunctions.FiguresPattern import *
from matplotlib.lines import Line2D

font = {'size'   : 16}

matplotlib.rc('font', **font)

custom_lines = [Line2D([0], [0], color=colors[c_list[k]], lw=4) for k in range(4)]+[Line2D([0], [0], marker=line_style[l][0], color=colors['black']) for l in range(3)]

num=1

if num==0 :
    params_network_0 = {
        'P': 2,
        'homeostasy': 2.5,
        'Apostpre': -0.3,
        'Aprepost': -0.3,
        'J': np.array([[0.,0.],[0.,0.]]),
        'noise_stim': 0.,
        'noise_input': 0.,
        'epsilon': 0.1,
        'save': True,
    }
    params_network_1 = {
        'P': 2,
        'homeostasy': 2.5,
        'Apostpre': -0.3,
        'Aprepost': -0.3,
        'J': np.array([[0.,-0.4],[0.,0.]]),
        'noise_stim': 0.,
        'noise_input': 0.,
        'epsilon': 1.,
        'save': True,
    }

    params_simu = {
        'dt': 0.1,
        'num_training': 100,
        'num_simu': 1,
        'stop_learning': 'number_success',
        'test_iteration': 50,
        'dir': '../Test',
        'save': False,
        'num_spike_output': 3,
        'num_success_params': 0,
    }

    params_pattern = {
        'type': 'succession',
        'n_pattern': 2,
        'delay': 1.,
        'random_time': None,
        'duration': 200.,
        'reward': 2,
        'sample_pattern': False
    }
    SIMULATOR_0 = simulatorPatternDual(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network_0)
    SIMULATOR_0.run(name='Test')
    SIMULATOR_1 = simulatorPatternDual(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network_1)
    SIMULATOR_1.run(name='Test')
    gridspec_kw = gs_kw = dict(hspace=0., wspace=0.)
    fig, ax = plt.subplots(figsize=(6, 16), ncols=2, nrows=3, gridspec_kw=gs_kw)
    for j in np.arange(2):
        for k in np.arange(3):
            ax[k,j].set_xlim(SIMULATOR_0.network.NEURON.time[-1]-310.+j*200.,SIMULATOR_0.network.NEURON.time[-1]-250.+j*200.)
            ax[k,j].spines['right'].set_visible(False)
            ax[k,j].spines['top'].set_visible(False)
            ax[k,j].spines['left'].set_visible(False)
            ax[k,j].spines['bottom'].set_linewidth(2.)
            ax[k,j].set_xticks([])
            ax[k,j].get_xaxis().set_visible(False)
            ax[k,j].get_yaxis().set_visible(False)
        ax[0,j].set_ylim(0., 2.)
        ax[0,j].spines['bottom'].set_color(colors['green'])
        ax[0,j].plot(SIMULATOR_0.network.STIM.time,
                   [SIMULATOR_0.network.STIM.spike_count[i][0] for i in range(len(SIMULATOR_0.network.STIM.spike_count))],
                   color=colors['blue'])
        ax[0,j].plot(SIMULATOR_0.network.STIM.time,
                   [SIMULATOR_0.network.STIM.spike_count[i][1] for i in range(len(SIMULATOR_0.network.STIM.spike_count))],
                   color=colors['red'])
        ax[1,j].set_ylim(-100., 50.)
        ax[1,j].spines['bottom'].set_color(colors['brown'])
        ax[1,j].plot(SIMULATOR_0.network.NEURON.time,
                    [SIMULATOR_0.network.NEURON.potential[i][0]
                     for i in range(len(SIMULATOR_0.network.NEURON.potential))], color=colors['brown'])
        ax[1,j].plot(SIMULATOR_0.network.NEURON.time,
                    [SIMULATOR_0.network.NEURON.potential[i][1]
                     for i in range(len(SIMULATOR_0.network.NEURON.potential))], color=colors['yellow'])
        ax[1,j].patch.set_facecolor(colors['red'])
        ax[1,j].patch.set_alpha(0.1)
        ax[2,j].set_ylim(-100., 50.)
        ax[2,j].spines['bottom'].set_color(colors['brown'])
        ax[2,j].plot(SIMULATOR_1.network.NEURON.time,
                    [SIMULATOR_1.network.NEURON.potential[i][0]
                     for i in range(len(SIMULATOR_1.network.NEURON.potential))], color=colors['brown'])
        ax[2,j].plot(SIMULATOR_1.network.NEURON.time,
                    [SIMULATOR_1.network.NEURON.potential[i][1]
                     for i in range(len(SIMULATOR_1.network.NEURON.potential))], color=colors['yellow'])
        ax[2,j].patch.set_facecolor(colors['brown'])
        ax[2,j].patch.set_alpha(0.1)
        for k in range(3):
            ax[k,j].set_zorder(100 - k)
    plt.tight_layout()
    plt.savefig('../Figures/dual/succession_epsilon/Comparison.pdf')
    plt.show()

elif num==1 :
    name = ['Symmetric LTD', 'Anti-Hebbian', 'Hebbian', 'Symmetric LTP', 'P=10', 'P=20', 'P=30']


    def path_name(Apostpre, Aprepost, P, epsilon, homeostasy, name, noise_input, noise_stim, stim_by_pattern, n_pattern,
                  strategy):
        path = "../Cluster/results_single/" + name + "/Apostpre=" + str(Apostpre) + "_Aprepost=" + str(
            Aprepost) + "_P=" + str(P) + "_epsilon=" + str(epsilon) + "_homeostasy=" + str(
            homeostasy) + "_name='" + name + "'_noise_input=" + str(noise_input) + "_noise_stim=" + str(
            noise_stim) + "_random_seed=0_stim_by_pattern=" + str(stim_by_pattern) + "_stop_learning='" + strategy + "'"
        return path + "/Plot_data_" + str(n_pattern) + ".npy"
    params_simu = {
        'dt': 0.1,
        'num_training': 1000,
        'num_simu': 200,
        'test_iteration': 50,
    }

    homeostasy=0.95
    noise_stim=0.
    noise_input=0.
    P=10
    stim_by_pattern=3
    A = [[-1.0,-1.0],[1.0,-1.0],[-1.0,1.0],[1.0,1.0]]
    strategy = 'None'
    fig1, ax = plt.subplots(figsize=(10, 4), ncols=1, nrows=1, constrained_layout=True)
    for l, n_pattern in enumerate([10,20,30]):
        for k, params in enumerate(A):
            plot_data = np.load(
                path_name(params[0], params[1], P, 0.05, homeostasy, 'pattern_single',
                          noise_input, noise_stim, stim_by_pattern, n_pattern, strategy)).item()
            ax.plot(plot_data['accuracy_iteration_list'][0],
                       np.mean(plot_data['accuracy_list'], axis=0), line_style[l], color=colors[c_list[k]])
    ax.axvspan(0, params_simu['num_training'], alpha=0.1, color=colors['green'])
    ax.set_xlim(0, params_simu['num_training'])
    ax.plot([0,params_simu['num_training']],[0.5,0.5],'k--', linewidth=1)
    ax.legend(custom_lines, name, title='N=10, 3 stim/pattern')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    plt.savefig('../Figures/single/article/Comparison.pdf')
    plt.show()