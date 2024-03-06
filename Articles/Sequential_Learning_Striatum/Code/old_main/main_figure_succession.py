import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from FiguresFunctions.FiguresPattern import *

font = {'size'   : 16}

matplotlib.rc('font', **font)

name = ['Symmetric LTD']
name_J = ['No Inhibition', 'Bilateral Inhibition', '2 -> 1', '1 -> 2']

def path_name(Apostpre,Aprepost,J_matrix,P,epsilon,homeostasy,name,noise_input,noise_stim,strategy):
    path ="../Cluster/results_dual/"+name+"/Apostpre="+str(Apostpre)+"_Aprepost="+str(Aprepost)+"_J_matrix="+str(J_matrix)+"_P="+str(P)+"_epsilon="+str(epsilon)+"_homeostasy="+str(homeostasy)+"_name='"+name+"'_noise_input="+str(noise_input)+"_noise_stim="+str(noise_stim)+"_random_seed=0_stop_learning='"+strategy+"'"
    return path+"/Plot_data_"+str(n_pattern)+".npy"

def str_title(P,n_pattern,homeostasy,noise_input,noise_stim,J,strategy):
    return str(P)+'_'+str(n_pattern)+'_'+str(homeostasy)+'_'+str(noise_input)+'_'+str(noise_stim)+'_'+str(J)+'_'+strategy


num = 0

if num == 0 :
    params_simu = {
        'dt': 0.1,
        'num_training': 1000,
        'test_iteration': 50,
    }
    epsilon=0.01
    #homeostasy=2.5
    P=10
    noise_stim=0.
    noise_input=0.
    J=0
    gridspec_kw = gs_kw = dict(width_ratios=[1], height_ratios=[3,3,3,3])
    A = [[-0.3,-0.3]]

    def plot_results(plot_data,ax,value_list=None,k=0,l=0,params=0.,control=False):
        if not control :
            x_filtered_list = []
            for i in range(len(plot_data['reward_list'])):
                before = plot_data['reward_list'][i][:int(20)][::-1]
                after = plot_data['reward_list'][i][-int(20):][::-1]
                x_padded = np.concatenate([before, plot_data['reward_list'][i], after])
                x_filtered = np.zeros(len(plot_data['reward_list'][i]))
                for j in range(len(plot_data['reward_list'][i])):
                    x_filtered[j] = np.mean(x_padded[j:j + 2 * 20 + 1])
                x_filtered_list.append(np.interp(range(
                    params_simu['num_training']),
                    plot_data['reward_iteration_list'][i], x_filtered))
            ax[0].plot(np.mean(np.array(x_filtered_list), axis=0), color=colors[c_list[k]], label=str(params))
            ax[1].plot(plot_data['accuracy_iteration_list'][0],
                       np.mean(plot_data['accuracy_list'], axis=0), '+-', color=colors[c_list[k]])
        else :
            ax[1].plot(plot_data['accuracy_iteration_list'][0],
                       np.mean(plot_data['accuracy_list'], axis=0), '+-', color=colors['grey'])
        if not control:
            x_iteration = plot_data['accuracy_iteration_list'][0]
            x_scaled = np.mean(plot_data['success_list'], axis=0)
            value_list[k,l] = x_scaled[-1]
            ax[2].plot(x_iteration, x_scaled, '+-', color=colors[c_list[k]], label=name[k])
    for strategy in ['None','number_success','exponential_trace']:
        for homeostasy in [1.5,2.,2.5,3.]:
        #for epsilon in [ 0.01, 0.02, 0.05, 0.1]:
            value_list = np.zeros((4, 8))
            for J in [0, 1, 2, 3]:
                n_pattern_list = np.arange(2,10, dtype=np.int)
                for l, n_pattern in enumerate(n_pattern_list) :
                    try :
                        fig1, ax = plt.subplots(figsize=(10,15),ncols=1, nrows=4, constrained_layout=True, gridspec_kw=gs_kw)
                        for k, params in enumerate(A):
                            plot_data = np.load(
                                path_name(params[0], params[1],J, P, epsilon, homeostasy, 'pattern_succession_homeostasy_espilon',
                                          noise_input, noise_stim, strategy)).item()
                            plot_results(plot_data, ax, value_list=value_list, k=J,l=l, params=params)
                        for j in range(3):
                            ax[j].axvspan(0, params_simu['num_training'], alpha=0.1, color=colors[colors['green']])
                            ax[j].set_xlim(0, params_simu['num_training'])
                    except:
                        pass
                    ax[3].plot(np.arange(4), value_list[:,l], '+-')
                    ax[0].set_ylim(0.4, 1.2)
                    ax[1].set_ylim(0.4, 1.2)
                    ax[2].set_ylim(-0.1, 1.5)
                    ax[0].legend()
                    plt.suptitle(
                        str_title(P, n_pattern, homeostasy, noise_input, noise_stim,J, strategy))
                    plt.close()
            fig2, ax2 = plt.subplots(figsize=(8, 6), ncols=1, nrows=1, constrained_layout=True)
            ax2.bar(n_pattern_list - 0.3, value_list[0], color=colors[c_list[0]], width=0.2, label=name_J[0])
            ax2.bar(n_pattern_list - 0.1, value_list[1], color=colors[c_list[1]], width=0.2, label=name_J[1])
            ax2.bar(n_pattern_list + 0.1, value_list[2], color=colors[c_list[2]], width=0.2, label=name_J[2])
            ax2.bar(n_pattern_list + 0.3, value_list[3], color=colors[c_list[3]], width=0.2, label=name_J[3])
            ax2.legend()
            ax2.plot([1,10], [0.5, 0.5], 'k--')
            ax2.patch.set_facecolor(colors['grey'])
            ax2.patch.set_alpha(0.1)
            ax2.set_xlim(1,7)
            ax2.set_ylim(0,1)
            ax2.set_xlabel('N = Number of neurons')
            ax2.set_ylabel('Fraction of patterns combinations learned')
            plt.savefig(
                '../Figures/dual/succession_homeostasy_epsilon/succession_' + str(homeostasy) +'_'+strategy+ '_success.pdf')
            plt.close()