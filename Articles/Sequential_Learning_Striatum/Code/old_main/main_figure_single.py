import matplotlib
import matplotlib.pyplot as plt
import os
from Articles.Sequential_Learning_Striatum.Code.FigureSequential import *

font = {'size'   : 16}

matplotlib.rc('font', **font)

name_STDP = ['Symmetric LTD','Anti-Hebbian','Hebbian', 'Symmetric LTP']

name='Bernstein'
P_list=[10,20]
stim_by_pattern_list=[3,4,5]
A_STDP_list = [[-1.0,-1.0],[1.0,-1.0],[-1.0,1.0],[1.0,1.0]]
epsilon=0.05
homeostasy=0.95
noise_stim=0.
noise_input=0.
stop_learning_list=['None','number_success','exponential_trace']
num_success_params=0
p_reward=0.5
stim_duration=50.
stim_delay=1.
dt=0.1
num_training=2000
num_simu=100
save='True'
random_seed=0

params_simu = {
    'dt': dt,
    'num_training': num_training,
    'num_simu': num_simu,
}

path_figure = '../Figures/single/'+name+'/'

if not os.path.exists(path_figure):
    os.makedirs(path_figure)

gridspec_kw = gs_kw = dict(width_ratios=[1], height_ratios=[1,1])

for stop_learning in stop_learning_list:
    for P in P_list:
        for stim_by_pattern in stim_by_pattern_list:
            n_pattern_list = np.arange(P / 2, 3 * P, P / 2, dtype=np.int)
            value_analysis = {
                'value_list_accuracy' : np.zeros((len(A_STDP_list), len(n_pattern_list)))
            }
            for l, n_pattern in enumerate(n_pattern_list) :
                fig1, ax1 = plt.subplots(figsize=(12,12),ncols=1, nrows=2, constrained_layout=True, gridspec_kw=gs_kw)
                for k, A_STDP in enumerate(A_STDP_list):
                    Apostpre=A_STDP[0]
                    Aprepost=A_STDP[1]
                    plot_data = np.load(
                        path_name(name + '/Figure_Single',n_pattern,P,stim_by_pattern,Apostpre,Aprepost,
                        homeostasy,epsilon,noise_stim,noise_input,stop_learning,num_success_params,
                        p_reward,stim_duration,stim_delay,dt,num_training,num_simu,save,random_seed)).item()
                    plot_results(plot_data, ax1[0],params_simu=params_simu, value_analysis=value_analysis, k=k,l=l, params=name_STDP)
                figure_stats(ax1[1],None,value_analysis['value_list_accuracy'][:,l],color=colors['red'],xlabel='Type of STDP',ylabel='Accuracy',label=A_STDP_list)
                title=path_name(name,n_pattern,P,stim_by_pattern,Apostpre,Aprepost,
                            homeostasy,epsilon,noise_stim,noise_input,stop_learning,num_success_params,
                            p_reward,stim_duration,stim_delay,dt,num_training,num_simu,save,random_seed,title=True)
                ax1[0].set_title(title+str(n_pattern), pad=10)
                plt.savefig(path_figure+title+str(n_pattern)+'.pdf')
                plt.close()
            fig2, ax2 = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)
            ax2.bar(n_pattern_list - 3*P/20., value_analysis['value_list_accuracy'][0], color=colors[c_list[0]], width=P/10., label=name_STDP[0])
            ax2.bar(n_pattern_list - P/20., value_analysis['value_list_accuracy'][1], color=colors[c_list[1]], width=P/10., label=name_STDP[1])
            ax2.bar(n_pattern_list + P/20., value_analysis['value_list_accuracy'][2], color=colors[c_list[2]], width=P/10., label=name_STDP[2])
            ax2.bar(n_pattern_list + 3*P/20., value_analysis['value_list_accuracy'][3], color=colors[c_list[3]], width=P/10., label=name_STDP[3])
            ax2.plot([0,n_pattern_list[-1]],[0.5,0.5], 'k--')
            ax2.patch.set_facecolor(colors['grey'])
            ax2.patch.set_alpha(0.1)
            ax2.legend()
            ax2.set_xlim(0.,n_pattern_list[-1])
            ax2.set_ylim(0.,1.)
            ax2.set_xlabel('P = Number of patterns')
            ax2.set_ylabel('Accuracy')
            plt.title(title[:-1],pad=10)
            plt.savefig(
                path_figure+'/general_'+title[:-1]+'.pdf')
            plt.close()