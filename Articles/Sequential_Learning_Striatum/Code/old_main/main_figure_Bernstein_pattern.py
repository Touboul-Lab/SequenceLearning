import matplotlib
import matplotlib.pyplot as plt
import os
from Articles.Sequential_Learning_Striatum.Code.FigureSequential import *

name_Bernstein = ['IaF - Symmetric LTP','IaF - Hebbian','IaF - Symmetric LTD','IaF - Anti-Hebbian', 'Nonlinear - AH - No Inhibition', 'Nonlinear - AH - Inhibition']

name_single='single/Bernstein'
name_dual='dual/Bernstein'

P_list=[10]
stim_by_pattern_list=[3,4,5]
epsilon=0.05
homeostasy_single=0.95
homeostasy_dual=3.
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

Bernstein_list = [
[name_single, [1.,1.], homeostasy_single, None],
[name_single, [-1.,1.], homeostasy_single, None],
[name_single, [-1.,-1.], homeostasy_single, None],
[name_single, [1.,-1.], homeostasy_single, None],
[name_dual, [1.,-1.], homeostasy_dual, 0],
[name_dual, [1.,-1.], homeostasy_dual, 2]
]

params_simu = {
    'dt': dt,
    'num_training': num_training,
    'num_simu': num_simu,
}

path_figure = '../Figures/Bernstein/'

if not os.path.exists(path_figure):
    os.makedirs(path_figure)

gridspec_kw = gs_kw = dict(width_ratios=[1], height_ratios=[1])

for stop_learning in stop_learning_list:
    for P in P_list:
        for stim_by_pattern in stim_by_pattern_list:
            n_pattern_list = np.arange(P / 2, 2 * P + P/2, P / 2, dtype=np.int)
            value_analysis = {
                'value_list_accuracy' : np.zeros((len(Bernstein_list), len(n_pattern_list))),
                'value_list_accuracy_std': np.zeros((len(Bernstein_list), len(n_pattern_list)))
            }
            for l, n_pattern in enumerate(n_pattern_list) :
                fig1, ax1 = plt.subplots(figsize=(4.5*6./7. * 22./12., 2.5),ncols=1, nrows=1, gridspec_kw=gs_kw)
                fig1.subplots_adjust(top=1., bottom=0., left=0., right=1.)
                for k, [name, A_STDP, homeostasy, J] in enumerate(Bernstein_list):
                    Apostpre=A_STDP[0]
                    Aprepost=A_STDP[1]
                    if J is None:
                        plot_data = np.load(
                            path_name(name + '/Figure_Single',n_pattern,P,stim_by_pattern,Apostpre,Aprepost,
                            homeostasy,epsilon,noise_stim,noise_input,stop_learning,num_success_params,
                            p_reward,stim_duration,stim_delay,dt,num_training,num_simu,save,random_seed)).item()
                    else:
                        plot_data = np.load(
                            path_name(name + '/Figure_Dual',n_pattern,P,stim_by_pattern,Apostpre,Aprepost,
                            homeostasy,epsilon,noise_stim,noise_input,stop_learning,num_success_params,
                            p_reward,stim_duration,stim_delay,dt,num_training,num_simu,J,save,random_seed)).item()
                    plot_results(plot_data, ax1,params_simu=params_simu, value_analysis=value_analysis, k=k,l=l, params=name_Bernstein)
                ax1.plot([0, params_simu['num_training']], [0.5, 0.5], 'k--')
                ax1.set_xlim(0, params_simu['num_training'])
                ax1.set_ylim(0.3, 1.)
                ax1.set_xlabel('Iterations')
                ax1.set_ylabel('Accuracy')
                ax1.legend(loc=8, ncol=3)
                ax1.patch.set_facecolor(colors['grey'])
                ax1.patch.set_alpha(0.1)
                title=path_name(name,n_pattern,P,stim_by_pattern,Apostpre,Aprepost,
                            homeostasy,epsilon,noise_stim,noise_input,stop_learning,num_success_params,
                            p_reward,stim_duration,stim_delay,dt,num_training,num_simu,J,save,random_seed,title=True)
                fig1.suptitle(
                    'Set-up: $P=' + str(P) + '$ input neurons, $' + str(
                        stim_by_pattern) + '$ stimulations by pattern and $N_p=' + str(
                        n_pattern) + '$ number of patterns',
                    y=-0.15)
                plt.savefig(path_figure+title+str(n_pattern)+'.svg')
                plt.close()
            fig2, ax2 = plt.subplots(figsize=(4.5*6./7. * 22./12., 2.5), ncols=1, nrows=1)
            fig2.subplots_adjust(top=1., bottom=0., left=0., right=1.)
            ax2.bar(n_pattern_list - 5*P/40., value_analysis['value_list_accuracy'][0], yerr=value_analysis['value_list_accuracy_std'][0]/2., color=colors[c_list[0]], width=P/20., label=name_Bernstein[0])
            ax2.bar(n_pattern_list - 3*P/40., value_analysis['value_list_accuracy'][1], yerr=value_analysis['value_list_accuracy_std'][1]/2., color=colors[c_list[1]], width=P/20., label=name_Bernstein[1])
            ax2.bar(n_pattern_list - P/40., value_analysis['value_list_accuracy'][2], yerr=value_analysis['value_list_accuracy_std'][2]/2., color=colors[c_list[2]], width=P/20., label=name_Bernstein[2])
            ax2.bar(n_pattern_list + P/40., value_analysis['value_list_accuracy'][3], yerr=value_analysis['value_list_accuracy_std'][3]/2., color=colors[c_list[3]], width=P/20., label=name_Bernstein[3])
            ax2.bar(n_pattern_list + 3*P / 40., value_analysis['value_list_accuracy'][4], yerr=value_analysis['value_list_accuracy_std'][4]/2., color=colors[c_list[4]],
                    width=P / 20., label=name_Bernstein[4])
            ax2.bar(n_pattern_list + 5*P / 40., value_analysis['value_list_accuracy'][5], yerr=value_analysis['value_list_accuracy_std'][5]/2., color=colors[c_list[5]],
                    width=P / 20., label=name_Bernstein[5])
            ax2.plot([0,n_pattern_list[-1]+P],[0.5,0.5], 'k--')
            ax2.patch.set_facecolor(colors['grey'])
            ax2.patch.set_alpha(0.1)
            ax2.legend()
            ax2.set_xlim(0.,n_pattern_list[-1]+P)
            ax2.set_xticks(n_pattern_list)
            ax2.set_ylim(0.,1.)
            ax2.set_xlabel('$N_p$ = Number of patterns', labelpad=5)
            ax2.set_ylabel('Accuracy')
            fig2.suptitle(
                'Set-up: $P=' + str(P) + '$ input neurons, $' + str(
                    stim_by_pattern) + '$ stimulations by pattern',
                y=-0.15)
            plt.savefig(
                path_figure+'/general_'+title[:-1]+'.svg')
            plt.close()