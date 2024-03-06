import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
from Simulator.SimulatorPattern import simulatorPattern
from Simulator.SimulatorPatternDual import simulatorPatternDual
from matplotlib.lines import Line2D
from Articles.Sequential_Learning_Striatum.Code.FigureSequential import *

num=0

if num == 0 :
    A_list = np.array([[1.,-1.],[-1.,1.]])

    def f_STDP(x,Apostpre, Aprepost):
        y = np.zeros(len(x))
        for i in np.arange(len(x)):
            if x[i]>0 :
                y[i]= Aprepost*np.exp(-x[i])
            else :
                y[i] = Apostpre * np.exp(x[i])
        return y
    c_list_figure=[c_list[3],c_list[1]]

    gs_kw = dict(hspace = 0., width_ratios=[1.5,0.5])
    fig, ax = plt.subplots(figsize=(2.,1.2), ncols=2, nrows=1, gridspec_kw=gs_kw)
    fig.subplots_adjust(top=1., bottom=0., left=0., right=1.)

    set_blank_axis(ax[1])

    ax[0].set_zorder(10)

    ax[0].plot([-5., 5.], [0., 0.], 'k--')
    ax[0].plot([0., 0.], [-1., 1.5], 'k--')
    x_1 =np.linspace(0.,5.,1000)[1:]
    x_2 = np.linspace(-5., 0., 1000)[:-1]
    for k, (Apostpre, Aprepost) in enumerate(A_list):
        ax[0].plot(x_1, f_STDP(x_1, Apostpre, Aprepost), color=colors[c_list_figure[k]])
        ax[0].plot(x_2, f_STDP(x_2, Apostpre, Aprepost), color=colors[c_list_figure[k]])
    ax[0].set_xlim(-5.,5.)
    ax[0].patch.set_facecolor(colors['white'])
    ax[0].set_yticks([-1.,0.,1.])
    ax[0].set_ylim(-1.,1.5)
    ax[0].set_xticks([])
    ax[0].set_xlabel('$\Delta t=t_{post}-t_{pre}$')
    ax[0].set_ylabel('$\Delta W$', labelpad=-2)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    legend_elements = [Line2D([0], [0], color=colors[c_list_figure[k]], lw=0.5, label='('+str(Apostpre)+','+str(Aprepost)+')') for k, (Apostpre, Aprepost) in enumerate(A_list)]

    ax[1].legend(handles=legend_elements, loc='center', fontsize=8, title='$(A_{-},A_{+})$', frameon=False, handlelength=1.)
    ax[0].set_title(u'$\Delta W = A_{-}\exp(\Delta t)$, $\Delta t < 0$\n$\Delta W = A_{+}\exp(-\Delta t)$, $\Delta t > 0$', x=0.8, y=-0.6, fontsize=10)
    fig.savefig('../Figures/STDP.svg')

elif num==1 :
    num_training_figure = 6
    params_simu = {
        'dt': 0.1,
        'num_training': 50,
        'num_simu': 1,
        'new_set': 0,
        'stim_recall': 0.5,
        'stop_learning': 'number_success',
        'test_iteration': None,
        'test_iteration_detail': None,
        'save': False,
        'num_success_params': 0,
        'num_spike_output' : 2
    }

    params_network = {
        'P': 3,
        'homeostasy': 3.,
        'Apostpre': 0.,
        'Aprepost': -1.,
        'noise_input': 0.01,
        'noise_stim': 0.02,
        'epsilon': 0.05,
        'save': True,
        'init_weight': ['uniform', [0.05,0.1]],
    }

    params_network_0 = {
        'P': 3,
        'homeostasy': 4.,
        'Apostpre': 0.,
        'Aprepost': -1.,
        'J': np.array([[0., 0.], [0., 0.]]),
        'noise_stim': 0.01,
        'noise_input': 0.02,
        'epsilon': 0.05,
        'save': True,
        'init_weight': ['uniform', [0.05, 0.1]],
    }
    params_network_1 = {
        'P': 3,
        'homeostasy': 4.,
        'Apostpre': 0.,
        'Aprepost': -1.,
        'J': np.array([[0., -0.4], [0., 0.]]),
        'noise_stim': 0.01,
        'noise_input': 0.02,
        'epsilon': 0.05,
        'save': True,
        'init_weight': ['uniform', [0.05, 0.1]],
    }

    SETS = list(itertools.combinations(np.arange(4),3))
    params_pattern = {
            'type': 'succession',
            'n_pattern': 2,
            'delay': 1.,
            'random_time': None,
            'duration': 50.,
            'reward': 2,
            'sample_pattern': False,
        }

    np.random.seed(35)
    SIMULATOR = simulatorPattern(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network)
    np.random.seed(50)
    SIMULATOR.init_pattern()
    SIMULATOR.run(name='Test')

    np.random.seed(35)
    SIMULATOR_0 = simulatorPatternDual(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network_0)
    np.random.seed(50)
    SIMULATOR_0.init_pattern()
    SIMULATOR_0.run(name='Test')

    np.random.seed(35)
    SIMULATOR_1 = simulatorPatternDual(params_simu=params_simu, params_pattern=params_pattern, params_network=params_network_1)
    np.random.seed(50)
    SIMULATOR_1.init_pattern()
    SIMULATOR_1.run(name='Test')

    x_0 = 0.
    x_1 = params_pattern['n_pattern'] * params_pattern['duration']
    x_2 = x_1 + num_training_figure * params_pattern['duration']
    x_2_ = x_2 + params_pattern['duration']
    x_3 = x_1 + params_simu['num_training'] * params_pattern['duration']
    x_4 = x_1 + params_simu['num_training']* params_pattern['duration'] + params_pattern['n_pattern'] * params_pattern['duration']

    gs_kw = dict(hspace=0., wspace=0., height_ratios=[0.5, 0.5, 1., 1., 1., 1., 1., 1., 1.], width_ratios = [2., 6., 1., 2.])
    fig, ax = plt.subplots(figsize=(10., 6.), ncols=4, nrows=9, gridspec_kw=gs_kw)
    fig.subplots_adjust(top=1., bottom=0., left=0., right=1.)

    gs = ax[0,0].get_gridspec()

    ax_legend_spike = set_subplot_legend(fig, gs[2:5,0], u'$P$ cortical\ninput neurons\n+ Random noise $\lambda_{stim}$')
    ax_legend_output = set_subplot_legend(fig, gs[6:,0], u'Different output\nneurons (Vm)')
    ax_legend_phase = set_subplot_legend(fig, gs[:2, 0], u'Phases\n$N_p$ patterns')

    ax[0,1].remove()
    ax[0,2].remove()
    ax_legend_training = fig.add_subplot(gs[0, 1:3])
    ax_legend_training.set_xticks([])
    ax_legend_training.set_yticks([])

    ax_legend_training.set_zorder(200)

    for l in [0, 1, 2, 3]:
        for k in np.arange(9):
            if k > 1 :
                ax[k,l].spines['right'].set_visible(False)
                ax[k,l].spines['top'].set_visible(False)
            ax[k,l].spines['left'].set_zorder(10)
            ax[k,l].set_xticks([])
            ax[k,l].set_yticks([])
            if l == 0:
                ax[k,l].set_xlim(x_0, x_1)
            elif l == 1:
                if k > 0:
                    ax[k,l].set_xlim(x_1, x_2)
                else:
                    ax_legend_training.set_xlim(x_1, x_2_)
            elif l == 2 :
                if k > 0:
                    ax[k,l].set_xlim(x_2, x_2_)
            elif l == 3:
                ax[k,l].set_xlim(x_3, x_4)
        for k in [2,3,4]:
            ax[k,l].set_ylim(0., 2.)
            if l == 2 :
                ax[k, l].spines['bottom'].set_color(colors['green'])
                ax[k, l].spines['bottom'].set_linestyle('--')
                ax[k, l].spines['left'].set_visible(False)
            else:
                ax[k,l].spines['bottom'].set_color(colors['green'])
                ax[k,l].plot(SIMULATOR.network.STIM.time, [SIMULATOR.network.STIM.spike_count[i][k - 2] for i in
                                                         range(len(SIMULATOR.network.STIM.spike_count))], color=colors['green'])

    for k in [2, 3, 4]:
        for i in range(len(SIMULATOR.stim_iteration)):
            if SIMULATOR.stim_iteration[i] == 1:
                if SIMULATOR.pattern_list[SIMULATOR.pattern_iteration[i]].reward > 0:
                    ax[k, 1].axvspan(x_1 + (i + 0.5) * params_pattern['duration'] - 3.,
                                     x_1 + (i + 0.5) * params_pattern['duration'] + 3.,
                                     facecolor=colors['pale purple'], edgecolor=colors['pale purple'], linewidth=0.5)
                else:
                    ax[k, 1].axvspan(x_1 + (i + 0.5) * params_pattern['duration'] - 3.,
                                     x_1 + (i + 0.5) * params_pattern['duration'] + 3.,
                                     facecolor=colors['white'], edgecolor=colors['pale purple'], linewidth=0.5)

    for num_net, simul in enumerate([SIMULATOR, SIMULATOR_0, SIMULATOR_1]):
        for j in np.arange(params_pattern['n_pattern']):
            if SIMULATOR.accuracy_test_list[0][j] > 0:
                ax[num_net+6,0].axvspan(j * params_pattern['duration'],
                                 (j + 1.) * params_pattern['duration'], ymin=0., ymax=0.1, facecolor=colors['green'])
            else:
                ax[num_net+6,0].axvspan(j * params_pattern['duration'],
                                  (j + 1) * params_pattern['duration'] + 3., ymin=0., ymax=0.1, facecolor=colors['red'])

        for j in np.arange(params_pattern['n_pattern']):
            if simul.accuracy_test_list[1][j] > 0:
                ax[num_net+6,3].axvspan(x_1+(len(simul.stim_iteration) + j) * params_pattern['duration'],
                                  x_1+(len(simul.stim_iteration) + j + 1.) * params_pattern['duration'], ymin=0., ymax=0.1, facecolor=colors['green'])
            else:
                ax[num_net+6,3].axvspan(x_1+(len(simul.stim_iteration) + j ) * params_pattern['duration'],
                                  x_1+(len(simul.stim_iteration) + j + 1) * params_pattern['duration'] + 3., ymin=0., ymax=0.1, facecolor=colors['red'])

        ax[num_net+6, 2].spines['bottom'].set_color(colors['brown'])
        ax[num_net+6, 2].spines['bottom'].set_linestyle('--')
        ax[num_net+6, 2].spines['left'].set_visible(False)

        for l in [0, 1, 3]:
            ax[num_net+6,l].set_ylim(-150., 50.)
            ax[num_net+6,l].spines['bottom'].set_color(colors['brown'])
            if num_net == 0:
                ax[num_net+6,l].plot(simul.network.NEURON.time,
                           [simul.network.NEURON.potential[i][0] + simul.network.NEURON.spike_count[i][0] * (
                                   simul.network.NEURON.parameters['E_reset'] - simul.network.NEURON.parameters[
                               'V_th'])
                            for i in range(len(simul.network.NEURON.potential))], color=colors['brown'])
            else:
                ax[num_net+6, 2].spines['bottom'].set_color(colors['brown'])
                ax[num_net+6, 2].spines['bottom'].set_linestyle('--')
                ax[num_net+6, 2].spines['left'].set_visible(False)
                for l in [0, 1, 3]:
                    ax[num_net+6, l].set_ylim(-150., 50.)
                    ax[num_net+6, l].spines['bottom'].set_color(colors['brown'])
                    ax[num_net+6, l].plot(simul.network.NEURON.time,
                               [simul.network.NEURON.potential[i][0] for i in range(len(simul.network.NEURON.potential))], color=colors['brown'])
                    ax[num_net+6, l].plot(simul.network.NEURON.time,
                               [simul.network.NEURON.potential[i][1] for i in range(len(simul.network.NEURON.potential))], color=colors['orange'], alpha=0.4)

    ax[5, 2].spines['bottom'].set_color(colors['yellow'])
    ax[5, 2].spines['bottom'].set_linestyle('--')
    ax[5, 2].spines['left'].set_visible(False)
    for l in [0, 1, 3]:
        ax[5,l].set_ylim(0., 2.)
        ax[5,l].spines['bottom'].set_color(colors['yellow'])
        ax[5,l].plot(SIMULATOR.network.RANDOM_INPUT.time,
                   [SIMULATOR.network.RANDOM_INPUT.spike_count[i][0] for i in range(len(SIMULATOR.network.RANDOM_INPUT.spike_count))],
                   color=colors['yellow'])
    ax[5,0].set_ylabel(u'Random input\n$\lambda_{input}$', labelpad=5)

    for i in np.arange(num_training_figure):
        ax[1, 1].set_ylim(0., 2.)
        if SIMULATOR.stim_iteration[i] == 1:
            if SIMULATOR.pattern_iteration[i] == 0:
                pattern_name = 'A (+)'
                ax[1, 1].add_patch(
                    patches.Rectangle((x_1 + i * params_pattern['duration'], 0.), params_pattern['duration'],
                                      2.,
                                      facecolor=colors['pale purple'], edgecolor=colors['black'],
                                      linewidth=0.5))
            elif SIMULATOR.pattern_iteration[i] == 1:
                pattern_name = 'B (-)'
                ax[1, 1].add_patch(
                    patches.Rectangle((x_1 + i * params_pattern['duration'], 0.), params_pattern['duration'],
                                      2.,
                                      facecolor=colors['white'], edgecolor=colors['black'],
                                      linewidth=0.5))
            ax[1, 1].text(x_1 + (i + 0.5) * params_pattern['duration'], 1., pattern_name,
                              horizontalalignment='center', verticalalignment='center', fontsize=12)

    ax[1, 2].set_ylim(0., 2.)
    pattern_name = '...'
    ax[1, 2].add_patch(
        patches.Rectangle((x_2, 0.),
                          params_pattern['duration'],
                          2.,
                          facecolor=colors['white'], edgecolor=colors['black'],
                          linewidth=0.5))
    ax[1, 2].text(x_2 + (0.5) * params_pattern['duration'], 1., pattern_name,
                  horizontalalignment='center', verticalalignment='center', fontsize=12)

    for l in [0,3] :
        if l == 0:
            x_init= x_0
        elif l == 3:
            x_init = x_3
        ax[1, l].set_ylim(0., 2.)
        pattern_name = 'A (+)'
        ax[1, l].add_patch(
            patches.Rectangle((x_init, 0.),
                              params_pattern['duration'],
                              2.,
                              facecolor=colors['pale purple'], edgecolor=colors['black'],
                              linewidth=0.5))
        ax[1, l].text(x_init + (0 + 0.5) * params_pattern['duration'], 1., pattern_name,
                      horizontalalignment='center', verticalalignment='center', fontsize=12)

        pattern_name = 'B (-)'
        ax[1, l].add_patch(
            patches.Rectangle((x_init+params_pattern['duration'], 0.),
                              params_pattern['duration'],
                              2.,
                              facecolor=colors['white'], edgecolor=colors['black'],
                              linewidth=0.5))
        ax[1, l].text(x_init + (1 + 0.5) * params_pattern['duration'], 1., pattern_name,
                      horizontalalignment='center', verticalalignment='center', fontsize=12)

    ax[0,0].set_ylim(0., 2.)
    ax[0,0].add_patch(patches.Rectangle((x_0, 0.), x_1 - x_0, 2.,
                                         facecolor=colors['white'], edgecolor=colors['black'],
                                         linewidth=0.5))
    ax_legend_training.set_ylim(0., 2.)
    ax_legend_training.add_patch(patches.Rectangle((x_1, 0.), x_2_ - x_1, 2.,
                                         facecolor=colors['pale purple'], edgecolor=colors['black'],
                                         linewidth=0.5))
    ax[0,3].set_ylim(0., 2.)
    ax[0,3].add_patch(patches.Rectangle((x_3, 0.), x_4 - x_3, 2.,
                                         facecolor=colors['white'], edgecolor=colors['black'],
                                         linewidth=0.5))

    ax[0,0].text((x_0 + x_1) / 2., 1., u'Test Before',
               horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax_legend_training.text((x_1 + x_2_) / 2., 1., u'Learning',
                  horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax[0,3].text((x_3 + x_4) / 2., 1., u'Test After',
               horizontalalignment='center', verticalalignment='center', fontsize=12)

    for l in [0,1,2,3]:
        for k in np.arange(9):
            ax[k,l].set_zorder(100-k)

    ax[8,1].set_xlabel('Iterations', labelpad=5)
    fig.savefig('../Figures/Setup.svg')