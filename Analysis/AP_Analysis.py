from Analysis.Filter.filter import filter_trace, savgol_filter
from InitFunctions import dirac, gaussian
from scipy.optimize import curve_fit
from Analysis.dataloader import dataloader_AP
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, pearsonr
from scipy.signal import find_peaks
import os
import shutil

import warnings
from scipy.optimize import OptimizeWarning

from Articles.Sequential_Learning_Striatum.Figures.FigureSequential import *
from NeuronModel import integrate_and_fire, integrate_and_fire_MSN

np.seterr(over='ignore', invalid='ignore')
warnings.simplefilter('ignore', OptimizeWarning)

dict_neuronModel = dict(IAF=integrate_and_fire, IAF_MSN=integrate_and_fire_MSN)

y_label = ['mV', 'MOhm', 'ms', 'microF', 'UA']

def step_AP(t, beg, end, I_value):
    if t < beg or t > end:
        return 0
    else:
        return I_value

def R_func(R_fast):
    def func(x, tau, t, R_slow):
        return R_slow + (R_fast - R_slow) / (1. + np.exp((x-t)/tau))
    return func

def exp_tau(baselineS, baselineV):
    def func(x, tau, t):
        return np.where(x < t, baselineS, baselineV + (baselineS - baselineV) * np.exp(-(x-t)/tau))
    return func

def firing_rate(I_value, params):
    d = np.where(I_value > -1000.*(params['E_l'] - params['V_th']) / params['R_rate'], 1000. / (
                params['Delta_abs'] + params['R_rate'] * params['C'] * np.log(
                    (I_value + 1000.*(params['E_l'] - params['E_r']) / params['R_rate']) / (
                            I_value + 1000.*(params['E_l'] - params['V_th']) / params['R_rate']))), 0.)
    return d

def find_percentile_d(x, data):
    sol = 0
    while sol < len(data) and data[sol]-x > 0:
        sol += 1
    if sol == len(data):
        sol = np.argmin(np.abs(data-x))
    return sol

def add(dict1, dict2):
    for name in dict2.keys():
        if name in dict1.keys():
            dict1[name].append(dict2[name])

class AP_analysis:
    def __init__(self, x, experimentalist, path='', name='', trace=0, num_neuron=0, plot=False):
        self.x = x
        self.experimentalist = experimentalist
        self.freq = int(1/(x[1, 0]-x[0, 0]))

        self.path = path
        self.name = name
        self.trace = trace
        self.parameters = {}
        if plot:
            plt.plot(self.x[:, 0], self.x[:, 1])
            plt.show()
        self.lag_protocol = int(0.005 * self.freq)
        if self.experimentalist in ['Merie']:
            if num_neuron == 0:
                self.beg_protocol = int(0.1 * self.freq)
                self.end_protocol = int(0.6 * self.freq)
            elif num_neuron == 1:
                self.beg_protocol = int(1. * self.freq)
                self.end_protocol = int(1.5 * self.freq)
        elif self.experimentalist in ['Elodie', 'Willy']:
            if self.x[-1, 0] > 1.05:
                self.beg_protocol = int(0.3 * self.freq)
                self.end_protocol = int(0.8 * self.freq)
            else:
                self.beg_protocol = int(0.25 * self.freq)
                self.end_protocol = int(0.75 * self.freq)

        self.x_1 = np.concatenate([self.x[:self.beg_protocol + self.lag_protocol],
                                   self.x[self.end_protocol - self.lag_protocol:]])
        self.x_2 = self.x[self.beg_protocol - self.lag_protocol:self.end_protocol + self.lag_protocol]

    def compute_kde(self):
        xvals_1 = np.linspace(self.x_1[:, 1].min(), self.x_1[:, 1].max(), 300)
        f_1 = gaussian_kde(self.x_1[:, 1])
        y_1 = f_1(xvals_1)
        xvals_2 = np.linspace(self.x_2[:, 1].min(), self.x_2[:, 1].max(), 300)
        f_2 = gaussian_kde(self.x_2[:, 1])
        y_2 = f_2(xvals_2)
        return xvals_1, y_1, xvals_2, y_2

    def find_RI(self):
        self.kde = self.compute_kde()
        self.ind_kde_1 = np.argmax(self.kde[1])
        self.parameters['baselineS'] = self.kde[0][self.ind_kde_1]
        self.ind_kde_2 = np.argmax(self.kde[3])
        self.parameters['baselineV'] = self.kde[2][self.ind_kde_2]
        popt, pcov = curve_fit(exp_tau(self.parameters['baselineS'], self.parameters['baselineV']),
                               self.x[self.beg_protocol - 2*self.lag_protocol:
                                      self.end_protocol - 2*self.lag_protocol, 0],
                               self.x[self.beg_protocol - 2*self.lag_protocol:
                                      self.end_protocol - 2*self.lag_protocol, 1],
                               p0=(0.001, self.x[self.beg_protocol, 0]), maxfev=100000)

        self.parameters['tau_RC'] = popt[0]

    def find_SAG(self):
        self.parameters['SAG'] = None
        self.parameters['SAG_loc'] = None
        if self.parameters['baselineS'] > self.parameters['baselineV']:
            FILTER_CHUNG = filter_trace(range_filter=20, type_filter='mean')
            self.x_filtered_chung = FILTER_CHUNG.compute(self.x_2)
            self.parameters['SAG'] = np.min(self.x_filtered_chung[:, 1])-self.parameters['baselineV']
            self.parameters['SAG_loc'] = np.argmin(self.x_filtered_chung[:, 1])

    def find_AP(self):
        self.parameters['phase_plane'] = []
        self.parameters['thrV'] = []
        self.parameters['amplitude'] = []
        self.parameters['half_width'] = []
        self.parameters['rise_time'] = []
        self.parameters['decay_time'] = []
        self.parameters['ratio_time'] = []
        self.parameters['after_depol'] = []
        self.parameters['AP_rise'] = []
        self.parameters['AP_decay'] = []
        self.parameters['AP_max'] = []
        self.parameters['AP_min'] = []
        if self.experimentalist == 'Willy':
            self.APs = find_peaks(self.x_2[:, 1], prominence=0.01, distance=30, height=0.)[
                            0] + self.beg_protocol - self.lag_protocol
        else:
            self.APs = find_peaks(self.x_2[:, 1], prominence=0.035, distance=30)[
                            0] + self.beg_protocol - self.lag_protocol
        self.parameters['num_spike'] = len(self.APs)
        if self.parameters['num_spike'] > 0:
            for p in range(self.parameters['num_spike']):
                self.properties_spike(p)

    def properties_spike(self, p):
        AP_beg = self.APs[p] - int(0.005 * self.freq)
        AP_end = self.APs[p] + int(0.005 * self.freq)
        SAVGOL = savgol_filter(range_filter=9, normalize=True)
        spike_derived = SAVGOL.compute(self.x)[AP_beg:AP_end, 2]
        rise_list = np.argwhere(np.diff(np.sign(spike_derived - 5.))).flatten()
        if len(rise_list) > 0:
            AP_rise = rise_list[0] + AP_beg
        else:
            AP_rise = AP_beg
        thrV = self.x[AP_rise, 1]
        self.parameters['thrV'].append(thrV)
        self.parameters['AP_rise'].append(AP_rise)
        argmax_AP = np.argmax(self.x[AP_rise:AP_end, 1])
        max_AP = np.max(self.x[AP_rise:AP_end, 1])
        self.parameters['AP_max'].append(max_AP)
        self.parameters['AP_min'].append(np.min(self.x[AP_rise:AP_end, 1]))
        self.parameters['amplitude'].append(max_AP-thrV)

        AP_decay = np.argmin(np.abs(self.x[argmax_AP+AP_rise:AP_end, 1] - thrV)) + argmax_AP + AP_rise
        self.parameters['AP_decay'].append(AP_decay)

        self.parameters['phase_plane'].append(np.array([self.x[AP_beg:AP_end, 1], spike_derived]))
        if AP_rise < AP_decay:
            spike = self.x[AP_rise:AP_decay]
            dt = 0.00001
            x_vals = np.arange(spike[0, 0], spike[-1, 0] + dt, dt)
            spike_val = np.array([x_vals, np.interp(x_vals, spike[:, 0], spike[:, 1])]).transpose()

            duration = len(spike_val[:, 0])
            percentile = np.array([0.05, 0.5, 0.95]) * (max_AP - thrV) + thrV
            arg_percentile = np.zeros(6, dtype=np.int16)
            argmax_val = np.argmax(spike_val[:, 1])
            if argmax_val == 0:
                self.parameters['half_width'].append(np.nan)
                self.parameters['rise_time'].append(np.nan)
                self.parameters['decay_time'].append(np.nan)
                self.parameters['ratio_time'].append(np.nan)
            else:
                for i in range(3):
                    arg_percentile[i] = np.argmin(np.abs(spike_val[:argmax_val, 1] - percentile[i]))
                    arg_percentile[5-i] = duration-np.argmin(np.abs(spike_val[argmax_val:][::-1, 1] - percentile[i]))-1
                self.parameters['half_width'].append(spike_val[arg_percentile[4], 0]-spike_val[arg_percentile[1], 0])
                self.parameters['rise_time'].append(spike_val[arg_percentile[2], 0]-spike_val[arg_percentile[0], 0])
                self.parameters['decay_time'].append(spike_val[arg_percentile[5], 0] - spike_val[arg_percentile[3], 0])
                if self.parameters['rise_time'][-1] > 0.:
                    self.parameters['ratio_time'].append(
                        self.parameters['decay_time'][-1]/self.parameters['rise_time'][-1])
                else:
                    self.parameters['ratio_time'].append(np.nan)
        else:
            self.parameters['half_width'].append(np.nan)
            self.parameters['rise_time'].append(np.nan)
            self.parameters['decay_time'].append(np.nan)
            self.parameters['ratio_time'].append(np.nan)

    def compute_ISI(self):
        if self.parameters['num_spike'] > 1:
            self.parameters['ISI'] = np.mean([self.x[self.APs[i+1], 0] -
                                              self.x[self.APs[i], 0] for i in range(self.parameters['num_spike']-1)])
        else:
            self.parameters['ISI'] = np.inf

    def plot(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(self.x[:, 0], self.x[:, 1], 'b-', label='V', linewidth=0.2)
        if self.parameters['baselineS'] > self.parameters['baselineV']:
            ax.plot(self.x_filtered_chung[:, 0], self.x_filtered_chung[:, 1], 'g--', linewidth=0.2)
            ax.plot([self.x_filtered_chung[self.parameters['SAG_loc'], 0]]*2,
                    [self.x_filtered_chung[self.parameters['SAG_loc'], 1],
                     self.x_filtered_chung[self.parameters['SAG_loc'], 1] - self.parameters['SAG']], 'r-')
        for p in range(self.parameters['num_spike']):
            ax.plot(self.x[self.parameters['AP_rise'][p], 0],
                    self.x[self.parameters['AP_rise'][p], 1], 'ro')
            ax.plot(self.x[self.parameters['AP_rise'][p]:self.parameters['AP_decay'][p], 0],
                    self.x[self.parameters['AP_rise'][p]:self.parameters['AP_decay'][p], 1], 'r-', linewidth=0.5)
        ax.plot(self.x[self.APs, 0], self.x[self.APs, 1], '*r', label='V')
        ax.plot(ax.get_xlim(), [self.parameters['baselineS']]*2, ':k',  label='baselineS', linewidth=0.5)
        ax.plot(ax.get_xlim(), [self.parameters['baselineV']]*2, '--k', label='baselineV', linewidth=0.5)
        ax.legend()
        self.parameters['plot'] = fig

class AP_protocol:
    def __init__(self, path, file, duration, step_value, min_value, lag,  experimentalist, path_tot, num_neuron=0):
        self.path = path
        self.num_neuron = num_neuron
        self.DATALOADER = dataloader_AP(path_tot, num_neuron=self.num_neuron)
        self.length = self.DATALOADER.__len__()

        self.poop = self.path + file + '_' + str(self.num_neuron+1) + '/'
        if os.path.isdir(self.poop):
            shutil.rmtree(self.poop)
        os.mkdir(self.poop)

        self.rheobase_ok = False
        self.first_spike = False
        self.PARAMETERS_LIST = {
            'file': file,
            'duration': duration,
            'step': step_value,
            'min': min_value,
            'lag': lag,
            'experimentalist': experimentalist,
            'ISI': [],
            'SAG': [],
            'half_width': [],
            'rise_time': [],
            'decay_time': [],
            'num_spike': [],
            'ratio_time': [],
            'thrV': [],
            'phase_plane': [],
            'diff_V': [],
            'rate_ISI': [],
            'rate': [],
            'I': [],
            'baselineS': [],
            'baselineV': [],
            'tau_RC': [],
            'amplitude': [],
            'AP_max': [],
            'AP_min': [],
            'adaptation_ampl': [],
            'adaptation_max': [],
            'rheobase': None,
            'plot': [],

        }
        self.current = np.arange(self.PARAMETERS_LIST['min']-lag,
                                 self.PARAMETERS_LIST['min'] + self.PARAMETERS_LIST['step'] * self.length - lag,
                                 self.PARAMETERS_LIST['step'])

    def iterate(self):
        self.x = []
        for p in range(self.length):
            current = self.current[p]
            x = self.DATALOADER.__getitem__(p)
            self.x.append(x)
            ANALYSIS = AP_analysis(x, self.PARAMETERS_LIST['experimentalist'], plot=False, num_neuron=self.num_neuron)
            ANALYSIS.compute_kde()
            ANALYSIS.find_RI()
            ANALYSIS.find_SAG()
            ANALYSIS.find_AP()
            ANALYSIS.compute_ISI()
            ANALYSIS.plot()
            PARAMETERS = ANALYSIS.parameters
            add(self.PARAMETERS_LIST, PARAMETERS)
            self.PARAMETERS_LIST['I'].append(current)
            self.PARAMETERS_LIST['rate_ISI'].append(1 / PARAMETERS['ISI'])
            self.PARAMETERS_LIST['rate'].append(PARAMETERS['num_spike'] / self.PARAMETERS_LIST['duration'])
            self.PARAMETERS_LIST['diff_V'].append(PARAMETERS['baselineV'] - PARAMETERS['baselineS'])
            if self.first_spike and (not self.rheobase_ok):
                if PARAMETERS['num_spike'] > 0:
                    self.PARAMETERS_LIST['rheobase'] = self.PARAMETERS_LIST['I'][-2]
                    self.p_rheobase = p - 1
                    self.rheobase_ok = True
                else:
                    self.first_spike = False
            elif (not self.first_spike) and PARAMETERS['num_spike'] > 0:
                self.first_spike = True

            PARAMETERS['plot'].savefig(self.poop + 'AP_' + str(p) + '.pdf')
            plt.close(PARAMETERS['plot'])

        self.dt = self.x[0][1, 0] - self.x[0][0, 0]
        self.freq = 1 / self.dt
        if self.PARAMETERS_LIST['experimentalist'] in ['Merie']:
            if self.num_neuron == 0:
                self.beg_protocol = int(0.1 * self.freq)
                self.end_protocol = int(0.6 * self.freq)
            elif self.num_neuron == 1:
                self.beg_protocol = int(1. * self.freq)
                self.end_protocol = int(1.5 * self.freq)
        elif self.PARAMETERS_LIST['experimentalist'] in ['Elodie', 'Willy']:
            if self.x[0][-1, 0] > 1.05:
                self.beg_protocol = int(0.3 * self.freq)
                self.end_protocol = int(0.8 * self.freq)
            else:
                self.beg_protocol = int(0.25 * self.freq)
                self.end_protocol = int(0.75 * self.freq)

    def plot_phase(self):
        for i in np.arange(self.p_rheobase, len(self.PARAMETERS_LIST['phase_plane'])):
            if len(self.PARAMETERS_LIST['phase_plane'][i]) > 0:
                plt.plot(self.PARAMETERS_LIST['phase_plane'][i][0][0], self.PARAMETERS_LIST['phase_plane'][i][0][1])
        plt.savefig(self.poop + 'Phase_Plane.pdf')
        plt.close()

    def plot_rate(self):
        I_value = np.array(self.PARAMETERS_LIST['I'])
        rate = np.array(self.PARAMETERS_LIST['rate'])
        ax = plt.subplot(111)
        R_rate, b_rate = np.polyfit(I_value[(I_value >= self.PARAMETERS_LIST['rheobase'])],
                                    rate[(I_value >= self.PARAMETERS_LIST['rheobase'])], 1)
        ax.plot(I_value, self.PARAMETERS_LIST['rate_ISI'], 'r-', label='rate_ISI')
        ax.plot(I_value, rate, 'g-', label='rate')
        ax.plot(I_value[(I_value >= self.PARAMETERS_LIST['rheobase'])],
                I_value[(I_value >= self.PARAMETERS_LIST['rheobase'])]*R_rate + b_rate)
        ax.plot([self.PARAMETERS_LIST['rheobase']] * 2, ax.get_ylim(), 'b--', label='rheobase')
        ax.legend()
        plt.savefig(self.poop + 'Rate.pdf')
        plt.close()
        self.PARAMETERS_LIST['coeff_rate'] = R_rate

    def plot_IV(self):
        I_value = np.array(self.PARAMETERS_LIST['I'])
        diff_V = np.array(self.PARAMETERS_LIST['diff_V'])
        I_select = I_value[(0.01 >= diff_V) & (diff_V >= -0.01)]
        diff_V_select = diff_V[(0.01 >= diff_V) & (diff_V >= -0.01)]
        R_I, b_I = np.polyfit(I_select, diff_V_select, 1)
        plt.plot(I, diff_V, '+-')
        plt.plot(I_select, R_I * I_select + b_I)
        plt.savefig(self.poop + 'IV.pdf')
        plt.close()
        self.PARAMETERS_LIST['R'] = R_I

    def plot_adaptation(self):
        AP_max = np.array(self.PARAMETERS_LIST['AP_max'][-1])
        AP_ampl = np.array(self.PARAMETERS_LIST['amplitude'][-1])
        a_max, _ = pearsonr(np.arange(len(AP_max)), AP_max)
        a_ampl, _ = pearsonr(np.arange(len(AP_ampl)), AP_ampl)
        plt.plot(np.arange(len(AP_max)), AP_max, label=str(a_max))
        plt.plot(np.arange(len(AP_max)), AP_ampl, label=str(a_ampl))
        plt.legend()
        plt.savefig(self.poop + 'Adaptation.pdf')
        plt.close()
        self.PARAMETERS_LIST['adaptation_max'] = a_max
        self.PARAMETERS_LIST['adaptation_ampl'] = a_ampl

    def plot_integrate_and_fire(self, ax, fig, count, neuronModel, noise_value=None, delta_abs_value=None):
        gs_article = ax[0].get_gridspec()
        ax_AP_EXP = ax[0]
        ax_RI = ax[1]
        ax_RI_MSN = ax[2] if neuronModel == 'IAF_MSN' else None
        ax_IF = ax[3] if neuronModel == 'IAF_MSN' else ax[2]
        ax_AP_IAF = ax[4] if neuronModel == 'IAF_MSN' else ax[3]
        ax_legend = gs_article[5, count] if neuronModel == 'IAF_MSN' else gs_article[4, count]

        ax_AP_EXP.set_title(self.PARAMETERS_LIST['file'], fontsize=12, pad=10)

        fig.delaxes(ax[5]) if neuronModel == 'IAF_MSN' else fig.delaxes(ax[4])
        I_value = np.array(self.PARAMETERS_LIST['I'])
        diff_V = 1000. * np.array(self.PARAMETERS_LIST['diff_V'])
        Rate = np.array(self.PARAMETERS_LIST['rate'])
        tau_RC = np.array(self.PARAMETERS_LIST['tau_RC'])

        self.params_integrate_and_fire = {'file': self.PARAMETERS_LIST['file'],
                                          'E_l': 1000. * np.mean(self.PARAMETERS_LIST['baselineS']),
                                          'E_reset': 1000. * np.mean(np.concatenate(self.PARAMETERS_LIST['AP_max'])),
                                          'rheobase': I_value[self.p_rheobase-1],
                                          'E_rheobase': 1000. * self.PARAMETERS_LIST['baselineV'][self.p_rheobase - 1],
                                          'E_r': 1000. * np.mean([np.mean(list_AP_MIN) for list_AP_MIN
                                                                  in self.PARAMETERS_LIST['AP_min']
                                                                  if len(list_AP_MIN) > 0]),
                                          'noise': noise_value,
                                          'scale_I': None
                                          }

        save_AP_EXP = dict(time=[], V=[])
        for x in self.x:
            ax_AP_EXP.plot(1000. * x[:, 0], 1000. * x[:, 1], color=colors['blue'], alpha=0.2)
            save_AP_EXP['time'].append(1000. * x[:, 0])
            save_AP_EXP['V'].append(1000. * x[:, 1])
        ax_AP_EXP.set_xlim(1000. * x[0, 0], 1000. * x[-1, 0])
        ax_AP_EXP.set_xlabel('Time (ms)')

        if neuronModel == 'IAF':
            RI_filter = (0. <= diff_V[:self.p_rheobase])
            I_RI = I_value[:self.p_rheobase][RI_filter]
            diff_V_RI = diff_V[:self.p_rheobase][RI_filter]
            tau_RC_RI = np.mean(tau_RC)
            self.params_integrate_and_fire['V_th'] = \
                np.maximum(1000. * np.mean(self.PARAMETERS_LIST['baselineV'][self.p_rheobase:]),
                           self.params_integrate_and_fire['E_r'] + 1.)
            self.params_integrate_and_fire['thrV'] = 1000. * np.mean([np.mean(list_VTH) for list_VTH
                                                                      in self.PARAMETERS_LIST['thrV']
                                                                      if len(list_VTH) > 0])
        else:
            if self.PARAMETERS_LIST['experimentalist'] == 'Willy':
                RI_filter = (0. >= diff_V)
            else:
                RI_filter = (0. >= diff_V) & (diff_V >= -20.)
            I_RI = I_value[RI_filter]
            diff_V_RI = diff_V[RI_filter]
            tau_RC_RI = tau_RC[RI_filter]
            tau_RC_RI = np.mean(tau_RC_RI)
            self.params_integrate_and_fire['V_th'] = 1000. * np.mean([np.mean(list_VTH) for list_VTH
                                                                      in self.PARAMETERS_LIST['thrV']
                                                                      if len(list_VTH) > 0])
        R_RI, b_RI = np.polyfit(I_RI, diff_V_RI, 1)
        C_RI = tau_RC_RI/R_RI

        diff_V_linear = R_RI * I_RI + b_RI
        ax_RI.plot(I_value, diff_V, '+-', color=colors['blue'])
        ax_RI.plot(I_RI, diff_V_linear, color=colors['green'])
        ax_RI.set_xlabel('I (pA)')
        save_RI = dict(I=I_value, I_RI=I_RI, diff_V=diff_V, diff_V_linear=diff_V_linear)

        if neuronModel == 'IAF_MSN':
            SAV_GOL = savgol_filter(range_filter=31, poly=3)
            V_list = []
            R_list = []
            C_list = []
            for x, current_I_value, baseline in zip(self.x, I_value, self.PARAMETERS_LIST['baselineS']):
                if current_I_value < -20.:
                    V = 1000. * SAV_GOL.compute(x)[:, 1] - 1000. * baseline
                    V_1 = V[self.end_protocol - int(0.005 * self.freq):self.end_protocol + int(0.01 * self.freq)]
                    I_1 = 0.
                    V_filter_1 = (np.abs((V_1[1:]-V_1[:-1])) > 0.02) * (~((V_1[1:] < 2.) * (V_1[1:] > - 10.)))
                    V_2 = V[self.beg_protocol - int(0.005 * self.freq):self.beg_protocol + int(0.02 * self.freq)]
                    I_2 = current_I_value
                    V_filter_2 = (np.abs((V_2[1:] - V_2[:-1])) > 0.02) * (~((V_2[1:] < 2.) * (V_2[1:] > - 10.)))
                    if any(V_filter_1) and any(V_filter_2):
                        C = C_RI
                        R_1_mean = None
                        n_iter = 0
                        while (R_1_mean is None or np.abs(R_1_mean - R_2_mean) > 0.0001) and n_iter < 1000:
                            if R_1_mean is not None:
                                C = np.maximum(0., C + 0.1 * (R_1_mean - R_2_mean))
                            R_1 = np.minimum(np.maximum(V_1[:-1][V_filter_1] * (I_1 - C / self.dt * (
                                        V_1[1:][V_filter_1] - V_1[:-1][V_filter_1])) ** (-1), 0.), 5 * R_RI)
                            R_1_mean = np.mean(R_1)
                            R_2 = np.minimum(np.maximum(V_2[:-1][V_filter_2] * (I_2 - C / self.dt * (
                                        V_2[1:][V_filter_2] - V_2[:-1][V_filter_2])) ** (-1), 0.), 5 * R_RI)
                            R_2_mean = np.mean(R_2)
                            n_iter = n_iter + 1
                        if n_iter < 1000:
                            C_list.append(C)
            if len(C_list) == 0:
                C_MEAN = C_RI
            else:
                C_MEAN = np.mean(C_list)
            self.params_integrate_and_fire['C'] = C_MEAN

            for x, current_I_value, baseline in zip(self.x, I_value, self.PARAMETERS_LIST['baselineS']):
                V = 1000. * SAV_GOL.compute(x)[:, 1] - 1000. * baseline
                if current_I_value < 0.:
                    V_1 = V[self.end_protocol - int(0.005 * self.freq):self.end_protocol]
                else:
                    V_1 = V[self.end_protocol - int(0.005 * self.freq):self.end_protocol - int(0.004 * self.freq)]
                V_filter_1 = (np.abs((V_1[1:] - V_1[:-1])) > 0.2)
                R_1 = V_1[:-1][V_filter_1] * (0. - C_MEAN / self.dt * (
                        V_1[1:][V_filter_1] - V_1[:-1][V_filter_1])) ** (-1)
                V_list.append(V_1[:-1][V_filter_1])
                R_list.append(R_1)
            V_list = np.concatenate(V_list)
            R_list = np.concatenate(R_list)

            V_list_mean = []
            R_list_mean = []
            V_x = np.linspace(np.min(V_list), np.minimum(np.max(V_list), 25.), 100)

            for j in np.arange(len(V_x)-1):
                if len(R_list[(V_x[j+1] >= V_list) & (V_list >= V_x[j])]) > 0:
                    V_list_mean.append((V_x[j] + V_x[j + 1]) / 2.)
                    R_list_mean.append(np.mean(R_list[(V_x[j+1] >= V_list) & (V_list >= V_x[j])]))
            V_list_mean = np.array(V_list_mean)
            R_list_mean = np.array(R_list_mean)

            RI_fast = np.mean(R_list_mean[V_list_mean < 0.])

            popt, pcov = curve_fit(R_func(RI_fast), V_list_mean, R_list_mean, p0=(20., 10., 2.*RI_fast), bounds=(
                [2., 0., 0.], [100., 10., 5. * RI_fast]), maxfev=100000)
            tau_V_RI, V_RI, RI_slow = popt
            x_v = np.linspace(V_list_mean[0], V_list_mean[-1], 1000)

            ax_RI_MSN.plot(V_list, R_list, '+', color=colors['blue'])
            ax_RI_MSN.plot(V_list_mean, R_list_mean, color=colors['light blue'])
            ax_RI_MSN.plot(x_v, R_func(RI_fast)(x_v, *popt), color=colors['green'])
            ax_RI_MSN.set_xlabel('V-E_l (mV)')
            ax_RI_MSN.set_ylim(0., 0.3)
            ax_RI_MSN.set_xlim(-20., 40.)

            self.params_integrate_and_fire['RI_slow'] = RI_slow * 1000.
            self.params_integrate_and_fire['RI_fast'] = RI_fast * 1000.
            self.params_integrate_and_fire['R_rate'] = R_RI * 1000.
            self.params_integrate_and_fire['R'] = R_RI * 1000.
            self.params_integrate_and_fire['V_RI'] = V_RI + self.params_integrate_and_fire['E_l']
            self.params_integrate_and_fire['tau_V_RI'] = tau_V_RI
            self.params_integrate_and_fire['tau'] = tau_RC_RI * 1000.
        else:
            self.params_integrate_and_fire['R_rate'] = R_RI * 1000.
            self.params_integrate_and_fire['R'] = R_RI * 1000.
            self.params_integrate_and_fire['tau'] = tau_RC_RI * 1000.
            self.params_integrate_and_fire['C'] = tau_RC_RI / R_RI

        dt_simu = 0.1

        if delta_abs_value is None:
            Delta_abs = 0.
            Rate_SIMU_mean = None
            Rate_mean = np.mean(Rate[Rate > 0.])

            n_iter = 0
            while (Rate_SIMU_mean is None or (Rate_SIMU_mean - Rate_mean) > 2.) and n_iter < 100:
                Rate_SIMU = []
                if Rate_SIMU_mean is not None:
                    Delta_abs = Delta_abs + 0.1 * (Rate_SIMU_mean - Rate_mean)
                self.params_integrate_and_fire['Delta_abs'] = Delta_abs
                for current_I_value in I_value:
                    NEURON = dict_neuronModel[neuronModel](P=1,
                                                           init=dirac(np.array(
                                                                [self.params_integrate_and_fire['E_l']])),
                                                           **self.params_integrate_and_fire)
                    for j in range(int(1000/dt_simu)):
                        NEURON.iterate(dt_simu, I=step_AP(j*dt_simu, 250., 750., current_I_value/1000.))
                        NEURON.update()
                    Rate_SIMU.append(1000. * np.sum(NEURON.spike_count) / 500.)
                Rate_SIMU = np.array(Rate_SIMU)
                if len(Rate_SIMU[Rate_SIMU > 0.]) == 0:
                    Rate_SIMU_mean = 0.
                else:
                    Rate_SIMU_mean = np.mean(Rate_SIMU[Rate_SIMU > 0.])
                n_iter = n_iter + 1

            if n_iter == 100:
                Delta_abs = 0.
        else:
            Delta_abs = 10.

        self.params_integrate_and_fire['Delta_abs'] = Delta_abs

        ax_AP_IAF.get_shared_y_axes().join(ax_AP_IAF, ax_AP_EXP)

        Rate_SIMU = []
        save_AP_IAF = dict(time=[], V=[])
        for current_I_value in I_value:
            NEURON = dict_neuronModel[neuronModel](P=1, init=gaussian(np.array([self.params_integrate_and_fire['E_l']]),
                                                                      1.), **self.params_integrate_and_fire)
            for j in range(int(1000/dt_simu)):
                NEURON.iterate(dt_simu, I=step_AP(j*dt_simu, 250., 750., current_I_value/1000.))
                NEURON.update()
            Rate_SIMU.append(1000. * np.sum(NEURON.spike_count) / 500.)
            ax_AP_IAF.set_xlim(NEURON.time[0], NEURON.time[-1])
            ax_AP_IAF.plot(NEURON.time, [NEURON.potential[i][0] + NEURON.spike_count[i][0] * (
                    NEURON.parameters['E_reset'] - NEURON.parameters['V_th']) for i in range(len(NEURON.potential))],
                    color=colors['green'], alpha=0.2)
            save_AP_IAF['time'].append(NEURON.time)
            save_AP_IAF['V'].append([NEURON.potential[i][0] + NEURON.spike_count[i][0] * (
                    NEURON.parameters['E_reset'] - NEURON.parameters['V_th']) for i in range(len(NEURON.potential))])
        ax_AP_IAF.set_xlabel('Time (ms)')
        Rate_SIMU = np.array(Rate_SIMU)

        Rate_EXACT = firing_rate(I_value, self.params_integrate_and_fire)
        ax_IF.plot(I_value, Rate_SIMU, 'o', label='rate_SIMU', color=colors['green'])
        ax_IF.plot(I_value, Rate, '+', label='rate_ISI', color=colors['blue'])
        ax_IF.plot(I_value, Rate_EXACT, '-', label='rate_EXACT', color=colors['light green'])
        save_IF = dict(I=I_value, Rate_SIMU=Rate_SIMU, Rate=Rate, Rate_EXACT=Rate_EXACT)
        ax_IF.set_xlabel('I (pA)')

        list_PARAMS_V = ['E_l', 'V_th', 'E_reset', 'E_r']
        list_PARAMS_R = ['R']
        list_PARAMS_T = ['tau', 'Delta_abs']
        list_PARAMS_C = ['C']
        if neuronModel == 'IAF_MSN':
            list_PARAMS_R = list_PARAMS_R + ['RI_fast', 'RI_slow']
            list_PARAMS_V = list_PARAMS_V + ['tau_V_RI', 'V_RI']

        inner = gridspec.GridSpecFromSubplotSpec(1, 7,
                                                 subplot_spec=ax_legend,
                                                 **dict(hspace=0., wspace=0.,
                                                        width_ratios=[len(list_PARAMS_V), 2.,
                                                                      len(list_PARAMS_R), 2.,
                                                                      len(list_PARAMS_T), 2.,
                                                                      len(list_PARAMS_C)]))

        for k, list_PARAMS_ in enumerate([list_PARAMS_V, list_PARAMS_R, list_PARAMS_T, list_PARAMS_C]):
            ax_ = fig.add_subplot(inner[2*k])
            for i, key_ in enumerate(list_PARAMS_):
                ax_.plot(i, self.params_integrate_and_fire[key_], 'o', alpha=1,
                         color=colors[list(colors.keys())[i]], ms=10)
            ax_.set_xticks(np.arange(len(list_PARAMS_)))
            ax_.set_xlim(-0.5, len(list_PARAMS_)-0.5)
            ax_.set_xticklabels(
                [key_ + '=' + str(np.round(self.params_integrate_and_fire[key_], decimals=2)) for key_ in list_PARAMS_])
            ax_.set_xticklabels(ax_.get_xticklabels(), rotation=45, ha='right')
            ax_.set_ylabel(y_label[k])
        np.save(self.poop + self.PARAMETERS_LIST['file'] + '_IAF.npy', self.params_integrate_and_fire)
        np.save(self.poop + self.PARAMETERS_LIST['file'] + '_plot_IAF.npy', dict(AP_EXP=save_AP_EXP, AP_IAF=save_AP_IAF,
                                                                                 RI=save_RI, IF=save_IF))

    def save(self):
        np.save(self.poop + self.PARAMETERS_LIST['file'] + '.npy', self.PARAMETERS_LIST)
