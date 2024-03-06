import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../../../'))

import Articles.Sequential_Learning_Striatum.Figures.cfg_pdf as cfg
from NetworkModel.StriatumNeurons import find_params_neuron

def step_AP(t, beg, end, I_value):
    if t < beg or t > end:
        return 0
    else:
        return I_value

def step_event(t, event, I_value_event):
    if t == event:
        return I_value_event
    else:
        return 0


neuronType_latency, neurons_params_latency = find_params_neuron('MSN_Izhi')

path_save = 'Figures/article/'

fig_y, fig_x = 16., 4.
top, bottom, left, right = 1., 1., 2., 1.
dict_margins = dict(top=1.-top/fig_x, bottom=bottom/fig_x, left=left/fig_y, right=1.-right/fig_y)
wspace = 1.5
width_average = (fig_y - left - right - wspace) / 2.
dict_margins['width_ratios'] = [width_average, width_average]
dict_margins['wspace'] = wspace / width_average
fig, ax = plt.subplots(1, 2, figsize=(fig_y * cfg.cm, fig_x * cfg.cm),
                       gridspec_kw=dict(**dict_margins))

ax_AP_latency = ax[0]
ax_event_latency = ax[1]

dt_simu = 0.01

current_ax_AP, current_ax_event, current_neuronType, current_neuron_params = \
        (ax_AP_latency, ax_event_latency, neuronType_latency, neurons_params_latency)

I_value = np.arange(-50, 50, 5)
spike = True
for current_I_value in I_value:
    NEURON = current_neuronType(P=1, **current_neuron_params)
    for j in range(int(1000 / dt_simu)):
        NEURON.iterate(dt_simu, I=step_AP(j * dt_simu, 250., 750., current_I_value / 1000.))
        NEURON.update()
    current_ax_AP.set_xlim(NEURON.time[0], NEURON.time[-1])
    if spike and np.sum([NEURON.spike_count[i][0] for i in range(len(NEURON.potential))]) > 0:
        current_ax_AP.plot(NEURON.time, [NEURON.potential[i][0] for i in range(len(NEURON.potential))],
                           color=cfg.colors['black'], alpha=1., zorder=10)
        spike = False
    else:
        current_ax_AP.plot(NEURON.time, [NEURON.potential[i][0] for i in range(len(NEURON.potential))],
                           color=cfg.colors['dark green'], alpha=0.2)

I_value_event = np.arange(0, 100, 5)
spike = True
for current_I_value in I_value_event:
    NEURON = current_neuronType(P=1, **current_neuron_params)
    for j in range(int(20 / dt_simu)):
        NEURON.iterate(dt_simu, I=step_event(j * dt_simu, 5., current_I_value / 1000. / dt_simu))
        NEURON.update()
    current_ax_event.set_xlim(NEURON.time[0], NEURON.time[-1])
    if spike and np.sum([NEURON.spike_count[i][0] for i in range(len(NEURON.potential))]) > 0:
        current_ax_event.plot(NEURON.time, [NEURON.potential[i][0] for i in range(len(NEURON.potential))],
                              color=cfg.colors['black'], alpha=1., zorder=10)
        spike = False
    else:
        current_ax_event.plot(NEURON.time, [NEURON.potential[i][0] for i in range(len(NEURON.potential))],
                              color=cfg.colors['dark green'], alpha=0.2)
current_ax_AP.set_ylabel(u'Membrane potential (mV)')

current_ax_AP.set_title('Step protocol')
current_ax_event.set_title('Event protocol')

for ax_ in [current_ax_AP, current_ax_event]:
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.set_ylim(ax_.get_ylim()[0], 50)

current_ax_AP.set_yticks([-150, -100, -50, 0, 50])
current_ax_AP.set_xlabel('Time (ms)')
current_ax_event.set_xlabel('Time (ms)')

plt.savefig('{}Figure_4_a.png'.format(path_save), dpi=1000)
plt.close()
