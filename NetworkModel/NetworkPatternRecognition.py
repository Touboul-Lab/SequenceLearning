import numpy as np
import InitFunctions
from NetworkModel import networkClass
from NeuronModel import stim, poissonPop
from WeightModel import synapticClass, spikeBased, STDP, weightClass
from NetworkModel.StriatumNeurons import find_params_neuron

class patternRecognition(networkClass):
    parameters = {'P': 1,
                  'neuronClass': 'MSN_IAF_EXP',
                  'noise_stim': None,
                  'noise_input': None,
                  'Aprepost': -1., 'Apostpre': 0., 'tprepost': 10., 'tpostpre': 10., 'nearest': False, 'exp': False,
                  'init_weight': ['uniform', (0.2, 0.5)],
                  'clip_weight': (0., 2.),
                  'homeostasy': 1.,
                  'epsilon': 0.1,
                  'save': True,
                  }

    def __init__(self, **kwargs):
        networkClass.subclass(self, patternRecognition.parameters, **kwargs)
        self.STIM = stim(P=self.parameters['P'], type='pattern', noise=self.parameters['noise_stim'],
                         save=self.parameters['save'])
        neuronType, neuronParams = find_params_neuron(self.parameters['neuronClass'])
        self.NEURON = neuronType(P=1, **neuronParams, save=self.parameters['save'])
        self.WEIGHT = STDP(self.STIM, self.NEURON, spikeBased,
                           init_weight=getattr(InitFunctions, self.parameters['init_weight'][0])(
                               *self.parameters['init_weight'][1]), save=self.parameters['save'],
                           Aprepost=self.parameters['Aprepost'] * self.parameters['epsilon'],
                           Apostpre=self.parameters['Apostpre'] * self.parameters['epsilon'],
                           tprepost=self.parameters['tprepost'],
                           tpostpre=self.parameters['tpostpre'], nearest=self.parameters['nearest'],
                           homeostasy=self.parameters['homeostasy'] * self.parameters['epsilon'],
                           alpha='clip',
                           w_min=self.parameters['clip_weight'][0], w_max=self.parameters['clip_weight'][1],
                           exp=self.parameters['exp']
                           )
        if self.parameters['noise_input'] is None:
            self.populations = [self.STIM, self.NEURON]
            self.inputs = []
            self.weights = [self.WEIGHT]
        else:
            self.RANDOM_INPUT = poissonPop(P=1, N=1, b=self.parameters['noise_input'], save=self.parameters['save'])
            self.WEIGHT_RANDOM_INPUT = weightClass(self.RANDOM_INPUT, self.NEURON, spikeBased,
                                                   init_weight=InitFunctions.dirac(1.0), save=self.parameters['save'])
            self.populations = [self.STIM, self.NEURON, self.RANDOM_INPUT]
            self.inputs = []
            self.weights = [self.WEIGHT, self.WEIGHT_RANDOM_INPUT]

    def iterate_pattern(self, dt, pattern, output):
        output['spike_after_pattern'] = [None for _ in np.arange(self.NEURON.P)]
        output['num_spike'] = np.zeros(self.NEURON.P, dtype=np.int)
        output['timing_spike'] = [[] for _ in np.arange(self.NEURON.P)]
        output['timing_spike_input'] = [[] for _ in np.arange(self.parameters['P'])]
        count = 0
        if output['stim_recall'] is not None:
            output['stim'] = np.random.binomial(1, output['stim_recall'])
        else:
            output['stim'] = 1
        for i in np.arange(pattern.duration):
            self.STIM.pattern = np.zeros(self.parameters['P'])
            if output['stim'] == 1:
                for k in np.arange(len(pattern.index)):
                    if i + 1 == pattern.timing[k]:
                        self.STIM.pattern[pattern.index[k]] += 1
                    elif i == pattern.timing[k]:
                        count += 1
            self.iterate(dt)
            self.update()
            current_spike_input = self.STIM.spike_count[-1]
            for k in np.arange(self.parameters['P']):
                if current_spike_input[k] == 1:
                    output['timing_spike_input'][k].append(i * dt)
            for j in np.arange(self.NEURON.P):
                if self.NEURON.spike_count[-1][j] == 1:
                    output['num_spike'][j] = output['num_spike'][j] + 1
                    output['timing_spike'][j].append(i * dt)
                    if count < len(pattern.index):
                        output['spike_after_pattern'][j] = -1
                    elif count == len(pattern.index):
                        if output['spike_after_pattern'][j] is None:
                            output['spike_after_pattern'][j] = 1
                        elif output['spike_after_pattern'][j] > 0:
                            output['spike_after_pattern'][j] += 1
        return output

class patternRecognitionDual(patternRecognition):
    parameters = {'P': 1,
                  'neuronClass': 'MSN_Burst',
                  'pattern': None, 'noise_stim': None,
                  'noise_input': None,
                  'Aprepost': -1., 'Apostpre': 0., 'tprepost': 10., 'tpostpre': 10., 'nearest': False, 'exp': False,
                  'init_weight': ['uniform', (0.2, 0.5)],
                  'clip_weight': (0., 2.),
                  'homeostasy': 1.,
                  'epsilon': 0.1,
                  'J_matrix': np.zeros((2, 2)),
                  'J_value': None,
                  'save': True,
                  }

    def __init__(self, **kwargs):
        networkClass.subclass(self, patternRecognitionDual.parameters, **kwargs)
        self.STIM = stim(P=self.parameters['P'], type='pattern', noise=self.parameters['noise_stim'],
                         save=self.parameters['save'])
        neuronType, neuronParams = find_params_neuron(self.parameters['neuronClass'])
        self.NEURON = neuronType(P=2, **neuronParams, save=self.parameters['save'])
        self.WEIGHT = STDP(self.STIM, self.NEURON, spikeBased,
                           init_weight=getattr(InitFunctions, self.parameters['init_weight'][0])(
                                               *self.parameters['init_weight'][1]), save=self.parameters['save'],
                           Aprepost=self.parameters['Aprepost']*self.parameters['epsilon'],
                           Apostpre=self.parameters['Apostpre']*self.parameters['epsilon'],
                           tprepost=self.parameters['tprepost'],
                           tpostpre=self.parameters['tpostpre'], nearest=self.parameters['nearest'],
                           homeostasy=self.parameters['homeostasy']*self.parameters['epsilon'],
                           alpha='clip',
                           w_min=self.parameters['clip_weight'][0], w_max=self.parameters['clip_weight'][1],
                           exp=self.parameters['exp']
                           )
        if isinstance(self.parameters['J_value'], np.float):
            self.WEIGHT_II = weightClass(self.NEURON, self.NEURON, spikeBased,
                                         init_weight=InitFunctions.dirac(self.parameters['J_value']),
                                         connectivity=InitFunctions.dirac(self.parameters['J_matrix']),
                                         save=self.parameters['save'])
        else:
            if self.parameters['J_value'] == 'random':
                self.WEIGHT_II = weightClass(self.NEURON, self.NEURON, spikeBased,
                                             init_weight=InitFunctions.uniform(-1., 0.),
                                             connectivity=InitFunctions.dirac(self.parameters['J']),
                                             save=self.parameters['save'])
        if self.parameters['noise_input'] is None:
            self.populations = [self.STIM, self.NEURON]
            self.inputs = []
            self.weights = [self.WEIGHT, self.WEIGHT_II]
        else:
            self.RANDOM_INPUT = poissonPop(P=1, N=1, b=self.parameters['noise_input'], save=self.parameters['save'])
            self.WEIGHT_RANDOM_INPUT = weightClass(self.RANDOM_INPUT, self.NEURON, spikeBased,
                                                   init_weight=InitFunctions.dirac(1.0), save=self.parameters['save'])
            self.populations = [self.STIM, self.NEURON, self.RANDOM_INPUT]
            self.inputs = []
            self.weights = [self.WEIGHT, self.WEIGHT_II, self.WEIGHT_RANDOM_INPUT]

class patternRecognitionExample(networkClass):
    parameters = {'P': 1,
                  'neuronClass': 'MSN_IAF_EXP',
                  'noise_stim': None,
                  'noise_input': None,
                  'Aprepost': -1., 'Apostpre': 0., 'tprepost': 10., 'tpostpre': 10., 'nearest': False, 'exp': False,
                  'init_weight': ['uniform', (0.2, 0.5)],
                  'clip_weight': (0., 2.),
                  'homeostasy': 1.,
                  'epsilon': 0.1,
                  'save': True,
                  }

    def __init__(self, **kwargs):
        networkClass.subclass(self, patternRecognition.parameters, **kwargs)
        self.STIM = stim(P=self.parameters['P'], type='pattern', noise=self.parameters['noise_stim'],
                         save=self.parameters['save'])
        self.NEURON = stim(P=1, type='pattern', save=self.parameters['save'])
        self.WEIGHT = STDP(self.STIM, self.NEURON, synapticClass,
                           init_weight=getattr(InitFunctions, self.parameters['init_weight'][0])(
                               *self.parameters['init_weight'][1]), save=self.parameters['save'],
                           Aprepost=self.parameters['Aprepost'] * self.parameters['epsilon'],
                           Apostpre=self.parameters['Apostpre'] * self.parameters['epsilon'],
                           tprepost=self.parameters['tprepost'],
                           tpostpre=self.parameters['tpostpre'], nearest=self.parameters['nearest'],
                           homeostasy=self.parameters['homeostasy'] * self.parameters['epsilon'],
                           alpha='clip',
                           w_min=self.parameters['clip_weight'][0], w_max=self.parameters['clip_weight'][1],
                           exp=self.parameters['exp']
                           )
        self.populations = [self.STIM, self.NEURON]
        self.inputs = []
        self.weights = [self.WEIGHT]

    def iterate_pattern(self, dt, pattern, output):
        output['spike_after_pattern'] = [None for _ in np.arange(self.NEURON.P)]
        output['num_spike'] = np.zeros(self.NEURON.P, dtype=np.int)
        output['timing_spike'] = [[] for _ in np.arange(self.NEURON.P)]
        output['timing_spike_input'] = [[] for _ in np.arange(self.parameters['P'])]
        count = 0
        if output['stim_recall'] is not None:
            output['stim'] = np.random.binomial(1, output['stim_recall'])
        else:
            output['stim'] = 1
        for i in np.arange(pattern.duration):
            self.STIM.pattern = np.zeros(self.parameters['P'])
            if output['stim'] == 1:
                for k in np.arange(len(pattern.index)):
                    if i + 1 == pattern.timing[k]:
                        self.STIM.pattern[pattern.index[k]] += 1
                    elif i == pattern.timing[k]:
                        count += 1
            self.NEURON.pattern = np.zeros((1,))
            for k in np.arange(len(output['pattern_MSN'])):
                if i + 1 == output['pattern_MSN'][k]:
                    self.NEURON.pattern[0] += 1
            self.iterate(dt)
            self.update()
            current_spike_input = self.STIM.spike_count[-1]
            for k in np.arange(self.parameters['P']):
                if current_spike_input[k] == 1:
                    output['timing_spike_input'][k].append(i * dt)
            for j in np.arange(self.NEURON.P):
                if self.NEURON.spike_count[-1][j] == 1:
                    output['num_spike'][j] = output['num_spike'][j] + 1
                    output['timing_spike'][j].append(i * dt)
                    if count < len(pattern.index):
                        output['spike_after_pattern'][j] = -1
                    elif count == len(pattern.index):
                        if output['spike_after_pattern'][j] is None:
                            output['spike_after_pattern'][j] = 1
                        elif output['spike_after_pattern'][j] > 0:
                            output['spike_after_pattern'][j] += 1
        return output
