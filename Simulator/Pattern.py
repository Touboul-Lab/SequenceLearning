import numpy as np
import itertools

def get_subpattern(pattern, simulator):
    list_pattern = []
    if pattern.reward > 0.:
        for i in np.arange(1, len(pattern.index)):
            for subset in itertools.combinations(pattern.index, i):
                list_pattern.append(subpattern(simulator, list(subset), pattern))
    return list_pattern

class pattern_None:
    def __init__(self, simulator):
        self.simulator = simulator
        self.index = []

        self.stim_by_pattern = 0
        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])

        self.reward = None
        self.reward_vector = None

        self.init_reward()

        self.timing_constant = np.array([])
        self.timing = self.timing_constant
        self.input = None

    def init_reward(self):
        choice = self.simulator.params_pattern['no_reward']
        self.reward = - 2. * choice + 1.
        if self.simulator.network.NEURON.P == 1:
            if choice == 0:
                self.reward_vector = np.ones(1)
            else:
                self.reward_vector = np.zeros(1)
        elif self.simulator.network.NEURON.P == 2:
            if self.simulator.params_pattern['J_reward'] == 'differential':
                self.reward_vector = np.zeros(2)
                self.reward_vector[choice] = 1.
            elif self.simulator.params_pattern['J_reward'] == 'same':
                if choice == 0:
                    self.reward_vector = np.ones(2)
                else:
                    self.reward_vector = np.zeros(2)

class patternClass:
    def __init__(self, simulator, index):
        self.simulator = simulator
        self.index = index

        self.stim_by_pattern = len(self.index)
        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])
        self.offset = int(self.simulator.params_pattern['offset'] / self.simulator.params_simu['dt'])
        if self.offset == 0:
            raise IndexError('Offset should be non null')
        self.reward = None
        self.reward_vector = None
        self.init_reward()

        self.timing_constant = self.offset + np.arange(0, self.stim_by_pattern) * int(
            self.simulator.params_pattern['delay'] / self.simulator.params_simu['dt'])
        self.timing = self.timing_constant

        self.input = None
        self.create_input()

    def init_reward(self):
        if self.simulator.params_pattern['p_reward'] is None:
            choice = self.simulator.params_pattern['no_reward']
        else:
            choice = np.random.binomial(1, 1. - self.simulator.params_pattern['p_reward'])
        self.reward = - 2. * choice + 1.
        if self.simulator.network.NEURON.P == 1:
            if choice == 0:
                self.reward_vector = np.ones(1)
            else:
                self.reward_vector = np.zeros(1)
        elif self.simulator.network.NEURON.P == 2:
            if self.simulator.params_pattern['J_reward'] == 'differential':
                self.reward_vector = np.zeros(2)
                self.reward_vector[choice] = 1.
            elif self.simulator.params_pattern['J_reward'] == 'same':
                if choice == 0:
                    self.reward_vector = np.ones(2)
                else:
                    self.reward_vector = np.zeros(2)

    def create_input(self):
        self.input = (-1.) * np.ones(self.simulator.params_network['P'])
        index_count = np.ones(self.simulator.params_network['P'])
        for current_index, current_timing in zip(self.index, self.timing_constant):
            if self.input[current_index] < 0.:
                self.input[current_index] = current_timing
            else:
                self.input[current_index] += current_timing
                index_count[current_index] += 1.
        self.input /= index_count

    def sample(self, noise_pattern):
        if noise_pattern is None:
            self.timing = self.timing_constant
        else:
            self.timing = np.maximum(np.minimum(self.duration - 1,
                                     self.timing_constant + noise_pattern
                                     * np.random.normal(0, 1, len(
                                        self.timing_constant)) / self.simulator.params_simu['dt']), 1).astype(np.int16)


class pattern_jitter(patternClass):
    def __init__(self, simulator, index):
        self.simulator = simulator
        self.index = index

        self.stim_by_pattern = len(self.index)
        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])
        self.offset = int(self.simulator.params_pattern['offset'] / self.simulator.params_simu['dt'])
        if self.offset == 0:
            raise IndexError('Offset should be non null')
        self.reward = None
        self.reward_vector = None
        self.init_reward()

        timing_constant_pattern = self.offset + np.arange(0, self.stim_by_pattern) * int(
            self.simulator.params_pattern['delay'] / self.simulator.params_simu['dt'])
        if int(self.simulator.params_pattern['delay'] / self.simulator.params_simu['dt']) % 2 == 0:
            half_delay = int(
                self.simulator.params_pattern['delay'] / self.simulator.params_simu['dt']) // 2
            half_delay_lim = (- half_delay, half_delay)
        else:
            half_delay = int(
                self.simulator.params_pattern['delay'] / self.simulator.params_simu['dt']) // 2
            half_delay_lim = (- half_delay, half_delay + 1)
        self.timing_constant = np.zeros(self.stim_by_pattern, dtype=np.int16)
        for current_index in np.arange(self.stim_by_pattern):
            self.timing_constant[current_index] = timing_constant_pattern[current_index] + \
                np.random.randint(half_delay_lim[0], half_delay_lim[1])
        self.timing_constant = np.maximum(np.minimum(self.duration - 1,
                                          self.timing_constant), 1).astype(np.int16)
        self.timing = self.timing_constant

        self.input = None
        self.create_input()

class patternConstant:
    def __init__(self, simulator, index, timing, reward):
        self.simulator = simulator
        self.index = index

        self.stim_by_pattern = len(self.index)
        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])

        self.current_reward = reward
        self.reward = None
        self.reward_vector = None
        self.init_reward()

        self.timing_constant = [int(current_timing / self.simulator.params_simu['dt']) for current_timing in timing]
        self.timing = self.timing_constant
        self.input = None
        if np.min(self.timing_constant) == 0:
            raise IndexError('Should be non null')

    def init_reward(self):
        choice = self.current_reward
        self.reward = - 2. * choice + 1.
        if self.simulator.network.NEURON.P == 1:
            if choice == 0:
                self.reward_vector = np.ones(1)
            else:
                self.reward_vector = np.zeros(1)
        elif self.simulator.network.NEURON.P == 2:
            if self.simulator.params_pattern['J_reward'] == 'differential':
                self.reward_vector = np.zeros(2)
                self.reward_vector[choice] = 1.
            elif self.simulator.params_pattern['J_reward'] == 'same':
                if choice == 0:
                    self.reward_vector = np.ones(2)
                else:
                    self.reward_vector = np.zeros(2)

class subpattern(patternClass):
    def __init__(self, simulator, index, pattern):
        self.simulator = simulator
        self.index = index

        self.pattern = pattern
        self.stim_by_pattern = len(self.index)

        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])
        self.offset = int(self.simulator.params_pattern['offset'] / self.simulator.params_simu['dt'])
        if self.offset == 0:
            raise IndexError('Offset should be non null')

        self.reward = None
        self.reward_vector = None
        self.init_reward()

        self.timing_constant = []
        for j in self.index:
            index_j = np.argwhere(j == np.array(self.pattern.index))[0][0]
            self.timing_constant.append(self.pattern.timing_constant[index_j])
        self.timing = self.timing_constant

        self.input = None
        self.create_input()

    def init_reward(self):
        choice = 1
        self.reward = - 2. * choice + 1.
        if self.simulator.network.NEURON.P == 1:
            if choice == 0:
                self.reward_vector = np.ones(1)
            else:
                self.reward_vector = np.zeros(1)
        elif self.simulator.network.NEURON.P == 2:
            if self.simulator.params_pattern['J_reward'] == 'differential':
                self.reward_vector = np.zeros(2)
                self.reward_vector[choice] = 1.
            elif self.simulator.params_pattern['J_reward'] == 'same':
                if choice == 0:
                    self.reward_vector = np.ones(2)
                else:
                    self.reward_vector = np.zeros(2)

    def sample(self, noise_pattern):
        if self.simulator.params_pattern['random_time'] is None:
            self.timing = self.timing_constant
        else:
            raise NameError('sample in subpattern')

class pattern_DMS(patternClass):
    def __init__(self, simulator, index):
        self.simulator = simulator
        self.index = index
        self.stim_by_pattern = len(self.index)
        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])
        self.offset = int(self.simulator.params_pattern['offset'] / self.simulator.params_simu['dt'])
        if self.offset == 0:
            raise IndexError('Offset should be non null')
        self.init_reward()

        self.timing_constant = self.offset * np.ones(self.stim_by_pattern)
        self.timing = self.timing_constant

        self.input = None
        self.create_input()


class pattern_Succession(patternClass):
    def __init__(self, simulator, num=None):
        self.simulator = simulator
        self.stim_by_pattern = num + 1
        self.index = np.arange(0, self.stim_by_pattern)
        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])
        self.offset = int(self.simulator.params_pattern['offset'] / self.simulator.params_simu['dt'])
        if self.offset == 0:
            raise IndexError('Offset should be non null')
        self.init_reward()
        self.timing_constant = self.offset + np.arange(0, self.stim_by_pattern) * int(
            self.simulator.params_pattern['delay'] / self.simulator.params_simu['dt'])
        self.timing = self.timing_constant

        self.input = None
        self.create_input()

    def init_reward(self):
        choice = np.unpackbits(np.array(self.simulator.params_pattern['no_reward'], dtype=np.uint8))[-self.stim_by_pattern]
        self.reward = - 2. * choice + 1.
        if self.simulator.network.NEURON.P == 1:
            if choice == 0:
                self.reward_vector = np.ones(1)
            else:
                self.reward_vector = np.zeros(1)
        elif self.simulator.network.NEURON.P == 2:
            if self.simulator.params_pattern['J_reward'] == 'differential':
                self.reward_vector = np.zeros(2)
                self.reward_vector[choice] = 1.
            elif self.simulator.params_pattern['J_reward'] == 'same':
                if choice == 0:
                    self.reward_vector = np.ones(2)
                else:
                    self.reward_vector = np.zeros(2)

class pattern_Poisson(patternClass):
    def __init__(self, simulator):
        self.simulator = simulator
        self.duration = int(self.simulator.params_pattern['duration'] / self.simulator.params_simu['dt'])
        self.duration_poisson = int(self.simulator.params_pattern['duration_poisson'] /
                                    self.simulator.params_simu['dt'])
        self.offset = int(self.simulator.params_pattern['offset'] / self.simulator.params_simu['dt'])
        if self.offset == 0:
            raise IndexError('Offset should be non null')
        self.init_reward()
        self.create_time()
        self.timing = self.timing_constant

        self.input = None
        self.create_input()

    def create_time(self):
        timing = []
        for j in np.arange(self.duration_poisson):
            event = np.random.binomial(1, self.simulator.params_simu['dt']
                                       * self.simulator.params_pattern['noise_poisson'])
            if event == 1:
                time_event = j
                timing.append(time_event)
        if len(timing) > self.simulator.params_network['P'] or len(timing) < 2:
            self.create_time()
        else:
            self.index = np.random.choice(np.arange(self.simulator.params_network['P']),
                                          len(timing), replace=False)
            self.timing_constant = self.offset + np.array(timing)
