import numpy as np
import copy

from NeuronModel import integrate_and_fire, izhikevich, integrate_and_fire_MSN

class synapticClass:
    parameters = {'save': False}

    def __init__(self, weightInstance, **kwargs):
        self.weight = weightInstance
        self.I_output = [np.zeros(self.weight.neurons_output.P)]
        self.current_I_output = self.I_output[-1]
        self.save = self.parameters['save']

    def subclass(self, params, **kwargs):
        self.parameters = copy.deepcopy(params)
        for key in params.keys():
            if key in kwargs.keys():
                self.parameters[key] = kwargs[key]

    def iterate(self, dt):
        pass

    def update(self):
        if self.save:
            self.I_output.append(self.current_I_output)
        else:
            self.I_output = [self.current_I_output]

    def plot_I_output(self, ax):
        ax.plot(self.weight.time, [self.I_output[i][0] for i in range(len(self.weight.time))])


class rateBased(synapticClass):
    parameters = {'save': False, 'rho': lambda x: x}

    def __init__(self, weightInstance, **kwargs):
        synapticClass.__init__(self, weightInstance, **kwargs)
        synapticClass.subclass(self, rateBased.parameters, **kwargs)
        if self.weight.neurons_input.__class__.__name__ == 'rate':
            pass
        else:
            raise NameError('Error in the input neurons class')

    def iterate(self, dt):
        if self.weight.connectivity:
            self.current_I_output = np.matmul(self.weight.weight[-1] * self.weight.connectivity_matrix,
                                              self.parameters['rho'](self.weight.neurons_input.value(index=1)))
        else:
            self.current_I_output = np.matmul(self.weight.weight[-1],
                                              self.parameters['rho'](self.weight.neurons_input.value(index=1)))


class potentialBased(synapticClass):
    parameters = {'save': False}

    def __init__(self, weightInstance, **kwargs):
        synapticClass.__init__(self, weightInstance, **kwargs)
        synapticClass.subclass(self, rateBased.parameters, **kwargs)

    def iterate(self, dt):
        if self.weight.connectivity:
            self.current_I_output = np.matmul(self.weight.weight[-1] * self.weight.connectivity_matrix,
                                              self.weight.neurons_input.value(index=1))
        else:
            self.current_I_output = np.matmul(self.weight.weight[-1], self.weight.neurons_input.value(index=1))


class gapJuctions(synapticClass):
    parameters = {'save': False, 'rho': lambda x: x}

    def __init__(self, weightInstance, **kwargs):
        synapticClass.__init__(self, weightInstance, **kwargs)
        synapticClass.subclass(self, rateBased.parameters, **kwargs)

    def iterate(self, dt):
        if self.weight.connectivity:
            self.current_I_output = np.matmul(self.weight.weight[-1] * self.weight.connectivity_matrix,
                                              self.weight.neurons_input.value(index=1)) - np.matmul(
                self.weight.weight[-1] * self.weight.connectivity_matrix,
                np.ones_like(self.weight.neurons_output.value(index=1))) * self.weight.neurons_output.value(index=1)
        else:
            self.current_I_output = np.matmul(self.weight.weight[-1], self.weight.neurons_input.value(index=1)) \
                                    - np.matmul(self.weight.weight[-1], np.ones_like(
                                        self.weight.neurons_output.value(index=1))) * \
                                    self.weight.neurons_output.value(index=1)


class spikeBased(synapticClass):
    parameters = {'save': False, 'taus': None, 'delay': 0., 'Vs': None, 'gs': 1., 'alpha': False}

    def __init__(self, weightInstance, **kwargs):
        synapticClass.__init__(self, weightInstance, **kwargs)
        synapticClass.subclass(self, spikeBased.parameters, **kwargs)
        if self.weight.neurons_input.parameters['spike']:
            pass
        else:
            raise NameError('Error in the input neurons class')
        self.P = self.weight.neurons_input.P
        self.presynaptic_spike = [[] for _ in range(self.P)]

    def compute_g(self, t):
        g = np.zeros(self.P)
        for i in range(self.P):
            for ti in self.presynaptic_spike[i]:
                if self.parameters['alpha']:
                    g[i] += (t-ti)/1000.*np.exp(-(t-ti)/self.parameters['taus'])
                else:
                    g[i] += np.exp(-(t - ti) / self.parameters['taus'])
        if self.weight.connectivity:
            g = np.matmul(self.weight.weight[-1] * self.weight.connectivity_matrix, g)
        else:
            g = np.matmul(self.weight.weight[-1], g)
        return g

    def iterate(self, dt):
        if self.parameters['taus'] is None:
            if self.weight.connectivity:
                if self.weight.neurons_input.parameters['save']:
                    self.current_I_output = np.matmul(self.weight.weight[-1] * self.weight.connectivity_matrix,
                                                      self.weight.neurons_input.spike_count[
                                                                  max(len(self.weight.neurons_input.time) - 1,
                                                                      -1 - self.parameters['delay'])])
                else:
                    self.current_I_output = np.matmul(self.weight.weight[-1] * self.weight.connectivity_matrix,
                                                      self.weight.neurons_input.spike_count[-1])
            else:
                if self.weight.neurons_input.parameters['save']:
                    self.current_I_output = np.matmul(self.weight.weight[-1], self.weight.neurons_input.spike_count[
                       max(len(self.weight.neurons_input.time) - 1, -1 - self.parameters['delay'])])
                else:
                    self.current_I_output = np.matmul(self.weight.weight[-1], self.weight.neurons_input.spike_count[-1])
            if isinstance(self.weight.neurons_output, integrate_and_fire) or isinstance(
                    self.weight.neurons_output, integrate_and_fire_MSN):
                self.current_I_output *= self.weight.neurons_output.parameters['tau']/dt
            elif isinstance(self.weight.neurons_output, izhikevich):
                self.current_I_output *= self.weight.neurons_output.parameters['C']/dt
            else:
                raise NameError('Not set up for neurons class')
        else:
            for i in range(self.weight.neurons_input.P):
                if self.weight.neurons_input.spike_count[-1][i] == 1:
                    self.presynaptic_spike[i][:] = [x for x in self.presynaptic_spike[i]
                                                    if x >= self.weight.time[-1]-20.]
                    self.presynaptic_spike[i].append(self.weight.time[-1])
            if self.parameters['Vs'] is None:
                self.current_I_output = self.compute_g(self.weight.time[-1])*self.parameters['gs']
            else:
                self.current_I_output = self.compute_g(self.weight.time[-1]) * self.parameters['gs'] * (
                            self.parameters['Vs'] - self.weight.neurons_output.potential[-1])
