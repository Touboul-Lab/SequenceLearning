import pickle
import copy
import numpy as np
import scipy
import scipy.stats

class networkClass:
    parameters = {}

    def __init__(self, populations, weights, inputs):
        self.populations = populations
        self.weights = weights
        self.inputs = inputs

    def subclass(self, params, **kwargs):
        self.parameters = copy.deepcopy(params)
        for key in params.keys():
            if key in kwargs.keys():
                self.parameters[key] = kwargs[key]

    def iterate(self, dt):
        for current_input in self.inputs:
            current_input.iterate()
        for current_weight in self.weights:
            current_weight.iterate(dt)
        for current_pop in self.populations:
            current_pop.iterate(dt, I=sum([u.current_I_input for u in current_pop.input]) +
                                sum([u.synaptic_current.current_I_output for u in current_pop.input_weight]))

    def update(self):
        for current_input in self.inputs:
            current_input.update()
        for current_weight in self.weights:
            current_weight.update()
        for current_pop in self.populations:
            current_pop.update()

    def save(self, path):
        with open(path + ".network", "wb") as file:
            pickle.dump(self, file, -1)

    def correlation(self, bins, dt=0, cdf=None, ax=None):
        N = len(self.populations[0].time)
        stat = []
        count = 0
        num = 0
        i = 0
        while i < N:
            if self.populations[1].spike_count[i] == 1:
                stat.append(count)
                if self.populations[0].spike_count[i] == 1:
                    count = 0
                    i += 1
                    num += 1
                i += 1
            elif self.populations[0].spike_count[i] == 1:
                count = 0
                i += 1
                num += 1
            else:
                i += 1
                count += dt
        if ax is None:
            pass
        else:
            ax.hist(stat, bins=bins, weights=np.ones_like(stat)/np.sum(stat), density=1, histtype='step',
                    label='Empirical')
        if cdf is None:
            return num, stat, None
        else:
            ks = scipy.stats.kstest(stat, lambda x: cdf(x))
            return num, stat, ks


