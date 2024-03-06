from NeuronModel.NeuronClass import neuronClass
import numpy as np

class izhikevich(neuronClass):
    parameters = {'save': True, 'spike': True, 'C': 50., 'k_input': 1., 'v_rest': -80., 'v_t': -20., 'c': -55.,
                  'v_peak': 40., 'a': 0.01, 'b': -20., 'd': 150., 'noise': None, 'scale_I': None, 'tau': None}

    def __init__(self, **kwargs):
        neuronClass.__init__(self, **kwargs)
        neuronClass.subclass(self, izhikevich.parameters, **kwargs)
        self.w = [np.zeros(self.P)]
        self.current_w = self.w[-1]
        self.refractory = np.inf

    def f(self, u, current_I):
        sol = self.parameters['k_input'] * (self.parameters['v_rest'] - u) * (self.parameters['v_t']-u) + current_I
        return sol

    def iterate(self, dt, **kwargs):
        if self.parameters['scale_I'] is None:
            current_I = kwargs['I']
        else:
            current_I = kwargs['I'] * self.parameters['scale_I']
        potential = np.where(self.refractory <= 0., self.parameters['c'], self.potential[-1])
        w = self.w[-1]
        self.refractory += dt
        if self.parameters['noise'] is not None:
            next_potential = potential + dt / self.parameters['C'] * (
                        self.f(potential, current_I) - w) \
                        + np.sqrt(dt/self.parameters['C']) * np.random.normal(0., self.parameters['noise'])
        else:
            next_potential = potential + dt / self.parameters['C'] * (
                        self.f(potential, current_I) - w)
        next_w = w + dt * self.parameters['a'] * (self.parameters['b'] * (potential - self.parameters['v_rest']) - w)
        event = 1 * (next_potential > self.parameters['v_peak'])
        next_w += self.parameters['d'] * event

        self.refractory = np.where(event, 0., self.refractory)

        self.current_time = self.time[-1] + dt
        self.current_potential = np.where(event, self.parameters['v_peak'], next_potential)
        self.current_spike_count = event
        self.current_w = next_w

    def update(self):
        neuronClass.update(self)
        if self.parameters['save']:
            self.w.append(self.current_w)
        else:
            self.w = [self.current_w]

    def init_eq(self):
        self.current_time = self.time[-1]
        self.current_potential = self.parameters['v_rest'] * np.ones(self.P)
        self.current_spike_count = np.zeros(self.P)
        self.current_w = np.zeros(self.P)
        self.refractory = np.inf*np.ones(self.P)
        self.update()

    def plot_phase(self, ax, current_I=0):
        potential = np.linspace(-100., 100., 100)
        ax.plot(potential, self.f(potential, current_I), label='noise input')
