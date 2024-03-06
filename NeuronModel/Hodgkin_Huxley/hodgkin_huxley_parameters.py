import numpy as np

def alpha_n(x):
    return (0.1 - 0.01 * (x + 65.)) / (np.exp(1. - 0.1 * (x + 65.)) - 1.)


def alpha_m(x):
    return (2.5 - 0.1 * (x + 65.)) / (np.exp(2.5 - 0.1 * (x + 65.)) - 1.)


def alpha_h(x):
    return 0.07 * np.exp(-(x + 65.) / 20.)


def beta_n(x):
    return 0.125 * np.exp(-(x + 65.) / 80.)


def beta_m(x):
    return 4. * np.exp(-(x + 65.) / 18.)


def beta_h(x):
    return 1. / (np.exp(3. - 0.1 * (x + 65.)) + 1.)


def tau_n(x):
    return 1. / (alpha_n(x) + beta_n(x))


def tau_m(x):
    return 1. / (alpha_m(x) + beta_m(x))


def tau_h(x):
    return 1. / (alpha_h(x) + beta_h(x))


def x_n(x):
    return alpha_n(x) / (alpha_n(x) + beta_n(x))


def x_m(x):
    return alpha_m(x) / (alpha_m(x) + beta_m(x))


def x_h(x):
    return alpha_h(x) / (alpha_h(x) + beta_h(x))


parameters = {'save': True, 'spike': False, 'C': 1.,
              'E_l': 10.6 - 65., 'g_l': 0.3,
              'E_Na': 115. - 65., 'g_Na': 120.,
              'E_K': -12. - 65., 'g_K': 36.,
              }