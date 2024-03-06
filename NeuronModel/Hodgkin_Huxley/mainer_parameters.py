import numpy as np

def alpha_n(x):
    return 0.02 * (x - 20.) / (1. - np.exp(-(x - 20.) / 9.))


def alpha_m(x):
    return 0.182 * (x + 35.) / (1. - np.exp(-(x + 35.) / 9.))


def alpha_h(x):
    return 0.024 * (x + 50.) / (1. - np.exp(-(x + 50.) / 5.))


def beta_n(x):
    return -0.002 * (x - 20.) / (1. - np.exp((x - 20.) / 9.))


def beta_m(x):
    return -0.124 * (x + 35.) / (1. - np.exp((x + 35.) / 9.))


def beta_h(x):
    return -0.0091 * (x + 75.) / (1. - np.exp((x + 75.) / 5.))


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
    return 1 / (1 + np.exp((x + 65) / 6.2))

parameters = {'save': True, 'spike': False, 'C': 1.,
              'E_l': -65., 'g_l': 0.3,
              'E_Na': 55., 'g_Na': 40.,
              'E_K': -77., 'g_K': 35.,
              }