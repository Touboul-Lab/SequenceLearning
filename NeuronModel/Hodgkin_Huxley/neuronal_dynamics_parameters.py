import numpy as np

def alpha_n(x):
    return 0.02 * (x - 25.) / (1. - np.exp(-(x - 25.) / 9.))


def alpha_m(x):
    return 0.182 * (x + 35.) / (1. - np.exp(-(x + 35.) / 9.))


def alpha_h(x):
    return 0.25 * np.exp(-(x + 90.) / 12.)


def beta_n(x):
    return -0.002 * (x - 25.) / (1. - np.exp((x - 25.) / 9.))


def beta_m(x):
    return -0.124 * (x + 35.) / (1. - np.exp((x + 35.) / 9.))


def beta_h(x):
    return 0.25 * np.exp(-(x + 90.) / 12.) * np.exp((x + 62.) / 6.)


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
              'E_l': -65., 'g_l': 0.3,
              'E_Na': 55., 'g_Na': 40.,
              'E_K': -77., 'g_K': 35.,
              }