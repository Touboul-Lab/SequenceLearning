import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from Articles.DMS_DLS.Code.FiguresDMS import burr_funct, burr_funct_fix

burr = 1

t = 0
init = 0.5

if burr == 0:

    x = np.linspace(0, 10., 10000)
    a = np.random.uniform(0.5, 1.)
    c = np.random.uniform(1., 2.)
    d = np.random.uniform(0., 2.)
    g = np.random.uniform(1., 2.)

    popt = a, c, d, g

    mode = popt[3]*((popt[1]-1.)/(popt[1]*popt[2]+1.))**(1./popt[1])
    y = burr_funct(t, init)(x, a, c, d, g)

    y_diff_1 = savgol_filter(y, 5, 3, deriv=1, delta=x[1]-x[0])
    y_diff_2 = savgol_filter(y, 5, 3, deriv=2, delta=x[1]-x[0])
    plt.plot(x, y, label='function')
    plt.plot(x, y_diff_1, label='first derivative')
    plt.plot(x, y_diff_2, label='second derivative')
    plt.plot([0., 10.], [0., 0.], label='xaxis')
    plt.plot([mode, mode], [-1., 1.], label='mode')

    plt.xlim(0., 10.)
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()

elif burr == 1:

    x = np.linspace(0, 10., 10000)
    a = np.random.uniform(0., 0.5)
    d = np.random.uniform(0., 2.)
    g = np.random.uniform(1., 2.)

    popt = a, d, g

    T = 1./(init-popt[0])*popt[2]/popt[1]
    y = burr_funct_fix(t, init)(x, a, d, g)
    y_diff = savgol_filter(y, 5, 3, deriv=1, delta=x[1]-x[0])
    plt.plot(x, y, label='function')
    plt.plot(x, y_diff, label='derivative')
    plt.plot([0., 10.], [init, -10./T + init], label='tangent origin')
    plt.plot([0., 10.], [-1./T, -1./T], label='derivative value')

    plt.xlim(0., 10.)
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()
