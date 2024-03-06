import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

for name in ['DLS1Hz','DLS2Hz','DLS5Hz','DMS1Hz','DMS2Hz','DMS5Hz','DMS2HzLTDpostpre','DMS2HzLTPpostpre']:
    print('../Data/'+name+'.txt')
    data = np.loadtxt('../Data/'+name+'.txt')
    fig, ax = plt.subplots(figsize=(10., 5.), ncols=1, nrows=1)
    fig.subplots_adjust(top=1., bottom=0., left=0., right=1.)

    x_interp = np.linspace(np.min(data[:,0]),np.max(data[:,0]),100)

    ax.plot(data[:,0], data[:,1], '+', label='Experimental Data')
    ax.set_xlim(-100.,100.)
    ax.set_ylim(0., 400.)
    ax.plot([0.,0.], [0.,400.], '--', color='grey')
    ax.plot([-100.,100.], [100.,100.], '--', color='grey')
    f = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear')
    y_interp = f(x_interp)
    ax.plot(x_interp, y_interp, label='interp')
    for i in np.arange(5):
        y_filter = savgol_filter(y_interp,25, i)
        ax.plot(x_interp, y_filter, label='filtered_'+str(i))
        np.savetxt('../Data/'+name+'_filtered_'+str(i)+'.txt', np.array([x_interp, y_filter]).transpose())
    ax.legend()
    plt.savefig('../Data/'+name+'_filtered.pdf')
    plt.close()
