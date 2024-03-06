import scipy
import scipy.signal
from scipy.signal import butter, lfilter, freqz
import numpy as np

class butter_filter():
    def __init__(self, cutoff, fs, order=2, btype='bandstop'):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        self.b = b
        self.a = a

    def compute(self,x):
        x_filtered = np.zeros(np.shape(x))
        x_filtered[:,0] = x[:,0]        
        x_filtered[:,1] = scipy.signal.filtfilt(self.b, self.a, x[:,1])
        return x_filtered
