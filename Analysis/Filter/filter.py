import numpy as np
import scipy
import scipy.signal

def pad(x,K):
    before = x[:K][::-1]
    after = x[-K:][::-1]
    x_padded = np.concatenate([before,x,after])
    return x_padded

class filter_trace():
    def __init__(self,range_filter = 1,type_filter = 'mean', M=None, p=None) :
        self.range_filter = range_filter
        self.type_filter = type_filter
        if self.type_filter == 'chung' :
            self.K = len(self.range_filter)
            self.M = M
            self.p = p
    def compute(self,x):
        #Init by adding range_filter value before/after
        P = np.shape(x)[0]
        if self.type_filter == 'chung' :
            filter_f = []
            filter_b = []
            w_f = []
            w_b = []
            for k in range(self.K) :
                x_padded = pad(x,self.range_filter[k])
                x_filtered_f = np.zeros(P)
                x_filtered_b = np.zeros(P)
                for i in range(P) :
                    x_filtered_f[i] = np.mean(x_padded[i:i+self.range_filter[k]+1,1]) 
                    x_filtered_b[i] = np.mean(x_padded[i+self.range_filter[k]+1:i+2*self.range_filter[k]+1,1])  
                x_padded = pad(x,self.M)
                x_filtered_f_padded = pad(x_filtered_f,self.M)
                x_filtered_b_padded = pad(x_filtered_b,self.M)
                f_i = np.zeros(P)
                b_i = np.zeros(P)
                for i in range(P) :
                    f_i[i] = (np.sum(np.square(x_padded[i:i+self.M+1,1]-x_filtered_f_padded[i:i+self.M+1])))**(-self.p)
                    b_i[i] = (np.sum(np.square(x_padded[i+self.M+1:i+2*self.M+1,1]-x_filtered_b_padded[i+self.M+1:i+2*self.M+1])))**(-self.p)
                filter_f.append(x_filtered_f)
                filter_b.append(x_filtered_b)
                w_f.append(f_i)
                w_b.append(b_i)
            filter_f=np.array(filter_f)
            filter_b=np.array(filter_b)
            w_f=np.array(w_f)
            w_b=np.array(w_b)
            #normalize
            normalize = np.sum(w_f,axis=0) + np.sum(w_b,axis=0)
            w_f=w_f/normalize
            w_b=w_b/normalize
            x_filtered = np.zeros(np.shape(x))
            x_filtered[:,0] = x[:,0]
            x_filtered[:,1] = np.sum(filter_f*w_f+filter_b*w_b,axis=0)
            np.put(x_filtered[:,1],np.argwhere(np.isnan(x_filtered[:,1])),x[:,1])
            return x_filtered
                
        else :
            before = np.repeat(np.array([x[0]]),self.range_filter,axis=0)
            after = np.repeat(np.array([x[-1]]),self.range_filter,axis=0)
            x_padded = np.concatenate([before,x,after])
            x_filtered = np.zeros(np.shape(x))
            x_filtered[:,0] = x[:,0]
            for i in range(P) :
                if self.type_filter == 'mean' :
                    x_filtered[i,1] = np.mean(x_padded[i:i+2*self.range_filter+1,1])
                elif self.type_filter == 'median' :
                    x_filtered[i,1] = np.median(x_padded[i:i+2*self.range_filter+1,1])
            np.put(x_filtered[:,1],np.argwhere(np.isnan(x_filtered[:,1])),x[:,1])
            return x_filtered
        
class savgol_filter():
    def __init__(self,range_filter = 31, poly = 5, normalize = False) :
        self.range_filter = range_filter
        self.poly = poly
        self.normalize = normalize

    def compute(self,x):
        if self.normalize:
            self.delta = x[1, 0] - x[0, 0]
        else:
            self.delta = 1
        x_derived = np.zeros((np.shape(x)[0],4))
        x_derived[:,0] = x[:,0]
        x_derived[:,1] = scipy.signal.savgol_filter(x[:,1],self.range_filter,self.poly,deriv=0, delta = self.delta)
        x_derived[:,2] = scipy.signal.savgol_filter(x[:,1],self.range_filter,self.poly,deriv=1, delta = self.delta)
        x_derived[:,3] = scipy.signal.savgol_filter(x[:,1],self.range_filter,self.poly,deriv=2, delta = self.delta)
        return x_derived