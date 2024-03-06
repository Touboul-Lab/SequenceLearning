import scipy.io as sio

class dataloader_AP():
    def __init__(self,path,num_neuron=0) :
        self.path = path
        self.data = sio.loadmat(self.path)
        self.__header__ = self.data['__header__']
        self.__version__ = self.data['__version__']
        self.__globals__ = self.data['__globals__']
        self.name_trace = [u for u in self.data.keys() if 'Trace' in u and u[-1]==str(num_neuron+1)]
    def __len__(self):
        return len(self.name_trace)
    def __getitem__(self,index):
        return self.data[self.name_trace[index]]