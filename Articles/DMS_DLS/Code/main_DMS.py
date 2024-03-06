import sys
import os
import time

import warnings
from sklearn.exceptions import ConvergenceWarning

import gc
import psutil

import numpy as np

warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='error', category=np.VisibleDeprecationWarning)

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

current_n_shutpoint = 0
while current_n_shutpoint < 10:
    try:
        from FiguresFunctions.OutputPatternClass import outputPatternDMS
        from Articles.DMS_DLS.Code.InitParams import paramsGestDMS

        current_n_shutpoint = 10
    except BrokenPipeError:
        time.sleep(np.random.randint(0, 20))
        current_n_shutpoint += 1

def mem(index, previous_time):
    process = psutil.Process(os.getpid())
    current_time = time.time()
    print('-' * 30)
    print('Iteration {:d}'.format(index))
    print('-' * 30)
    print('Memory usage : {:2.2f} MB'.format(round(process.memory_info().rss/1024.0/1024.0, 1)))
    print('RAM : {:2.2f}%'.format(psutil.virtual_memory().percent))
    print('Time since last : {:2.2f} s'.format(current_time - previous_time))
    return current_time


PARAMS_GEST = paramsGestDMS()
PARAMS_GEST.init_main_pattern()

OUTPUT = outputPatternDMS(PARAMS_GEST.args, PARAMS_GEST.params_simu, PARAMS_GEST.params_pattern,
                          PARAMS_GEST.params_network)
last_time = time.time()
for n_pattern in PARAMS_GEST.params_simu['n_pattern_list']:
    PARAMS_GEST.update_main_pattern_n_pattern(n_pattern)
    OUTPUT.create_n_pattern(n_pattern)
    if not OUTPUT.rerun:
        for num_simu in range(PARAMS_GEST.params_simu['num_simu']):
            PARAMS_GEST.update_main_pattern_n_simu(num_simu)
            SIMULATOR = PARAMS_GEST.simulator(params_simu=PARAMS_GEST.params_simu,
                                              params_pattern=PARAMS_GEST.params_pattern,
                                              params_network=PARAMS_GEST.params_network)
            if PARAMS_GEST.args.random_seed is not None:
                np.random.seed(PARAMS_GEST.random_seed[num_simu])
            SIMULATOR.init_pattern()
            SIMULATOR.run(name=str(n_pattern)+'_'+str(num_simu))
            if PARAMS_GEST.params_simu['plot']:
                SIMULATOR.plot()
            OUTPUT.add_output(SIMULATOR.params_output)
            del SIMULATOR
            gc.collect()
            last_time = mem(num_simu, last_time)
        OUTPUT.end_n_pattern()
OUTPUT.end()
