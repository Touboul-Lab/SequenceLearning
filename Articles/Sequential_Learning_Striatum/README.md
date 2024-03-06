# Code Implementation of ...

# Python Environment

Install conda and use
conda env create -file environment.yml

# In Cluster/params_pattern_article_single/dual/poisson/example.txt Parameters used in the article figures


# Use Code/main_DMS.py to reproduce our simulations

```
Compute the accuracy of the network


optional arguments:

  -h, --help            show this help message and exit
  
  --save_dir SAVE_DIR Directory where to save the output of the script

  --name NAME Name of save directory (str) (DIRECTORY1/DIRECTORY2)
  
  --P P                 Number of neurons in the input cortical layer (int)
  
  --neuronClass NEURONCLASS Choice of neurons parameters (str) can be chosen from:
                                * MSN_Izhi
                                * MSN_IAF_EXP
                                * MSN_Yim
                                and others defined in NetworkModel/StriatumNeurons.py

  --Apostpre APOSTPRE   Maximum amplitude of post-pre STDP (float)
  
  --Aprepost APREPOST   Maximum amplitude of pre-post STDP(float)
  
  --homeostasy HOMEOSTASY   Value of the LTP reward factor in learning and relearning phases (float)

  --epsilon EPSILON     Scaling factor for both plasticities (positive float)
  
  --noise_stim NOISE_STIM  Noise in the input cortical neurons (positive float)

  --noise_input NOISE_INPUT  Noise in the the random input neuron  (positive float)

  --noise_pattern NOISE_PATTERN   Noise in pattern generation (positive float)
    
  --stop_learning STOP_LEARNING Methods to stop learning from:
            * None (no stopping mechanism) (USED IN THE ARTICLE)
            * exponential_trace (modulated by an exponential memory of errors)
            * number_success (no update when a pattern has been classified correctly for num_success_params times)

  --num_success_params NUM_SUCCESS_PARAMS Parameter for when STOP_LEARNING=number_success

  --dt DT Timestep (positive float)
  
  --num_training NUM_TRAINING Number of pattern iterations (int)
  
  --stim_duration STIM_DURATION Duration of pattern (positive float)
  
  --stim_offset STIM_OFFSET Beginning of pattern (positive float)
  
  --save Flag for saving results or not
  
  --plot Flag for plots, if not set, a plot will be generated at each simulation
  
  --random_seed RANDOM_SEED Numpy random seed used (None or int, 0 in the article)

  network:
  {single,dual}
    single       Only one MSN in the network
    dual         Network with two MSNs
        --J_matrix J_MATRIX (int), choose from different connectivity patterns
            if J_matrix = 0: #no connections
                J_matrix = np.zeros((2, 2))
            elif J_matrix = 1: # both connected
                J_matrix = np.array([[0., -1.], [-1., 0.]])
            elif J_matrix = 2: # 1->2
                J_matrix = np.array([[0., -1.], [0., 0.]])
            elif J_matrix = 3: # 2->1
                J_matrix = np.array([[0., 0.], [-1., 0.]])
        --J_value J_VALUE (float, or 'random')
        --J_reward J_VREWARD (str, {'differential', 'same'})
        
  pattern:
  {list_pattern, jitter, poisson, succession, example}
    list_pattern    Task 1
        --stim_by_pattern STIM_BY_PATTERN Max number of stimulation by pattern (int)
        --repartition REPARTITION Choice of repartition of patterns, choose from {uniform, uniform_stim}
                                        * uniform, patterns chosen randomly without consideration of the number of stim
                                        * uniform_stim, patterns chosen randomly with equal 
                                            repartition between number of stim (in the article)
        --p_reward P_REWARD   Probability of a pattern to be rewarded (positive float, between 0 and 1)
        --stim_delay STIM_DELAY   Duration between each stim (positive float)
        --num_simu NUM_SIMU   Number of simulations (int)
    succession      Task 2
        --stim_delay STIM_DELAY   Duration between each stim (positive float)
    jitter          Task 3
        --stim_by_pattern STIM_BY_PATTERN Max number of stimulation by pattern (int)
        --repartition REPARTITION Choice of repartition of patterns, choose from {uniform, uniform_stim}
                                        * uniform, patterns chosen randomly without consideration of the number of stim
                                        * uniform_stim, patterns chosen randomly with equal 
                                            repartition between number of stim (in the article)
        --p_reward P_REWARD   Probability of a pattern to be rewarded (positive float, between 0 and 1)
        --stim_delay STIM_DELAY   Duration between each stim (positive float)
        --num_simu NUM_SIMU   Number of simulations (int)
    poisson         Task 4
        --p_reward P_REWARD   Probability of a pattern to be rewarded (positive float, between 0 and 1)
        --duration_poisson DURATION_POISSON Duration of the Poisson spike train (positive float)
        --noise_poisson DURATION_POISSON Intensity of the Poisson spike train (positive float)
        --num_simu NUM_SIMU   Number of simulations (int)
    example         Figure 2b
        --no_reward NO_REWARD   Presence (0) or absence (1) of reward {0,1}
        --num_simu NUM_SIMU   Number of simulations (int)
        --pattern_example PATTERN_EXAMPLE   Random noise example A or classification of a single Poisson pattern B {A,B}
        --start_weight START_WEIGHT Starting from low or high synaptic weights {low, high} (str)
```