# Code Implementation of 
Perez, Sylvie and Cui, Yihui and Vignoud, Gaëtan and Perrin, Elodie and Mendes, Alexandre and Touboul, Jonathan and Venance, Laurent, Striatum Expresses Region-Specific Plasticity Consistent with Distinct Memory Abilities. Available at SSRN: https://ssrn.com/abstract=3890377 or http://dx.doi.org/10.2139/ssrn.3890377

# Python Environment

Install conda and use
conda env create -file environment.yml

# In Cluster/params_DMS_article.txt Parameters used in the article figures

# In Code/FiguresDMS.py, code for all figures and associated analysis

# Use Code/main_DMS.py to reproduce our simulations

```
Compute the accuracy of the network

optional arguments:

  -h, --help            show this help message and exit
  
  --save_dir SAVE_DIR Directory where to save the output of the script

  --name NAME Name of save directory (str) (DIRECTORY1/DIRECTORY2)
  
  --P P                 Number of neurons in the input cortical layer (int)
  
  --stim_by_pattern STIM_BY_PATTERN Number of stimulation by pattern (int)

  --Apostpre APOSTPRE   Maximum amplitude of post-pre STDP (float)
  
  --Aprepost APREPOST   Maximum amplitude of pre-post STDP(float)
  
  --homeostasy HOMEOSTASY   Value of the LTP reward factor in learning and relearning phases (float)

  --epsilon EPSILON     Scaling factor for both plasticities (positive float)
  
  --noise_pattern NOISE_PATTERN   Noise in pattern generation (positive float)

  --noise_input NOISE_INPUT  Noise in the input cortical neurons and the random input neuron  (positive float)
    
  --stop_learning STOP_LEARNING Methods to stop learning from:
            * None (no stopping mechanism) (USED IN THE ARTICLE)
            * exponential_trace (modulated by an exponential memory of errors)
            * number_success (no update when a pattern has been classified correctly for num_success_params times)

  --num_success_params NUM_SUCCESS_PARAMS Parameter for when STOP_LEARNING=number_success

  --p_reward P_REWARD   Probability of a pattern to be rewarded (positive float, between 0 and 1)
  
  --stim_duration STIM_DURATION Duration of pattern (positive float)
  
  --stim_offset STIM_OFFSET Beginning of pattern (positive float)
  
  --stim_recall STIM_RECALL  Proportion of pattern presentation eta_m (positive float, between 0 and 1)
  
  --homeostasy_post HOMEOSTASY_POST Value of the LTP reward factor in initial and maintenance phases (float, 0. in the article)

  --dt DT Timestep (positive float)
  
  --num_training_initial NUM_TRAINING_INITIAL Number of pattern iterations in the initial phase (int)

  --num_training_learning NUM_TRAINING_LEARNING Number of pattern iterations in the learning phase (int)

  --num_training_maintenance NUM_TRAINING_MAINTENANCE Number of pattern iterations in the maintenance phase (int)

  --num_training_recall NUM_TRAINING_RECALL Number of pattern iterations in the recall phase (int)

  --num_simu NUM_SIMU   Number of simulations (int)
  
  --new_set NEW_SET     1 for changes of patterns at the end of the maintenance phase, 0 to keep the same (int, 0 in the article)
  
  --save Flag for saving results or not
  
  --plot Flag for plots, if not set, a plot will be generated at each simulation
  
  --random_seed RANDOM_SEED Numpy random seed used (None or int, 0 in the article)
```