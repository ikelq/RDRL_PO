# the ieee 33 bus system has been stalled with 3 IB-ERs
# change the position of those IB-ER to show a clear simulation results   
# generating the loading data
# run 118 with penality cofficient 50 for voltage violation 
# correct the state, include the PQ of slack bus for all environment
# not use sam optimizator

# add reactive power into state

import copy
import random
from typing import Dict, List, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandapower as pp
from pandas.core.frame import DataFrame
from torch.distributions import Normal
import pandas as pd
import argparse
# from sam import SAM
import datetime

from IPython.display import clear_output

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--partial", default=24, type=int) 
    parser.add_argument("--qr", default=0, type=int) 
    parser.add_argument("--env_name", default=33, type=int) 
    args = parser.parse_args()

    load_pu = np.load('load96.npy')
    gene_pu = np.load('gen96.npy')

    # parameters
    num_frames = 100000
    # test_frames = 96*360
    memory_size = 100000
    batch_size = 128
    initial_random_steps = 10000
    decay = 0    # decay time for sampling data form data buffer
    decay_average = 0   # eahc dacay_average_reward data have the same reward

    # partial = 1 : partial power loss; partial = 2 : partial power loss and voltage violation;
    # partial = 3 : partial state power loss voltage violation   # partial = 4 : partial state #partial = 5 partial state, reduce loss
    partial = args.partial
    # qr = args.qr
    env_name = args.env_name

    if partial == 0:
        import Env as Env
    if partial == 24:
        import Env_partial_state_reward_li_np as Env
        # torch.cuda.set_device(5)
   

    print(partial)
    
    for seed in [777]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        for qr in [0,1]:
            # for env_name in [33,69,118]:
            print(env_name)

            if env_name == 69:
                # ieee_model = pc.from_mpc('case69.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
                id_iber = [5, 22, 44, 63]
                id_svc = [52]  #, 33
                line_f_bus = [2,7,11,36]
                penalty_cofficient = 50
                id_svc_capacity = 2
                iber_re_capacity = 3
            if env_name ==33:
                # ieee_model = pc.from_mpc('case33_bw.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
                id_iber = [16, 31]
                id_svc = [7]
                line_f_bus = [2,5,10]
                penalty_cofficient = 50
                id_svc_capacity = 2
                iber_re_capacity = 3
            if env_name == 118:
                # id_iber = [33, 50, 53, 69, 76, 97, 106, 111]
                # id_svc = [44, 104]
                id_iber = [33, 44, 50, 53, 76, 97, 106, 111]
                id_svc = [69, 84]
                line_f_bus = [1,10,28,29,64,78,99]
                penalty_cofficient = 50
                id_svc_capacity = 2
                iber_re_capacity = 3
            #
            
            env = Env.grid_case(env_name, load_pu, gene_pu, id_iber, id_svc, line_f_bus, iber_re_capacity,
                id_svc_capacity)
            