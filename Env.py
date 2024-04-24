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
import pandapower.converter as pc
from pandas.core.frame import DataFrame
from torch.distributions import Normal
import pandas as pd


def Relu(x: np.ndarray):
    return np.maximum(0, x)


class grid_case:

    def __init__(self,
                 env_name,
                 load_pu: np.ndarray,
                 gene_pu: np.ndarray,
                 id_iber,
                 id_svc,
                 line_f_bus,
                 iber_re_capacity,
                 id_svc_capacity):
        """Initializate."""
        self.id_iber = id_iber
        self.id_svc = id_svc
        self.env_name = env_name
        if self.env_name == 33:
            self.model = pc.from_mpc('case33_bw.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        if self.env_name == 69:
            self.model = pc.from_mpc('case69.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        if self.env_name == 118:
            self.model = pc.from_mpc('case1180zh.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        self.load_pu = load_pu
        self.gene_pu = gene_pu
        self.action_dim = len(self.id_iber) + len(self.id_svc)
        self.step_n = 0
        self.done = False
        self.n_bus = len(self.model.bus)

        # 1.32
        for i in self.id_iber:
            pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='IBVR', scaling=1.0, in_service=True,
                           max_p_mw=4, min_p_mw=0, max_q_mvar=iber_re_capacity, min_q_mvar=-iber_re_capacity, controllable=True)
        for i in self.id_svc:
            pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='SVC', scaling=1.0, in_service=True,
                           max_p_mw=0, min_p_mw=0, max_q_mvar=id_svc_capacity, min_q_mvar=0, controllable=True)

        pp.runpp(self.model, algorithm='bfsw')
        self.observation_space = copy.deepcopy(np.hstack((np.array(self.model.res_bus.vm_pu),  np.array(self.model.res_ext_grid.iloc[0]),
                                                          np.array(self.model.res_load.p_mw), np.array(self.model.res_load.q_mvar),
                                                          np.array(self.model.res_sgen.p_mw), np.array(self.model.res_sgen.q_mvar))))

        # self.id_Gp = self.id_iber
        # self.id_Gp = [x-1 for x in self.id_Gp]
        # self.id_Gq = self.id_iber + self.id_svc
        # self.id_Gq = [x+self.n_bus-1-1 for x in self.id_Gq]

        self.id_bus_load = self.model.load.bus.values

        # self.observation_space = np.hstack((np.ones(33), np.zeros(33*2+4)))
        # np.zeros(self.net.mu.shape[1] + self.net.label_mu.shape[1] + len(self.id_Gq))
        # self.injection_pq = np.zeros(self.net.mu.shape[1])
        # self.injection_pq = np.hstack((self.model.res_bus.p_mw.values[1:], self.model.res_bus.q_mvar.values[1:]))
        self.action_space = np.zeros(self.action_dim)

        self.init_load_p_mw = copy.deepcopy(self.model.load.p_mw)
        self.init_load_q_mvar = copy.deepcopy(self.model.load.q_mvar)
        self.init_max_q_mavr = copy.deepcopy(self.model.sgen.max_q_mvar)
        self.init_min_q_mavr = copy.deepcopy(self.model.sgen.min_q_mvar)

        self.init_line_r_ohm_per_km = copy.deepcopy(self.model.line.r_ohm_per_km)
        self.init_line_x_ohm_per_km = copy.deepcopy(self.model.line.x_ohm_per_km)

        # self.load_pu = self.load_pu[:,np.newaxis] * np.ones_like(self.init_load_p_mw)[np.newaxis,:]
        # self.load_pu[:370*96,:] = self.load_pu[:370*96,:] * np.random.uniform(low=0.8,high=1.2,size= self.load_pu[:370*96,:].shape)
        # self.load_pu[96*380:96*385,:] = 1.1 * self.load_pu[96*380:96*385,:]
        # self.load_pu[96*385:96*395, :] = 1.2 * self.load_pu[96*385:96*395, :]
        #
        # self.gene_pu = self.gene_pu[:,np.newaxis] * np.ones_like(self.id_iber)[np.newaxis,:]
        # self.gene_pu[:370*96,:] = self.gene_pu[:370*96,:] * np.random.uniform(low=0.8,high=1.2,size= self.gene_pu[:370*96,:].shape)
        # self.gene_pu[96*380:96*385,:] = 1.1 * self.gene_pu[96*380:96*385,:]
        # self.gene_pu[96*385:96*395, :] = 1.2 * self.gene_pu[96*385:96*395, :]
        #
        # np.save('two'+str(self.n_bus)+'load', self.load_pu)
        # np.save( 'two'+str(self.n_bus) + 'gen', self.gene_pu)


        # self.load_pu = self.load_pu[:, np.newaxis] * np.ones_like(self.init_load_p_mw)[np.newaxis, :]
        # self.load_pu = self.load_pu * np.random.uniform(low=0.8, high=1.2, size=self.load_pu.shape)
        # #
        # self.gene_pu = self.gene_pu[:, np.newaxis] * np.ones_like(self.id_iber)[np.newaxis, :]
        # self.gene_pu = self.gene_pu * np.random.uniform(low=0.8, high=1.2, size=self.gene_pu.shape)
        #
        # #

        # if env_name == 118:
        # #     low = 
        # self.load_pu = np.random.uniform(low=0.5, high=1.5, size=[110000,self.init_load_p_mw.size])
        # self.gene_pu = np.random.uniform(low=1, high=2, size=[110000,len(self.id_iber)])
        # np.save('two'+str(self.n_bus)+'load15', self.load_pu)
        # np.save( 'two'+str(self.n_bus) + 'gen15', self.gene_pu)

        self.load_pu = np.load('two'+str(self.n_bus)+'load15.npy')
        self.gene_pu = np.load('two'+str(self.n_bus) + 'gen15.npy')
