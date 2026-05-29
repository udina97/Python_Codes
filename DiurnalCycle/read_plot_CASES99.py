#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:11:56 2025

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
import os
import sys
import copy

#%%Import csv data for CASES99 timeseries

path = '/uufs/chpc.utah.edu/common/home/u1450851/CASES99/'

u10_K = pd.read_csv(path+'Kumar_u10.csv',header=None,index_col=False)
u10_V = pd.read_csv(path+'Varun_u10.csv',header=None,index_col=False)
u10_O = pd.read_csv(path+'Obs_u10.csv',header=None,index_col=False)

HF_K = pd.read_csv(path+'Kumar_HF.csv',header=None,index_col=False)
HF_V = pd.read_csv(path+'Varun_HF.csv',header=None,index_col=False)
HF_O = pd.read_csv(path+'Obs_HF.csv',header=None,index_col=False)

uS_K = pd.read_csv(path+'Kumar_ustar.csv',header=None,index_col=False)
uS_V = pd.read_csv(path+'Varun_ustar.csv',header=None,index_col=False)


#%%Plot timeseries

fig,axs = plt.subplots(3,1,tight_layout=True,figsize=(10,10))

axs[0].plot(u10_K[0],u10_K[1],c='k',label='Kumar')
axs[0].plot(u10_V[0],u10_V[1],c='r',label='Varun')
axs[0].plot(u10_O[0],u10_O[1],c='g',label='Obs')

axs[1].plot(HF_K[0],HF_K[1],c='k',label='Kumar')
axs[1].plot(HF_V[0],HF_V[1],c='r',label='Varun')
axs[1].plot(HF_O[0],HF_O[1],c='g',label='Obs')

axs[2].plot(uS_K[0],uS_K[1],c='k',label='Kumar')
axs[2].plot(uS_V[0],uS_V[1],c='r',label='Varun')

axs[0].set_ylim(1,7)
axs[1].set_ylim(-150,300)
axs[2].set_ylim(0.1,0.5)

for i in range(len(axs)):
    axs[i].legend()
    axs[i].set_xlim(0,48)

plt.show()
























































