#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:02:21 2025

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

#%%#Simulation parameters

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/DiurnalCycle/'
sim_a = 'diurnal_c_aniso_L3D'
path_a = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_a+'/data/Scalar2D/'
sim_na = 'diurnal_c_noaniso_L3D_fix'
path_na = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_na+'/data/Scalar2D/'
StartTime = 1296000
AvgIter = 24000

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.075 #05; %0.000005;
zi = 3000.0 #in m
uscale = 0.4 #set equal to whatever is in parameters.py
Tscale = 288 #in K
textsize = 20
wbase = 1000

nx = 128
ny = 128
nz = 384
lx = 1000*np.pi/zi
ly = 1000*np.pi/zi
lz = 3000/zi
dx = lx/nx
dy = ly/ny
dz = lz/nz

plot_profile = 0
plot_color = 0

#%% Simulation path

# simPath = path

#%%Import variables

Nfiles = 52
heat_flux_a = dict()
heat_flux_na = dict()
for i in range(0,Nfiles):
    # dataS2D_a = xr.open_dataarray(path_a+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc')
    heat_flux_a[str(StartTime + AvgIter*i)] = xr.open_dataarray(path_a+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,-1]
    # dataS2D_na = xr.open_dataarray(path_na+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc')
    heat_flux_na[str(StartTime + AvgIter*i)] = xr.open_dataarray(path_na+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,-1]
    # del dataS2D_a,dataS2D_na
    print(f'Done with File: {i}')
    
#%%Load csv data for comparison

path_csv = '/uufs/chpc.utah.edu/common/home/u1450851/CASES99/'

HF_K = pd.read_csv(path_csv+'Kumar_HF.csv',header=None,index_col=False)
HF_V = pd.read_csv(path_csv+'Varun_HF.csv',header=None,index_col=False)
HF_O = pd.read_csv(path_csv+'Obs_HF.csv',header=None,index_col=False)

#%%Plot time series of ustar horizontally averaged
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Start time
start_time = datetime(2023, 1, 1, 17, 0, 0)  # Example date, adjust as needed
# Time step and total duration
total_seconds = 55 * 60 * 60  # 56 hours
num_points = int(total_seconds / dt)
# Create time vector in seconds
time_seconds = np.linspace(0, total_seconds, num_points)
# Convert to datetime objects (optional, slower but human-readable)
time_vector = [start_time + timedelta(seconds=float(t)) for t in time_seconds]

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(10,5))

# for i in range(0,Nfiles):
#     event_time = start_time + timedelta(hours=(StartTime + AvgIter*i)*dt/3600)
#     axs.plot(event_time,np.mean(heat_flux_a[str(StartTime + AvgIter*i)])*uscale*Tscale*1005,'o',c='k',label='Aniso')
#     axs.plot(event_time,np.mean(heat_flux_na[str(StartTime + AvgIter*i)])*uscale*Tscale*1005,'o',c='r',label='NoAniso')
    
for i in range(0,Nfiles):
    event_time = start_time + timedelta(hours=(StartTime + AvgIter*i)*dt/3600)
    axs.plot((StartTime + AvgIter*i)*0.075/3600,np.mean(heat_flux_a[str(StartTime + AvgIter*i)])*uscale*Tscale*1005,'o',c='k',label='Aniso')
    axs.plot((StartTime + AvgIter*i)*0.075/3600,np.mean(heat_flux_na[str(StartTime + AvgIter*i)])*uscale*Tscale*1005,'o',c='r',label='NoAniso')
    
axs.plot(HF_K[0]+7,HF_K[1],c='k',label='Kumar')
axs.plot(HF_V[0]+7,HF_V[1],c='r',label='Sharma')
axs.plot(HF_O[0]+7,HF_O[1],c='g',label='Obs')
    
axs.set_ylabel(r"$\overline{w'\theta'}$ $[W m^{-2}]$",fontsize=15)
axs.set_xlabel('Time',fontsize=15)
axs.set_ylim(-100,300)
axs.axhline(0,c='k',ls='--')
handles, labels = axs.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs.legend(by_label.values(), by_label.keys(),fontsize=15)

# first_tick = start_time.replace(hour=18, minute=0, second=0, microsecond=0)
# if first_tick < start_time:
#     first_tick += timedelta(days=1)  # make sure it's after start_time

# tick_positions = []
# current_tick = first_tick
# while current_tick <= time_vector[-1]:
#     tick_positions.append(current_tick)
#     current_tick += timedelta(hours=3)

# # Apply the custom ticks and labels
# axs.set_xticks(tick_positions)
# axs.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

# xlim_start = start_time + timedelta(hours=24)
# xlim_end = start_time + timedelta(hours=55)

# # Apply x-axis limits
# axs.set_xlim([xlim_start, xlim_end])
axs.grid()
plt.tick_params(axis='both',which='major',labelsize=15)

# plt.savefig(path_fig+'HF_ts_sfc.png',dpi=300)
plt.show()

#%%