#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 07:46:05 2025

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
sim_na = 'diurnal_c_noaniso_L3D'
path_na = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_na+'/data/Scalar2D/'
StartTime = 1296000
AvgIter = 24000

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.075 #05; %0.000005;
zi = 3000.0 #in m
uscale = 0.4 #set equal to whatever is in parameters.py
ug = 9.5
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

#%%Bicheng functions

def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_h[:, :, -1]
  return phi_c

#%%Import variables

Nfiles = 52
phiM_a = np.zeros((Nfiles),order='F')
phiM_na = np.zeros((Nfiles),order='F')

phiH_a = np.zeros((Nfiles),order='F')
phiH_na = np.zeros((Nfiles),order='F')

for i in range(0,Nfiles):
    phiM = xr.open_dataarray(path_a+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,2] #momentum
    phiH = xr.open_dataarray(path_a+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,4] #heat
    
    phiM_a[i] = np.mean(phiM)
    phiH_a[i] = np.mean(phiH)
    
    phiM = xr.open_dataarray(path_na+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,2]
    phiH = xr.open_dataarray(path_na+'Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,4]
    
    phiM_na[i] = np.mean(phiM)
    phiH_na[i] = np.mean(phiH)
    
    print(f'Done with File: {i}')
    
#%%

from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Start time
start_time = datetime(2023, 1, 1, 20, 0, 0, 36818)  # Example date, adjust as needed
# Time step and total duration
total_seconds = 55 * 60 * 60  # 56 hours
num_points = int(total_seconds / dt)
# Create time vector in seconds
time_seconds = np.linspace(0, total_seconds, num_points)
# Convert to datetime objects (optional, slower but human-readable)
# time_vector = [start_time + timedelta(seconds=float(t)) for t in time_seconds]
time_vector = [start_time + timedelta(seconds=AvgIter * dt * i) for i in range(Nfiles)]

fig,axs = plt.subplots(2,1,tight_layout=True,figsize=(10,5),sharex=True)

p1 = axs[0].plot(mdates.date2num(time_vector),phiM_a,c='k',label='Aniso')
p1 = axs[0].plot(mdates.date2num(time_vector),phiM_na,c='r',label='NoAniso')
p2 = axs[1].plot(mdates.date2num(time_vector),phiH_a,c='k')
p2 = axs[1].plot(mdates.date2num(time_vector),phiH_na,c='r')
    
axs[0].set_ylabel(r"$\phi_M$",fontsize=15)
axs[1].set_ylabel(r"$\phi_H$",fontsize=15)
axs[1].set_xlabel('Time',fontsize=15)
for i in range(len(axs)):
    axs[i].set_ylim(0,3)
# axs[1].set_ylim(0,np.sqrt(3)/2)

axs[1].set_xlim(mdates.date2num(time_vector[0]), mdates.date2num(time_vector[-1]))

# Use HourLocator to get ticks at 18, 21, 0, 3, 6, etc.
axs[1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))

fig.autofmt_xdate()

# axs[0].set_title('Aniso',fontsize=15)
# axs[1].set_title('No Aniso',fontsize=15)

for i in range(len(axs)):
    axs[i].legend(loc='upper left')
    axs[i].axhline(1,c='k',ls='--')

# plt.savefig(path_fig+'phi_ts.png',dpi=300)
plt.show()
    











































