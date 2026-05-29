#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 07:04:50 2025

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
path_a = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_a+'/data/Momentum3D/'
sim_na = 'diurnal_c_noaniso_L3D'
path_na = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_na+'/data/Momentum3D/'
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

tke_a = np.zeros((nz,Nfiles),order='F')
tke_na = np.zeros((nz,Nfiles),order='F')

for i in range(0,Nfiles):
    u = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,0]
    u_w = uvpnode2wnode(u)
    v = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,1]
    v_w = uvpnode2wnode(v)
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,2]
    uu = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,4]
    uu_w = uvpnode2wnode(uu)
    vv = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,5]
    vv_w = uvpnode2wnode(vv)
    ww = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,6]
    
    tke_a[:,i] = 0.5*np.mean((uu_w-u_w*u_w) + (vv_w-v_w*v_w) + (ww-w*w),axis=(0,1))
    
    u = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,0]
    u_w = uvpnode2wnode(u)
    v = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,1]
    v_w = uvpnode2wnode(v)
    w = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,2]
    uu = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,4]
    uu_w = uvpnode2wnode(uu)
    vv = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,5]
    vv_w = uvpnode2wnode(vv)
    ww = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,:,6]
    
    tke_na[:,i] = 0.5*np.mean((uu_w-u_w*u_w) + (vv_w-v_w*v_w) + (ww-w*w),axis=(0,1))
    
    print(f'Done with File: {i}')
    
#%%Compute 2D velocity magnitude time and z

# Rxz = np.zeros((nz,Nfiles),order='F')
# Ryz = np.zeros((nz,Nfiles),order='F')

# for i in range(0,Nfiles):
#     Rxz[:,i] = np.mean(uw[str(StartTime + AvgIter*i)] - u[str(StartTime + AvgIter*i)]*w[str(StartTime + AvgIter*i)] - txz[str(StartTime + AvgIter*i)],axis=(0,1))
#     Ryz[:,i] = np.mean(vw[str(StartTime + AvgIter*i)] - v[str(StartTime + AvgIter*i)]*w[str(StartTime + AvgIter*i)] - tyz[str(StartTime + AvgIter*i)],axis=(0,1))

# Mom_flux_a = np.sqrt(Rxz_a**2 + Ryz_a**2)
# Mom_flux_na = np.sqrt(Rxz_na**2 + Ryz_na**2)
# ustar_a = (Rxz_a**2 + Ryz_a**2)**(1/4)
# ustar_na = (Rxz_na**2 + Ryz_na**2)**(1/4)

#%%Plot

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

# p1 = axs[0].pcolormesh(mdates.date2num(time_vector),np.arange(0,nz)*dz*zi,Mom_flux_a*(uscale**2)/(ug**2),cmap='hot',vmin=0,vmax=0.001,shading='auto')
# p2 = axs[1].pcolormesh(mdates.date2num(time_vector),np.arange(0,nz)*dz*zi,Mom_flux_na*(uscale**2)/(ug**2),cmap='hot',vmin=0,vmax=0.001,shading='auto')
p1 = axs[0].pcolormesh(mdates.date2num(time_vector),np.arange(0,nz)*dz*zi,tke_a*(uscale**2)/(ug**2)*1000,vmin=0,vmax=4,cmap='hot',shading='auto')
p2 = axs[1].pcolormesh(mdates.date2num(time_vector),np.arange(0,nz)*dz*zi,tke_na*(uscale**2)/(ug**2)*1000,vmin=0,vmax=4,cmap='hot',shading='auto')
    
axs[0].set_ylabel(r"$z$ [m]",fontsize=15)
axs[1].set_ylabel(r"$z$ [m]",fontsize=15)
axs[1].set_xlabel('Time',fontsize=15)
axs[0].set_ylim(dz*zi/2,1200)
axs[1].set_ylim(dz*zi/2,1200)

axs[1].set_xlim(mdates.date2num(time_vector[0]), mdates.date2num(time_vector[-1]))

# Use HourLocator to get ticks at 18, 21, 0, 3, 6, etc.
axs[1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))

fig.autofmt_xdate()

axs[0].set_title('Aniso',fontsize=15)
axs[1].set_title('No Aniso',fontsize=15)


cbar = plt.colorbar(p1)
cbar.set_label(label=r'$TKE/u_G^2$',fontsize=15)
cbar = plt.colorbar(p2)
cbar.set_label(label=r'$TKE/u_G^2$',fontsize=15)

# plt.savefig(path_fig+'TKE_ts_prof.png',dpi=300)
plt.show()

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

height = [1,5,10,15,20,30,50]

fig,axs = plt.subplots(2,1,tight_layout=True,figsize=(10,5),sharex=True)

for i in range(len(height)):
    p1 = axs[0].plot(mdates.date2num(time_vector),tke_a[height[i],:]*(uscale**2)/(ug**2)*1000,label=f'{height[i]*dz*zi:.2f}')
    p2 = axs[1].plot(mdates.date2num(time_vector),tke_na[height[i],:]*(uscale**2)/(ug**2)*1000,label=f'{height[i]*dz*zi:.2f}')
    
axs[0].set_ylabel(r"$TKE/u_G^2$",fontsize=15)
axs[1].set_ylabel(r"$TKE/u_G^2$",fontsize=15)
axs[1].set_xlabel('Time',fontsize=15)
# axs[0].set_ylim(0,np.sqrt(3)/2)
# axs[1].set_ylim(0,np.sqrt(3)/2)

axs[1].set_xlim(mdates.date2num(time_vector[0]), mdates.date2num(time_vector[-1]))

# Use HourLocator to get ticks at 18, 21, 0, 3, 6, etc.
axs[1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))

fig.autofmt_xdate()

axs[0].set_title('Aniso',fontsize=15)
axs[1].set_title('No Aniso',fontsize=15)

for i in range(len(axs)):
    axs[i].legend()
    # axs[i].axhline(0.38,c='k',ls='--')

# plt.savefig(path_fig+'TKE_ts_heights.png',dpi=300)
plt.show()


