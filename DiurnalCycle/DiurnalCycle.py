#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:48:05 2025

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

sim = 'diurnal_c_aniso_L3D'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/data/'
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

simPath = path

#%%Import variables

#Momentum fields 3D

var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC','avgL3D']
dataM = xr.open_dataarray(path+'Momentum3D/Data_Momentum_'+str(StartTime)+'.nc')
data_mom = dict()
for i in range(len(var)):
    data_mom[var[i]] = dataM.data[:,:,:,i]
del dataM
keys_list = list(data_mom.keys())
if keys_list[-1]=='avgL3D':
    data_mom['avgL3D'] = data_mom['avgL3D']*AvgIter

# Surface momentum

var2D = ['avgUstar']
dataM2D = xr.open_dataarray(path+'Momentum2D/Data_Momentum_2D_'+str(StartTime)+'.nc')
data_mom_2D = dict()
for i in range(len(var2D)):
    data_mom_2D[var2D[i]] = dataM2D.data[:,:,i]
del dataM2D

# Scalar fields 3D
    
varS = ['avgT','avgT2','avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus','avg_ds']
dataS = xr.open_dataarray(path+'Scalar3D/Data_Scalar_'+str(StartTime)+'.nc')
data_sc = dict()
for i in range(len(varS)):
    data_sc[varS[i]] = dataS.data[:,:,:,i]
del dataS

# Surface Scalar

varS2D = ['avgWstar','avgL','avgPHIm','avgPSIm','avgPHIh','avgPSIh','avgSFCval','avgSFCflux']
dataS2D = xr.open_dataarray(path+'Scalar2D/Data_Scalar_2D_'+str(StartTime)+'.nc')
data_sc_2D = dict()
for i in range(len(varS2D)):
    data_sc_2D[varS2D[i]] = dataS2D.data[:,:,i]
del dataS2D

# Anisotropy

varA = ['avgXB','avgYB']#,'avgPHIM','avgPHIH','avgPSIM','avgPSIH','avgL3D','avgustar3D','avgSCF3D']
dataA = xr.open_dataarray(path+'Anisotropy/Data_Anisotropy_'+str(StartTime)+'.nc')
data_aniso = dict()
for i in range(len(varA)):
    data_aniso[varA[i]] = dataA.data[:,:,:,i]
del dataA

#%% Bicheng Functions:
    
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

#%%Flow overview plots, mean profile of temperature, wind speed, shear stress

u_w = uvpnode2wnode(data_mom['avgU'])
v_w = uvpnode2wnode(data_mom['avgV'])

Re = {
      'uu' : data_mom['avgU2'] - data_mom['avgU']*data_mom['avgU'],
      'vv' : data_mom['avgV2'] - data_mom['avgV']*data_mom['avgV'],
      'ww' : data_mom['avgW2'] - data_mom['avgW']*data_mom['avgW'],
      'uv' : data_mom['avgUV'] - data_mom['avgU']*data_mom['avgV'],
      'uw' : data_mom['avgUW'] - u_w*data_mom['avgW'],
      'vw' : data_mom['avgVW'] - v_w*data_mom['avgW']
      }

xlabels = ['U [m/s]','V [m/s]','W [m/s]','T [K]',r"$\overline{u'w'}$ $[m^2s^{-2}]$",r"$\overline{v'w'}$ $[m^2s^{-2}]$"]

fig,axs = plt.subplots(1,6,figsize=(14,5),tight_layout=True)
axs[0].plot(np.mean(data_mom['avgU'],axis=(0,1))*uscale,np.arange(0,nz)*dz+dz/2,c='k',label='U')
axs[1].plot(np.mean(data_mom['avgV'],axis=(0,1))*uscale,np.arange(0,nz)*dz+dz/2,c='k',label='V')
axs[2].plot(np.mean(data_mom['avgW'],axis=(0,1))*uscale,np.arange(0,nz)*dz,c='k',label='W')
# axs[3].plot(np.mean(data_sc['avgT'],axis=(0,1))*Tscale,np.arange(0,nz)*dz+dz/2,c='k',label='T')
axs[4].plot(np.mean(Re['uw'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='k',label='Ruw')
axs[4].plot(np.mean(-data_mom['avgtxz'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='r',label='txz')
axs[5].plot(np.mean(Re['vw'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='k',label='Rvw')
axs[5].plot(np.mean(-data_mom['avgtyz'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='r',label='tyz')

for i in range(len(axs)):
    axs[i].set_ylabel('z/zi',fontsize=15)
    axs[i].set_ylim(0,nz*dz)
    axs[i].legend()
    axs[i].axhline(0.8,c='k',ls='--')
    axs[i].set_xlabel(xlabels[i],fontsize=15)
    
plt.show()

#%% Graphical Representation of U,W,TKE:

yslice = 64
    
tke = (Re['uu'] + Re['vv'] + wnode2uvpnode(Re['ww']))/2  

#Mean Velocity colorplots
fig, axs=plt.subplots(3,1,figsize=(8,6), constrained_layout=True)
# plt1 = axs[0].pcolormesh(x_ax,z_ax,np.nanmean(tmp_u,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
# plt2 = axs[1].pcolormesh(x_ax,z_ax,np.nanmean(tmp_w,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
# plt3 = axs[2].pcolormesh(x_ax,z_ax,np.nanmean(tmp_tke,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
plt1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,data_mom['avgU'][:,yslice,:].T,cmap='YlGnBu',shading='gouraud')
plt2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_mom['avgW'][:,yslice,:].T,cmap='YlGnBu',shading='gouraud')
plt3 = axs[2].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,tke[:,yslice,:].T,cmap='YlGnBu',shading='gouraud')

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])


# axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
# axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1]);axs[2].set_xlim(x_ax[0],x_ax[-1])
axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\overline{u}(x,y_{nz/2},z)/u_*$')

axs[1].set_ylabel(r'$z/z_i$');#axs[1].set_xlabel(r'$x/z_i$') 
axs[1].set_title(r'$\overline{w}(x,y_{nz/2},z)/u_*$')

axs[2].set_xlabel(r'$x/z_i$'); axs[2].set_ylabel(r'$z/z_i$')
axs[2].set_title(r'$\overline{e}(x,y_{nz/2},z)/u_*^2$')


plt.show()

#%% Surface temperature

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D['avgSFCval']*Tscale).T,cmap='jet')
axs.set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs.set_ylabel(r'$y/z_i$ [m]', fontsize=15)
axs.set_title('Surface temperature', fontsize=15)
axs.text(0.1, 0.9,f"{np.mean(data_sc_2D['avgSFCval'] * Tscale):.2f}", transform=axs.transAxes, fontsize=15)
cbar = plt.colorbar(p)
cbar.set_label(label='T [K]',fontsize=15)
plt.show()

#%%Surface heat flux

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D['avgSFCflux']*Tscale*uscale*1.2*1005).T,cmap='jet')
axs.set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs.set_ylabel(r'$y/z_i$ [m]', fontsize=15)
axs.set_title('Surface heat flux', fontsize=15)
axs.text(0.1, 0.9,f"{np.mean(data_sc_2D['avgSFCflux'] * Tscale*uscale*1.2*1005):.2f}", transform=axs.transAxes, fontsize=15)
cbar = plt.colorbar(p)
cbar.set_label(label=r"$\overline{w'\theta'}$ [$K-ms^{-1}$]",fontsize=15)
plt.show()

#%%Surface friction velocity

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_mom_2D['avgUstar']*uscale).T,cmap='jet')
axs.set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs.set_ylabel(r'$y/z_i$ [m]', fontsize=15)
axs.set_title('Surface friction velocity', fontsize=15)
axs.text(0.1, 0.9,f"{np.mean(data_mom_2D['avgUstar'] * uscale):.2f}", transform=axs.transAxes, fontsize=15)
cbar = plt.colorbar(p)
cbar.set_label(label=r"$u_*$ [$ms^{-1}$]",fontsize=15)
plt.show()

#%%Surface obukhov length

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D['avgL']*zi).T,cmap='jet')
axs.set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs.set_ylabel(r'$y/z_i$ [m]', fontsize=15)
axs.set_title('Surface Obukhov length', fontsize=15)
axs.text(0.1, 0.9,f"{np.mean(data_sc_2D['avgL'] * zi):.2f}", transform=axs.transAxes, fontsize=15)
cbar = plt.colorbar(p)
cbar.set_label(label=r"$L$ [$m$]",fontsize=15)
plt.show()

#%%3D Obukhov length 

ustar = ((Re['uw']-data_mom['avgtxz'])**2 + (Re['vw']-data_mom['avgtyz'])**2)**(1/4)
# heat_flux = data_sc['avgWT'] - data_mom['avgW']*uvpnode2wnode(data_sc['avgT']) - data_sc['avgWT_sgs']
# T_w = uvpnode2wnode(data_sc['avgT'])
# T_w[:,:,0] = data_sc_2D['avgSFCval']
# L = -(ustar**3)*T_w/(0.4*9.81*heat_flux)

fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(np.mean(L,axis=(0,1)),np.arange(0,nz)*dz,c='k',label='L')
axs.plot(np.mean(data_mom['avgL3D'],axis=(0,1))*zi,np.arange(0,nz)*dz,c='k',label='L')
# axs.plot(np.mean(heat_flux,axis=(0,1))*Tscale*uscale,np.arange(0,nz)*dz,c='k',label='heat flux')
# axs.plot(np.mean(data_sc_2D['avgSFCflux'])*Tscale*uscale,dz/2,'o')
# axs.plot(np.mean(T_w*Tscale,axis=(0,1)),np.arange(0,nz)*dz,c='k',label='T')
# axs.plot(np.mean(data_sc['avgT'],axis=(0,1))*Tscale,np.arange(0,nz)*dz,c='r')
# axs.plot(np.mean(data_sc_2D['avgSFCval'])*Tscale,dz/2,'o')
# axs.plot(np.mean(ustar,axis=(0,1)),np.arange(0,nz)*dz,c='k',label='ustar')
# axs.plot(np.mean(data_mom_2D['avgUstar'])*uscale,dz/2,'o')
axs.set_ylabel('z',fontsize=15)
axs.set_xlabel('L',fontsize=15)
axs.legend()

plt.show()

#%%Variances vs z/L

phi_u = np.sqrt(Re['uu']-data_mom['avgtxx'])/ustar
phi_u_1D = phi_u.flatten()

z3D = np.zeros((nx,ny,nz),order='F')
for i in range(0,nx):
    for j in range(0,ny):
        z3D[i,j,:] = np.arange(0,nz)*dz*zi + dz*zi/2
z1D = z3D.flatten()
L1D = (data_mom['avgL3D']*zi).flatten()
zeta = z1D/L1D

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.scatter(zeta,phi_u_1D,c='k')

# axs.set_xlim(1e-4,1e2)
# axs.set_xscale('log')

# axs.set_ylim(0,10)

plt.show()

#%% Check values of anisotropy invariants for the simulation with the anisotropy module

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering, Anisotropy

xslice = 64
yslice = 64
zslice = 128

fig,axs = plt.subplots(1,1,tight_layout=True)

# p=axs.pcolormesh(np.arange(0,ny)*dy,np.arange(0,nz)*dz,data_aniso['avgYB'][xslice,:,:].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
p=axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_aniso['avgYB'][:,yslice,:].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
# p=axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,data_aniso['avgYB'][:,:,zslice].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
cbar = plt.colorbar(p)

plt.show()

#%% Anisotropy for the simulation with the anisotropy module

#%%Calculate GABLS time and sfc temperature cycle

dt = 0.075
jtt = 58*60*60/dt
diurnal_start = 1
jtt_vec = np.arange(1,int(jtt))
sfc_T = np.zeros_like(jtt_vec,dtype='float64')

for i in range(len(jtt_vec)):
    
    gabls_time = 13.869054597370143 + ((jtt_vec[i]-diurnal_start)*dt)/3600
    
    # print(f'GABLS time for jt = {jtt} is {gabls_time}')
    
    if gabls_time <= 17.400266746568022:
        sfc_T[i] = -10.0 - 25.0 * np.cos(gabls_time * 0.22 + 0.2)  # DAY 1
    elif 17.400266746568022 < gabls_time <= 29.984273434101986:
        sfc_T[i] = -0.54 * gabls_time + 15.2  # NIGHT 1
    elif 29.984273434101986 < gabls_time <= 41.935606343477687:
        sfc_T[i] = -7.0 - 25.0 * np.cos(gabls_time * 0.21 + 1.8)  # DAY 2
    elif 41.935606343477687 < gabls_time <= 53.310447312949172:
        sfc_T[i] = -0.37 * gabls_time + 18.0  # NIGHT 2
    elif 53.310447312949172 < gabls_time <= 65.618603812525905:
        sfc_T[i] = -4.0 - 25.0 * np.cos(gabls_time * 0.22 + 2.5)  # DAY 3
    elif gabls_time > 65.618603812525905:
        sfc_T[i] = 4.4  # NIGHT 3
    
    # print(f'The surface temp should be {(sc_sfcval+273.15)}')
    
#%%Plot sfc temperature cycle

import matplotlib.dates as mdates
from datetime import datetime, timedelta

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/DiurnalCycle/'

# Start time
start_time = datetime(2023, 1, 1, 17, 0, 0)  # Example date, adjust as needed

# Time step and total duration
dt = 0.075  # seconds
total_seconds = 58 * 60 * 60  # 56 hours
num_points = int(total_seconds / dt)

# Create time vector in seconds
time_seconds = np.linspace(0, total_seconds, num_points)

# Convert to datetime objects (optional, slower but human-readable)
time_vector = [start_time + timedelta(seconds=float(t)) for t in time_seconds]

jt = 1296000 + 43*AvgIter
event_time = start_time + timedelta(hours=jt*0.075/3600)
diurnal_start = 1
    
gabls_time = 13.869054597370143 + ((jt-diurnal_start)*dt)/3600

# print(f'GABLS time for jt = {jtt} is {gabls_time}')

if gabls_time <= 17.400266746568022:
    T = -10.0 - 25.0 * np.cos(gabls_time * 0.22 + 0.2)  # DAY 1
elif 17.400266746568022 < gabls_time <= 29.984273434101986:
    T = -0.54 * gabls_time + 15.2  # NIGHT 1
elif 29.984273434101986 < gabls_time <= 41.935606343477687:
    T = -7.0 - 25.0 * np.cos(gabls_time * 0.21 + 1.8)  # DAY 2
elif 41.935606343477687 < gabls_time <= 53.310447312949172:
    T = -0.37 * gabls_time + 18.0  # NIGHT 2
elif 53.310447312949172 < gabls_time <= 65.618603812525905:
    T = -4.0 - 25.0 * np.cos(gabls_time * 0.22 + 2.5)  # DAY 3
elif gabls_time > 65.618603812525905:
    T = 4.4  # NIGHT 3

fig,axs = plt.subplots(1,1,figsize=(10,4),tight_layout=True)
axs.plot(time_vector[:-1],sfc_T+273.15,c='k',linewidth=2.0)
axs.plot(event_time,T+273.15,'o')
axs.set_xlabel('Time',fontsize=18)
axs.set_ylabel(r'$T_S$ [K]', fontsize=18)
axs.set_ylim(271,295)

first_tick = start_time.replace(hour=18, minute=0, second=0, microsecond=0)
if first_tick < start_time:
    first_tick += timedelta(days=1)  # make sure it's after start_time

tick_positions = []
current_tick = first_tick
while current_tick <= time_vector[-1]:
    tick_positions.append(current_tick)
    current_tick += timedelta(hours=3)

# Apply the custom ticks and labels
axs.set_xticks(tick_positions)
axs.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

xlim_start = start_time  # 16:00 on the first day
xlim_end = start_time + timedelta(hours=55)  # Midnight after 55 full hours

# Apply x-axis limits
axs.set_xlim([xlim_start, xlim_end])
# axs.set_title(f"t = {(jt-1)*dt/3600} and T = {T+273.15}")
yticks = [272, 276, 280, 284, 288, 292]
axs.set_yticks(yticks)
axs.grid()
axs.tick_params(axis='both',labelsize=15)

# plt.savefig(path_fig+'sfcT_cycle.png',dpi=300)
plt.show()












































