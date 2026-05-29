#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 06:22:16 2025

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
import mpl_scatter_density

#%%#Simulation parameters

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/DiurnalCycle/'
sim_a = 'diurnal_c_aniso_L3D'
path_a = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_a+'/data/Momentum3D/'
sim_na = 'diurnal_c_noaniso_L3D_fix'
path_na = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_na+'/data/Momentum3D/'
StartTime = 1296000
AvgIter = 24000
Nfiles = 52
StartFile = 27
EndFile = 40

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

#%%Import RAV L from RAV surface files

L_a = dict()
L_na = dict()
kvonk = 0.41
g=9.81*zi/(uscale**2)

for i in range(StartFile,EndFile):

    L = xr.open_dataarray(path_a+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,1]
    L_a[str(StartTime + AvgIter*i)] = L
    
    #-----------------------------------------------------------------------------------------------------------------
    
    L = xr.open_dataarray(path_na+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,1]
    L_na[str(StartTime + AvgIter*i)] = L
    
    print(f'Done with File: {i}')
    
#%%Box plot of L

z0 = 0.03
L_flat_a = np.array([])
L_flat_na = np.array([])

for i in range(Nfiles):
    L_flat_a = np.concatenate((L_flat_a,((L_a[str(StartTime + i*AvgIter)]*zi)).flatten()))
    L_flat_na = np.concatenate((L_flat_na,((L_na[str(StartTime + i*AvgIter)]*zi)).flatten()))

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,4))

axs.boxplot([L_flat_a[(L_flat_a>0)],L_flat_na[(L_flat_na>0)]],showfliers=False)

axs.set_title(r"$z/L$",fontsize=15)

axs.set_xticklabels(['A','NA'],fontsize=15)

plt.show()

#%%PLot a histogram for L

tmp = L_flat_na[(L_flat_na>0)]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.hist(tmp,bins=100,log=True,cumulative=False,range=(np.median(tmp)-3*np.std(tmp),np.median(tmp)+3*np.std(tmp)))
axs.set_xscale('log')
plt.show()

#%%Compute L from RAV

L_rav_a = dict()
L_rav_na = dict()
ustar_a = dict()
ustar_na = dict()
kvonk = 0.41
g=9.81*zi/(uscale**2)

for i in range(StartFile,EndFile):
    T = xr.open_dataarray(path_a+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,-2]
    
    tsz = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,7]

    txz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,20]
    tyz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,21]
    
    ustar_a[str(StartTime + AvgIter*i)] = ((- txz)**2 + (- tyz)**2)**(1/4)
    
    wT_a = (- tsz)
    
    L_rav_a[str(StartTime + AvgIter*i)] = -(ustar_a[str(StartTime + AvgIter*i)]**3)*T/(kvonk*g*wT_a)
    
    #-----------------------------------------------------------------------------------------------------------------
    
    T = xr.open_dataarray(path_na+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,-2]
    tsz = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,7]

    txz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,20]
    tyz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,21]
    
    ustar_na[str(StartTime + AvgIter*i)] = ((- txz)**2 + (- tyz)**2)**(1/4)
    
    wT_na = (- tsz)
    
    L_rav_na[str(StartTime + AvgIter*i)] = -(ustar_na[str(StartTime + AvgIter*i)]**3)*T/(kvonk*g*wT_na)
    
    print(f'Done with File: {i}')

#%%Import temperature

T3D_a = dict()
T3D_na = dict()

for i in range(StartFile,EndFile):
    T3D_a[str(StartTime + i*AvgIter)] = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + i*AvgIter)+'.nc').data[:,:,0:50,0]
    T3D_na[str(StartTime + i*AvgIter)] = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + i*AvgIter)+'.nc').data[:,:,0:50,0]
    print(f'Done with File: {i}')
    
#%%Compute ustar 3D

ustar_a = dict()
ustar_na = dict()

for i in range(StartFile,EndFile):
    u = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    v = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    
    uw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,14]
    uw_uvp = wnode2uvpnode(uw)
    vw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,15]
    vw_uvp = wnode2uvpnode(vw)
    txz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,20]
    txz_uvp = wnode2uvpnode(txz)
    tyz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,21]
    tyz_uvp = wnode2uvpnode(tyz)
    
    ustar_a[str(StartTime + AvgIter*i)] = ((uw_uvp - u*w_uvp - txz_uvp)**2 + (vw_uvp - v*w_uvp - tyz_uvp)**2)**(1/4)
    
    u = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    v = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    w = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    
    uw = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,14]
    uw_uvp = wnode2uvpnode(uw)
    vw = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,15]
    vw_uvp = wnode2uvpnode(vw)
    txz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,20]
    txz_uvp = wnode2uvpnode(txz)
    tyz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,21]
    tyz_uvp = wnode2uvpnode(tyz)
    
    ustar_na[str(StartTime + AvgIter*i)] = ((uw_uvp - u*w_uvp - txz_uvp)**2 + (vw_uvp - v*w_uvp - tyz_uvp)**2)**(1/4)
    print(f'Done with File: {i}')
    
#%%Compute the heat flux

wT_a = dict()
wT_na =dict()

for i in range(StartFile,EndFile):
    
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    wT = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    wT_uvp = wnode2uvpnode(wT)
    tsz = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,7]
    tsz_uvp = wnode2uvpnode(tsz)
    
    wT_a[str(StartTime + i*AvgIter)] = (wT_uvp - T3D_a[str(StartTime + i*AvgIter)]*w_uvp - tsz_uvp)
    
    w = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    wT = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    wT_uvp = wnode2uvpnode(wT)
    tsz = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,7]
    tsz_uvp = wnode2uvpnode(tsz)
    
    wT_na[str(StartTime + i*AvgIter)] = (wT_uvp - T3D_na[str(StartTime + i*AvgIter)]*w_uvp - tsz_uvp)

#%%Compute L

L_rav_a = dict()
L_rav_na = dict()

for i in range(StartFile,EndFile):
    L_rav_a[str(StartTime+i*AvgIter)] = -T3D_a[str(StartTime+i*AvgIter)]*ustar_a[str(StartTime+i*AvgIter)]**3/(g*kvonk*wT_a[str(StartTime+i*AvgIter)])
    L_rav_na[str(StartTime+i*AvgIter)] = -T3D_na[str(StartTime+i*AvgIter)]*ustar_na[str(StartTime+i*AvgIter)]**3/(g*kvonk*wT_na[str(StartTime+i*AvgIter)])

#%%

idx = 35
tmp_a = L_rav_a[str(StartTime + idx*AvgIter)][:,:,8]
tmp_na = L_rav_na[str(StartTime + idx*AvgIter)][:,:,8]

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,4))

p1=axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,zi*tmp_a.T,cmap='jet')
p2=axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,zi*tmp_na.T,cmap='jet')

for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=15)
    
axs[0].set_ylabel(r"$y/z_i$",fontsize=15)
axs[0].set_title(r"Aniso", fontsize=15)
axs[1].set_title(r"No Aniso",fontsize=15)

axs[0].text(0.1,0.8,f"Median = {np.median(tmp_a*zi):.2f}",fontsize=15,color='black',bbox=dict(facecolor='white',alpha=0.6))
axs[1].text(0.1,0.8,f"Median = {np.median(tmp_na*zi):.2f}",fontsize=15,color='black',bbox=dict(facecolor='white',alpha=0.6))
            
cbar=plt.colorbar(p1)
cbar=plt.colorbar(p2)

plt.show()

#%%Compare L from the RAV file and L computed with RAV outputs

z0 = 0.03
L_rav_flat_a = np.array([])
L_rav_flat_na = np.array([])

for i in range(Nfiles):
    L_rav_flat_a = np.concatenate((L_rav_flat_a,(z0/(L_rav_a[str(StartTime + i*AvgIter)]*zi)).flatten()))
    L_rav_flat_na = np.concatenate((L_rav_flat_na,(z0/(L_rav_na[str(StartTime + i*AvgIter)]*zi)).flatten()))

#%%
fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,4))

axs.scatter(L_rav_flat_na,L_flat_na,s=1)
axs.set_yscale('log')
axs.set_xscale('log')

plt.show()

#%%Compute sigma U






































