#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 15:19:04 2025

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from functions import get_dphidx,get_dphidy,get_dphidz,uvpnode2wnode,wnode2uvpnode,compute_d_twr,find_coordinates,average_over_selected_coords

#%%Load data

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat']

case = 8

if case < 6:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc')
else:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc')

#%%Simulation Parameters

if case < 6:
    lx = 2*np.pi
    ly = 2*np.pi
    lz = 1
    nx,ny,nz = np.shape(data[:,:,:,0])
else:
    lx = 2.88
    ly = 2.88
    lz = 0.96
    mpiProc = 32
    nx,ny,nz = np.shape(data[:,:,:,0])
    nz = nz + 5
    
zi = 1000
canopyH = 39/zi

dx = lx/nx
dy = ly/ny
dz = lz/nz

z_w = np.arange(0,nz)*dz
z_uvp = np.arange(0,nz)*dz + dz/2

#%%

ls = ['-','--']
colors = ['k','b','g','r','c','m','y','lime','violet']
Ntwr = 100
Nz_SLayer = 200

#%%Giulia data profiles

# sfc = np.load(path_to_data+'../input_txt_files/'+cases[case]+'/sfc.npy')

# coord_f = np.argwhere(sfc==1.6)
# coord_p = np.argwhere(sfc==0)
# idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
# idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)
# sel_coord_f = coord_f[idx_f]
# sel_coord_p = coord_p[idx_p]
# del coord_f,coord_p,idx_f,idx_p

# # T_13 = wnode2uvpnode(((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23])))
# # T_23 = wnode2uvpnode(((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24])))
# # cov_turb = -np.sqrt((T_13)**2 + (T_23)**2)
# # ustar = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height
# # del T_13,T_23,cov_turb

# TwrAvgProf_f = np.zeros((Nz_SLayer),order='F')
# TwrAvgProf_p = np.zeros((Nz_SLayer),order='F')

# w3_t = wnode2uvpnode(data.data[:,:,:,33] - 3*data.data[:,:,:,2]*data.data[:,:,:,6] + 2*data.data[:,:,:,2]**3)
# w2_t = wnode2uvpnode(data.data[:,:,:,6] - data.data[:,:,:,2]**2)
# # w3_t = wnode2uvpnode(data.data[:,:,:,25] - 3*data.data[:,:,:,0]*data.data[:,:,:,4] + 2*data.data[:,:,:,0]**3)
# # w2_t = wnode2uvpnode(data.data[:,:,:,4] - data.data[:,:,:,0]**2)
# skewness = w3_t/(w2_t**1.5)

# TwrAvgProf_f = average_over_selected_coords(skewness[:,:,:Nz_SLayer], sel_coord_f)
# TwrAvgProf_p = average_over_selected_coords(skewness[:,:,:Nz_SLayer], sel_coord_p)

# fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(TwrAvgProf_f,z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[0],label='Forested')
# axs.plot(TwrAvgProf_p,z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Patches')

# axs.set_xlabel(r"$Sk_w$",fontsize=15)
# axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
# axs.set_ylim(0,2)
# axs.axhline(1,ls='--',c='k')

# plt.show() 

#%%Save profiles

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'Skew_' + cases[case] + '_f.npy',TwrAvgProf_f)
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'Skew_' + cases[case] + '_p.npy',TwrAvgProf_p)

#%%Ben data profiles

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord_f = find_coordinates(intf,Ntwr,'flat')
# coord_p = find_coordinates(intf,Ntwr,'max')
# coord_v = find_coordinates(intf,Ntwr,'min')

z_profile = np.arange(nz) * dz * zi
zeds = np.ones((nx, ny, 1)) * z_profile
dist = copy.deepcopy(zeds)
del zeds,z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
mask4D = np.expand_dims(mask, axis=-1)
data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
del mask,mask4D

# Pre-extract needed variable indices
idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

w3_t = data.data[:,:,:,33] - 3*data.data[:,:,:,2]*data.data[:,:,:,6] + 2*data.data[:,:,:,2]**3
w2_t = data.data[:,:,:,6] - data.data[:,:,:,2]**2
skewness = w3_t/(w2_t)**1.5

# skew_p = np.zeros((Ntwr,Nz_SLayer))
# skew_v = np.zeros((Ntwr,Nz_SLayer))
skew_f = np.zeros((Ntwr,Nz_SLayer))

# for idx, (ix,iy) in enumerate(coord_p):
#     z_start = np.argmax(dist[ix,iy,:] > 0) - 5
#     skew_p[idx,:] = skewness[ix,iy,z_start:z_start+Nz_SLayer]
    
# for idx, (ix,iy) in enumerate(coord_v):
#     z_start = np.argmax(dist[ix,iy,:] > 0) - 5
#     skew_v[idx,:] = skewness[ix,iy,z_start:z_start+Nz_SLayer]
    
for idx, (ix,iy) in enumerate(coord_f):
    z_start = np.argmax(dist[ix,iy,:] > 0) - 5
    skew_f[idx,:] = skewness[ix,iy,z_start:z_start+Nz_SLayer]

del w3_t,w2_t,skewness

fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(np.mean(skew_p,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[0],label='Peak')
# axs.plot(np.mean(skew_v,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Valley')
axs.plot(np.mean(skew_f,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Flat')

axs.set_xlabel(r"$\overline{U}/u_*$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
axs.set_ylim(0,10)
axs.axhline(1,ls='--',c='k')

plt.show() 

#%%Save profiles

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'Skew_' + cases[case] + '_p.npy',np.mean(skew_p,axis=(0)))
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'Skew_' + cases[case] + '_v.npy',np.mean(skew_v,axis=(0)))
np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'Skew_' + cases[case] + '.npy',np.mean(skew_f,axis=(0)))

















































