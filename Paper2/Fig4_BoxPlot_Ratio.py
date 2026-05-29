#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:13:34 2026

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy

#%%Set up cases, paths, names

B_path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
G_path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'

B_cases = ['Flat','Sinusoidal','ATTO']
G_cases = ['Empty_9mps','Gap_8_9mps','Patch_8_9mps']

SGS = False

if SGS:
    Full = '/anisotropy.nc'
    Diag = '/anisotropy_D.nc'
else:
    Full = '/anisotropy_NoSGS.nc'
    Diag = '/anisotropy_D_NoSGS.nc'

#%%Simulation parameters

Nx_G = 256
Ny_G = 256
Nz_G = 256
Lx_G = 2*np.pi
Lz_G = 1
dx_G = Lx_G/Nx_G
dz_G = Lz_G/Nz_G
x_G = np.arange(0,Nx_G)*dx_G
z_w_G = np.arange(0,Nz_G)*dz_G
z_uvp_G = np.arange(0,Nz_G)*dz_G + dz_G/2
zi_G = 1000
Hcanopy_G = 39/zi_G

Nx_B = 256
Ny_B = 256
Nz_B = 384
Lx_B = 2.880
Lz_B = 0.960
dx_B = Lx_B/Nx_B
dz_B = Lz_B/Nz_B
x_B = np.arange(0,Nx_B)*dx_B
z_w_B = np.arange(0,Nz_B)*dz_B
z_uvp_B = np.arange(0,Nz_B)*dz_B + dz_B/2
zi_B = 1000
Hcanopy_B = 39/zi_B

#%%Giulia data

case = 0

aniso = xr.open_dataarray(G_path + G_cases[case] + Full)
anisoD = xr.open_dataarray(G_path + G_cases[case] + Diag)
mom = xr.open_dataarray(G_path + G_cases[case] + '/Data_Momentum_4TKE.nc')
TKE_terms = xr.open_dataarray(G_path + G_cases[case] + '/TKE_terms.nc')
tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
# tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)

ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data

fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.boxplot(ratio[:,:,15:30].flatten(), sym='')
axs.boxplot(tmpNorm[:,:,2].flatten(), sym='')

plt.show()

#%%Ben data

case = 0

aniso = xr.open_dataarray(B_path + B_cases[case] + Full)
anisoD = xr.open_dataarray(B_path + B_cases[case] + Diag)
mom = xr.open_dataarray(B_path + B_cases[case] + '/dataTKE.nc')
dist = xr.open_dataarray(B_path + B_cases[case] + '/dist.nc')[:,:,:,0].data
TKE_terms = xr.open_dataarray(B_path + B_cases[case] + '/TKE_terms.nc')

for j in range(0,3):
    (aniso[:,:,:, j].data)[dist<0] = np.nan
    (anisoD[:,:,:, j].data)[dist<0] = np.nan
    
tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
tmpDIS[dist < 0] = np.nan
# tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
tmpRES[dist < 0] = np.nan
tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data

ratio[dist < 0] = np.nan

plt.subplots(1,1,tight_layout=True)

plt.boxplot(tmpNorm[(dist > 1.5*39) & (dist < 3*39)].flatten(), sym='')

plt.show()

#%%Tyler data

import pickle

with open('/uufs/chpc.utah.edu/common/home/u1450851/Tyler_yB/yb_ratio_NEON_neutral2.p','rb') as file:
    Ty_data = pickle.load(file)

#%%Dictionary with ratio values from tower heights 

bp_dict = dict()

for i in range(len(G_cases)):
    aniso = xr.open_dataarray(G_path + G_cases[i] + Full)
    anisoD = xr.open_dataarray(G_path + G_cases[i] + Diag)
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    if i==0:
        bp_dict[G_cases[i]] = ratio[:,:,2].flatten()
    else:
        bp_dict[G_cases[i]] = ratio[:,:,15:30].flatten()
    
for i in range(len(B_cases)):
    aniso = xr.open_dataarray(B_path + B_cases[i] + Full)
    anisoD = xr.open_dataarray(B_path + B_cases[i] + Diag)
    dist = xr.open_dataarray(B_path + B_cases[i] + '/dist.nc')[:,:,:,0].data
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    ratio[dist < 0] = np.nan
    
    bp_dict[B_cases[i]] = ratio[(dist > 1.5*39) & (dist < 3*39)].flatten()

for i in range(len(Ty_data)):
    keys = list(Ty_data.keys())
    bp_dict[keys[i]] = Ty_data[keys[i]]

new_keys = ['Empty','g800','i800','Flat','Sin','ATTO','G_S','G_U','F_S','F_U']
bp_dict = dict(zip(new_keys, bp_dict.values()))

#%%Dictionary with TKE residual from tower heights

tke_dict = dict()

for i in range(len(G_cases)):
    TKE_terms = xr.open_dataarray(G_path + G_cases[i] + '/TKE_terms.nc')
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
    tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)
        
    if i==0:
        tke_dict[G_cases[i]] = tmpNorm[:,:,2].flatten()
    else:
        tke_dict[G_cases[i]] = tmpNorm[:,:,15:30].flatten()
    
for i in range(len(B_cases)):
    TKE_terms = xr.open_dataarray(B_path + B_cases[i] + '/TKE_terms.nc')
    dist = xr.open_dataarray(B_path + B_cases[i] + '/dist.nc')[:,:,:,0].data
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
    val = np.nanmedian(tmpRES[(dist>38) & (dist<40)])
    tmpRES[abs(tmpRES) < 0.05*val] = 0
    # tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)
    
    tke_dict[B_cases[i]] = tmpNorm[(dist > 1.5*39) & (dist < 3*39)].flatten()

for i in range(len(Ty_data)):
    keys = list(Ty_data.keys())
    tke_dict[keys[i]] = []

new_keys = ['Empty','g800','i800','Flat','Sin','ATTO','G_S','G_U','F_S','F_U']
tke_dict = dict(zip(new_keys, tke_dict.values()))


#%%Plot Box plot

fig,axs = plt.subplots(2,1,tight_layout=True,figsize=(14,6),sharex=True)

axs[0].boxplot(bp_dict.values(), labels=bp_dict.keys(), sym='')
axs[1].boxplot(tke_dict.values(), labels=bp_dict.keys(), sym='')
axs[1].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
axs[1].set_yticks([-100, -50, 0, 50, 100])
axs[0].set_ylabel(r"$\frac{y_B}{y_{B,d}}$", fontsize=16)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_ylabel(r"$\frac{P - \varepsilon}{|\varepsilon|}$", fontsize=16)
axs[1].grid(alpha=0.2)
axs[0].grid(alpha=0.2)
plt.show()



























































