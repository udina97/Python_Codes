#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:27:50 2025

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


#%%Set path to the profiles

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat']

case = 6

if case < 6:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
    terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc').data
    data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc').data
    anisotropy = xr.open_dataarray(path_to_data + cases[case] + '/anisotropy.nc').data
else:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc').data
    data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc').data
    anisotropy = xr.open_dataarray(path_to_data + cases[case] + '/anisotropy.nc').data
    
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

#%%Plot TKE res vs yB for the RSL

# fig, axs = plt.subplots(1, 1, figsize=(6, 6))

# # Deep copy the arrays
# tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
# tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])
# tmpRES[abs(tmpRES) < 0.01*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
# # Avoid divide-by-zero
# with np.errstate(divide='ignore', invalid='ignore'):
#     tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

# # Select y_B and TKE slices
# x_yB = anisotropy[:, :, 10:60, 1]
# y_TKE = tmpNorm[:, :, 10:60]

# # Compute binned statistics
# TKE_median = []
# # TKE_std = []
# TKE_q25 = []
# TKE_q75 = []
# yB_mean = []

# j = 0.1
# for i in range(28):
#     if i == 0:
#         mask = x_yB < j
#         yB_mean.append(0.05)
#     else:
#         mask = (x_yB > j) & (x_yB < j + 0.025)
#         yB_mean.append(j + 0.0125)
#     TKE_vals = y_TKE[mask]
#     TKE_median.append(np.nanmedian(TKE_vals))
#     # TKE_std.append(np.nanstd(TKE_vals))
#     TKE_q25.append(np.nanpercentile(TKE_vals, 25))
#     TKE_q75.append(np.nanpercentile(TKE_vals, 75))
#     j += 0.025

# # Plot results
# axs.plot(yB_mean, TKE_median, c='k', marker='o')
# axs.fill_between(yB_mean, np.array(TKE_q25), np.array(TKE_q75), alpha=0.5)

# axs.axhline(0, color='k', linestyle='-.')
# axs.axvline(0.38, color='k', linestyle='-.')
# axs.text(0.38 + 0.01, 200, 'yB = 0.38', rotation=90, va='center', ha='left', color='black')
# axs.axvline(0.36, color='k', linestyle='-.')
# axs.text(0.36 - 0.02, 200, 'yB = 0.36', rotation=90, va='center', ha='left', color='black')

# axs.set_xlabel(r'$y_B$', fontsize=18)
# axs.set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=21)
# axs.set_xlim(0.15, 0.6)
# axs.set_ylim(-70, 250)
# axs.tick_params(axis='x', labelsize=12)
# axs.tick_params(axis='y', labelsize=12)
# axs.set_title(f'{cases[case]}',fontsize=15)

# # Correlation text
# # corr = np.corrcoef(x_yB.values.flatten(), y_TKE.flatten())[0, 1]
# # axs.text(0.1, 0.9, f'r = {round(corr, 2)}', transform=axs.transAxes, fontsize=18)

# plt.tight_layout()
# plt.show()

#%%Save the profile

# save_Prof = np.zeros((4,28),order='F')

# save_Prof[0,:] = TKE_median
# save_Prof[1,:] = TKE_q25
# save_Prof[2,:] = TKE_q75
# save_Prof[3,:] = yB_mean

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'ResTKEvsYB_' + cases[case] + '_Q_v2.npy',save_Prof)


#%%Residual vs yB correlation plot

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

z_profile = np.arange(nz) * dz * zi
zeds = np.ones((nx, ny, 1)) * z_profile
dist = copy.deepcopy(zeds)
del zeds,z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
mask4D = np.expand_dims(mask, axis=-1)
data[:, :, :, :] = np.where(mask4D, np.nan, data[:, :, :, :])
terms_bdg[:, :, :, :] = np.where(mask4D, np.nan, terms_bdg[:, :, :, :])
anisotropy[:, :, :, :] = np.where(mask4D, np.nan, anisotropy[:, :, :, :])
del mask,mask4D

# Optional: remove if not needed
# from scipy.stats import gaussian_kde

level1 = 16*dz*zi
level2 = 120*dz*zi

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

# Deep copy the arrays
tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] - terms_bdg[:, :, :, 11])
val = np.nanmedian(tmpRES[(dist[:,:,5:]>38) & (dist[:,:,5:]<42)])
tmpRES[abs(tmpRES) < 0.01*val] = 0
# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

# Select y_B and TKE slices
x_yB = anisotropy[:, :, :, 1]
y_TKE = tmpNorm
mask_dist = (dist[:,:,5:] > level1) & (dist[:,:,5:] < level2)

# Compute binned statistics
TKE_median = []
# TKE_std = []
TKE_q25 = []
TKE_q75 = []
yB_mean = []

j = 0.1
for i in range(28):
    if i == 0:
        mask = x_yB < j
        yB_mean.append(0.05)
    else:
        mask = (x_yB > j) & (x_yB < j + 0.025)
        yB_mean.append(j + 0.0125)
        
    combined_mask = mask & mask_dist
    TKE_vals = y_TKE[combined_mask]
    TKE_median.append(np.nanmedian(TKE_vals))
    # TKE_std.append(np.nanstd(TKE_vals))
    TKE_q25.append(np.nanpercentile(TKE_vals, 25))
    TKE_q75.append(np.nanpercentile(TKE_vals, 75))
    j += 0.025

# Plot results
axs.plot(yB_mean, TKE_median, c='k', marker='o')
axs.fill_between(yB_mean, np.array(TKE_q25), np.array(TKE_q75), alpha=0.5)

axs.axhline(0, color='k', linestyle='-.')
axs.axvline(0.38, color='k', linestyle='-.')
axs.text(0.38 + 0.01, 200, 'yB = 0.38', rotation=90, va='center', ha='left', color='black')
axs.axvline(0.36, color='k', linestyle='-.')
axs.text(0.36 - 0.02, 200, 'yB = 0.36', rotation=90, va='center', ha='left', color='black')

axs.set_xlabel(r'$y_B$', fontsize=18)
axs.set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=21)
axs.set_xlim(0.15, 0.6)
axs.set_ylim(-70, 250)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)
# axs.set_title(f'{cases[case]}',fontsize=15)

# Correlation text
# x_yB_flat = x_yB.values[mask_dist]
# y_TKE_flat = y_TKE[mask_dist]

# Compute correlation, avoiding NaNs
# valid = ~np.isnan(x_yB_flat) & ~np.isnan(y_TKE_flat)
# corr = np.corrcoef(x_yB_flat[valid], y_TKE_flat[valid])[0, 1]
# axs.text(0.1, 0.9, f'r = {round(corr, 2)}', transform=axs.transAxes, fontsize=18)

plt.tight_layout()
plt.show()


#%%Save the profile

# save_Prof = np.zeros((4,28),order='F')

# save_Prof[0,:] = TKE_median
# save_Prof[1,:] = TKE_q25
# save_Prof[2,:] = TKE_q75
# save_Prof[3,:] = yB_mean

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'ResTKEvsYB_' + cases[case] + '_Q_v2.npy',save_Prof)















































