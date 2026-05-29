#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:55:55 2026

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

#%%Load anisotropy and TKE data

fig,axs = plt.subplots(2,3,tight_layout=True,sharey=True,figsize=(10,6), sharex=True)

for i in range(len(G_cases)):
    
    if G_cases[i] == 'Empty_9mps':
        zstart = 1
    else:
        zstart = 11
    
    TKE_terms = xr.open_dataarray(G_path + G_cases[i] + '/TKE_terms.nc')
    aniso = xr.open_dataarray(G_path + G_cases[i] + Full)
    anisoD = xr.open_dataarray(G_path + G_cases[i] + Diag)
    
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    # ratio_prime = np.zeros_like(ratio)
    # for i in range(0,Nz_G):
    #     ratio_prime[:,:,i] = (ratio[:,:,i] - np.nanmedian(ratio[:,:,i]))/np.nanmedian(ratio[:,:,i])*100
    
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    # tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
    tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = np.nan
    
    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)
    
    # tmpNorm[abs(tmpNorm) < 0.1*np.max(np.nanmedian(tmpNorm,axis=(0,1)))] = 0

    axs[0,i].hexbin(tmpNorm[:,:,zstart:Ny_G//2].flatten(),ratio[:,:,zstart:Ny_G//2].flatten(),cmap='hot_r',extent=(-120,120,0.6,1),gridsize=100)
    # axs[0,i].scatter(tmpNorm[:,:,zstart:-26].flatten(),ratio[:,:,zstart:-26].flatten(),s=1)

for i in range(len(B_cases)):

    TKE_terms = xr.open_dataarray(B_path + B_cases[i] + '/TKE_terms.nc')
    aniso = xr.open_dataarray(B_path + B_cases[i] + Full)
    anisoD = xr.open_dataarray(B_path + B_cases[i] + Diag)
    dist = xr.open_dataarray(B_path + B_cases[i] + '/dist.nc')
    
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    ratio[dist[:,:,:,0].data < 39] = np.nan
    # ratio_prime = np.zeros_like(ratio)
    # for i in range(0,Nz_B-5):
    #     ratio_prime[:,:,i] = (ratio[:,:,i] - np.nanmedian(ratio[:,:,i]))/np.nanmedian(ratio[:,:,i])*100
    
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    # tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
    # tmpRES[abs(tmpRES) < 0.01*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
    tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = np.nan
    
    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
    
    tmpNorm[dist[:,:,:,0].data < 39] = np.nan
    # tmpNorm[abs(tmpNorm) < 0.1*np.max(np.nanmedian(tmpNorm,axis=(0,1)))] = 0
    
    axs[1,i].hexbin(tmpNorm[:,:,:-40].flatten(),ratio[:,:,:-40].flatten(),cmap='hot_r',extent=(-120,120,0.6,1))

axs[0,0].set_ylabel(r"$\frac{y_B}{y_{B,d}}$",fontsize=15)
axs[1,0].set_ylabel(r"$\frac{y_B}{y_{B,d}}$",fontsize=15)
axs[0,0].tick_params(axis='y', labelsize=14)
axs[1,0].tick_params(axis='y', labelsize=14)
for i in range(len(axs[0])):
    axs[1,i].set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=15)
    axs[1,i].tick_params(axis='x', labelsize=14)

import string

labels = string.ascii_lowercase

k = 0
for row in range(axs.shape[0]):
    for col in range(axs.shape[1]):
        axs[row,col].grid(alpha=0.3)
        ax = axs[row, col]
        ax.text(-0.01, 1.1, f"{labels[k]})", transform=ax.transAxes, fontsize=14, fontfamily="serif", va="top", ha="left", clip_on=False)
        k += 1
        
plt.show()

#%%Ratio vs TKE residual Plot version 2: compute boxplots for TKE res bins

Res_bins = [-np.inf,-150, -100, -80, -60, -40, -20, -1, 1, 20, 40, 60, 80, 100, 150,np.inf]

case = 0

TKE_terms = xr.open_dataarray(G_path + G_cases[case] + '/TKE_terms.nc')
aniso = xr.open_dataarray(G_path + G_cases[case] + Full)
anisoD = xr.open_dataarray(G_path + G_cases[case] + Diag)

ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data

tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
# tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)

ratio_flat = ratio[:,:,1:Nz_G//2].flatten()
tmpNorm_flat = tmpNorm[:,:,1:Nz_G//2].flatten()

bin_indices = np.digitize(tmpNorm_flat, Res_bins)

box_data = []

for i in range(1, len(Res_bins)):
    mask = bin_indices == i
    vals = ratio_flat[mask]

    # remove NaNs if needed
    vals = vals[~np.isnan(vals)]

    box_data.append(vals)

fig, ax = plt.subplots(figsize=(10,5),tight_layout=True)

positions = np.arange(len(box_data))

ax.boxplot(box_data, positions=positions, widths=1.0, showfliers=False)

ax.set_xlim(-0.5, len(box_data)-0.5)
ax.set_xticks(positions)
ax.set_xticklabels([f"{Res_bins[i]} to {Res_bins[i+1]}" for i in range(len(Res_bins)-1)],rotation=45)

ax.set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$",fontsize=14)
ax.set_ylabel(r"$y_B/y_{B,d}$",fontsize=14)

plt.show()

#%%

Res_bins = [-np.inf,-150, -100, -80, -60, -40, -20, -1, 1, 20, 40, 60, 80, 100,150, np.inf]

case = 0

TKE_terms = xr.open_dataarray(B_path + B_cases[case] + '/TKE_terms.nc')
aniso = xr.open_dataarray(B_path + B_cases[case] + Full)
anisoD = xr.open_dataarray(B_path + B_cases[case] + Diag)
dist = xr.open_dataarray(B_path + B_cases[case] + '/dist.nc')[:,:,:,0].data

ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data

tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
# tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
tmpRES[dist<0] = np.nan
tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)

ratio_flat = ratio[(dist > 39) & (dist<15*39)].flatten()
tmpNorm_flat = tmpNorm[(dist > 39) & (dist<15*39)].flatten()

bin_indices = np.digitize(tmpNorm_flat, Res_bins)

box_data = []

for i in range(1, len(Res_bins)):
    mask = bin_indices == i
    vals = ratio_flat[mask]
    vals = vals[~np.isnan(vals)]
    if len(vals)<100:
        vals = []

    box_data.append(vals)

fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)

positions = np.arange(len(box_data))

ax.boxplot(box_data, positions=positions, widths=1.0, showfliers=False)

ax.set_xlim(-0.5, len(box_data)-0.5)
ax.set_xticks(positions)
ax.set_xticklabels([f"{Res_bins[i]} to {Res_bins[i+1]}" for i in range(len(Res_bins)-1)],rotation=45)

ax.set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$",fontsize=14)
ax.set_ylabel(r"$y_B/y_{B,d}$",fontsize=14)

plt.show()

#%%

Res_bins = [-np.inf,-150, -100, -75, -50, -25, -1, 1, 25, 50, 75, 100, 150, np.inf]
case = 2

TKE_terms = xr.open_dataarray(B_path + B_cases[case] + '/TKE_terms.nc')
dist = xr.open_dataarray(B_path + B_cases[case] + '/dist.nc')[:,:,:,0].data

tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
# tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
tmpRES[dist<0] = np.nan
tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)

tmpNorm_flat = tmpNorm[(dist > 39) & (dist<10*39)].flatten()

plt.subplots()
plt.hist(tmpNorm_flat,bins=Res_bins,density=True)
plt.show()

#%%

# Res_bins = [-np.inf,-150, -100, -80, -60, -40, -20, -1, 1, 20, 40, 60, 80, 100, 150, np.inf]
Res_bins = [-np.inf,-150, -100, -75, -50, -25, -1, 1, 25, 50, 75, 100, 150, np.inf]

fig,axs = plt.subplots(2,3,tight_layout=True,sharey=True,figsize=(10,6), sharex=True)

for i in range(len(G_cases)):
    
    if G_cases[i] == 'Empty_9mps':
        zstart = 1
    else:
        zstart = 11
    
    TKE_terms = xr.open_dataarray(G_path + G_cases[i] + '/TKE_terms.nc')
    aniso = xr.open_dataarray(G_path + G_cases[i] + Full)
    anisoD = xr.open_dataarray(G_path + G_cases[i] + Diag)
    
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    # tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
    tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
    
    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)
    
    ratio_flat = ratio[:,:,zstart:-26].flatten()
    tmpNorm_flat = tmpNorm[:,:,zstart:-26].flatten()

    bin_indices = np.digitize(tmpNorm_flat, Res_bins)

    box_data = []

    for j in range(1, len(Res_bins)):
        mask = bin_indices == j
        vals = ratio_flat[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals)<100:
            vals=[]
        box_data.append(vals)
        
    positions = np.arange(len(box_data))
    axs[0,i].boxplot(box_data, positions=positions, widths=1.0, showfliers=False, capwidths=0.3)

for i in range(len(B_cases)):

    TKE_terms = xr.open_dataarray(B_path + B_cases[i] + '/TKE_terms.nc')
    aniso = xr.open_dataarray(B_path + B_cases[i] + Full)
    anisoD = xr.open_dataarray(B_path + B_cases[i] + Diag)
    dist = xr.open_dataarray(B_path + B_cases[i] + '/dist.nc')[:,:,:,0].data
    
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    ratio[dist < 39] = np.nan
    
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    # tmpDIS[abs(tmpDIS) < 0.01*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
    # tmpRES[abs(tmpRES) < 0.01*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
    tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
    
    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)
    
    tmpNorm[dist < 39] = np.nan
    # tmpNorm[abs(tmpNorm) < 0.1*np.max(np.nanmedian(tmpNorm,axis=(0,1)))] = 0
    
    ratio_flat = ratio[(dist > 39) & (dist<20*39)].flatten()
    tmpNorm_flat = tmpNorm[(dist > 39) & (dist<20*39)].flatten()

    bin_indices = np.digitize(tmpNorm_flat, Res_bins)

    box_data = []

    for j in range(1, len(Res_bins)):
        mask = bin_indices == j
        vals = ratio_flat[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals)<100:
            vals = []
        box_data.append(vals)
        
    positions = np.arange(len(box_data))
    axs[1,i].boxplot(box_data, positions=positions, widths=1.0, showfliers=False, capwidths=0.3)

axs[0,0].set_ylabel(r"$\frac{y_B}{y_{B,d}}$",fontsize=15)
axs[1,0].set_ylabel(r"$\frac{y_B}{y_{B,d}}$",fontsize=15)
axs[0,0].tick_params(axis='y', labelsize=14)
axs[1,0].tick_params(axis='y', labelsize=14)
for i in range(len(axs[0])):
    axs[1,i].set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=15)
    axs[1,i].tick_params(axis='x', labelsize=10)
    axs[1,i].set_xlim(-0.5, len(box_data)-0.5)
    axs[1,i].set_xticks(positions)
    axs[1,i].set_xticklabels([f"{Res_bins[i]} ; {Res_bins[i+1]}" for i in range(len(Res_bins)-1)],rotation=90)

import string

labels = string.ascii_lowercase

k = 0
for row in range(axs.shape[0]):
    for col in range(axs.shape[1]):
        axs[row,col].grid(alpha=0.3)
        ax = axs[row, col]
        ax.text(-0.01, 1.1, f"{labels[k]})", transform=ax.transAxes, fontsize=14, fontfamily="serif", va="top", ha="left", clip_on=False)
        k += 1
        
plt.show()

#%%

# VERSION 2 OF THE PLOT: MEDIAN LINE WITH FILL IN COLOUR BETWEEN THE 25 AND 75 QUARTILES

#%%

case = 0
zstart = 1
N = 10
inner_edges = np.linspace(-150, 150, N+1)
Res_bins = np.concatenate(([-np.inf], inner_edges, [np.inf]))
# Res_bins = [-np.inf,-150, -100, -75, -50, -25, -1, 1, 25, 50, 75, 100, 150, np.inf]

TKE_terms = xr.open_dataarray(G_path + G_cases[case] + '/TKE_terms.nc')
aniso = xr.open_dataarray(G_path + G_cases[case] + Full)
anisoD = xr.open_dataarray(G_path + G_cases[case] + Diag)

ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data

tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)

ratio_flat = ratio[:,:,zstart:-128].flatten()
tmpNorm_flat = tmpNorm[:,:,zstart:-128].flatten()

bin_indices = np.digitize(tmpNorm_flat, Res_bins)

box_data = []
med_tke = np.zeros((len(Res_bins)-1))
med_yb = np.zeros((len(Res_bins)-1))
iq25 = np.zeros((len(Res_bins)-1))
iq75 = np.zeros((len(Res_bins)-1))

for j in range(1, len(Res_bins)):
    mask = bin_indices == j
    vals = ratio_flat[mask]
    vals = vals[~np.isnan(vals)]
    if len(vals)<1:
        vals=[]
        med_tke[j-1] = np.nan
        med_yb[j-1] = np.nan
        iq25[j-1] = np.nan
        iq75[j-1] = np.nan
    else:
        med_tke[j-1] = np.median(tmpNorm_flat[mask])
        med_yb[j-1] = np.median(vals)
        iq25[j-1] = np.quantile(vals,0.25)
        iq75[j-1] = np.quantile(vals,0.75)
    box_data.append(vals)
    
positions = np.arange(len(box_data))

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(med_tke,med_yb,c='k')
axs.fill_between(med_tke,iq25,iq75,alpha=0.5)

plt.show()

#%%

case = 2
N = 10
inner_edges = np.linspace(-150, 150, N+1)
Res_bins = np.concatenate(([-np.inf], inner_edges, [np.inf]))
# Res_bins = [-np.inf,-150, -100, -75, -50, -25, -1, 1, 25, 50, 75, 100, 150, np.inf]

TKE_terms = xr.open_dataarray(B_path + B_cases[case] + '/TKE_terms.nc')
aniso = xr.open_dataarray(B_path + B_cases[case] + Full)
anisoD = xr.open_dataarray(B_path + B_cases[case] + Diag)
dist = xr.open_dataarray(B_path + B_cases[case] + '/dist.nc')[:,:,:,0].data

ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
ratio[dist < 39] = np.nan

tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)

tmpNorm[dist < 39] = np.nan

ratio_flat = ratio[(dist > 39) & (dist<20*39)].flatten()
tmpNorm_flat = tmpNorm[(dist > 39) & (dist<20*39)].flatten()

bin_indices = np.digitize(tmpNorm_flat, Res_bins)

box_data = []
med_tke = np.zeros((len(Res_bins)-1))
med_yb = np.zeros((len(Res_bins)-1))
iq25 = np.zeros((len(Res_bins)-1))
iq75 = np.zeros((len(Res_bins)-1))

for j in range(1, len(Res_bins)):
    mask = bin_indices == j
    vals = ratio_flat[mask]
    vals = vals[~np.isnan(vals)]
    if len(vals)<1:
        vals=[]
        med_tke[j-1] = np.nan
        med_yb[j-1] = np.nan
        iq25[j-1] = np.nan
        iq75[j-1] = np.nan
    else:
        med_tke[j-1] = np.median(tmpNorm_flat[mask])
        med_yb[j-1] = np.median(vals)
        iq25[j-1] = np.quantile(vals,0.25)
        iq75[j-1] = np.quantile(vals,0.75)
    box_data.append(vals)
    
positions = np.arange(len(box_data))

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(med_tke,med_yb,c='k')
axs.fill_between(med_tke,iq25,iq75,alpha=0.5)

plt.show()

#%%

N = 10
inner_edges = np.linspace(-150, 150, N+1)
Res_bins = np.concatenate(([-np.inf], inner_edges, [np.inf]))

fig,axs = plt.subplots(2,3,tight_layout=True,sharey=True,figsize=(11,5), sharex=True)

for i in range(len(G_cases)):
    
    if G_cases[i] == 'Empty_9mps':
        zstart = 1
    else:
        zstart = 11
    
    TKE_terms = xr.open_dataarray(G_path + G_cases[i] + '/TKE_terms.nc')
    aniso = xr.open_dataarray(G_path + G_cases[i] + Full)
    anisoD = xr.open_dataarray(G_path + G_cases[i] + Diag)
    
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
    tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)
    
    ratio_flat = ratio[:,:,zstart:-128].flatten()
    tmpNorm_flat = tmpNorm[:,:,zstart:-128].flatten()

    bin_indices = np.digitize(tmpNorm_flat, Res_bins)

    box_data = []

    med_tke = np.zeros((len(Res_bins)-1))
    med_yb = np.zeros((len(Res_bins)-1))
    iq25 = np.zeros((len(Res_bins)-1))
    iq75 = np.zeros((len(Res_bins)-1))

    for j in range(1, len(Res_bins)):
        mask = bin_indices == j
        vals = ratio_flat[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals)<1:
            vals=[]
            med_tke[j-1] = np.nan
            med_yb[j-1] = np.nan
            iq25[j-1] = np.nan
            iq75[j-1] = np.nan
        else:
            med_tke[j-1] = np.median(tmpNorm_flat[mask])
            med_yb[j-1] = np.median(vals)
            iq25[j-1] = np.quantile(vals,0.25)
            iq75[j-1] = np.quantile(vals,0.75)
        box_data.append(vals)
        
    positions = np.arange(len(box_data))
    axs[0,i].plot(med_tke,med_yb,c='k')
    axs[0,i].fill_between(med_tke,iq25,iq75,alpha=0.5)

for i in range(len(B_cases)):

    TKE_terms = xr.open_dataarray(B_path + B_cases[i] + '/TKE_terms.nc')
    aniso = xr.open_dataarray(B_path + B_cases[i] + Full)
    anisoD = xr.open_dataarray(B_path + B_cases[i] + Diag)
    dist = xr.open_dataarray(B_path + B_cases[i] + '/dist.nc')[:,:,:,0].data
    
    ratio = aniso[:,:,:,1].data/anisoD[:,:,:,1].data
    ratio[dist < 39] = np.nan
    
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
    val = np.nanmedian(tmpRES[(dist>38) & (dist<40)])
    tmpRES[abs(tmpRES) < 0.05*val] = 0
    # tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)

    tmpNorm[dist < 39] = np.nan

    ratio_flat = ratio[(dist > 39) & (dist<20*39)].flatten()
    tmpNorm_flat = tmpNorm[(dist > 39) & (dist<20*39)].flatten()

    bin_indices = np.digitize(tmpNorm_flat, Res_bins)

    box_data = []
    med_tke = np.zeros((len(Res_bins)-1))
    med_yb = np.zeros((len(Res_bins)-1))
    iq25 = np.zeros((len(Res_bins)-1))
    iq75 = np.zeros((len(Res_bins)-1))

    for j in range(1, len(Res_bins)):
        mask = bin_indices == j
        vals = ratio_flat[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals)<1:
            vals=[]
            med_tke[j-1] = np.nan
            med_yb[j-1] = np.nan
            iq25[j-1] = np.nan
            iq75[j-1] = np.nan
        else:
            med_tke[j-1] = np.median(tmpNorm_flat[mask])
            med_yb[j-1] = np.median(vals)
            iq25[j-1] = np.quantile(vals,0.25)
            iq75[j-1] = np.quantile(vals,0.75)
        box_data.append(vals)
        
    positions = np.arange(len(box_data))
    axs[1,i].plot(med_tke,med_yb,c='k')
    axs[1,i].fill_between(med_tke,iq25,iq75,alpha=0.5)

axs[0,0].set_ylabel(r"$\frac{y_B}{y_{B,d}}$",fontsize=15)
axs[1,0].set_ylabel(r"$\frac{y_B}{y_{B,d}}$",fontsize=15)
axs[0,0].tick_params(axis='y', labelsize=12)
axs[1,0].tick_params(axis='y', labelsize=12)
for i in range(len(axs[0])):
    axs[1,i].set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=15)
    axs[1,i].tick_params(axis='x', labelsize=12)
    axs[1,i].set_xlim(-150, 150)
    # axs[1,i].set_xticks(positions)
    # axs[1,i].set_xticklabels([f"{Res_bins[i]} ; {Res_bins[i+1]}" for i in range(len(Res_bins)-1)],rotation=90)

import string

labels = string.ascii_lowercase

k = 0
for row in range(axs.shape[0]):
    for col in range(axs.shape[1]):
        axs[row,col].grid(alpha=0.3)
        ax = axs[row, col]
        ax.text(-0.01, 1.1, f"{labels[k]})", transform=ax.transAxes, fontsize=14, fontfamily="serif", va="top", ha="left", clip_on=False)
        k += 1
        
plt.show()





















