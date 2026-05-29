#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:01:54 2026

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
    
#%%

colorsG = ['peachpuff','sandybrown','saddlebrown']
markersG = ['o','s','v']
colorsB = ['yellowgreen','olivedrab','darkolivegreen']
markersB = ['<','P','*']

fig,axs = plt.subplots(1,1,tight_layout=True)

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

heights = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]

for i in range(len(G_cases)):
    aniso = xr.open_dataarray(G_path + G_cases[i] + Full)
    anisoD = xr.open_dataarray(G_path + G_cases[i] + Diag)
    
    R = aniso[:,:,:,1].data/anisoD[:,:,:,1].data

    R_med = np.nanmedian(R,axis=(0,1))
    R_10 = np.nanquantile(R, 0.1, axis=(0,1))
    R_90 = np.nanquantile(R, 0.9, axis=(0,1))

    axs.plot(R_med,z_uvp_G,c=colorsG[i])
    # axs.fill_betweenx(z_uvp_G,R_10,R_90,alpha=0.3,color=colorsG[i])

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

for i in range(len(B_cases)):
    aniso = xr.open_dataarray(B_path + B_cases[i] + Full)
    anisoD = xr.open_dataarray(B_path + B_cases[i] + Diag)
    dist = xr.open_dataarray(B_path + B_cases[i] + '/dist.nc')
    mask = dist[:,:,:,0] < 0
    for j in range(0,3):
        aniso[:,:,:, j] = np.where(mask, np.nan, aniso[:,:,:, j])
        anisoD[:,:,:, j] = np.where(mask, np.nan, anisoD[:,:,:, j])
    R = aniso[:,:,:,1].data/anisoD[:,:,:,1].data

    R_med = np.nanmedian(R,axis=(0,1))
    R_10 = np.nanquantile(R, 0.1, axis=(0,1))
    R_90 = np.nanquantile(R, 0.9, axis=(0,1))

    axs.plot(R_med,z_uvp_B[:-5],c=colorsB[i])
    
axs.set_xlim(0.5,1)
axs.set_ylim(0,1)
axs.set_xlabel(r"$y_B/y_{B,d}$",fontsize=14)
axs.set_ylabel(r"$z/z_i$",fontsize=14)

plt.show()























































