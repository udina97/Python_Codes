#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:44:38 2026

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

for i in range(len(G_cases)):
    if i==0:
        zstart = 1
    else:
        zstart = 11

    anisoD = xr.open_dataarray(G_path + G_cases[i] + Diag)
    terms_ptb = xr.open_dataarray(G_path + G_cases[i] + '/terms_ptb.nc')
    
    l3 = anisoD[:,:,zstart:,2].data
    sw = terms_ptb[:,:,zstart:,5].data/(2*terms_ptb[:,:,zstart:,6].data) - 1/3
    
    print(f'For case "{G_cases[i]}" lamba3 is sigmaW:',100*(np.sum(l3==sw)/l3.size))

for i in range(len(B_cases)):

    anisoD = xr.open_dataarray(B_path + B_cases[i] + Diag)
    terms_ptb = xr.open_dataarray(B_path + B_cases[i] + '/terms_ptb.nc')
    dist = xr.open_dataarray(B_path + B_cases[i] + '/dist.nc')[:,:,:,0].data
    
    l3 = (anisoD[:,:,:,2].data)[dist>39]
    sw = (terms_ptb[:,:,:,5].data/(2*terms_ptb[:,:,:,6].data) - 1/3)[dist>39]
    
    print(f'For case "{B_cases[i]}" lamba3 is sigmaW:',100*(np.sum(l3==sw)/l3.size))



















































