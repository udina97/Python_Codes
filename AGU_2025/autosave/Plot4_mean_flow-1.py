#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:01:23 2025

@author: u1450851
"""

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import xarray as xr

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 128
ny = 128
nz = 128

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.4
Tscale = 290
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load the data

aniso = 'patch128_aniso_1ms'
classic = 'patch128_classic_hog96_1ms'

mom3D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

mom3D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

#%%

sfcT = sc2D_a.data[:,:,-2]*Tscale

fig,axs = plt.subplots(1,1)

p = axs.pcolormesh(x,y,sfcT.T,cmap='hot_r')
cbar = plt.colorbar(p)

plt.show()

#%%Compute ustar

uw = mo





























































