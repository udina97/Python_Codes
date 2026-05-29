#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:38:47 2025

@author: u1450851
"""

import numpy as np
from scipy.stats import ks_2samp,wasserstein_distance
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from scipy import stats

#%%Set some simulation parameters

Nx = 128
Ny = 128
Nz = 128
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

zi = 1000
uscale = 0.4
Tscale = 290
x = np.arange(0,Nx)*dx
y = np.arange(0,Ny)*dy
z_uvp = np.arange(0,Nz)*dz + dz/2
z_w = np.arange(0,Nz)*dz

kvonk = 0.4
g_hat = 9.81*zi/(uscale**2)

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

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

#%%Set the path to the simulation and data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

sim_a = 'Tpatch800_a288m290s2_1ms_a'
sim_na = 'Tpatch800_a288m290s2_1ms_na'
path_to_data_na = path + sim_na + '/data/'
path_to_data_a = path + sim_a + '/data/'

mom3D_na = xr.open_dataarray(path_to_data_na + 'Momentum3D/Data_Momentum_2hr.nc')
mom2D_na = xr.open_dataarray(path_to_data_na + 'Momentum2D/Data_Momentum_2D_2hr.nc')
sc3D_na = xr.open_dataarray(path_to_data_na + 'Scalar3D/Data_Scalar_2hr.nc')
sc2D_na = xr.open_dataarray(path_to_data_na + 'Scalar2D/Data_Scalar_2D_2hr.nc')

mom3D_a = xr.open_dataarray(path_to_data_a + 'Momentum3D/Data_Momentum_2hr.nc')
mom2D_a = xr.open_dataarray(path_to_data_a + 'Momentum2D/Data_Momentum_2D_2hr.nc')
sc3D_a = xr.open_dataarray(path_to_data_a + 'Scalar3D/Data_Scalar_2hr.nc')
sc2D_a = xr.open_dataarray(path_to_data_a + 'Scalar2D/Data_Scalar_2D_2hr.nc')

#%%Compute t-test for ustar, heatflux, zeta

ustar3D_a = (((mom3D_a[:,:,:,14] - uvpnode2wnode(mom3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,20])**2 + \
          (mom3D_a[:,:,:,15] - uvpnode2wnode(mom3D_a[:,:,:,1])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,21])**2)**0.25).values
ustar3D_na = (((mom3D_na[:,:,:,14] - uvpnode2wnode(mom3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,20])**2 + \
          (mom3D_na[:,:,:,15] - uvpnode2wnode(mom3D_na[:,:,:,1])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,21])**2)**0.25).values
    
heatflux3D_a = (sc3D_a[:,:,:,4] - uvpnode2wnode(sc3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - sc3D_a[:,:,:,7]).values
heatflux3D_na = (sc3D_na[:,:,:,4] - uvpnode2wnode(sc3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - sc3D_na[:,:,:,7]).values

zoverL3D_a = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_a)**3)*sc3D_a[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_a)))
zoverL3D_na = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_na)**3)*sc3D_na[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_na)))

zlevel = 0

var_a = ustar3D_a[:,:,zlevel].flatten()
var_na = ustar3D_na[:,:,zlevel].flatten()

t_stat, p_val = stats.ttest_ind(var_a, var_na, equal_var=False)

print(f"t = {t_stat:.3f}, p = {p_val:.3e}")































































