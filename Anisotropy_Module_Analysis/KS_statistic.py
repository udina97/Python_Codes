#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:36:58 2025

@author: u1450851
"""

import numpy as np
from scipy.stats import ks_2samp,wasserstein_distance
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

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


#%%Compute the Kolmogorov-Smirnov test for velocity components

zlevel = 1
var = 0

var_a = (mom3D_a[:,:,zlevel,var].values).ravel()
var_na = (mom3D_na[:,:,zlevel,var].values).ravel()

statistic, p_value = ks_2samp(var_a, var_na)
w_dist = wasserstein_distance(var_a, var_na)

# print(f'The KS statistic is: {statistic} while the P-value is: {p_value}')

x1, y1 = ecdf(var_a)
x2, y2 = ecdf(var_na)

# Find point of maximum difference
all_x = np.sort(np.concatenate([x1, x2]))
cdf1 = np.searchsorted(x1, all_x, side="right") / len(x1)
cdf2 = np.searchsorted(x2, all_x, side="right") / len(x2)
diff = np.abs(cdf1 - cdf2)
imax = np.argmax(diff)

fig,axs = plt.subplots(1,1,figsize=(7,5))
axs.step(x1, y1, where="post", label="Anisotropy")
axs.step(x2, y2, where="post", label="No Anisotropy")
plt.vlines(all_x[imax], cdf1[imax], cdf2[imax], colors="red", linestyles="--", label=f"KS statistic = {statistic:.3f}")
plt.fill_between(all_x, cdf1, cdf2, color='gray', alpha=0.3, label=f'Wasserstein = {w_dist :.03f}')
axs.set_xlabel(f"Var")
axs.set_ylabel("CDF")
axs.legend()
axs.set_title(f"KS test: p = {p_value:.2e}")

plt.show()

#%%ustar

zlevel = 0

ustar3D_a = (((mom3D_a[:,:,:,14] - uvpnode2wnode(mom3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,20])**2 + \
          (mom3D_a[:,:,:,15] - uvpnode2wnode(mom3D_a[:,:,:,1])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,21])**2)**0.25).values
    
ustar3D_na = (((mom3D_na[:,:,:,14] - uvpnode2wnode(mom3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,20])**2 + \
          (mom3D_na[:,:,:,15] - uvpnode2wnode(mom3D_na[:,:,:,1])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,21])**2)**0.25).values

var_a = ustar3D_a[:,:,zlevel].ravel()
var_na = ustar3D_na[:,:,zlevel].ravel()

statistic, p_value = ks_2samp(var_a,var_na)
w_dist = wasserstein_distance(var_a, var_na)

# print(f'The KS statistic is: {statistic} while the P-value is: {p_value}')

x1, y1 = ecdf(var_a)
x2, y2 = ecdf(var_na)

all_x = np.sort(np.concatenate([x1, x2]))
cdf1 = np.searchsorted(x1, all_x, side="right") / len(x1)
cdf2 = np.searchsorted(x2, all_x, side="right") / len(x2)
diff = np.abs(cdf1 - cdf2)
imax = np.argmax(diff)

fig,axs = plt.subplots(1,1,figsize=(7,5))
axs.step(x1, y1, where="post", label="Anisotropy")
axs.step(x2, y2, where="post", label="No Anisotropy")
plt.vlines(all_x[imax], cdf1[imax], cdf2[imax], colors="red", linestyles="--", label=f"KS statistic = {statistic:.3f}")
plt.fill_between(all_x, cdf1, cdf2, color='gray', alpha=0.3, label=f'Wasserstein = {w_dist :.03f}')
axs.set_xlabel(r"$u_*$ at "+f"{zlevel*dz*zi}m")
axs.set_ylabel("CDF")
axs.legend()
axs.set_title(f"KS test: p = {p_value:.2e}")

plt.show()

#%%heatflux

zlevel = 0

heatflux3D_a = (sc3D_a[:,:,:,4] - uvpnode2wnode(sc3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - sc3D_a[:,:,:,7]).values

heatflux3D_na = (sc3D_na[:,:,:,4] - uvpnode2wnode(sc3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - sc3D_na[:,:,:,7]).values

var_a = heatflux3D_a[:,:,zlevel].ravel()
var_na = heatflux3D_na[:,:,zlevel].ravel()

statistic, p_value = ks_2samp(var_a,var_na)
w_dist = wasserstein_distance(var_a, var_na)

# print(f'The KS statistic is: {statistic} while the P-value is: {p_value}')

x1, y1 = ecdf(var_a)
x2, y2 = ecdf(var_na)

all_x = np.sort(np.concatenate([x1, x2]))
cdf1 = np.searchsorted(x1, all_x, side="right") / len(x1)
cdf2 = np.searchsorted(x2, all_x, side="right") / len(x2)
diff = np.abs(cdf1 - cdf2)
imax = np.argmax(diff)

fig,axs = plt.subplots(1,1,figsize=(7,5))
axs.step(x1, y1, where="post", label="Anisotropy")
axs.step(x2, y2, where="post", label="No Anisotropy")
plt.vlines(all_x[imax], cdf1[imax], cdf2[imax], colors="red", linestyles="--", label=f"KS statistic = {statistic:.3f}")
plt.fill_between(all_x, cdf1, cdf2, color='gray', alpha=0.3, label=f'Wasserstein = {w_dist :.05f}')
axs.set_xlabel(r"$\overline{wT}$ at "+f"{zlevel*dz*zi}m")
axs.set_ylabel("CDF")
axs.legend()
axs.set_title(f"KS test: p = {p_value:.2e}")

plt.show()

#%%zoverL

zlevel = 50

zoverL3D_a = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_a)**3)*sc3D_a[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_a)))

zoverL3D_na = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_na)**3)*sc3D_na[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_na)))

var_a = zoverL3D_a[:,:,zlevel].ravel()
var_na = zoverL3D_na[:,:,zlevel].ravel()

statistic, p_value = ks_2samp(var_a,var_na)
w_dist = wasserstein_distance(var_a, var_na)

# print(f'The KS statistic is: {statistic} while the P-value is: {p_value}')

x1, y1 = ecdf(var_a)
x2, y2 = ecdf(var_na)

all_x = np.sort(np.concatenate([x1, x2]))
cdf1 = np.searchsorted(x1, all_x, side="right") / len(x1)
cdf2 = np.searchsorted(x2, all_x, side="right") / len(x2)
diff = np.abs(cdf1 - cdf2)
imax = np.argmax(diff)

fig,axs = plt.subplots(1,1,figsize=(7,5))
axs.step(x1, y1, where="post", label="Anisotropy")
axs.step(x2, y2, where="post", label="No Anisotropy")
plt.vlines(all_x[imax], cdf1[imax], cdf2[imax], colors="red", linestyles="--", label=f"KS statistic = {statistic:.3f}")
plt.fill_between(all_x, cdf1, cdf2, color='gray', alpha=0.3, label=f'Wasserstein = {w_dist :.05f}')
axs.set_xlabel(r"$\zeta$ at "+f"{zlevel*dz*zi + dz*zi/2}m")
axs.set_ylabel("CDF")
axs.legend()
axs.set_title(f"KS test: p = {p_value:.2e}")

plt.show()

#%%Reynolds stresses

Rxx_a = mom3D_a[:,:,:,4].values - mom3D_a[:,:,:,0].values*mom3D_a[:,:,:,0].values - mom3D_a[:,:,:,16].values
Ryy_a = mom3D_a[:,:,:,5].values - mom3D_a[:,:,:,1].values*mom3D_a[:,:,:,1].values - mom3D_a[:,:,:,17].values
Rzz_a = wnode2uvpnode(mom3D_a[:,:,:,6].values - mom3D_a[:,:,:,2].values*mom3D_a[:,:,:,2].values)

Rxx_na = mom3D_na[:,:,:,4].values - mom3D_na[:,:,:,0].values*mom3D_na[:,:,:,0].values - mom3D_na[:,:,:,16].values
Ryy_na = mom3D_na[:,:,:,5].values - mom3D_na[:,:,:,1].values*mom3D_na[:,:,:,1].values - mom3D_na[:,:,:,17].values
Rzz_na = wnode2uvpnode(mom3D_na[:,:,:,6].values - mom3D_na[:,:,:,2].values*mom3D_na[:,:,:,2].values)

zlevel = 10

var_a = (Rzz_a[:,:,zlevel]).ravel()
var_na = (Rzz_na[:,:,zlevel]).ravel()

statistic, p_value = ks_2samp(var_a,var_na)
w_dist = wasserstein_distance(var_a, var_na)

x1, y1 = ecdf(var_a)
x2, y2 = ecdf(var_na)

all_x = np.sort(np.concatenate([x1, x2]))
cdf1 = np.searchsorted(x1, all_x, side="right") / len(x1)
cdf2 = np.searchsorted(x2, all_x, side="right") / len(x2)
diff = np.abs(cdf1 - cdf2)
imax = np.argmax(diff)

fig,axs = plt.subplots(1,1,figsize=(7,5))
axs.step(x1, y1, where="post", label="Anisotropy")
axs.step(x2, y2, where="post", label="No Anisotropy")
plt.vlines(all_x[imax], cdf1[imax], cdf2[imax], colors="red", linestyles="--", label=f"KS statistic = {statistic:.3f}")
plt.fill_between(all_x, cdf1, cdf2, color='gray', alpha=0.3, label=f'Wasserstein = {w_dist :.05f}')
axs.set_xlabel(r"$R_{ii}$")
axs.set_ylabel("Cumulative probability")
axs.legend()
axs.set_title(f"KS test: p = {p_value:.2e}")

plt.show()




























