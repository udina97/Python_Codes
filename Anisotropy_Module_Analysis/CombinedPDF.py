#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:47:06 2025

@author: u1450851
"""

import numpy as np
from scipy.stats import ks_2samp,wasserstein_distance
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

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

#%%

ustar3D_a = (((mom3D_a[:,:,:,14] - uvpnode2wnode(mom3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,20])**2 + \
          (mom3D_a[:,:,:,15] - uvpnode2wnode(mom3D_a[:,:,:,1])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,21])**2)**0.25).values
ustar3D_na = (((mom3D_na[:,:,:,14] - uvpnode2wnode(mom3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,20])**2 + \
          (mom3D_na[:,:,:,15] - uvpnode2wnode(mom3D_na[:,:,:,1])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,21])**2)**0.25).values
    
heatflux3D_a = (sc3D_a[:,:,:,4] - uvpnode2wnode(sc3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - sc3D_a[:,:,:,7]).values
heatflux3D_na = (sc3D_na[:,:,:,4] - uvpnode2wnode(sc3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - sc3D_na[:,:,:,7]).values

zoverL3D_a = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_a)**3)*sc3D_a[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_a)))
zoverL3D_na = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_na)**3)*sc3D_na[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_na)))

zlevel = 0

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((ustar3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((ustar3D_na[:,:,zlevel]).flatten()), max((ustar3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,ustar3D_na[:,:,zlevel].T,cmap='bwr',vmin=0,vmax=1)
axs[1,0].hist((ustar3D_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((heatflux3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((heatflux3D_na[:,:,zlevel]).flatten()), max((heatflux3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,heatflux3D_na[:,:,zlevel].T,cmap='bwr',vmin=-0.005,vmax=0.005)
axs[1,1].hist((heatflux3D_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((zoverL3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((zoverL3D_na[:,:,zlevel]).flatten()), max((zoverL3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,zoverL3D_na[:,:,zlevel].T,cmap='bwr',vmin=-10,vmax=10)
axs[1,2].hist((zoverL3D_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)
# axs[1,2].set_xscale('log')

axs[1,0].set_xlabel(r'$u_*/u_s$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs[1,1].set_xlabel(r'$\overline{wT}/u_sT_s$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs[1,2].set_xlabel(r'$\zeta$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

# for i in range(2):
#     axs[0,i+1].set_yticklabels([])

fig.suptitle(sim_na,fontsize=14)

plt.show()

#%%

ustar3D_a = (((mom3D_a[:,:,:,14] - uvpnode2wnode(mom3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,20])**2 + \
          (mom3D_a[:,:,:,15] - uvpnode2wnode(mom3D_a[:,:,:,1])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,21])**2)**0.25).values
ustar3D_na = (((mom3D_na[:,:,:,14] - uvpnode2wnode(mom3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,20])**2 + \
          (mom3D_na[:,:,:,15] - uvpnode2wnode(mom3D_na[:,:,:,1])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,21])**2)**0.25).values
    
heatflux3D_a = (sc3D_a[:,:,:,4] - uvpnode2wnode(sc3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - sc3D_a[:,:,:,7]).values
heatflux3D_na = (sc3D_na[:,:,:,4] - uvpnode2wnode(sc3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - sc3D_na[:,:,:,7]).values

zoverL3D_a = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_a)**3)*sc3D_a[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_a)))
zoverL3D_na = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_na)**3)*sc3D_na[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_na)))

zlevel = 0

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,4))

kde = gaussian_kde((ustar3D_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((ustar3D_a[:,:,zlevel]).flatten()), max((ustar3D_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist((ustar3D_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[0].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((ustar3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((ustar3D_na[:,:,zlevel]).flatten()), max((ustar3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist((ustar3D_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[0].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
axs[0].axvline(np.mean((ustar3D_a[:,:,zlevel]).flatten()),c='r',ls='--')
axs[0].axvline(np.mean((ustar3D_na[:,:,zlevel]).flatten()),c='g',ls='--')

kde = gaussian_kde((heatflux3D_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((heatflux3D_a[:,:,zlevel]).flatten()), max((heatflux3D_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((heatflux3D_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[1].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((heatflux3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((heatflux3D_na[:,:,zlevel]).flatten()), max((heatflux3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((heatflux3D_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[1].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
axs[1].axvline(np.mean((heatflux3D_a[:,:,zlevel]).flatten()),c='r',ls='--')
axs[1].axvline(np.mean((heatflux3D_na[:,:,zlevel]).flatten()),c='g',ls='--')

kde = gaussian_kde((zoverL3D_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((zoverL3D_a[:,:,zlevel]).flatten()), max((zoverL3D_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[2].hist((zoverL3D_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[2].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((zoverL3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((zoverL3D_na[:,:,zlevel]).flatten()), max((zoverL3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[2].hist((zoverL3D_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[2].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
axs[2].axvline(np.mean((zoverL3D_a[:,:,zlevel]).flatten()),c='r',ls='--')
axs[2].axvline(np.mean((zoverL3D_na[:,:,zlevel]).flatten()),c='g',ls='--')
# axs[1,2].set_xscale('log')

axs[0].set_xlabel(r'$u_*/u_s$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs[1].set_xlabel(r'$\overline{wT}/u_sT_s$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs[2].set_xlabel(r'$\zeta$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[0].set_ylabel(r'$PDF$',fontsize=14)
axs[2].legend()

# for i in range(2):
#     axs[i+1].set_yticklabels([])

plt.show()

#%%

Rxx_a = mom3D_a[:,:,:,4].values - mom3D_a[:,:,:,0].values*mom3D_a[:,:,:,0].values - mom3D_a[:,:,:,16].values
Ryy_a = mom3D_a[:,:,:,5].values - mom3D_a[:,:,:,1].values*mom3D_a[:,:,:,1].values - mom3D_a[:,:,:,17].values
Rzz_a = wnode2uvpnode(mom3D_a[:,:,:,6].values - mom3D_a[:,:,:,2].values*mom3D_a[:,:,:,2].values)

Rxx_na = mom3D_na[:,:,:,4].values - mom3D_na[:,:,:,0].values*mom3D_na[:,:,:,0].values - mom3D_na[:,:,:,16].values
Ryy_na = mom3D_na[:,:,:,5].values - mom3D_na[:,:,:,1].values*mom3D_na[:,:,:,1].values - mom3D_na[:,:,:,17].values
Rzz_na = wnode2uvpnode(mom3D_na[:,:,:,6].values - mom3D_na[:,:,:,2].values*mom3D_na[:,:,:,2].values)

zlevel = 0

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((Rxx_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rxx_a[:,:,zlevel]).flatten()), max((Rxx_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,Rxx_a[:,:,zlevel].T,cmap='bwr')
axs[1,0].hist((Rxx_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((Ryy_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Ryy_a[:,:,zlevel]).flatten()), max((Ryy_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,Ryy_a[:,:,zlevel].T,cmap='bwr')
axs[1,1].hist((Ryy_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((Rzz_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rzz_a[:,:,zlevel]).flatten()), max((Rzz_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,Rzz_a[:,:,zlevel].T,cmap='bwr')
axs[1,2].hist((Rzz_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)
# axs[1,2].set_xscale('log')

axs[1,0].set_xlabel(r'$R_{xx}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1,1].set_xlabel(r'$R_{yy}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1,2].set_xlabel(r'$R_{zz}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

# for i in range(2):
#     axs[0,i+1].set_yticklabels([])

fig.suptitle(sim_a,fontsize=14)

plt.show()

#%%

Rxx_a = mom3D_a[:,:,:,4].values - mom3D_a[:,:,:,0].values*mom3D_a[:,:,:,0].values - mom3D_a[:,:,:,16].values
Ryy_a = mom3D_a[:,:,:,5].values - mom3D_a[:,:,:,1].values*mom3D_a[:,:,:,1].values - mom3D_a[:,:,:,17].values
Rzz_a = wnode2uvpnode(mom3D_a[:,:,:,6].values - mom3D_a[:,:,:,2].values*mom3D_a[:,:,:,2].values)

Rxx_na = mom3D_na[:,:,:,4].values - mom3D_na[:,:,:,0].values*mom3D_na[:,:,:,0].values - mom3D_na[:,:,:,16].values
Ryy_na = mom3D_na[:,:,:,5].values - mom3D_na[:,:,:,1].values*mom3D_na[:,:,:,1].values - mom3D_na[:,:,:,17].values
Rzz_na = wnode2uvpnode(mom3D_na[:,:,:,6].values - mom3D_na[:,:,:,2].values*mom3D_na[:,:,:,2].values)

zlevel = 0

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,4))

kde = gaussian_kde((Rxx_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rxx_na[:,:,zlevel]).flatten()), max((Rxx_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist((Rxx_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[0].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
kde = gaussian_kde((Rxx_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rxx_a[:,:,zlevel]).flatten()), max((Rxx_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist((Rxx_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[0].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
axs[0].axvline(np.mean((Rxx_a[:,:,zlevel]).flatten()),c='r',ls='--')
axs[0].axvline(np.mean((Rxx_na[:,:,zlevel]).flatten()),c='g',ls='--')

kde = gaussian_kde((Ryy_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Ryy_na[:,:,zlevel]).flatten()), max((Ryy_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((Ryy_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[1].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
kde = gaussian_kde((Ryy_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Ryy_a[:,:,zlevel]).flatten()), max((Ryy_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((Ryy_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[1].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
axs[1].axvline(np.mean((Ryy_a[:,:,zlevel]).flatten()),c='r',ls='--')
axs[1].axvline(np.mean((Ryy_na[:,:,zlevel]).flatten()),c='g',ls='--')

kde = gaussian_kde((Rzz_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rzz_na[:,:,zlevel]).flatten()), max((Rzz_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[2].hist((Rzz_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[2].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
kde = gaussian_kde((Rzz_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rzz_a[:,:,zlevel]).flatten()), max((Rzz_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[2].hist((Rzz_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[2].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
axs[2].axvline(np.mean((Rzz_a[:,:,zlevel]).flatten()),c='r',ls='--')
axs[2].axvline(np.mean((Rzz_na[:,:,zlevel]).flatten()),c='g',ls='--')
# axs[1,2].set_xscale('log')

axs[0].set_xlabel(r'$R_{xx}/u_s^2$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1].set_xlabel(r'$R_{yy}/u_s^2$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[2].set_xlabel(r'$R_{zz}/u_s^2$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[0].set_ylabel(r'$PDF$',fontsize=14)
axs[2].legend()

# for i in range(2):
#     axs[i+1].set_yticklabels([])

plt.show()

#%%

TT_a = (sc3D_a[:,:,:,1].values - sc3D_a[:,:,:,0].values*sc3D_a[:,:,:,0].values)*(Tscale**2)
TT_na = (sc3D_na[:,:,:,1].values - sc3D_na[:,:,:,0].values*sc3D_na[:,:,:,0].values)*(Tscale**2)

from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(10,5))

kde = gaussian_kde((sc3D_na[:,:,zlevel,0].values*Tscale).flatten())
x_pdf = np.linspace(min((sc3D_na[:,:,zlevel,0].values*Tscale).flatten()), max((sc3D_na[:,:,zlevel,0].values*Tscale).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(sc3D_na[:,:,zlevel,0].values*Tscale).T,cmap='bwr')
axs[1,0].hist((sc3D_na[:,:,zlevel,0].values*Tscale).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((TT_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((TT_na[:,:,zlevel]).flatten()), max((TT_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(TT_na[:,:,zlevel]).T,cmap='bwr')
axs[1,1].hist((TT_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$T$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1,1].set_xlabel(r'$\overline{TT}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)

axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

# for i in range(1):
#     axs[0,i+1].set_yticklabels([])

fig.suptitle(sim_na,fontsize=14)

plt.show()

#%%

TT_a = (sc3D_a[:,:,:,1].values - sc3D_a[:,:,:,0].values*sc3D_a[:,:,:,0].values)*(Tscale**2)
TT_na = (sc3D_na[:,:,:,1].values - sc3D_na[:,:,:,0].values*sc3D_na[:,:,:,0].values)*(Tscale**2)

from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

kde = gaussian_kde((sc3D_na[:,:,zlevel,0].values*Tscale).flatten())
x_pdf = np.linspace(min((sc3D_na[:,:,zlevel,0].values*Tscale).flatten()), max((sc3D_na[:,:,zlevel,0].values*Tscale).flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist((sc3D_na[:,:,zlevel,0].values*Tscale).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[0].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
kde = gaussian_kde((sc3D_a[:,:,zlevel,0].values*Tscale).flatten())
x_pdf = np.linspace(min((sc3D_a[:,:,zlevel,0].values*Tscale).flatten()), max((sc3D_a[:,:,zlevel,0].values*Tscale).flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist((sc3D_a[:,:,zlevel,0].values*Tscale).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[0].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
axs[0].axvline(np.mean((sc3D_a[:,:,zlevel,0].values*Tscale).flatten()),c='r',ls='--')
axs[0].axvline(np.mean((sc3D_na[:,:,zlevel,0].values*Tscale).flatten()),c='g',ls='--')

kde = gaussian_kde((TT_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((TT_na[:,:,zlevel]).flatten()), max((TT_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((TT_na[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="No Correction", color='grey')
axs[1].plot(x_pdf, pdf, 'g-', label="PDF No Correction")
kde = gaussian_kde((TT_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((TT_a[:,:,zlevel]).flatten()), max((TT_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((TT_a[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="W/ Correction")
axs[1].plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
axs[1].axvline(np.mean((TT_a[:,:,zlevel]).flatten()),c='r',ls='--')
axs[1].axvline(np.mean((TT_na[:,:,zlevel]).flatten()),c='g',ls='--')


axs[0].set_xlabel(r'$T$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1].set_xlabel(r'$\overline{TT}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)

axs[0].set_ylabel(r'$PDF$',fontsize=14)
axs[1].legend()

# for i in range(1):
#     axs[0,i+1].set_yticklabels([])

plt.show()








































































