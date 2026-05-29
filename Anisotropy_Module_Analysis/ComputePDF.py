#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:16:19 2025

@author: u1450851
"""

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

#%%Set the path to the simulation and data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

# sim = 'Tpatch800_a288m290s2_1ms_a'
sim = 'Tpatch800_a285m290s5_1ms_na'
# sim = 'Tpatch800_a285m290s5_1ms_a'
# sim = 'test_hom_delta5_clipH'
# sim = 'test_hom_delta5_na_stable'
# sim = 'test_hom_delta5_a'
path_to_data = path + sim + '/data/'

extension = str(250000)
# extension = '1hr'

mom3D = xr.open_dataarray(path_to_data + 'Momentum3D/Data_Momentum_'+extension+'.nc')
mom2D = xr.open_dataarray(path_to_data + 'Momentum2D/Data_Momentum_2D_'+extension+'.nc')
sc3D = xr.open_dataarray(path_to_data + 'Scalar3D/Data_Scalar_'+extension+'.nc')
sc2D = xr.open_dataarray(path_to_data + 'Scalar2D/Data_Scalar_2D_'+extension+'.nc')
# aniso = xr.open_dataarray(path_to_data + 'Anisotropy/Data_Anisotropy_' + str(nRAV_start[i]) + '.nc')

#%%Plot surface temperature

Tsfc = sc2D[:,:,-2].values*Tscale

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(8,4))

# kde = gaussian_kde((Tsfc).flatten())
# x_pdf = np.linspace(min((Tsfc).flatten()), max((Tsfc).flatten()), 1000)
# pdf = kde(x_pdf)
# axs[0].hist((Tsfc).flatten(), bins=30, density=True, alpha=0.4, label="Histogram")
# axs[0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
axs[0].set_xlabel(r"$T_{sfc}$ [K]",fontsize=15)
axs[0].set_ylabel(f'PDF',fontsize=15)
p = axs[1].pcolormesh(x,y,(Tsfc).T,cmap='hot_r')
axs[1].set_xlabel(r"$x/z_i$",fontsize=15)
axs[1].set_ylabel(r"$y/z_i$",fontsize=15)
fig.suptitle(r"Surface T", fontsize=12)
cbar = plt.colorbar(p)

plt.show()

#%%Plot histogram and pdf of u,v,w values at different heights

from scipy.stats import gaussian_kde

zlevel = 5

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

for i in range(3):
    kde = gaussian_kde((mom3D[:,:,zlevel,i].values).flatten())
    x_pdf = np.linspace(min((mom3D[:,:,zlevel,i].values).flatten()), max((mom3D[:,:,zlevel,i].values).flatten()), 1000)
    pdf = kde(x_pdf)
    
    p = axs[0,i].pcolormesh(x,y,(mom3D[:,:,zlevel,i].values).T,cmap='bwr')
    axs[1,i].hist((mom3D[:,:,zlevel,i].values).flatten(), bins=30, density=True, alpha=0.4, label="Histogram")
    axs[1,i].plot(x_pdf, pdf, 'r-', label="KDE PDF")
    cbar = plt.colorbar(p)
    axs[0,i].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(f'u at {zlevel*dz*zi + dz*zi/2 :.02f}m', fontsize=14)
axs[1,1].set_xlabel(f'v at {zlevel*dz*zi + dz*zi/2 :.02f}m', fontsize=14)
axs[1,2].set_xlabel(f'w at {zlevel*dz*zi :.02f}m', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(2):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Plot pdf for ustar and heatflux and zoverL

ustar3D = (((mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2] - mom3D[:,:,:,20])**2 + \
          (mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2] - mom3D[:,:,:,21])**2)**0.25).values
    
heatflux3D = (sc3D[:,:,:,4] - uvpnode2wnode(sc3D[:,:,:,0])*mom3D[:,:,:,2] - sc3D[:,:,:,7]).values

L3D = -(wnode2uvpnode(ustar3D)**3)*sc3D[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D))

zoverL3D = np.ones((Nx,Ny,Nz),order='F')*z_uvp/L3D

from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((ustar3D[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((ustar3D[:,:,zlevel]).flatten()), max((ustar3D[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,ustar3D[:,:,zlevel].T,cmap='bwr',vmin=0,vmax=1)
axs[1,0].hist((ustar3D[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((heatflux3D[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((heatflux3D[:,:,zlevel]).flatten()), max((heatflux3D[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,heatflux3D[:,:,zlevel].T,cmap='bwr',vmin=-0.001,vmax=0.001)
axs[1,1].hist((heatflux3D[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((zoverL3D[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((zoverL3D[:,:,zlevel]).flatten()), max((zoverL3D[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,zoverL3D[:,:,zlevel].T,cmap='bwr',vmin=-10,vmax=10)
axs[1,2].hist((zoverL3D[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)
# axs[1,2].set_xscale('log')

axs[1,0].set_xlabel(r'$u_*$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs[1,1].set_xlabel(r'$\overline{wT}$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs[1,2].set_xlabel(r'$\zeta$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(2):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Plot the pdf of temperature and temperature variance

TT = (sc3D[:,:,:,1].values - sc3D[:,:,:,0].values*sc3D[:,:,:,0].values)*(Tscale**2)

from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(10,7))

kde = gaussian_kde((sc3D[:,:,zlevel,0].values*Tscale).flatten())
x_pdf = np.linspace(min((sc3D[:,:,zlevel,0].values*Tscale).flatten()), max((sc3D[:,:,zlevel,0].values*Tscale).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(sc3D[:,:,zlevel,0].values*Tscale).T,cmap='bwr')
axs[1,0].hist((sc3D[:,:,zlevel,0].values*Tscale).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((TT[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((TT[:,:,zlevel]).flatten()), max((TT[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(TT[:,:,zlevel]).T,cmap='bwr')
axs[1,1].hist((TT[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$T$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1,1].set_xlabel(r'$\overline{TT}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)

axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Plot pdf of the Reynolds stresses

Rxx = mom3D[:,:,:,4].values - mom3D[:,:,:,0].values*mom3D[:,:,:,0].values - mom3D[:,:,:,16].values
Ryy = mom3D[:,:,:,5].values - mom3D[:,:,:,1].values*mom3D[:,:,:,1].values - mom3D[:,:,:,17].values
Rzz = wnode2uvpnode(mom3D[:,:,:,6].values - mom3D[:,:,:,2].values*mom3D[:,:,:,2].values) - mom3D[:,:,:,18].values
Rxy = mom3D[:,:,:,13].values - mom3D[:,:,:,0].values*mom3D[:,:,:,1].values - mom3D[:,:,:,19].values
Rxz = wnode2uvpnode(mom3D[:,:,:,14].values - uvpnode2wnode(mom3D[:,:,:,0].values)*mom3D[:,:,:,2].values - mom3D[:,:,:,20].values)
Ryz = wnode2uvpnode(mom3D[:,:,:,15].values - uvpnode2wnode(mom3D[:,:,:,1].values)*mom3D[:,:,:,2].values - mom3D[:,:,:,21].values)

from scipy.stats import gaussian_kde

zlevel = 00

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((Rxx[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rxx[:,:,zlevel]).flatten()), max((Rxx[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,Rxx[:,:,zlevel].T,cmap='bwr')
axs[1,0].hist((Rxx[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((Ryy[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Ryy[:,:,zlevel]).flatten()), max((Ryy[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,Ryy[:,:,zlevel].T,cmap='bwr')
axs[1,1].hist((Ryy[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((Rzz[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((Rzz[:,:,zlevel]).flatten()), max((Rzz[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,Rzz[:,:,zlevel].T,cmap='bwr')
axs[1,2].hist((Rzz[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)
# axs[1,2].set_xscale('log')

axs[1,0].set_xlabel(r'$R_{xx}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1,1].set_xlabel(r'$R_{yy}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[1,2].set_xlabel(r'$R_{zz}$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(2):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Streamwise and spanwise velocity gradient

from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(10,5))

kde = gaussian_kde((mom3D[:,:,zlevel,22].values).flatten())
x_pdf = np.linspace(min((mom3D[:,:,zlevel,22].values).flatten()), max((mom3D[:,:,zlevel,22].values).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(mom3D[:,:,zlevel,22].values).T,cmap='bwr')
axs[1,0].hist((mom3D[:,:,zlevel,22].values).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((mom3D[:,:,zlevel,23].values).flatten())
x_pdf = np.linspace(min((mom3D[:,:,zlevel,23].values).flatten()), max((mom3D[:,:,zlevel,23].values).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(mom3D[:,:,zlevel,23].values).T,cmap='bwr')
axs[1,1].hist((mom3D[:,:,zlevel,23].values).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$\partial u/\partial z$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs[1,1].set_xlabel(r'$\partial v/\partial z$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)

axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Surface scaling functions

phiM = sc2D[:,:,2].values
phiH = sc2D[:,:,4].values
psiM = sc2D[:,:,3].values
psiH = sc2D[:,:,5].values


from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,4,tight_layout=True,figsize=(10,5))

kde = gaussian_kde((phiM[:,:]).flatten())
x_pdf = np.linspace(min((phiM[:,:]).flatten()), max((phiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(phiM[:,:]).T,cmap='bwr')
axs[1,0].hist((phiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((phiH[:,:]).flatten())
x_pdf = np.linspace(min((phiH[:,:]).flatten()), max((phiH[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(phiH[:,:]).T,cmap='bwr')
axs[1,1].hist((phiH[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiM[:,:]).flatten())
x_pdf = np.linspace(min((psiM[:,:]).flatten()), max((psiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,(psiM[:,:]).T,cmap='bwr')
axs[1,2].hist((psiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiH[:,:]).flatten())
x_pdf = np.linspace(min((psiH[:,:]).flatten()), max((psiH[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,3].pcolormesh(x,y,(psiH[:,:]).T,cmap='bwr')
axs[1,3].hist((psiH[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,3].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,3].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$\phi_M$', fontsize=14)
axs[1,1].set_xlabel(r'$\phi_H$', fontsize=14)
axs[1,2].set_xlabel(r'$\psi_M$', fontsize=14)
axs[1,3].set_xlabel(r'$\psi_H$', fontsize=14)

axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()
































