#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 13:27:19 2025

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

#%%sfc T patches

Tsfc = sc2D_a[:,:,-2].values

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,(Tsfc*Tscale).T,cmap='hot_r',vmin=285,vmax=295)
axs.set_xlabel(r"$x/z_i$",fontsize=15)
axs.set_ylabel(r"$y/z_i$",fontsize=15)
# axs.set_title(r"Surface T", fontsize=12)
cbar = plt.colorbar(p)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/SfcT_patches.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%ustar


ustar3D_a = (((mom3D_a[:,:,:,14] - uvpnode2wnode(mom3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,20])**2 + \
          (mom3D_a[:,:,:,15] - uvpnode2wnode(mom3D_a[:,:,:,1])*mom3D_a[:,:,:,2] - mom3D_a[:,:,:,21])**2)**0.25).values
ustar3D_na = (((mom3D_na[:,:,:,14] - uvpnode2wnode(mom3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,20])**2 + \
          (mom3D_na[:,:,:,15] - uvpnode2wnode(mom3D_na[:,:,:,1])*mom3D_na[:,:,:,2] - mom3D_na[:,:,:,21])**2)**0.25).values
    
zlevel = 0

diff = ustar3D_a[:,:,zlevel] - ustar3D_na[:,:,zlevel]

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

p = axs.pcolormesh(x,y,diff.T,cmap='bwr',vmin=-0.2,vmax=0.2)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$y/z_i$',fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
cbar = plt.colorbar(p)
cbar.ax.tick_params(labelsize=12)
plt.show()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/ustar_pcolor_z0.png',dpi=300,facecolor='white', edgecolor='white')

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

kde = gaussian_kde((ustar3D_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((ustar3D_a[:,:,zlevel]).flatten()), max((ustar3D_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((ustar3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((ustar3D_na[:,:,zlevel]).flatten()), max((ustar3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'b--', label="PDF No Correction")
axs.axvline(np.mean((ustar3D_a[:,:,zlevel]).flatten()),c='r',ls=':')
axs.axvline(np.mean((ustar3D_na[:,:,zlevel]).flatten()),c='b',ls=':')
axs.set_xlabel(r'$u_*/u_s$ at '+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs.set_ylabel(r'$PDF$',fontsize=14)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_ylim(0)

plt.show()


# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/ustar_pdf_z10.png',dpi=300,facecolor='white', edgecolor='white')

#%%

heatflux3D_a = (sc3D_a[:,:,:,4] - uvpnode2wnode(sc3D_a[:,:,:,0])*mom3D_a[:,:,:,2] - sc3D_a[:,:,:,7]).values
heatflux3D_na = (sc3D_na[:,:,:,4] - uvpnode2wnode(sc3D_na[:,:,:,0])*mom3D_na[:,:,:,2] - sc3D_na[:,:,:,7]).values

zlevel = 0

diff = heatflux3D_a[:,:,zlevel] - heatflux3D_na[:,:,zlevel]

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

p = axs.pcolormesh(x,y,diff.T,cmap='bwr',vmin=-0.0005,vmax=0.0005)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$y/z_i$',fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
cbar = plt.colorbar(p)
cbar.ax.tick_params(labelsize=12)
plt.show()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/heatflux_pcolor_z0.png',dpi=300,facecolor='white', edgecolor='white')

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

kde = gaussian_kde((heatflux3D_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((heatflux3D_a[:,:,zlevel]).flatten()), max((heatflux3D_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((heatflux3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((heatflux3D_na[:,:,zlevel]).flatten()), max((heatflux3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'b--', label="PDF No Correction")
axs.axvline(np.mean((heatflux3D_a[:,:,zlevel]).flatten()),c='r',ls=':')
axs.axvline(np.mean((heatflux3D_na[:,:,zlevel]).flatten()),c='b',ls=':')
axs.set_xlabel(r"$\overline{w'T'}/u_sT_s$ at "+f'{zlevel*dz*zi :.02f}m', fontsize=14)
axs.set_ylabel(r'$PDF$',fontsize=14)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_ylim(0)

plt.show()


# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/heatflux_pdf_z10.png',dpi=300,facecolor='white', edgecolor='white')

#%%

zoverL3D_a = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_a)**3)*sc3D_a[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_a)))
zoverL3D_na = np.ones((Nx,Ny,Nz),order='F')*z_uvp/(-(wnode2uvpnode(ustar3D_na)**3)*sc3D_na[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D_na)))

zlevel = 10

diff = zoverL3D_a[:,:,0] - zoverL3D_na[:,:,0]

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

p = axs.pcolormesh(x,y,diff.T,cmap='bwr',vmin=-10,vmax=10)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$y/z_i$',fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
cbar = plt.colorbar(p)
cbar.ax.tick_params(labelsize=12)
plt.show()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/zeta_pcolor_z10.png',dpi=300,facecolor='white', edgecolor='white')

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

kde = gaussian_kde((zoverL3D_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((zoverL3D_a[:,:,zlevel]).flatten()), max((zoverL3D_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((zoverL3D_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((zoverL3D_na[:,:,zlevel]).flatten()), max((zoverL3D_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'b--', label="PDF No Correction")
axs.axvline(np.mean((zoverL3D_a[:,:,zlevel]).flatten()),c='r',ls=':')
axs.axvline(np.mean((zoverL3D_na[:,:,zlevel]).flatten()),c='b',ls=':')
axs.set_xlabel(r'$\zeta$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs.set_ylabel(r'$PDF$',fontsize=14)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_ylim(0)

plt.show()


# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/zeta_pdf_z10.png',dpi=300,facecolor='white', edgecolor='white')


#%%

T_a = sc3D_a[:,:,:,0].values*Tscale
T_na = sc3D_na[:,:,:,0].values*Tscale

zlevel = 10

diff = T_a[:,:,0] - T_na[:,:,0]

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

p = axs.pcolormesh(x,y,diff.T,cmap='bwr',vmin=-1,vmax=1)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$y/z_i$',fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
cbar = plt.colorbar(p)
cbar.ax.tick_params(labelsize=12)
plt.show()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/T_pcolor_z10.png',dpi=300,facecolor='white', edgecolor='white')

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

kde = gaussian_kde((T_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((T_a[:,:,zlevel]).flatten()), max((T_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((T_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((T_na[:,:,zlevel]).flatten()), max((T_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'b--', label="PDF No Correction")
axs.axvline(np.mean((T_a[:,:,zlevel]).flatten()),c='r',ls=':')
axs.axvline(np.mean((T_na[:,:,zlevel]).flatten()),c='b',ls=':')
axs.set_xlabel(r'$T$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs.set_ylabel(r'$PDF$',fontsize=14)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_ylim(0)

plt.show()


# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/T_pdf_z10.png',dpi=300,facecolor='white', edgecolor='white')

#%%

phiM_a = sc2D_a[:,:,5].values
phiM_na = sc2D_na[:,:,5].values

diff = phiM_a[:,:] - phiM_na[:,:]

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

p = axs.pcolormesh(x,y,diff.T,cmap='bwr',vmin=-10,vmax=10)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$y/z_i$',fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
cbar = plt.colorbar(p)
cbar.ax.tick_params(labelsize=12)
plt.show()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/phiM_pcolor.png',dpi=300,facecolor='white', edgecolor='white')

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

kde = gaussian_kde((phiM_a[:,:]).flatten())
x_pdf = np.linspace(min((phiM_a[:,:]).flatten()), max((phiM_a[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((phiM_na[:,:]).flatten())
x_pdf = np.linspace(min((phiM_na[:,:]).flatten()), max((phiM_na[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'b--', label="PDF No Correction")
axs.axvline(np.median((phiM_a[:,:]).flatten()),c='r',ls=':')
axs.axvline(np.median((phiM_na[:,:]).flatten()),c='b',ls=':')
axs.set_xlabel(r'$\phi_M$', fontsize=14)
axs.set_ylabel(r'$PDF$',fontsize=14)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_ylim(0)
# axs.set_xlim(-50,2)
# axs.set_xscale('log')

plt.show()


# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/phiM_pdf.png',dpi=300,facecolor='white', edgecolor='white')

#%%

u_a = mom3D_a[:,:,:,0].values
u_na = mom3D_na[:,:,:,0].values

zlevel = 10

diff = u_a[:,:,0] - u_na[:,:,0]

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

p = axs.pcolormesh(x,y,diff.T,cmap='bwr',vmin=-2,vmax=2)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$y/z_i$',fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
cbar = plt.colorbar(p)
cbar.ax.tick_params(labelsize=12)
plt.show()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/u_pcolor_z10.png',dpi=300,facecolor='white', edgecolor='white')

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,6))

kde = gaussian_kde((u_a[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((u_a[:,:,zlevel]).flatten()), max((u_a[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'r-', label="PDF W/ Correction")
kde = gaussian_kde((u_na[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((u_na[:,:,zlevel]).flatten()), max((u_na[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
axs.plot(x_pdf, pdf, 'b--', label="PDF No Correction")
axs.axvline(np.mean((u_a[:,:,zlevel]).flatten()),c='r',ls=':')
axs.axvline(np.mean((u_na[:,:,zlevel]).flatten()),c='b',ls=':')
axs.set_xlabel(r'$u/u_s$ at '+f'{zlevel*dz*zi + dz*zi/2:.02f}m', fontsize=14)
axs.set_ylabel(r'$PDF$',fontsize=14)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_ylim(0)

plt.show()


# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/MarcPresentaion/u_pdf_z10.png',dpi=300,facecolor='white', edgecolor='white')
























































