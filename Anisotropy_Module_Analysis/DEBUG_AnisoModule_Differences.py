#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:24:57 2025

@author: u1450851
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

#%%

#Set some simulation parameters

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

#%%
#Set the path to the simulation and data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
cases = ['800p_128_noaniso_1ms','test_mag_L_ustar','800p_128_aniso_1ms']
nRAV = 50000
mom3D = dict(); mom2D = dict(); sc3D = dict(); sc2D = dict(); aniso = dict()

for i in range(len(cases)):
    sim = cases[i]
    path_to_data = path + sim + '/data/'
    mom3D[cases[i]] = xr.open_dataarray(path_to_data + 'Momentum3D/Data_Momentum_' + str(nRAV) + '.nc')
    mom2D[cases[i]] = xr.open_dataarray(path_to_data + 'Momentum2D/Data_Momentum_2D_' + str(nRAV) + '.nc')
    sc3D[cases[i]] = xr.open_dataarray(path_to_data + 'Scalar3D/Data_Scalar_' + str(nRAV) + '.nc')
    sc2D[cases[i]] = xr.open_dataarray(path_to_data + 'Scalar2D/Data_Scalar_2D_' + str(nRAV) + '.nc')
    aniso[cases[i]] = xr.open_dataarray(path_to_data + 'Anisotropy/Data_Anisotropy_' + str(nRAV) + '.nc')


#%%Plot the surface temperature 

Tsfc = sc2D[cases[0]][:,:,-2].values

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,(Tsfc*Tscale).T,cmap='hot_r')
axs.set_xlabel(r"$x/z_i$",fontsize=15)
axs.set_ylabel(r"$y/z_i$",fontsize=15)
axs.set_title(r"Surface T", fontsize=12)
cbar = plt.colorbar(p)

plt.show()

#%%Plot the differences in ustar, wT and L

ustar = abs(mom2D[cases[0]][:,:,0].values - mom2D[cases[1]][:,:,0].values)
wT = abs(sc2D[cases[0]][:,:,-1].values - sc2D[cases[1]][:,:,-1].values)
L = abs(sc2D[cases[0]][:,:,1].values - sc2D[cases[1]][:,:,1].values)

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,4))

p1 = axs[0].pcolormesh(x,y,ustar.T,cmap = 'hot_r',vmin=0,vmax=1)
p2 = axs[1].pcolormesh(x,y,wT.T,cmap = 'hot_r',vmin=0,vmax=0.005)
p3 = axs[2].pcolormesh(x,y,L.T,cmap = 'hot_r',vmin=0,vmax=0.05)

cbar = plt.colorbar(p1)
cbar = plt.colorbar(p2)
cbar = plt.colorbar(p3)

plt.show()

#%%Plot the difference in the scaling functions

phiM = abs(sc2D[cases[0]][:,:,2].values - sc2D[cases[1]][:,:,2].values)
psiM = abs(sc2D[cases[0]][:,:,3].values - sc2D[cases[1]][:,:,3].values)
phiH = abs(sc2D[cases[0]][:,:,4].values - sc2D[cases[1]][:,:,4].values)
psiH = abs(sc2D[cases[0]][:,:,5].values - sc2D[cases[1]][:,:,5].values)

fig,axs = plt.subplots(1,4,tight_layout=True,figsize=(12,4))

p1 = axs[0].pcolormesh(x,y,phiM.T,cmap = 'hot_r',vmin=0,vmax=0.5)
p2 = axs[1].pcolormesh(x,y,psiM.T,cmap = 'hot_r',vmin=0,vmax=2)
p3 = axs[2].pcolormesh(x,y,phiH.T,cmap = 'hot_r',vmin=0,vmax=0.5)
p4 = axs[3].pcolormesh(x,y,psiH.T,cmap = 'hot_r',vmin=0,vmax=5)

cbar = plt.colorbar(p1)
cbar = plt.colorbar(p2)
cbar = plt.colorbar(p3)
cbar = plt.colorbar(p4)

plt.show()

#%%Plot the difference in the velocity field

zslice = 1
u = abs(mom3D[cases[0]][:,:,zslice,0].values - mom3D[cases[1]][:,:,zslice,0].values)
v = abs(mom3D[cases[0]][:,:,zslice,1].values - mom3D[cases[1]][:,:,zslice,1].values)
w = abs(mom3D[cases[0]][:,:,zslice,2].values - mom3D[cases[1]][:,:,zslice,2].values)

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,4))

p1 = axs[0].pcolormesh(x,y,u.T,cmap = 'hot_r',vmin=0)
p2 = axs[1].pcolormesh(x,y,v.T,cmap = 'hot_r',vmin=0)
p3 = axs[2].pcolormesh(x,y,w.T,cmap = 'hot_r',vmin=0)

cbar = plt.colorbar(p1)
cbar = plt.colorbar(p2)
cbar = plt.colorbar(p3)

plt.show()


















































