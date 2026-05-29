#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:04:19 2025

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
import os
import sys
import copy
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from read_checkpoint_SC import read_checkpoint

from read_checkpoint_SC import read_checkpoint_sfc

#%%#inputs

sim = '64x3_30min_noaniso'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/'

nx = 64
ny = 64
nz = 64
lx = 0.5*np.pi
ly = 0.5*np.pi
lz = 0.5
dx = lx/nx
dy = ly/ny
dz = lz/nz

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.1 #05; %0.000005;
zi = 500.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = path

#%%Import variables


var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']
    
var2D = ['avgUstar']
    
varS = ['avgT','avgT2','avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus','avg_ds']

varS2D = ['avgWstar','avgL','avgPHIm','avgPSIm','avgPHIh','avgPSIh','avgSFCval','avgSFCflux']

varA = ['avgXB','avgYB']#,'avgPHIM','avgPHIH','avgPSIM','avgPSIH','avgL3D','avgustar3D','avgSCF3D']
    
dataM = xr.open_dataarray(path+'Data_Momentum.nc')
dataM2D = xr.open_dataarray(path+'Data_Momentum_2D.nc')
dataS = xr.open_dataarray(path+'Data_Scalar.nc')
dataS2D = xr.open_dataarray(path+'Data_Scalar_2D.nc')
# dataA = xr.open_dataarray(path+'Data_Anisotropy.nc')

data_mom = dict()
data_mom_2D = dict()
data_sc = dict()
data_sc_2D = dict()
# data_aniso = dict()

for i in range(len(var)):
    data_mom[var[i]] = dataM.data[:,:,:,i]
    
for i in range(len(var2D)):
    data_mom_2D[var2D[i]] = dataM2D.data[:,:,i]

for i in range(len(varS)):
    data_sc[varS[i]] = dataS.data[:,:,:,i]

for i in range(len(varS2D)):
    data_sc_2D[varS2D[i]] = dataS2D.data[:,:,i]
    
    
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

wn_x = 2*np.pi*np.fft.rfftfreq(nx, dx)
wn_y = 2*np.pi*np.fft.rfftfreq(ny, dy)

#%%

def create_spectral_filter_rfft(nx, ny, dx, dy, alpha, ifilter=1):
    kx = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi  # shape (nx//2+1,)
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi   # shape (ny,)
    
    # Filter will match rfft2 output shape: (ny, nx//2+1)
    kky, kkx = np.meshgrid(ky, kx, indexing='ij')  # shape (ny, nx//2+1)
    kk2 = kkx**2 + kky**2

    G = np.ones((ny, nx // 2 + 1)) / (nx * ny)
    delta = alpha * np.sqrt(dx * dy)

    if ifilter == 1:  # Spectral cutoff
        kc2 = (np.pi / delta)**2
        G[kk2 >= kc2] = 0.0
    elif ifilter == 2:  # Gaussian
        G *= np.exp(-(2 * delta)**2 * kk2 / (4 * 6))
    elif ifilter == 3:  # Top-hat
        G *= (np.sin(kkx * delta / 2) * np.sin(kky * delta / 2) + 1e-8) / \
             (kkx * delta / 2 * kky * delta / 2 + 1e-8)

    # Manually zero Nyquist modes
    G[:, -1] = 0.0  # Nyquist in x
    G[ny // 2, :] = 0.0  # Nyquist in y

    return G


def test_filter_layer(f, G):
    """
    Apply spectral filter to a 2D field using rfft2.
    """
    nx, ny = f.shape
    f_hat = np.fft.rfft2(f)
    f_hat_filtered = f_hat * G
    f_filtered = np.fft.irfft2(f_hat_filtered, s=(nx, ny))
    return f_filtered


def get_layer_filter_3Din(var_in, dx=1.0, dy=1.0, alpha=2.0, layer=1):
    """
    Applies a spectral filter to a horizontal layer of a 3D array.

    Parameters:
    - var_in: 3D numpy array of shape (nx, ny, nz)
    - dx, dy: spatial resolution in x and y directions
    - alpha: filter sharpness (e.g., alpha=2)
    - layer: 1-based index of the layer to extract (Fortran style)

    Returns:
    - var_out: 2D filtered slice
    """
    # Convert layer from 1-based to 0-based index
    z = layer - 1
    nx, ny, nz = var_in.shape
    assert 0 <= z < nz, "Layer index out of bounds"

    # Extract 2D layer
    var_glob = var_in[:, :, z]

    # Create filter in rfft2 format
    # G = create_spectral_filter_rfft(nx, ny, dx, dy, alpha)
    G = create_spectral_filter_rfft(ny, nx, dy, dx, alpha)

    # Apply filter
    var_out = test_filter_layer(var_glob, G)

    return var_out

#%%Compute the filtered velocity at the surface like in the LES

u1 = np.zeros_like(data_mom['avgU'][:,:,0])
v1 = np.zeros_like(data_mom['avgU'][:,:,0])
u_avg = np.zeros_like(data_mom['avgU'][:,:,0])

u1 = get_layer_filter_3Din(data_mom['avgU'],dx,dy,2,1)
v1 = get_layer_filter_3Din(data_mom['avgV'],dx,dy,2,1)
u_avg = np.sqrt(u1**2 + v1**2)

#%%Compute the SGS stress terms in post process

nut = wnode2uvpnode(data_mom['avgNut'])

tij = {
       'txx' : -2*nut*get_dphidx(data_mom['avgU'], wn_x),
       'tyy' : -2*nut*get_dphidy(data_mom['avgV'], wn_y),
       'tzz' : -2*nut*get_dphidz(data_mom['avgW'], dz),
       'txy' : -2*nut*0.5*(get_dphidx(data_mom['avgV'], wn_x) + get_dphidy(data_mom['avgU'], wn_y)),
       'txz' : -2*nut*0.5*(get_dphidx(data_mom['avgW'], wn_x) + (data_mom['avgdudz'])),
       'tyz' : -2*nut*0.5*(get_dphidy(data_mom['avgW'], wn_y) + (data_mom['avgdvdz']))
       }

tij['txz'][:,:,0] = -(data_mom_2D['avgUstar']**2)*(u1/u_avg)
tij['tyz'][:,:,0] = -(data_mom_2D['avgUstar']**2)*(v1/u_avg)

l1 = ['txx','txy','txz','tyy','tyz','tzz']
l2 = ['avgtxx','avgtxy','avgtxz','avgtyy','avgtyz','avgtzz']

fig,axs = plt.subplots(1,6,figsize=(10,4))

for i in range(len(axs)):
    axs[i].plot(np.mean(tij[l1[i]],axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='k',label=l1[i])
    axs[i].plot(np.mean(-data_mom[l2[i]],axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='r',ls='--',label=l2[i])
    axs[i].legend()
    axs[i].set_ylim(0,nz*dz)
    
plt.show()


#%%Plot profile of txz pre and post interpolation

fig,axs = plt.subplots(1,1,figsize=(4,6))
axs.plot(np.mean(-data_mom['avgtxz'],axis=(0,1)),np.arange(0,nz)*dz, c='k')
axs.plot(np.mean(wnode2uvpnode(-data_mom['avgtxz']),axis=(0,1)),np.arange(0,nz)*dz,c='g')
plt.show()

#%%Plot profiles of the diagonal terms of the SGS tensor

fig,axs = plt.subplots(1,3,figsize=(6,4))
axs[0].plot(np.mean(data_mom['avgtxx'],axis=(0,1)),np.arange(0,nz)*dz + dz/2, c='k', label='txx')
axs[1].plot(np.mean(data_mom['avgtyy'],axis=(0,1)),np.arange(0,nz)*dz + dz/2, c='k', label='tyy')
axs[2].plot(np.mean(data_mom['avgtzz'],axis=(0,1)),np.arange(0,nz)*dz + dz/2, c='k', label='tzz')
for i in range(len(axs)):
    axs[i].legend()
    
plt.show()

#%%Plot the gradient of vertical velocity
dudx = get_dphidx(data_mom['avgU'], wn_x)
dwdz = get_dphidz(data_mom['avgW'], dz)
# fig,axs = plt.subplots(1,1)
# axs.plot(np.mean(get_dphidz(data_mom['avgW'], dz),axis=(0,1)),np.arange(0,nz)*dz)
# plt.show()













































