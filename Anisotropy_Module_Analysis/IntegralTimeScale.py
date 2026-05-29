#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 10:32:10 2026

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr
import os

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from functions import get_dphidx,get_dphidy,get_dphidz,uvpnode2wnode,wnode2uvpnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso,read_checkpoint_sfc, read_checkpoint_sfc_L

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

#%%Import Data

res = 128
step = 133000

sim = 'patch'+str(res)+'_classic_1ms_v2'
# sim = 'homo'+str(res)+'_classic_1ms'

path_data = '/scratch/general/nfs1/u1450851/LES_Sims/Patch/v2/'+str(res)+'/'+sim
# path_data = '/scratch/general/nfs1/u1450851/LES_Sims/Homog/'+str(res)+'/'+sim

data = xr.open_dataarray(path_data+'/data/Momentum3D/Data_Momentum_30min.nc')
checkpnt = read_checkpoint_aniso(path_data+'/output_checkpoint/',[step],nx,ny,nz)

#%%Autocorrelation function along x direction
#--------------------------------------------------------------------------------------------------
#%%Compute the autocorrelation function

# u = (data[:,:,:,0].data)*uscale
u = checkpnt['u'][:,:,:-1]*uscale
uprime = np.zeros_like(u)
for k in range(nz):
    uprime[:,:,k] = u[:,:,k] - np.mean(u[:,:,k],axis=(0,1))
# uvar = np.mean(uprime**2,axis=(0,1))
autocorr = np.zeros((nx//2,ny,nz))

# for k in range(nz):
#     print(k)
#     for j in range(ny):
#         uline = uprime[:, j, k]
#         var_line = np.mean(uline**2)
#         for i in range(nx//2):
#             autocorr[i, j, k] = np.mean(uline[:nx-i] * uline[i:nx]) / var_line

# #FFT faster 
for k in range(nz):
    u_slice = uprime[:, :, k]
    u_hat = np.fft.fft(u_slice, axis=0)
    R = np.fft.ifft(u_hat * np.conjugate(u_hat), axis=0).real / nx
    var = np.mean(u_slice**2, axis=0)
    for i in range(ny):
        R[:,i] = R[:,i]/var[i]
    # R /= var[np.newaxis, :]
    autocorr[:, :, k] = R[:nx//2, :]

#%%Compute integral length and time scale

auto_xz = np.mean(autocorr,axis=(1))

def first_zero_or_min(arr):
    arr = np.asarray(arr)

    zero_crossings = np.where(arr <= 0)[0]
    if zero_crossings.size > 0:
        return zero_crossings[0]

    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
            return i

    return np.argmin(arr)

L = np.zeros(nz)
L2D = np.zeros((ny,nz))
for k in range(nz):
    idx = first_zero_or_min(auto_xz[:, k])
    L[k] = np.trapz(auto_xz[:idx, k], dx=dx*zi)
    for j in range(ny):
        idx = first_zero_or_min(autocorr[:, j, k])
        L2D[j,k] = np.trapz(autocorr[:idx, j, k], dx=dx*zi)

U_mean = np.mean(u,axis=(0,1))
T = L/U_mean

#%%Plot Autocorrelation function corves

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(nz):
    if ~np.any(auto_xz[:,i]<0):
        axs.plot(x[:nx//2],auto_xz[:,i])
    
axs.axhline(0,c='k')
plt.show()

#%%Plot integral time/length scale profile

fig,axs = plt.subplots(1,1,tight_layout=True)
# axs.plot(np.mean(L2D,axis=(0)),z_uvp)
# axs.plot(L,z_uvp)
axs.plot(T/60,z_uvp)
# axs.set_ylim(0,1)
plt.show()

#%%Autocorrelation in the y direction
#-----------------------------------------------------------------------------------------------------------
#%%Compute the autocorrelation function

# u = (data[:,:,:,0].data)*uscale
u = checkpnt['u'][:,:,:-1]*uscale
uprime = np.zeros_like(u)
for k in range(nz):
    uprime[:,:,k] = u[:,:,k] - np.mean(u[:,:,k],axis=(0,1))
# uvar = np.mean(uprime**2,axis=(0,1))
autocorr = np.zeros((nx,ny//2,nz))

# for k in range(nz):
#     print(k)
#     for j in range(ny):
#         uline = uprime[:, j, k]
#         var_line = np.mean(uline**2)
#         for i in range(nx//2):
#             autocorr[i, j, k] = np.mean(uline[:nx-i] * uline[i:nx]) / var_line

# #FFT faster 
for k in range(nz):
    u_slice = uprime[:, :, k]
    u_hat = np.fft.fft(u_slice, axis=1)
    R = np.fft.ifft(u_hat * np.conjugate(u_hat), axis=1).real / ny
    var = np.mean(u_slice**2, axis=1)
    for i in range(nx):
        R[i,:] = R[i,:]/var[i]
    # R /= var[:, np.newaxis]
    autocorr[:, :, k] = R[:, :ny//2]

#%%Compute integral length and time scale

auto_yz = np.mean(autocorr,axis=(0))

def first_zero_or_min(arr):
    arr = np.asarray(arr)

    zero_crossings = np.where(arr <= 0)[0]
    if zero_crossings.size > 0:
        return zero_crossings[0]

    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
            return i

    return np.argmin(arr)

L = np.zeros(nz)
L2D = np.zeros((nx,nz))
for k in range(nz):
    idx = first_zero_or_min(auto_yz[:, k])
    L[k] = np.trapz(auto_yz[:idx, k], dx=dy*zi)
    for j in range(nx):
        idx = first_zero_or_min(autocorr[j, :, k])
        L2D[j,k] = np.trapz(autocorr[j, :idx, k], dx=dy*zi)

U_mean = np.mean(u,axis=(0,1))
T = L/U_mean

#%%Plot Autocorrelation function corves

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(nz):
    if ~np.any(auto_yz[:,i]<0):
        axs.plot(y[:ny//2],auto_yz[:,i])
    
axs.axhline(0,c='k')
plt.show()

#%%Plot integral time/length scale profile

fig,axs = plt.subplots(1,1,tight_layout=True)
# axs.plot(np.mean(L2D,axis=(0)),z_uvp)
# axs.plot(L,z_uvp)
axs.plot(T/60,z_uvp)
# axs.set_ylim(0,1)
plt.show()

#%%Computing the autocorrelation function averaging velocity over lines instead of planes

# u = (data[:,:,:,0].data)*uscale
u = checkpnt['v'][:,:,:-1]*uscale
uprime = np.zeros_like(u)
for k in range(nz):
    for j in range(ny):
        uprime[:,j,k] = u[:,j,k] - np.mean(u[:,j,k],axis=(0))
# uvar = np.mean(uprime**2,axis=(0,1))
autocorr = np.zeros((nx//2,ny,nz))

# #FFT faster 
for k in range(nz):
    u_slice = uprime[:, :, k]
    u_hat = np.fft.fft(u_slice, axis=0)
    R = np.fft.ifft(u_hat * np.conjugate(u_hat), axis=0).real / nx
    var = np.mean(u_slice**2, axis=0)
    for i in range(ny):
        R[:,i] = R[:,i]/var[i]
    # R /= var[:, np.newaxis]
    autocorr[:, :, k] = R[:nx//2, :]

auto_xz = np.mean(autocorr,axis=(1))

def first_zero_or_min(arr):
    arr = np.asarray(arr)

    zero_crossings = np.where(arr <= 0)[0]
    if zero_crossings.size > 0:
        return zero_crossings[0]

    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
            return i

    return np.argmin(arr)

L = np.zeros(nz)
L2D = np.zeros((ny,nz))
T2D = np.zeros((ny,nz))
for k in range(nz):
    idx = first_zero_or_min(auto_xz[:, k])
    L[k] = np.trapz(auto_xz[:idx, k], dx=dx*zi)
    for j in range(ny):
        idx = first_zero_or_min(autocorr[:, j, k])
        L2D[j,k] = np.trapz(autocorr[:idx, j, k], dx=dx*zi)
        T2D[j,k] = L2D[j,k]/np.mean(u[:,j,k])

U_mean = np.mean(u,axis=(0,1))
T = L/U_mean

#%%Plot Autocorrelation function corves

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(nz):
    if ~np.any(auto_xz[:,i]<0):
        axs.plot(x[:nx//2],auto_xz[:,i])
    
axs.axhline(0,c='k')
plt.show()

#%%Plot integral time/length scale profile

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.plot(L,z_uvp)
axs.plot(np.mean(L2D,axis=(0)),z_uvp)
# axs.plot(T/60,z_uvp)
# axs.plot(np.mean(T2D,axis=(0))/60,z_uvp)
# axs.set_ylim(0,1)
plt.show()

#%%

#Compute the integral time scale by taking the fluctuating velocity field as the last checkpoint minus the 30 minute RAV

#%%

u_ist = checkpnt['u'][:,:,:nz]
u_rav = data[:,:,:,0].data

u_prime = u_ist - u_rav
































