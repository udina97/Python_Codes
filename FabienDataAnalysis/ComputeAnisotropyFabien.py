# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 04:28:44 2025

@author: udina
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

#%%Simulation parameters

nx = 256
ny = 256
nz = 256
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

#%%Some functions

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

def Anisotropy(Nx, Ny, Nz, R11, R22, R33, R12, R13, R23):

    import numpy as np
    import xarray as xr

    # Calculate TKE
    e = R11 + R22 + R33
    Ntot = Nx * Ny * Nz

    # Flatten arrays
    e_flat = e.ravel(order='F')
    R11_flat = R11.ravel(order='F')
    R22_flat = R22.ravel(order='F')
    R33_flat = R33.ravel(order='F')
    R12_flat = R12.ravel(order='F')
    R13_flat = R13.ravel(order='F')
    R23_flat = R23.ravel(order='F')

    # Build full Reynolds stress tensor (Ntot, 3, 3)
    R_all = np.zeros((Ntot, 3, 3))
    R_all[:, 0, 0] = R11_flat
    R_all[:, 1, 1] = R22_flat
    R_all[:, 2, 2] = R33_flat
    R_all[:, 0, 1] = R12_flat
    R_all[:, 1, 0] = R12_flat
    R_all[:, 0, 2] = R13_flat
    R_all[:, 2, 0] = R13_flat
    R_all[:, 1, 2] = R23_flat
    R_all[:, 2, 1] = R23_flat

    # Avoid division by zero
    e_safe = np.where(e_flat == 0.0, 1e-12, e_flat)

    # Identity matrix for subtraction
    Id = np.eye(3)

    # Compute anisotropy tensor
    B_all = R_all / e_safe[:, None, None] - (1.0 / 3.0) * Id

    # Compute eigenvalues (symmetric tensor)
    eigvals_all = np.linalg.eigvalsh(B_all)
    eigvals_sorted = np.sort(eigvals_all, axis=1)[:, ::-1]

    lambda3_flat = eigvals_sorted[:, 2]
    C1c = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    C2c = 2 * (eigvals_sorted[:, 1] - eigvals_sorted[:, 2])
    C3c = 3 * eigvals_sorted[:, 2] + 1

    xB_flat = C1c + 0.5 * C3c
    yB_flat = C3c * (np.sqrt(3) / 2)

    # Reshape back to (Nx, Ny, Nz)
    shape = (Nx, Ny, Nz)
    xB = xB_flat.reshape(shape, order='F')
    yB = yB_flat.reshape(shape, order='F')
    lambda3 = lambda3_flat.reshape(shape, order='F')

    return xB, yB, lambda3

#%%Set path to data and load the data

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/FabienData/'

case = 'P800_Ug1_Data_Momentum'

data = xr.open_dataarray(path_to_data + case + '.nc')

#%%Compute the Reynolds stresses

Rxx = data.data[:,:,:,4] - data.data[:,:,:,0]*data.data[:,:,:,0]
Ryy = data.data[:,:,:,5] - data.data[:,:,:,1]*data.data[:,:,:,1]
Rzz = wnode2uvpnode(data.data[:,:,:,6] - data.data[:,:,:,2]*data.data[:,:,:,2])
Rxy = data.data[:,:,:,13] - data.data[:,:,:,0]*data.data[:,:,:,1]
Rxz = wnode2uvpnode(data.data[:,:,:,14]) - data.data[:,:,:,0]*wnode2uvpnode(data.data[:,:,:,2])
Ryz = wnode2uvpnode(data.data[:,:,:,15]) - data.data[:,:,:,1]*wnode2uvpnode(data.data[:,:,:,2])

#%%Compute anisotropy

# xB,yB,lambda3 = Anisotropy(nx, ny, nz, Rxx, Ryy, Rzz, Rxy, Rxz, Ryz)
xB,yB,lambda3 = Anisotropy(nx, ny, nz, Rxx-data.data[:,:,:,16], Ryy-data.data[:,:,:,17], Rzz-data.data[:,:,:,18],\
                           Rxy-data.data[:,:,:,19], Rxz-wnode2uvpnode(data.data[:,:,:,20]), Ryz-wnode2uvpnode(data.data[:,:,:,21]))

#%%Check for negative valuese of xB and yB

if np.any(xB < 0) or np.any(yB < 0):
    print(f"Anisotropy has negative values")
    if np.any(xB < 0) and np.any(yB < 0):
        print("Both xB and yB have negatives")
    elif np.any(xB < 0):
        print("Only xB has negatives")
    elif np.any(yB < 0):
        print("Only yB has negatives")
else:
    print(f"Anisotropy has all positive values")
    
#%%Plot Anisotropy

os.chdir('C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\functions\\')

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
xslice = 100
yslice = 100
zslice = 1

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,6))

axs[0,0].pcolormesh(x,y,xB[:,:,zslice].T,cmap=cmap,vmin = 0,vmax = 1)
axs[0,1].pcolormesh(x,z_uvp,xB[:,yslice,:].T,cmap=cmap,vmin = 0,vmax = 1)
p1 = axs[0,2].pcolormesh(y,z_uvp,xB[xslice,:,:].T,cmap=cmap,vmin = 0,vmax = 1)

axs[1,0].pcolormesh(x,y,yB[:,:,zslice].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)
axs[1,1].pcolormesh(x,z_uvp,yB[:,yslice,:].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)
p2 = axs[1,2].pcolormesh(y,z_uvp,yB[xslice,:,:].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)

# 1. x-y plane @ zslice
neg_xB = np.where(xB[:, :, zslice] < 0)
axs[0, 0].plot(x[neg_xB[0]], y[neg_xB[1]], 'r.', markersize=20)
neg_yB = np.where(yB[:, :, zslice] < 0)
axs[1, 0].plot(x[neg_yB[0]], y[neg_yB[1]], 'r.', markersize=20)

# 2. x-z plane @ yslice
neg_xB = np.where(xB[:, yslice, :] < 0)
axs[0, 1].plot(x[neg_xB[0]], z_uvp[neg_xB[1]], 'r.', markersize=20)
neg_yB = np.where(yB[:, yslice, :] < 0)
axs[1, 1].plot(x[neg_yB[0]], z_uvp[neg_yB[1]], 'r.', markersize=20)

# 3. y-z plane @ xslice
neg_xB = np.where(xB[xslice, :, :] < 0)
axs[0, 2].plot(y[neg_xB[0]], z_uvp[neg_xB[1]], 'r.', markersize=20)
neg_yB = np.where(yB[xslice, :, :] < 0)
axs[1, 2].plot(y[neg_yB[0]], z_uvp[neg_yB[1]], 'r.', markersize=20)

axs[1,1].contour(x,z_uvp,yB[:,yslice,:].T,levels=[0.38],colors='black')
axs[1,2].contour(y,z_uvp,yB[xslice,:,:].T,levels=[0.38],colors='black')

cbar1 = plt.colorbar(p1,label='xB')
cbar2 = plt.colorbar(p2,label='yB')

axs[0,0].set_title(f"zslice = {zslice}",fontsize=15)
axs[0,1].set_title(f"yslice = {yslice}",fontsize=15)
axs[0,2].set_title(f"xslice = {xslice}",fontsize=15)

plt.show()

#%%Reynolds stress profiles

fig,axs = plt.subplots(1,6,tight_layout=True,figsize=(12,6))

axs[0].plot(np.mean(Rxx,axis=(0,1)),z_uvp,'k',label='Res')
axs[1].plot(np.mean(Ryy,axis=(0,1)),z_uvp,'k',label='Res')
axs[2].plot(np.mean(Rzz,axis=(0,1)),z_uvp,'k',label='Res')
axs[3].plot(np.mean(Rxy,axis=(0,1)),z_uvp,'k',label='Res')
axs[4].plot(np.mean(Rxz,axis=(0,1)),z_uvp,'k',label='Res')
axs[5].plot(np.mean(Ryz,axis=(0,1)),z_uvp,'k',label='Res')

axs[0].plot(np.mean(-data.data[:,:,:,16],axis=(0,1)),z_uvp,'r',label='SGS')
axs[1].plot(np.mean(-data.data[:,:,:,17],axis=(0,1)),z_uvp,'r',label='SGS')
axs[2].plot(np.mean(-data.data[:,:,:,18],axis=(0,1)),z_uvp,'r',label='SGS')
axs[3].plot(np.mean(-data.data[:,:,:,19],axis=(0,1)),z_uvp,'r',label='SGS')
axs[4].plot(np.mean(-wnode2uvpnode(data.data[:,:,:,20]),axis=(0,1)),z_uvp,'r',label='SGS')
axs[5].plot(np.mean(-wnode2uvpnode(data.data[:,:,:,21]),axis=(0,1)),z_uvp,'r',label='SGS')

for i in range(len(axs)):
    axs[i].set_ylim(0,0.2)
    axs[i].axvline(0,c='k',ls='--')

plt.show()

#%%Histogram of Anisotropy for the surface level

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)

axs.hist(yB[:,:,0].flatten(),bins=100)

plt.show()

























