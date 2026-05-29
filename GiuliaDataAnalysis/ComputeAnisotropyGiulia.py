# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 08:46:38 2025

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
lz = 1
dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.313
# Tscale = 290

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
    """
    Compute barycentric invariants and smallest eigenvalue lambda3
    from Reynolds stress tensor components stored as xarray.DataArrays.
    """

    import numpy as np
    import xarray as xr

    # Extract dimensions and coordinates from one of the input arrays
    dims = R11.dims
    coords = R11.coords

    # Convert DataArrays to NumPy arrays
    R11v = R11.values
    R22v = R22.values
    R33v = R33.values
    R12v = R12.values
    R13v = R13.values
    R23v = R23.values

    # Calculate TKE
    e = R11v + R22v + R33v
    Ntot = Nx * Ny * Nz

    # Flatten arrays for batch processing
    e_flat = e.ravel(order='F')
    R11_flat = R11v.ravel(order='F')
    R22_flat = R22v.ravel(order='F')
    R33_flat = R33v.ravel(order='F')
    R12_flat = R12v.ravel(order='F')
    R13_flat = R13v.ravel(order='F')
    R23_flat = R23v.ravel(order='F')

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

    # Eigenvalues (symmetric case)
    eigvals_all = np.linalg.eigvalsh(B_all)
    eigvals_sorted = np.sort(eigvals_all, axis=1)[:, ::-1]

    lambda3_flat = eigvals_sorted[:, 2]
    C1c = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    C2c = 2 * (eigvals_sorted[:, 1] - eigvals_sorted[:, 2])
    C3c = 3 * eigvals_sorted[:, 2] + 1

    xB_flat = C1c + 0.5 * C3c
    yB_flat = C3c * (np.sqrt(3) / 2)

    # Reshape back to (Nx, Ny, Nz)
    xB = xr.DataArray(xB_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)
    yB = xr.DataArray(yB_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)
    lambda3 = xr.DataArray(lambda3_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)

    return xB, yB, lambda3

def Anisotropy1D(R11, R22, R33, R12, R13, R23):
   
    import numpy as np

    # TKE
    e = R11 + R22 + R33
    N = len(R11)

    # Build Reynolds stress tensor (N, 3, 3)
    R_all = np.zeros((N, 3, 3))
    R_all[:, 0, 0] = R11
    R_all[:, 1, 1] = R22
    R_all[:, 2, 2] = R33
    R_all[:, 0, 1] = R_all[:, 1, 0] = R12
    R_all[:, 0, 2] = R_all[:, 2, 0] = R13
    R_all[:, 1, 2] = R_all[:, 2, 1] = R23

    # Avoid division by zero
    e_safe = np.where(e == 0.0, 1e-12, e)

    # Identity matrix
    Id = np.eye(3)

    # Anisotropy tensor
    B_all = R_all / e_safe[:, None, None] - (1.0 / 3.0) * Id

    # Eigenvalues (symmetric case)
    eigvals_all = np.linalg.eigvalsh(B_all)
    eigvals_sorted = np.sort(eigvals_all, axis=1)[:, ::-1]

    lambda3 = eigvals_sorted[:, 2]
    C1c = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    C2c = 2 * (eigvals_sorted[:, 1] - eigvals_sorted[:, 2])
    C3c = 3 * eigvals_sorted[:, 2] + 1

    xB = C1c + 0.5 * C3c
    yB = C3c * (np.sqrt(3) / 2)

    return xB, yB, lambda3

#%%Set path to data and load the data

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','Hom_Amazon_9mps','Empty_9mps']
# case = 0

for case in range(len(cases)):
    
    data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc')
    terms_ptb = xr.open_dataarray(path_to_data + cases[case] + '/terms_ptb.nc')

# %Compute the Reynolds stresses

# Rxx = data.data[:,:,:,4] - data.data[:,:,:,0]*data.data[:,:,:,0] - data.data[:,:,:,19]
# Ryy = data.data[:,:,:,5] - data.data[:,:,:,1]*data.data[:,:,:,1] - data.data[:,:,:,20]
# Rzz = wnode2uvpnode(data.data[:,:,:,6] - data.data[:,:,:,2]*data.data[:,:,:,2]) - data.data[:,:,:,21]
# Rxy = data.data[:,:,:,7] - data.data[:,:,:,0]*data.data[:,:,:,1] - data.data[:,:,:,22]
# Rxz = wnode2uvpnode(data.data[:,:,:,8]) - data.data[:,:,:,0]*wnode2uvpnode(data.data[:,:,:,2]) - wnode2uvpnode(data.data[:,:,:,23])
# Ryz = wnode2uvpnode(data.data[:,:,:,9]) - data.data[:,:,:,1]*wnode2uvpnode(data.data[:,:,:,2]) - wnode2uvpnode(data.data[:,:,:,24])

# Rxxd = np.zeros((nz),'d',order='F')
# Ryyd = np.zeros((nz),'d',order='F')
# Rzzd = np.zeros((nz),'d',order='F')
# Rxyd = np.zeros((nz),'d',order='F')
# Rxzd = np.zeros((nz),'d',order='F')
# Ryzd = np.zeros((nz),'d',order='F')

# u = data.data[:,:,:,0]
# v = data.data[:,:,:,1]
# w = wnode2uvpnode(data.data[:,:,:,2])

# for k in range(0,nz):
#     Rxxd[k] = np.mean(data.data[:,:,k,4],axis=(0,1)) - np.mean(u[:,:,k],axis=(0,1))*np.mean(u[:,:,k],axis=(0,1))
#     Ryyd[k] = np.mean(data.data[:,:,k,5],axis=(0,1)) - np.mean(v[:,:,k],axis=(0,1))*np.mean(v[:,:,k],axis=(0,1))
#     Rzzd[k] = np.mean(data.data[:,:,k,6],axis=(0,1)) - np.mean(w[:,:,k],axis=(0,1))*np.mean(w[:,:,k],axis=(0,1))
#     Rxyd[k] = np.mean(data.data[:,:,k,7],axis=(0,1)) - np.mean(u[:,:,k],axis=(0,1))*np.mean(v[:,:,k],axis=(0,1))
#     Rxzd[k] = np.mean(data.data[:,:,k,8],axis=(0,1)) - np.mean(u[:,:,k],axis=(0,1))*np.mean(w[:,:,k],axis=(0,1))
#     Ryzd[k] = np.mean(data.data[:,:,k,9],axis=(0,1)) - np.mean(v[:,:,k],axis=(0,1))*np.mean(w[:,:,k],axis=(0,1))

# #%%

# [xB,yB,l3] = Anisotropy1D(np.mean(Rxx,axis=(0,1)), np.mean(Ryy,axis=(0,1)), np.mean(Rzz,axis=(0,1)), np.mean(Rxy,axis=(0,1)), np.mean(Rxz,axis=(0,1)), np.mean(Ryz,axis=(0,1)))
# [xB_d,yB_d,l3_d] = Anisotropy1D(np.mean(Rxx,axis=(0,1)) + Rxxd, np.mean(Ryy,axis=(0,1)) + Ryyd, np.mean(Rzz,axis=(0,1)) + Rzzd, \
#                                 np.mean(Rxy,axis=(0,1)) + Rxyd, np.mean(Rxz,axis=(0,1)) + Rxzd, np.mean(Ryz,axis=(0,1)) + Ryzd)

# # [xB,yB,l3] = Anisotropy1D(np.mean(terms_ptb[:,:,:,0],axis=(0,1)),np.mean(terms_ptb[:,:,:,3],axis=(0,1)),np.mean(terms_ptb[:,:,:,5],axis=(0,1)),\
# #                           np.mean(terms_ptb[:,:,:,1],axis=(0,1)),np.mean(terms_ptb[:,:,:,2],axis=(0,1)),np.mean(terms_ptb[:,:,:,4],axis=(0,1)))

# #%%Plot anisotropy profile 

# fig,axs = plt.subplots(1,1,tight_layout=True)
# axs.plot(yB,z_uvp/(39/zi),c='k')
# axs.plot(yB_d,z_uvp/(39/zi),c='r')
# axs.set_ylim(0,10)
# axs.set_xlim(0,0.5)
# axs.axvline(0.38,c='k',ls=':')

# plt.show()

#%Compute anisotropy

# xB,yB,lambda3 = Anisotropy(nx, ny, nz, Rxx, Ryy, Rzz, Rxy, Rxz, Ryz)
# xB,yB,lambda3 = Anisotropy(nx, ny, nz, Rxx-data.data[:,:,:,19], Ryy-data.data[:,:,:,20], Rzz-data.data[:,:,:,21],\
#                            Rxy-data.data[:,:,:,22], Rxz-wnode2uvpnode(data.data[:,:,:,23]), Ryz-wnode2uvpnode(data.data[:,:,:,24]))

    anisotropy = xr.DataArray(np.ones(shape = (nx,ny,nz,3),order='F'),\
                           dims=('x','y','z','variable'), coords = {'variable':['xB','yB','lambda3']})

# anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(nx, ny, nz, terms_ptb[:,:,:,0] - data[:,:,:,19],\
#                                                                           terms_ptb[:,:,:,3] - data[:,:,:,20],\
#                                                                               terms_ptb[:,:,:,5] - data[:,:,:,21],\
#                                                                                   terms_ptb[:,:,:,1] - data[:,:,:,22],\
#                                                                                       terms_ptb[:,:,:,2] - wnode2uvpnode(data[:,:,:,23]),\
#                                                                                           terms_ptb[:,:,:,4] - wnode2uvpnode(data[:,:,:,24]))

    anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(nx, ny, nz, terms_ptb[:,:,:,0],\
                                                                              terms_ptb[:,:,:,3],\
                                                                                  terms_ptb[:,:,:,5],\
                                                                                      terms_ptb[:,:,:,1],\
                                                                                          terms_ptb[:,:,:,2],\
                                                                                              terms_ptb[:,:,:,4])
        
# anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(nx, ny, nz, terms_ptb[:,:,:,0],\
#                                                                           terms_ptb[:,:,:,3],\
#                                                                               terms_ptb[:,:,:,5],\
#                                                                                   terms_ptb[:,:,:,1],\
#                                                                                       terms_ptb[:,:,:,2],\
#                                                                                           terms_ptb[:,:,:,4])

# anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(nx, ny, nz,-data[:,:,:,19],\
#                                                                          -data[:,:,:,20],\
#                                                                             -data[:,:,:,21],\
#                                                                                  -data[:,:,:,22],\
#                                                                                      -wnode2uvpnode(data[:,:,:,23]),\
#                                                                                          -wnode2uvpnode(data[:,:,:,24]))

    anisotropy.to_netcdf(path_to_data+cases[case] + '/anisotropy_NoSGS.nc')
    print(f'Done with case {case}')

# anisotropy = xr.open_dataarray(path+cases[case] + '\\anisotropy.nc')

#%%

xB = anisotropy[:,:,:,0]
yB = anisotropy[:,:,:,1]

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

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
xslice = 100
yslice = 100
zslice = 10

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,6))

axs[0,0].pcolormesh(x,y,xB[:,:,zslice].T,cmap=cmap,vmin = 0,vmax = 1)
axs[0,1].pcolormesh(x,z_uvp,xB[:,yslice,:].T,cmap=cmap,vmin = 0,vmax = 1)
p1 = axs[0,2].pcolormesh(y,z_uvp,xB[xslice,:,:].T,cmap=cmap,vmin = 0,vmax = 1)

axs[1,0].pcolormesh(x,y,yB[:,:,zslice].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)
axs[1,1].pcolormesh(x,z_uvp,yB[:,yslice,:].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)
p2 = axs[1,2].pcolormesh(y,z_uvp,yB[xslice,:,:].T,cmap=cmap,vmin = 0,vmax = np.sqrt(3)/2)

# 1. x-y plane @ zslice
neg_xB = np.where(xB[:, :, zslice] < 0)
axs[0, 0].plot(x[neg_xB[0]], y[neg_xB[1]], 'r.', markersize=10)
neg_yB = np.where(yB[:, :, zslice] < 0)
axs[1, 0].plot(x[neg_yB[0]], y[neg_yB[1]], 'r.', markersize=10)

# 2. x-z plane @ yslice
neg_xB = np.where(xB[:, yslice, :] < 0)
axs[0, 1].plot(x[neg_xB[0]], z_uvp[neg_xB[1]], 'r.', markersize=10)
neg_yB = np.where(yB[:, yslice, :] < 0)
axs[1, 1].plot(x[neg_yB[0]], z_uvp[neg_yB[1]], 'r.', markersize=10)

# 3. y-z plane @ xslice
neg_xB = np.where(xB[xslice, :, :] < 0)
axs[0, 2].plot(y[neg_xB[0]], z_uvp[neg_xB[1]], 'r.', markersize=10)
neg_yB = np.where(yB[xslice, :, :] < 0)
axs[1, 2].plot(y[neg_yB[0]], z_uvp[neg_yB[1]], 'r.', markersize=10)

axs[1,1].contour(x,z_uvp,yB[:,yslice,:].T,levels=[0.38],colors='black')
axs[1,2].contour(y,z_uvp,yB[xslice,:,:].T,levels=[0.38],colors='black')

cbar1 = plt.colorbar(p1,label='xB')
cbar2 = plt.colorbar(p2,label='yB')

axs[0,0].set_title(f"zslice = {zslice*dz*zi :.02f}m",fontsize=15)
axs[0,1].set_title(f"yslice = {yslice*dy*zi :.02f}m",fontsize=15)
axs[0,2].set_title(f"xslice = {xslice*dx*zi :.02f}m",fontsize=15)

fig.suptitle(f'{cases[case]}',fontsize=15)

plt.show()

#%%Reynolds stress profiles

fig,axs = plt.subplots(1,6,tight_layout=True,figsize=(12,6))

axs[0].plot(np.mean(Rxx,axis=(0,1)),z_uvp,'k',label='Res')
axs[1].plot(np.mean(Ryy,axis=(0,1)),z_uvp,'k',label='Res')
axs[2].plot(np.mean(Rzz,axis=(0,1)),z_uvp,'k',label='Res')
axs[3].plot(np.mean(Rxy,axis=(0,1)),z_uvp,'k',label='Res')
axs[4].plot(np.mean(Rxz,axis=(0,1)),z_uvp,'k',label='Res')
axs[5].plot(np.mean(Ryz,axis=(0,1)),z_uvp,'k',label='Res')

axs[0].plot(np.mean(-data.data[:,:,:,19],axis=(0,1)),z_uvp,'r',label='SGS')
axs[1].plot(np.mean(-data.data[:,:,:,20],axis=(0,1)),z_uvp,'r',label='SGS')
axs[2].plot(np.mean(-data.data[:,:,:,21],axis=(0,1)),z_uvp,'r',label='SGS')
axs[3].plot(np.mean(-data.data[:,:,:,22],axis=(0,1)),z_uvp,'r',label='SGS')
axs[4].plot(np.mean(-wnode2uvpnode(data.data[:,:,:,23]),axis=(0,1)),z_uvp,'r',label='SGS')
axs[5].plot(np.mean(-wnode2uvpnode(data.data[:,:,:,24]),axis=(0,1)),z_uvp,'r',label='SGS')

for i in range(len(axs)):
    axs[i].set_ylim(0,0.2)
    axs[i].axvline(0,c='k',ls='--')

plt.show()

#%%Histogram of Anisotropy for the surface level

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)

axs.hist(yB[:,:,0].flatten(),bins=100)

plt.show()
