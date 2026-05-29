#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:00:31 2026

@author: u1450851
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

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf

#%%#%%Paths to data

directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
cases = ['Flat','Sinusoidal','ATTO']
case = 2
# pathFig = directory + case + '\\Figures\\'
# pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/' 

path_sin = directory + cases[case] + '/'

#%%#%%Loading Topo data

zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py

Nx = 256
Ny = 256
Nz = 384
Lx = 2.880
Ly = 2.880
Lz = 0.960
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz
mpiProc = 32
nzTot = Nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz
z_short = z_uvp[:-5]

# Assuming you have a function build_phi to load phi data from a file
phi = build_phi(path_sin + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf, iintf = build_intf(phi, dz)

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
Nz_SLayer = 300 #int(Nz/2)

z_profile = np.arange(Nz) * dz * zi
zeds = np.ones((Nx, Ny, 1)) * z_profile
    
dist = copy.deepcopy(zeds)

dist -= intf[:, :, np.newaxis] * zi

#%%Save dist

distDA = xr.DataArray(np.zeros(shape = (Nx,Ny,Nz-5,1),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['dist']})

distDA[:,:,:,0] = dist[:,:,5:]
# distDA.to_netcdf(path_sin + 'dist.nc')

#%%Load netCDF data

NumVariables = 64
data = xr.open_dataarray(path_sin+'dataTKE.nc')
terms_ptb = xr.open_dataarray(path_sin+'terms_ptb.nc')

#%%Anisotropy function

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

#%%Compute anisotropy

anisotropy = xr.DataArray(np.ones(shape = (Nx,Ny,Nz-5,3),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['xB','yB','lambda3']})

# anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(Nx, Ny, Nz-5, terms_ptb[:,:,:,0] - data[:,:,:,19],\
#                                                                           terms_ptb[:,:,:,3] - data[:,:,:,20],\
#                                                                               terms_ptb[:,:,:,5] - data[:,:,:,21],\
#                                                                                   np.zeros_like(terms_ptb[:,:,:,0]),\
#                                                                                       np.zeros_like(terms_ptb[:,:,:,0]),\
#                                                                                           np.zeros_like(terms_ptb[:,:,:,0]))
    
anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(Nx, Ny, Nz-5, terms_ptb[:,:,:,0],\
                                                                          terms_ptb[:,:,:,3],\
                                                                              terms_ptb[:,:,:,5],\
                                                                                  terms_ptb[:,:,:,1],\
                                                                                      terms_ptb[:,:,:,2],\
                                                                                          terms_ptb[:,:,:,4])

# anisotropy.to_netcdf(path_sin + 'anisotropy_NoSGS.nc')

# anisotropy = xr.open_dataarray(path_sin + 'anisotropy.nc')


#%%

# yb = anisotropy[:,:,:,1].data

# #%%

# plt.plot(np.mean(data[:,:,:,19],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(data[:,:,:,20],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(data[:,:,:,21],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(data[:,:,:,22],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(data[:,:,:,23],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(data[:,:,:,24],axis=(0,1)),z_uvp[5:],)

# #%%

# plt.figure()
# plt.plot(np.mean(terms_ptb[:,:,:,0],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(terms_ptb[:,:,:,1],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(terms_ptb[:,:,:,2],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(terms_ptb[:,:,:,3],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(terms_ptb[:,:,:,4],axis=(0,1)),z_uvp[5:],)
# plt.plot(np.mean(terms_ptb[:,:,:,5],axis=(0,1)),z_uvp[5:],)



















































