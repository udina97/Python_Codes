#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:54:23 2026

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
import scipy.io

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from functions import get_dphidx,get_dphidy,get_dphidz,uvpnode2wnode,wnode2uvpnode,compute_d_twr,find_coordinates,average_over_selected_coords

#%%Load data

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat','simulation_G']

case = 4

if case < 6:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc')
elif case >= 6 and case < 9:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc')
    dist = xr.open_dataarray(path_to_data + cases[case] + '/dist.nc')
else:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/'
    data = scipy.io.loadmat(path_to_data + cases[case] + '/profiles.mat')
    
#%%Simulation Parameters

if case < 6:
    lx = 2*np.pi; ly = 2*np.pi; lz = 1
    nx,ny,nz = np.shape(data[:,:,:,0])
    zi = 1000
    canopyH = 39/zi
    dx = lx/nx; dy = ly/ny; dz = lz/nz
    x = np.arange(0,nx)*dx
    y = np.arange(0,ny)*dy
    z_w = np.arange(0,nz)*dz
    z_uvp = np.arange(0,nz)*dz + dz/2
    LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
    uscale = 0.313
elif case >= 6 and case < 9:
    lx = 2.88; ly = 2.88; lz = 0.96
    mpiProc = 32
    nx,ny,nz = np.shape(data[:,:,:,0])
    nz = nz + 5
    zi = 1000
    canopyH = 39/zi
    dx = lx/nx; dy = ly/ny; dz = lz/nz
    x = np.arange(0,nx)*dx
    y = np.arange(0,ny)*dy
    z_w = np.arange(0,nz-5)*dz
    z_uvp = np.arange(0,nz-5)*dz + dz/2
    LAD = [0.15655190, 0.20633190, 0.24492203, 0.28024144, 0.33267326, 0.33145316, 0.32065714, 0.28729650, 0.25240169, 0.17358901, 0.11740349, 0.064294815, \
    0.041340224, 0.023756023, 0.013134912, 0.013107183]
    uscale = 0.4
else:
    lx = data['lx'][0][0]; ly = data['ly'][0][0]; lz = data['lz'][0][0]
    nx = data['nx'][0][0]; ny = data['ny'][0][0]; nz = data['nz'][0][0]
    dx = data['dx'][0][0]; dy = data['dy'][0][0]; dz = data['dz'][0][0]
    z_uvp = data['z'][0]
    z_w = data['zi'][0]
    uscale = 1.23
    canopyH = 15.3

#%%Plot the IBL

tmp = np.sqrt(data[:,:,:,0].data**2 + data[:,:,:,1].data**2)
tmp2D = np.mean(tmp,axis=(1))
tmp2D = tmp2D - np.mean(tmp2D,axis=(0),keepdims=True)

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,z_uvp,tmp2D.T,cmap='jet')

cbar = plt.colorbar(p)

plt.show()


#%%Plot profiles 

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(0,nx,40):
    axs.plot(tmp2D[i,:],z_uvp,c='k')
    
plt.show()

#%%Compute Reynolds stresses

uw = data[:,:,:,8].data - data[:,:,:,0].data*data[:,:,:,2].data - data[:,:,:,23].data
vw = data[:,:,:,9].data - data[:,:,:,1].data*data[:,:,:,2].data - data[:,:,:,24].data
ustar = (uw**2 + vw**2)**(1/4)

#%%Compute mean profile with terrain following coordinates

dist_mask = np.where(dist[:,:,:,0].data<0,0,1)

def compute_terrain_height_from_mask(topo, z, fluid_value=1):
    """
    Estimate terrain height h(x,y) from a 3D topography/fluid mask.

    Parameters
    ----------
    topo : ndarray, shape (nx, ny, nz)
        Mask indicating whether each point is fluid/above terrain.
    z : ndarray, shape (nz,)
        Vertical coordinate.
    fluid_value : int or bool
        Value in topo corresponding to fluid / above terrain.

    Returns
    -------
    h : ndarray, shape (nx, ny)
        Estimated terrain height at each horizontal location.
    """

    topo = np.asarray(topo)
    z = np.asarray(z)

    fluid = topo == fluid_value

    nx, ny, nz = topo.shape
    h = np.full((nx, ny), np.nan)

    for i in range(nx):
        for j in range(ny):
            fluid_indices = np.where(fluid[i, j, :])[0]

            if len(fluid_indices) > 0:
                k0 = fluid_indices[0]
                h[i, j] = z[k0]

    return h

# h = compute_terrain_height_from_mask(dist_mask, z_w)

z_agl_levels = np.arange(0, 0.8, dz)  # 0 to 495 m AGL every 5 m

def terrain_following_mean_profile(u, topo, z, z_agl_levels, fluid_value=1):
    """
    Compute terrain-following horizontally averaged profile of streamwise velocity.

    Parameters
    ----------
    u : ndarray, shape (nx, ny, nz)
        Streamwise velocity.
    topo : ndarray, shape (nx, ny, nz)
        Mask indicating fluid/terrain points.
    z : ndarray, shape (nz,)
        Absolute vertical coordinate.
    z_agl_levels : ndarray
        Target height-above-ground levels.
    fluid_value : int or bool
        Value in topo corresponding to fluid / above terrain.

    Returns
    -------
    u_mean : ndarray, shape (len(z_agl_levels),)
        Terrain-following mean streamwise velocity profile.
    u_tf : ndarray, shape (nx, ny, len(z_agl_levels))
        Interpolated terrain-following velocity field.
    h : ndarray, shape (nx, ny)
        Terrain height field.
    """

    u = np.asarray(u)
    topo = np.asarray(topo)
    z = np.asarray(z)
    z_agl_levels = np.asarray(z_agl_levels)

    nx, ny, nz = u.shape

    fluid = topo == fluid_value

    h = compute_terrain_height_from_mask(topo, z, fluid_value=fluid_value)

    u_tf = np.full((nx, ny, len(z_agl_levels)), np.nan)

    for i in range(nx):
        for j in range(ny):

            if np.isnan(h[i, j]):
                continue

            u_col = u[i, j, :]
            fluid_col = fluid[i, j, :]

            # Keep only fluid points
            z_fluid = z[fluid_col]
            u_fluid = u_col[fluid_col]

            if len(z_fluid) < 2:
                continue

            # Convert absolute height to height above local terrain
            z_agl_col = z_fluid - h[i, j]

            # Remove NaNs if present
            valid = np.isfinite(z_agl_col) & np.isfinite(u_fluid)

            if np.sum(valid) < 2:
                continue

            z_agl_col = z_agl_col[valid]
            u_fluid = u_fluid[valid]

            # Ensure increasing vertical coordinate for interpolation
            sort_idx = np.argsort(z_agl_col)
            z_agl_col = z_agl_col[sort_idx]
            u_fluid = u_fluid[sort_idx]

            # Interpolate only where target levels are within the valid column range
            u_tf[i, j, :] = np.interp(z_agl_levels, z_agl_col, u_fluid, left=np.nan, right=np.nan)

    # Horizontal average at each terrain-following height
    u_mean = np.nanmean(u_tf, axis=(0, 1))

    return u_mean, u_tf, h


u_mean, u_tf, h = terrain_following_mean_profile(ustar,dist_mask,z_w,z_agl_levels,fluid_value=1)

plt.figure()
plt.plot(u_mean, z_agl_levels)
# plt.plot(np.mean(data[:,:,:,0].data,axis=(0,1)),z_w,c='r')
plt.xlabel("Mean Profile")
plt.ylabel("Height above local terrain")
plt.grid(True)
plt.axhline(canopyH,c='k',ls='--')
plt.ylim(0,z_agl_levels[-1])
plt.show()

#%%Plot pcolor of ustar at canopy top from the terrain following interpolated field

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x,y,u_tf[:,:,1].T,cmap='jet')

plt.show()


#%%Plot profiles 
tmp2D = np.mean(u_tf,axis=(1))
tmp = tmp2D - np.mean(tmp2D,axis=(0),keepdims=True)
fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(0,nx,40):
    axs.plot(tmp[i,:],z_agl_levels,c='k')
    
plt.show()


#%%

# BLENDING HEIGHT FOR GIULIA HETEROGENEOUS FOREST CASES

#%%derivative functions

def get_dphidz_2D(phi, dz):
    import numpy as np
    dphidz = np.zeros(phi.shape)
    dphidz[:,:-1] = (phi[:,1:]-phi[:,:-1]) / (dz)
    dphidz[:,-1] = dphidz[:,-2]
    return dphidz

#%%
U = np.sqrt(data[:,:,:,0].data**2 + data[:,:,:,1].data**2)
u_y = np.mean(data[:,:,:,0].data,axis=(1))*uscale
u_xy = np.mean(u_y,axis=(0),keepdims=True)
ustar_xy = np.mean(ustar,axis=(0,1))*uscale

dudz_y = get_dphidz_2D(u_y, dz*zi)
dudz_xy = get_dphidz_2D(u_xy, dz*zi)

diff = (dudz_y - dudz_xy)*zi/np.max(ustar_xy)

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,z_uvp/canopyH,diff.T,cmap='bwr',vmin=-10,vmax=10)
cbar = plt.colorbar(p)

plt.show()

#%%Computing lower and upper qurtile profile of strewise velocity as per Bou-Zeid 2004

u = data[:,:,:,0].data*uscale
u_y = np.mean(u,axis=(1))
u_xy = np.mean(u_y,axis=(0),keepdims=True)
diff = (u_y - u_xy)/np.max(ustar_xy)

lq = np.zeros((nz))
uq = np.zeros((nz))

for i in range(nz):
    lq[i] = np.quantile(diff[:,i], 0.25)
    uq[i] = np.quantile(diff[:,i], 0.75)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(lq,z_uvp/canopyH,c='k')
axs.plot(uq,z_uvp/canopyH,c='r')

axs.set_ylim(0,20)

plt.show()

































