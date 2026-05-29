#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 10:31:25 2025

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

case = 7

if case < 6:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc')
elif case >= 6 and case < 9:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc')
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
    z_w = np.arange(0,nz)*dz
    z_uvp = np.arange(0,nz)*dz + dz/2
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

#%%

ls = ['-','--']
colors = ['k','b','g','r','c','m','y','lime','violet']
Ntwr = 10
Nz_SLayer = 200

#%%

def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar(Nz_SLayer, z_d, dist, u, v, twr=False):    
    kappa = 0.4
    z_start = np.argmax(dist > 0) - 5

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[z_start:z_start+Nz_SLayer]

    # Levels to use for logarithmic fit
    fit_levels = [100, 110, 120, 130]

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]

    # Fit the log profile
    coefs, _ = curve_fit(log_fit, z_data, U_data, maxfev=10000)
    a, b = coefs

    # Fitted velocity profile across the entire surface layer
    u_fit = log_fit(z_d[:Nz_SLayer], a, b)

    # Compute z0hi and ustar
    z0hi = 1 / b
    ustar = U_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit

#%%Compute the shear

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord = find_coordinates(intf,Ntwr,'flat')
# coord = find_coordinates(intf,Ntwr,'max')
# coord = find_coordinates(intf,Ntwr,'min')

z_profile = np.arange(nz) * dz * zi
zeds = np.ones((nx, ny, 1)) * z_profile
dist = copy.deepcopy(zeds)
del zeds,z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
mask4D = np.expand_dims(mask, axis=-1)
data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
del mask,mask4D

shear = np.zeros((Ntwr, Nz_SLayer), dtype='float64', order='F')

# Pre-extract needed variable indices
idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

height = math.ceil(canopyH/dz)
    
d_dim = compute_d_twr(data,coord,dist,height+5,dz,zi,uscale,LAD)

z0hi = np.zeros((Ntwr),'d',order='F')
ustar = np.zeros((Ntwr),'d',order='F')
    
for k, (ix, iy) in enumerate(coord):
    # Find first index where dist > 0
    z_start = np.argmax(dist[ix, iy, :] > 0) - 5

    # Slice velocity components over Nz_SLayer starting from z_start
    z_end = z_start + Nz_SLayer
    uw = data.data[ix, iy, z_start:z_end, idx_uw] - data.data[ix, iy, z_start:z_end, idx_u]*data.data[ix, iy, z_start:z_end, idx_w] - \
        data.data[ix, iy, z_start:z_end, idx_txz]
    vw = data.data[ix, iy, z_start:z_end, idx_vw] - data.data[ix, iy, z_start:z_end, idx_v]*data.data[ix, iy, z_start:z_end, idx_w] - \
        data.data[ix, iy, z_start:z_end, idx_tyz]

    # Compute magnitude and normalize
    shear = np.sqrt(uw**2 + vw**2)
    shear[k, :] = shear
    
for k, (ix, iy) in enumerate(coord):
    loc = coord[k]

    z = z_uvp
    z_d = (z - ((d_dim[k])/zi))
    
    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                            data.data[loc[0],loc[1],:,1],True)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.mean(shear,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Flat')

axs.set_xlabel(r"$\sqrt{\overline{u'w'}^2 + \overline{v'w'}^2}/u_*^2$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
axs.set_ylim(0,10)
axs.axhline(1,ls='--',c='k')

plt.show() 

#%%Compute a linear fit for each tower

z_over_h = z_uvp[:Nz_SLayer]/canopyH
shear_fit = np.zeros((Ntwr,Nz_SLayer),'d',order='F')

for i in range(0,Ntwr):
    mask = (z_over_h>5) & (z_over_h<10)
    z_int = z_over_h[mask]
    shear_int = shear_f[i,:][mask]
    coef = np.polyfit(z_int,shear_int,1)
    shear_fit[i,:] = np.polyval(coef,z_over_h)
    
#%%

ustar_v2 = np.sqrt(shear_fit[:,0])

#%%
def phi_m_loc(Nx, Ny, Nz_SLayer, z_d, u, v, avgdUdz, avgdVdz, ustar):
    kappa = 0.4

    U_mag = np.sqrt(u**2 + v**2)                   # Magnitude of velocity
    mean_dUdz = (u * avgdUdz + v * avgdVdz) / U_mag  # Directional mean shear
    phi_m_1d = (kappa * z_d[:Nz_SLayer] / ustar) * mean_dUdz

    return phi_m_1d

PHI_f = np.zeros((Ntwr, Nz_SLayer), dtype='float64', order='F')

for k, (ix, iy) in enumerate(coord_f):
    z_start = np.argmax(dist[ix, iy, :] > 0) - 5
    loc = coord_f[k]
    z = z_uvp
    z_d = (z - ((d_dim_f[k])/zi))
    PHI_f[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
                          data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar_v2[k])
        
fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.mean(PHI_f,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Flat')

axs.axvline(1,c='k',ls='--')

axs.set_xlabel(r"$\phi_M$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
axs.set_ylim(0,10)
axs.axhline(1,ls='--',c='k')

plt.show() 

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.scatter(shear_int,z_int)
axs.plot(shear_fit[-1,:],z_uvp[:Nz_SLayer]/canopyH)

plt.show()
#%%Plot each tower

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(0,Ntwr):
    axs.plot(shear_f[i,:],z_uvp[:Nz_SLayer]/canopyH)
    axs.plot(shear_fit[i,:],z_uvp[:Nz_SLayer]/canopyH)
    
plt.show()

#%%Funsctioons to compute ustar from log fit

def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar_G(Nz_SLayer, z_d, u, v, twr=False):    

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]
    kappa = 0.4

    # Levels to use for logarithmic fit
    fit_levels = [50, 60, 70, 90]

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]

    # Fit the log profile
    coefs, _ = curve_fit(log_fit, z_data, U_data, maxfev=10000)
    a, b = coefs

    # Fitted velocity profile across the entire surface layer
    u_fit = log_fit(z_d[:Nz_SLayer], a, b)

    # Compute z0hi and ustar
    z0hi = 1 / b
    ustar = U_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit

def phi_m_loc(Nx, Ny, Nz_SLayer, z_d, u, v, avgdUdz, avgdVdz, ustar):
    kappa = 0.4

    U_mag = np.sqrt(u**2 + v**2)                   # Magnitude of velocity
    mean_dUdz = (u * avgdUdz + v * avgdVdz) / U_mag  # Directional mean shear
    phi_m_1d = (kappa * z_d[:Nz_SLayer] / ustar) * mean_dUdz

    return phi_m_1d

def compute_d_twr_G(data, coord, height, dz, zi, u_scale, LAD):
    
    d_dim = np.zeros(coord.shape[0])
    
    nlevels = len(LAD)

    z = (np.arange(1, height + 1)[:nlevels]) * dz * zi - (dz * zi) / 2
    Z = z  # Final vertical coordinates for integration

    for idx in range(coord.shape[0]):
        i, j = coord[idx]
        
        # Extract u, v and compute |U| above that level
        u = data.data[i, j, :,0]
        v = data.data[i, j, :,1]
        U = np.sqrt(u**2 + v**2)

        # Select the first `nlevels` points for integration (truncate if too short)
        U_slice = U[:nlevels]
        Z_slice = Z[:len(U_slice)]

        VAR = U_slice * u_scale
        Y = (VAR**2) * 0.4 * LAD

        num = trapezoid(Y * Z_slice, Z_slice)
        den = trapezoid(Y, Z_slice)

        d_dim[idx] = num / den if den != 0 else 0.0

    return d_dim

def compute_ustar(Nz_SLayer, z_d, dist, u, v, twr=False):    
    kappa = 0.4
    z_start = np.argmax(dist > 0) - 5

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[z_start:z_start+Nz_SLayer]

    # Levels to use for logarithmic fit
    fit_levels = [100, 110, 120, 130]

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]

    # Fit the log profile
    coefs, _ = curve_fit(log_fit, z_data, U_data, maxfev=10000)
    a, b = coefs

    # Fitted velocity profile across the entire surface layer
    u_fit = log_fit(z_d[:Nz_SLayer], a, b)

    # Compute z0hi and ustar
    z0hi = 1 / b
    ustar = U_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit


#%%

z0hi = np.zeros((Ntwr),'d',order='F')
ustar = np.zeros((Ntwr),'d',order='F')
u_fit = np.zeros((Ntwr,Nz_SLayer),'d',order='F')
U_mean = np.zeros((Ntwr,Nz_SLayer),'d',order='F')
z_d = np.zeros((Ntwr,nz),'d',order='F')
    
for k, (ix, iy) in enumerate(coord_p):
        
    loc = coord_p[k]
    
    z = z_uvp
    z_d[k,:] = (z - ((d_dim_p[k])/zi))
    
    [z0hi[k],ustar[k],U_mean[k,:],U_data,z_data,u_fit[k,:]] = compute_ustar(Nz_SLayer,z_d[k,:],dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                        data.data[loc[0],loc[1],:,1],True)


#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(0,Ntwr):
    axs.plot(U_mean[i,:],z_uvp[:Nz_SLayer])
    axs.plot(u_fit[k,:],z_uvp[:Nz_SLayer])
    # axs.axhline(d_dim_p[i]/zi + z0hi[i])
    # axs.axhline(d_dim_f[i]/zi)
    
# axs.axhline(canopyH,c='k')
axs.set_xlim(0)
axs.set_ylim(0)
plt.show()


















































