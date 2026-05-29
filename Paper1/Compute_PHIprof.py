#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:13:42 2025

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
    uscale = 0.4
    LAD = [0.15655190, 0.20633190, 0.24492203, 0.28024144, 0.33267326, 0.33145316, 0.32065714, 0.28729650, 0.25240169, 0.17358901, 0.11740349, 0.064294815, \
    0.041340224, 0.023756023, 0.013134912, 0.013107183]
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
Ntwr = 100
Nz_SLayer = 200

#%%Giulia data functions

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

#%%Giulia data PHI

sfc = np.load(path_to_data+'../input_txt_files/'+cases[case]+'/sfc.npy')

TwrAvgProf_f = np.zeros((Nz_SLayer),order='F')
TwrAvgProf_p = np.zeros((Nz_SLayer),order='F')

coord_f = np.argwhere(sfc==1.6)
coord_p = np.argwhere(sfc==0)
idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)
sel_coord_f = coord_f[idx_f]
sel_coord_p = coord_p[idx_p]
del coord_f,coord_p,idx_f,idx_p

height = math.ceil(canopyH/dz)

disp_f = compute_d_twr_G(data, sel_coord_f, height, dz, zi, uscale, LAD)
disp_p = np.zeros((Ntwr),order='F')
dudz_uvp = wnode2uvpnode(data.data[:,:,:,12])
dvdz_uvp = wnode2uvpnode(data.data[:,:,:,15])

phi_m_2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
z2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
z0hi = np.zeros((sel_coord_f.shape[0]),'d',order='F')
ustar = np.zeros((sel_coord_f.shape[0]),'d',order='F')

# T_13 = wnode2uvpnode(((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23])))
# T_23 = wnode2uvpnode(((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24])))
# cov_turb = -np.sqrt((T_13)**2 + (T_23)**2)
# ustar2D = np.sqrt(-cov_turb[:, :, :])  # slice at hc_n height
# ustar2D = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height

for k in range(sel_coord_f.shape[0]):
    loc = sel_coord_f[k]
    
    # inf_pt = int(np.ceil(disp_f[k]/(dz*zi)))
    
    # ustar[k] = ustar2D[loc[0],loc[1]]

    z = z_uvp
    z_d = (z - ((disp_f[k])/zi))
    
    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar_G(Nz_SLayer,z_d,data.data[loc[0],loc[1],:,0],\
                                                            data.data[loc[0],loc[1],:,1],True)
    
    phi_m_2D[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],:Nz_SLayer,0],data.data[loc[0],loc[1],:Nz_SLayer,1],\
                              dudz_uvp[loc[0],loc[1],:Nz_SLayer],dvdz_uvp[loc[0],loc[1],:Nz_SLayer],ustar[k])
        
    # print(f'Done with coord: {k}')

TwrAvgProf_f = np.mean(phi_m_2D,axis=(0))

phi_m_2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
z2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
z0hi = np.zeros((sel_coord_f.shape[0]),'d',order='F')
ustar = np.zeros((sel_coord_f.shape[0]),'d',order='F')

for k in range(sel_coord_p.shape[0]):
    loc = sel_coord_p[k]
    
    # inf_pt = int(np.ceil(disp_p[k]/(dz*zi)))
    
    # ustar[k] = ustar2D[loc[0],loc[1]]

    z = z_uvp
    z_d = (z - ((disp_p[k])/zi))
    
    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar_G(Nz_SLayer,z_d,data.data[loc[0],loc[1],:,0],\
                                                            data.data[loc[0],loc[1],:,1],True)
    
    phi_m_2D[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],:Nz_SLayer,0],data.data[loc[0],loc[1],:Nz_SLayer,1],\
                              dudz_uvp[loc[0],loc[1],:Nz_SLayer],dvdz_uvp[loc[0],loc[1],:Nz_SLayer],ustar[k])
        
    # print(f'Done with coord: {k}')

TwrAvgProf_p = np.mean(phi_m_2D,axis=(0))

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_f,z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[0],label='Forested')
axs.plot(TwrAvgProf_p,z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Patches')

axs.axvline(1,c='k',ls='--')

axs.set_xlabel(r"$\phi_M$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
axs.set_ylim(0,10)
axs.axhline(1,ls='--',c='k')

plt.show() 

#%%Save profiles

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'PHI_' + cases[case] + '_f.npy',TwrAvgProf_f)
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'PHI_' + cases[case] + '_p.npy',TwrAvgProf_p)

#%% Ben data profiles

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord_f = find_coordinates(intf,Ntwr,'flat')
# coord_p = find_coordinates(intf,Ntwr,'max')
# coord_v = find_coordinates(intf,Ntwr,'min')

z_profile = np.arange(nz) * dz * zi
zeds = np.ones((nx, ny, 1)) * z_profile
dist = copy.deepcopy(zeds)
del zeds,z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
mask4D = np.expand_dims(mask, axis=-1)
data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
del mask,mask4D

# PHI_p = np.zeros((Ntwr, Nz_SLayer), dtype='float64', order='F')
# PHI_v = np.zeros((Ntwr, Nz_SLayer), dtype='float64', order='F')
PHI_f = np.zeros((Ntwr, Nz_SLayer), dtype='float64', order='F')

# Pre-extract needed variable indices
idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

height = math.ceil(canopyH/dz)
    
# d_dim_p = compute_d_twr(data,coord_p,dist,height+5,dz,zi,uscale,LAD)
# d_dim_v = compute_d_twr(data,coord_v,dist,height+5,dz,zi,uscale,LAD)
d_dim_f = compute_d_twr(data,coord_f,dist,height+5,dz,zi,uscale,LAD)

# z0hi = np.zeros((Ntwr),'d',order='F')
# ustar = np.zeros((Ntwr),'d',order='F')

# for k, (ix, iy) in enumerate(coord_p):
#     # Find first index where dist > 0
#     z_start = np.argmax(dist[ix, iy, :] > 0) - 5
#     inf_pt = int(np.ceil(d_dim_p[k]/(dz*zi)))
    
#     # Extract Ruw and Rvw at z_start + 16
#     # z_idx_ustar = z_start + 16
#     # z_idx_ustar = z_start + inf_pt
#     # Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
#     #         data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
#     #         data.data[ix, iy, z_idx_ustar, idx_txz])

#     # Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
#     #         data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
#     #         data.data[ix, iy, z_idx_ustar, idx_tyz])
    
#     # ustar[k] = (Ruw**2 + Rvw**2)**0.25
    
#     loc = coord_p[k]

#     z = z_uvp
#     z_d = (z - ((d_dim_p[k])/zi))
    
#     [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
#                                                             data.data[loc[0],loc[1],:,1],True)
    
#     PHI_p[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
#                               data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])

# z0hi = np.zeros((Ntwr),'d',order='F')
# ustar = np.zeros((Ntwr),'d',order='F')
    
# for k, (ix, iy) in enumerate(coord_v):
#     # Find first index where dist > 0
#     z_start = np.argmax(dist[ix, iy, :] > 0) - 5
#     inf_pt = int(np.ceil(d_dim_v[k]/(dz*zi)))
    
#     # Extract Ruw and Rvw at z_start + 16
#     # z_idx_ustar = z_start + 16
#     # z_idx_ustar = z_start + inf_pt
#     # Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
#     #         data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
#     #         data.data[ix, iy, z_idx_ustar, idx_txz])

#     # Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
#     #         data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
#     #         data.data[ix, iy, z_idx_ustar, idx_tyz])
    
#     # ustar[k] = (Ruw**2 + Rvw**2)**0.25
    
#     loc = coord_v[k]

#     z = z_uvp
#     z_d = (z - ((d_dim_v[k])/zi))
    
#     [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
#                                                             data.data[loc[0],loc[1],:,1],True)
    
#     PHI_v[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
#                               data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])

z0hi = np.zeros((Ntwr),'d',order='F')
ustar = np.zeros((Ntwr),'d',order='F')
    
for k, (ix, iy) in enumerate(coord_f):
    # Find first index where dist > 0
    z_start = np.argmax(dist[ix, iy, :] > 0) - 5
    
    # Extract Ruw and Rvw at z_start + 16
    # z_idx_ustar = z_start + 16
    # Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
    #         data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
    #         data.data[ix, iy, z_idx_ustar, idx_txz])

    # Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
    #         data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
    #         data.data[ix, iy, z_idx_ustar, idx_tyz])

    # ustar[k] = (Ruw**2 + Rvw**2)**0.25
    
    loc = coord_f[k]
    
    z = z_uvp
    z_d = (z - ((d_dim_f[k])/zi))
    
    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                        data.data[loc[0],loc[1],:,1],True)
    
    PHI_f[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
                          data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])



fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(np.mean(PHI_p,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[0],label='Peak')
# axs.plot(np.mean(PHI_v,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Valley')
axs.plot(np.mean(PHI_f,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Flat')

axs.axvline(1,c='k',ls='--')

axs.set_xlabel(r"$\phi_M$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
axs.set_ylim(0,10)
axs.axhline(1,ls='--',c='k')

plt.show() 


#%%Save profiles

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'PHI_' + cases[case] + '_p.npy',np.mean(PHI_p,axis=(0)))
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'PHI_' + cases[case] + '_v.npy',np.mean(PHI_v,axis=(0)))
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'PHI_' + cases[case] + '.npy',np.mean(PHI_f,axis=(0)))


#%%Compute the velocity gradient for Giometto's data
kappa = 0.4


def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar(Nz_SLayer, z_d, u, v, twr=False):    
    from scipy.optimize import curve_fit
    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]

    # Levels to use for logarithmic fit
    fit_levels = [70, 80, 90]#, 110]

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

def phi_m_loc(Nz_SLayer, z_d, u, v, avgdUdz, avgdVdz, ustar):
    kappa = 0.4

    U_mag = np.sqrt(u**2 + v**2)                   # Magnitude of velocity
    mean_dUdz = (u * avgdUdz + v * avgdVdz) / U_mag  # Directional mean shear
    phi_m_1d = (kappa * z_d[:Nz_SLayer] / ustar) * mean_dUdz

    return phi_m_1d

R13_xy = data['uw_xy'].squeeze(); R13_tw = data['uw_tw'].squeeze()
R23_xy = data['vw_xy'].squeeze(); R23_tw = data['vw_tw'].squeeze()

R13d = data['uw_xy'].squeeze() + data['uwd_xy'].squeeze()
R23d = data['vw_xy'].squeeze() + data['vwd_xy'].squeeze()

ustar_xy = (((R13_xy + data['txz_xy'].squeeze())**2 + (R23_xy + data['tyz_xy'].squeeze())**2)**(0.25))[8:]
ustar_tw = (((R13_tw + data['txz_tw'].squeeze())**2 + (R23_tw + data['tyz_tw'].squeeze())**2)**(0.25))[8:]
ustar_d_xy = (((R13d + data['txz_xy'].squeeze())**2 + (R23d + data['tyz_xy'].squeeze())**2)**(0.25))[8:]

u_xy = (data['u_xy'].squeeze())[8:]; u_tw = (data['u_tw'].squeeze())[8:]
v_xy = (data['v_xy'].squeeze())[8:]; v_tw = (data['v_tw'].squeeze())[8:]
zd = (2/3)*canopyH
z_d = z_uvp[:-8] - zd
z0hi,ustar_v2,U_mean,U_data,z_data,u_fit = compute_ustar(nz-8, z_d, u_xy, v_xy)
phi_xy = phi_m_loc(nz-8, z_d, u_xy, v_xy, (data['dudz_xy'].squeeze())[8:], (data['dvdz_xy'].squeeze())[8:], ustar_v2)
z0hi,ustar_v2,U_mean,U_data,z_data,u_fit = compute_ustar(nz-8, z_d, u_tw, v_tw)
phi_tw = phi_m_loc(nz-8, z_d, u_tw, v_tw, (data['dudz_tw'].squeeze())[8:], (data['dvdz_tw'].squeeze())[8:], ustar_v2)

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(6,3))

axs.plot(phi_xy[:],z_uvp[:-8]/canopyH,c='k',ls='-')
axs.plot(phi_tw[:],z_uvp[:-8]/canopyH,c='k',ls='--')
axs.axvline(1,c='k',ls='--')

axs.set_xlabel(r"$\phi_M$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
axs.set_ylim(0,10)
axs.axhline(1,ls='--',c='k')

plt.show() 


#%%Save profiles

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'PHI_' + cases[case] + '_xy.npy',phi_xy)
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'PHI_' + cases[case] + '_tw.npy',phi_tw)


#%%Computing ustar differently

R13_xy = data['uw_xy'].squeeze(); R13_tw = data['uw_tw'].squeeze()
R23_xy = data['vw_xy'].squeeze(); R23_tw = data['uw_tw'].squeeze()

R13d = data['uw_xy'].squeeze() + data['uwd_xy'].squeeze()
R23d = data['vw_xy'].squeeze() + data['vwd_xy'].squeeze()

ustar_xy = ((R13_xy + data['txz_xy'].squeeze())**2 + (R23_xy + data['tyz_xy'].squeeze())**2)**(0.25)
ustar_tw = ((R13_tw + data['txz_tw'].squeeze())**2 + (R23_tw + data['tyz_tw'].squeeze())**2)**(0.25)
ustar_d_xy = ((R13d + data['txz_xy'].squeeze())**2 + (R23d + data['tyz_xy'].squeeze())**2)**(0.25)






























