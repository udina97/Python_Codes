#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:06:42 2025

@author: u1450851
"""
#%%

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

#%%Funsctioons to compute ustar from log fit

def log_fit(z, a, b):
    return a * np.log(b * z)

GIULIA_FIT_LEVELS = [80, 90, 100]
BEN_FIT_LEVELS = {
    'ATTO': [160, 170, 180],
    'Sinusoidal': [160, 170, 180],
    'Flat': [100, 110, 120],
}
URBAN_FIT_LEVELS = [100, 110, 120]

def compute_ustar_G(Nz_SLayer, z_d, u, v, twr=False, fit_levels=None):    

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]
    kappa = 0.4

    if fit_levels is None:
        fit_levels = GIULIA_FIT_LEVELS
    fit_levels = np.asarray(fit_levels, dtype=int)

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

def compute_ustar(Nz_SLayer, z_d, dist, u, v, twr=False, fit_levels=None):    
    kappa = 0.4
    z_start = np.argmax(dist > 0) - 5

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[z_start:z_start+Nz_SLayer]

    if fit_levels is None:
        fit_levels = BEN_FIT_LEVELS['ATTO']
    fit_levels = np.asarray(fit_levels, dtype=int)

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

#%%Create dictionaries where we store ustar,z0,d,U/ustar

ustar = dict()
z0 = dict()
dispH = dict()
UoverU = dict()

#%%Compute ustar,z0,d, and U/ustar for Giulia's data

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps']

for i in range(len(cases)):
    #load the data
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path_to_data + cases[i] + '/Data_Momentum_4TKE.nc')
    
    #Simulation parameters
    lx = 2*np.pi; ly = 2*np.pi; lz = 1
    nx,ny,nz = np.shape(data[:,:,:,0])
    zi = 1000
    canopyH = 39/zi
    dx = lx/nx; dy = ly/ny; dz = lz/nz
    z_w = np.arange(0,nz)*dz
    z_uvp = np.arange(0,nz)*dz + dz/2
    LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
    uscale = 0.313
    Ntwr = 100
    Nz_SLayer = 200
    height = math.ceil(canopyH/dz)
    
    #Load the surface data and select the virtual towers
    sfc = np.load(path_to_data+'../input_txt_files/'+cases[i]+'/sfc.npy')
    coord_f = np.argwhere(sfc==1.6)
    coord_p = np.argwhere(sfc==0)
    idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
    idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)
    sel_coord_f = coord_f[idx_f]
    sel_coord_p = coord_p[idx_p]
    del coord_f,coord_p,idx_f,idx_p
    
    #Compute the displacement height
    dispH[cases[i]+'_f'] = compute_d_twr_G(data, sel_coord_f, height, dz, zi, uscale, LAD)
    dispH[cases[i]+'_p'] = np.zeros((Ntwr),order='F')
    
    #Compute ustar,z0,U/ustar
    z0[cases[i]+'_f'] = np.zeros((Ntwr),'d',order='F')
    z0[cases[i]+'_p'] = np.zeros((Ntwr),'d',order='F')
    ustar[cases[i]+'_f'] = np.zeros((Ntwr), 'd', order='F')
    ustar[cases[i]+'_p'] = np.zeros((Ntwr), 'd', order='F')
    UoverU[cases[i]+'_f'] = np.zeros((Ntwr), 'd', order='F')
    UoverU[cases[i]+'_p'] = np.zeros((Ntwr), 'd', order='F')
    
    for k in range(sel_coord_f.shape[0]):
        loc = sel_coord_f[k]
        z = z_uvp
        z_d = (z - ((dispH[cases[i]+'_f'][k])/zi))
        
        [z0[cases[i]+'_f'][k],ustar[cases[i]+'_f'][k],U_mean,U_data,z_data,u_fit] = compute_ustar_G(Nz_SLayer,z_d,data.data[loc[0],loc[1],:,0],\
                                                                data.data[loc[0],loc[1],:,1],True)
        UoverU[cases[i]+'_f'][k] = U_mean[10]/ustar[cases[i]+'_f'][k]
        
    for k in range(sel_coord_p.shape[0]):
        loc = sel_coord_p[k]
        z = z_uvp
        z_d = (z - ((dispH[cases[i]+'_p'][k])/zi))
        
        [z0[cases[i]+'_p'][k],ustar[cases[i]+'_p'][k],U_mean,U_data,z_data,u_fit] = compute_ustar_G(Nz_SLayer,z_d,data.data[loc[0],loc[1],:,0],\
                                                                data.data[loc[0],loc[1],:,1],True)
        UoverU[cases[i]+'_p'][k] = U_mean[10]/ustar[cases[i]+'_p'][k]
    
    print('Done with case: ' + cases[i])
    
for i in range(len(cases)):
    ustar[cases[i]+'_f'] = ustar[cases[i]+'_f']*uscale
    ustar[cases[i]+'_p'] = ustar[cases[i]+'_p']*uscale
    z0[cases[i]+'_f'] = z0[cases[i]+'_f']/canopyH
    z0[cases[i]+'_p'] = z0[cases[i]+'_p']/canopyH

#%%Compute ustar,z0,d, and U/ustar for Ben's data

cases = ['ATTO','Sinusoidal','Flat']

for i in range(len(cases)):
    #Load the data
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + cases[i] + '/dataTKE.nc')
    
    #Simulation parameters
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
    Ntwr = 100
    Nz_SLayer = 200
    height = math.ceil(canopyH/dz)
    
    #load surface data
    from functions import build_phi, build_intf
    phi = build_phi(path_to_data + cases[i] + '/phi_functions/', nx, ny, nz, mpiProc)
    intf, iintf = build_intf(phi, dz)
    
    if cases[i]=='Flat':
        #Select coordinates
        coord_f = find_coordinates(intf,Ntwr,'flat')
        z_profile = np.arange(nz) * dz * zi
        zeds = np.ones((nx, ny, 1)) * z_profile
        dist = copy.deepcopy(zeds)
        del zeds,z_profile
        dist -= intf[:, :, np.newaxis] * zi
        mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
        mask4D = np.expand_dims(mask, axis=-1)
        data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
        del mask,mask4D
        
        idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
        idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

        dispH[cases[i]] = compute_d_twr(data,coord_f,dist,height+5,dz,zi,uscale,LAD)
        
        #Compute the ustar,z0,U/ustar
        z0[cases[i]] = np.zeros((Ntwr),'d',order='F')
        ustar[cases[i]] = np.zeros((Ntwr), 'd', order='F')
        UoverU[cases[i]] = np.zeros((Ntwr), 'd', order='F')
        
        for k, (ix, iy) in enumerate(coord_f):
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
    
            loc = coord_f[k]
            z = z_uvp
            z_d = (z - ((dispH[cases[i]][k])/zi))
            
            [z0[cases[i]][k],ustar[cases[i]][k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                                    data.data[loc[0],loc[1],:,1],True,fit_levels=BEN_FIT_LEVELS[cases[i]])
            UoverU[cases[i]][k] = U_mean[16]/ustar[cases[i]][k]
        
    else:
        #Select coordinates and compute displacement height
        coord_p = find_coordinates(intf,Ntwr,'max')
        coord_v = find_coordinates(intf,Ntwr,'min')
    
        z_profile = np.arange(nz) * dz * zi
        zeds = np.ones((nx, ny, 1)) * z_profile
        dist = copy.deepcopy(zeds)
        del zeds,z_profile
        dist -= intf[:, :, np.newaxis] * zi
        mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
        mask4D = np.expand_dims(mask, axis=-1)
        data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
        del mask,mask4D
        
        idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
        idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24
            
        dispH[cases[i]+'_p'] = compute_d_twr(data,coord_p,dist,height+5,dz,zi,uscale,LAD)
        dispH[cases[i]+'_v'] = compute_d_twr(data,coord_v,dist,height+5,dz,zi,uscale,LAD)
        
        #Compute the ustar,z0,U/ustar
        z0[cases[i]+'_p'] = np.zeros((Ntwr),'d',order='F')
        z0[cases[i]+'_v'] = np.zeros((Ntwr),'d',order='F')
        ustar[cases[i]+'_p'] = np.zeros((Ntwr), 'd', order='F')
        ustar[cases[i]+'_v'] = np.zeros((Ntwr), 'd', order='F')
        UoverU[cases[i]+'_p'] = np.zeros((Ntwr), 'd', order='F')
        UoverU[cases[i]+'_v'] = np.zeros((Ntwr), 'd', order='F')
        
        for k, (ix, iy) in enumerate(coord_p):
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
    
            loc = coord_p[k]
            z = z_uvp
            z_d = (z - ((dispH[cases[i]+'_p'][k])/zi))
            
            [z0[cases[i]+'_p'][k],ustar[cases[i]+'_p'][k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                                    data.data[loc[0],loc[1],:,1],True,fit_levels=BEN_FIT_LEVELS[cases[i]])
            UoverU[cases[i]+'_p'][k] = U_mean[16]/ustar[cases[i]+'_p'][k]
        
        for k, (ix, iy) in enumerate(coord_v):
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
    
            loc = coord_v[k]
            z = z_uvp
            z_d = (z - ((dispH[cases[i]+'_v'][k])/zi))
            
            [z0[cases[i]+'_v'][k],ustar[cases[i]+'_v'][k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                                    data.data[loc[0],loc[1],:,1],True,fit_levels=BEN_FIT_LEVELS[cases[i]])
            UoverU[cases[i]+'_v'][k] = U_mean[16]/ustar[cases[i]+'_v'][k]
            
    print('Done with case: '+cases[i])

for i in range(len(cases)):
    if cases[i] == 'Flat':
        ustar[cases[i]] = ustar[cases[i]]*uscale
        z0[cases[i]] = z0[cases[i]]/canopyH
    else:
        ustar[cases[i]+'_p'] = ustar[cases[i]+'_p']*uscale
        ustar[cases[i]+'_v'] = ustar[cases[i]+'_v']*uscale
        z0[cases[i]+'_p'] = z0[cases[i]+'_p']/canopyH
        z0[cases[i]+'_v'] = z0[cases[i]+'_v']/canopyH

#%%Compute ustar,z0,d, and U/ustar for Marco's data

case = 'simulation_G'

kappa = 0.4

def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar(Nz_SLayer, z_d, u, v, twr=False, fit_levels=None):    
    from scipy.optimize import curve_fit
    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]

    if fit_levels is None:
        fit_levels = URBAN_FIT_LEVELS
    fit_levels = np.asarray(fit_levels, dtype=int)

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

#Load the data
path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/'
data = scipy.io.loadmat(path_to_data + case + '/profiles.mat')

#Simulation parameters
lx = data['lx'][0][0]; ly = data['ly'][0][0]; lz = data['lz'][0][0]
nx = data['nx'][0][0]; ny = data['ny'][0][0]; nz = data['nz'][0][0]
dx = data['dx'][0][0]; dy = data['dy'][0][0]; dz = data['dz'][0][0]
z_uvp = data['z'][0]
z_w = data['zi'][0]
uscale = 1.23
canopyH = 15.3
Nz_SLayer = 200

#Compute displacement height
dispH[case] = (2/3)*canopyH
z_d = z_uvp[:-8] - dispH[case]
u_xy = (data['u_xy'].squeeze())[8:]; u_tw = (data['u_tw'].squeeze())[8:]
v_xy = (data['v_xy'].squeeze())[8:]; v_tw = (data['v_tw'].squeeze())[8:]

z0[case+'_xy'],ustar[case+'_xy'],U_mean,U_data,z_data,u_fit = compute_ustar(nz-8, z_d, u_xy, v_xy)
UoverU[case+'_xy'] = U_mean[16]/ustar[case+'_xy']
z0[case+'_xy'] = z0[case+'_xy']/canopyH

z0[case+'_tw'],ustar[case+'_tw'],U_mean,U_data,z_data,u_fit = compute_ustar(nz-8, z_d, u_tw, v_tw)
UoverU[case+'_tw'] = U_mean[16]/ustar[case+'_tw']
z0[case+'_tw'] = z0[case+'_tw']/canopyH

#%%Compute the means 

cases = list(ustar.keys())

for i in range(len(cases)):
    print(f'ustar for '+cases[i]+f': {np.mean(ustar[cases[i]]) :.02f}')
    
print('*'*80)

cases = list(z0.keys())

for i in range(len(cases)):
    print(f'z0 for '+cases[i]+f': {np.mean(z0[cases[i]]) :.02f}')

print('*'*80)

cases = list(dispH.keys())

for i in range(len(cases)):
    print(f'dispH for '+cases[i]+f': {np.mean(dispH[cases[i]]) :.00f}')

print('*'*80)

cases = list(UoverU.keys())

for i in range(len(cases)):
    print(f'UoverU for '+cases[i]+f': {np.mean(UoverU[cases[i]]) :.01f}')

























































































# %%
