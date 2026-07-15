#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:55:02 2025

@author: u1450851
"""

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

#%%

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


#%%Load data

cases = ['Gap_8_9mps','Patch_8_9mps','ATTO','Sinusoidal','Flat','simulation_G']
titles = ['g800','i800','ATTO','Sinusoidal','Flat','Urban']
phi_prof = dict()
aniso_prof= dict()
tke_prof = dict()

ls = ['-','--']
Ntwr = 100
Nz_SLayer = 200

for case in range(len(cases)):
#----------------------------------------------------------------Giulia's HC cases--------------------------------------------------------------------------
    if case < 2:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
        data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc')
        anisotropy = xr.open_dataarray(path_to_data + cases[case] + '/anisotropy.nc')
        terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc')
        
        lx = 2*np.pi; ly = 2*np.pi; lz = 1
        nx,ny,nz = np.shape(data[:,:,:,0])
        zi = 1000
        canopyH = 39/zi
        dx = lx/nx; dy = ly/ny; dz = lz/nz
        z_w = np.arange(0,nz)*dz
        z_uvp = np.arange(0,nz)*dz + dz/2
        LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
        uscale = 0.313
        
        sfc = np.load(path_to_data+'../input_txt_files/'+cases[case]+'/sfc.npy')
        
        tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
        tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])
        # tmpRES = np.where(np.abs(tmpRES) < 10, 0, tmpRES)
        # tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
        
        # tmpDIS = terms_bdg[:, :, :, 11].copy()
        # tmpRES = (terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11]).copy()
        
        threshold = 0.05 * np.nanmax(np.nanmedian(tmpRES, axis=(0, 1)))
        
        tmpRES = tmpRES.where(np.abs(tmpRES) >= threshold, 0)

        # Avoid divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
        
        phi_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['forest','patch']})
        aniso_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['forest','patch']})
        tke_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['forest','patch']})
        
        coord_f = np.argwhere(sfc==1.6)
        coord_p = np.argwhere(sfc==0)
        idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
        idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)
        sel_coord_f = coord_f[idx_f]
        sel_coord_p = coord_p[idx_p]
        del coord_f,coord_p,idx_f,idx_p
        
        T_13 = wnode2uvpnode(((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23])))
        T_23 = wnode2uvpnode(((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24])))
        cov_turb = -np.sqrt((T_13)**2 + (T_23)**2)
        ustar = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height
        del T_13,T_23,cov_turb
        # tke_xr[:,0] = average_over_selected_coords((terms_bdg.data[:,:,:Nz_SLayer,14] + terms_bdg.data[:,:,:Nz_SLayer,11])*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_f)
        # tke_xr[:,1] = average_over_selected_coords((terms_bdg.data[:,:,:Nz_SLayer,14] + terms_bdg.data[:,:,:Nz_SLayer,11])*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_p)
        
        tke_xr[:,0] = average_over_selected_coords(tmpNorm[:,:,:Nz_SLayer],sel_coord_f)
        tke_xr[:,1] = average_over_selected_coords(tmpNorm[:,:,:Nz_SLayer],sel_coord_p)
        del ustar

        height = math.ceil(canopyH/dz)
        
        disp_f = compute_d_twr_G(data, sel_coord_f, height, dz, zi, uscale, LAD)
        disp_p = np.zeros((Ntwr),order='F')
        dudz_uvp = wnode2uvpnode(data.data[:,:,:,12])
        dvdz_uvp = wnode2uvpnode(data.data[:,:,:,15])
        
        phi_m_2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
        z2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
        z0hi = np.zeros((sel_coord_f.shape[0]),'d',order='F')
        ustar = np.zeros((sel_coord_f.shape[0]),'d',order='F')
        yb_twr = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
        
        
        for k in range(sel_coord_f.shape[0]):
            loc = sel_coord_f[k]
            yb_twr[k,:] = anisotropy.data[loc[0],loc[1],:Nz_SLayer,1]
        
            z = z_uvp
            z_d = (z - ((disp_f[k])/zi))
            
            [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar_G(Nz_SLayer,z_d,data.data[loc[0],loc[1],:,0],\
                                                                    data.data[loc[0],loc[1],:,1],True)
            
            phi_m_2D[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],:Nz_SLayer,0],data.data[loc[0],loc[1],:Nz_SLayer,1],\
                                      dudz_uvp[loc[0],loc[1],:Nz_SLayer],dvdz_uvp[loc[0],loc[1],:Nz_SLayer],ustar[k])
        
        phi_xr[:,0] = np.mean(phi_m_2D,axis=(0))
        aniso_xr[:,0] = np.mean(yb_twr,axis=(0))
        
        phi_m_2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
        z2D = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
        z0hi = np.zeros((sel_coord_f.shape[0]),'d',order='F')
        ustar = np.zeros((sel_coord_f.shape[0]),'d',order='F')
        yb_twr = np.zeros((sel_coord_f.shape[0],Nz_SLayer))
        
        for k in range(sel_coord_p.shape[0]):
            loc = sel_coord_p[k]
            yb_twr[k,:] = anisotropy.data[loc[0],loc[1],:Nz_SLayer,1]
        
            z = z_uvp
            z_d = (z - ((disp_p[k])/zi))
            
            [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar_G(Nz_SLayer,z_d,data.data[loc[0],loc[1],:,0],\
                                                                    data.data[loc[0],loc[1],:,1],True)
            
            phi_m_2D[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],:Nz_SLayer,0],data.data[loc[0],loc[1],:Nz_SLayer,1],\
                                      dudz_uvp[loc[0],loc[1],:Nz_SLayer],dvdz_uvp[loc[0],loc[1],:Nz_SLayer],ustar[k])
        
        phi_xr[:,1] = np.mean(phi_m_2D,axis=(0))
        aniso_xr[:,1] = np.mean(yb_twr,axis=(0))
        
        phi_prof[cases[case]] = phi_xr
        aniso_prof[cases[case]] = aniso_xr
        tke_prof[cases[case]] = tke_xr
        
#------------------------------------------------------------Ben's TC cases------------------------------------------------------------------------------        
    elif case >= 2 and case < 4:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
        data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc')
        anisotropy = xr.open_dataarray(path_to_data + cases[case] + '/anisotropy.nc')
        terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc')
        
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
            
        from functions import build_phi, build_intf

        phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
        intf, iintf = build_intf(phi, dz)

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
        anisotropy.data[:, :, :, :] = np.where(mask4D, np.nan, anisotropy.data[:, :, :, :])
        terms_bdg.data[:, :, :, :] = np.where(mask4D, np.nan, terms_bdg.data[:, :, :, :])
        del mask,mask4D
        
        tmpRES = copy.deepcopy((terms_bdg.data[:,:,:,14] - terms_bdg.data[:,:,:,11]))
        tmpDIS = copy.deepcopy((terms_bdg.data[:,:,:,11]))
        val = np.nanmedian(tmpRES[(dist[:,:,5:]>38) & (dist[:,:,5:]<42)])
        tmpRES[abs(tmpRES) < 0.05*val] = 0
        # tmpRES[(abs(tmpRES)<1)] = 0
        # tmpNorm = ((tmpRES)/abs(tmpDIS))*100
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

        phi_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['ridge','valley']})
        aniso_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['ridge','valley']})
        tke_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['ridge','valley']})

        # Pre-extract needed variable indices
        idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
        idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24
        idx_P, idx_D = 14, 11
        
        ustar = np.zeros(Ntwr)
        res = np.zeros((Ntwr,Nz_SLayer))
        
        for k, (ix, iy) in enumerate(coord_p):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5

            # Extract Ruw and Rvw at z_start + 16
            z_idx_ustar = z_start + 16
            # Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
            #         data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
            #         data.data[ix, iy, z_idx_ustar, idx_txz])

            # Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
            #         data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
            #         data.data[ix, iy, z_idx_ustar, idx_tyz])

            # ustar[k] = (Ruw**2 + Rvw**2)**0.25

            # # Slice velocity components over Nz_SLayer starting from z_start
            z_end = z_start + Nz_SLayer
            # res[k,:] = (terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D])*canopyH / (ustar[k]**3)
            res[k,:] = tmpNorm[ix,iy,z_start:z_end]

        height = math.ceil(canopyH/dz)
            
        d_dim_p = compute_d_twr(data,coord_p,dist,height+5,dz,zi,uscale,LAD)
        d_dim_v = compute_d_twr(data,coord_v,dist,height+5,dz,zi,uscale,LAD)

        z0hi = np.zeros((Ntwr),'d',order='F')
        ustar = np.zeros((Ntwr),'d',order='F')
        phi_m_2D = np.zeros((coord_p.shape[0],Nz_SLayer))
        yb_twr = np.zeros((coord_p.shape[0],Nz_SLayer))

        for k, (ix, iy) in enumerate(coord_p):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
            inf_pt = int(np.ceil(d_dim_p[k]/(dz*zi)))
            
            loc = coord_p[k]
            yb_twr[k,:] = anisotropy.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1]

            z = z_uvp
            z_d = (z - ((d_dim_p[k])/zi))
            
            [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                                    data.data[loc[0],loc[1],:,1],True)
            
            phi_m_2D[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
                                      data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])
                
        phi_xr[:,0] = np.mean(phi_m_2D,axis=(0))
        aniso_xr[:,0] = np.mean(yb_twr,axis=(0))
        tke_xr[:,0] = np.mean(res,axis=(0))
        
        ustar = np.zeros(Ntwr)
        res = np.zeros((Ntwr,Nz_SLayer))
        
        for k, (ix, iy) in enumerate(coord_v):
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

            # Slice velocity components over Nz_SLayer starting from z_start
            z_end = z_start + Nz_SLayer
            # res[k,:] = (terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D])*canopyH / (ustar[k]**3)
            res[k,:] = tmpNorm[ix,iy,z_start:z_end]

        z0hi = np.zeros((Ntwr),'d',order='F')
        ustar = np.zeros((Ntwr),'d',order='F')
        phi_m_2D = np.zeros((coord_p.shape[0],Nz_SLayer))
        yb_twr = np.zeros((coord_p.shape[0],Nz_SLayer))
            
        for k, (ix, iy) in enumerate(coord_v):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
            inf_pt = int(np.ceil(d_dim_v[k]/(dz*zi)))
            
            loc = coord_v[k]
            yb_twr[k,:] = anisotropy.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1]

            z = z_uvp
            z_d = (z - ((d_dim_v[k])/zi))
            
            [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                                    data.data[loc[0],loc[1],:,1],True)
            
            phi_m_2D[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
                                      data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])
                
        phi_xr[:,1] = np.mean(phi_m_2D,axis=(0))
        aniso_xr[:,1] = np.mean(yb_twr,axis=(0))
        tke_xr[:,1] = np.mean(res,axis=(0))
        
        phi_prof[cases[case]] = phi_xr
        aniso_prof[cases[case]] = aniso_xr
        tke_prof[cases[case]] = tke_xr
        
    elif case == 4:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
        data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc')
        anisotropy = xr.open_dataarray(path_to_data + cases[case] + '/anisotropy.nc')
        terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc')
        
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
        from functions import build_phi, build_intf

        phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
        intf, iintf = build_intf(phi, dz)

        coord_f = find_coordinates(intf,Ntwr,'flat')

        z_profile = np.arange(nz) * dz * zi
        zeds = np.ones((nx, ny, 1)) * z_profile
        dist = copy.deepcopy(zeds)
        del zeds,z_profile
        dist -= intf[:, :, np.newaxis] * zi
        mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
        mask4D = np.expand_dims(mask, axis=-1)
        data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
        anisotropy.data[:, :, :, :] = np.where(mask4D, np.nan, anisotropy.data[:, :, :, :])
        terms_bdg.data[:, :, :, :] = np.where(mask4D, np.nan, terms_bdg.data[:, :, :, :])
        del mask,mask4D
        
        tmpRES = copy.deepcopy((terms_bdg.data[:,:,:,14] - terms_bdg.data[:,:,:,11]))
        tmpDIS = copy.deepcopy((terms_bdg.data[:,:,:,11]))
        val = np.nanmedian(tmpRES[(dist[:,:,5:]>38) & (dist[:,:,5:]<42)])
        tmpRES[abs(tmpRES) < 0.05*val] = 0
        # tmpRES[(abs(tmpRES)<1)] = 0
        # tmpNorm = ((tmpRES)/abs(tmpDIS))*100
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

        phi_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,),order='F'),\
                                dims=('z',),)
        aniso_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,),order='F'),\
                                dims=('z',),)
        tke_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,),order='F'),\
                                dims=('z',),)

        # Pre-extract needed variable indices
        idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
        idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24
        idx_P, idx_D = 14, 11
        
        ustar = np.zeros(Ntwr)
        res = np.zeros((Ntwr,Nz_SLayer))
        
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

            # Slice velocity components over Nz_SLayer starting from z_start
            z_end = z_start + Nz_SLayer
            # res[k,:] = (terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D])*canopyH / (ustar[k]**3)
            res[k,:] = tmpNorm[ix,iy,z_start:z_end]

        height = math.ceil(canopyH/dz)

        d_dim_f = compute_d_twr(data,coord_f,dist,height+5,dz,zi,uscale,LAD)
        
        z0hi = np.zeros((Ntwr),'d',order='F')
        ustar = np.zeros((Ntwr),'d',order='F')
        phi_m_2D = np.zeros((coord_p.shape[0],Nz_SLayer),'d',order='F')
        yb_twr = np.zeros((coord_p.shape[0],Nz_SLayer),'d',order='F')
            
        for k, (ix, iy) in enumerate(coord_f):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
            
            loc = coord_f[k]
            yb_twr[k,:] = anisotropy.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1]
            
            z = z_uvp
            z_d = (z - ((d_dim_f[k])/zi))
            
            [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dist[loc[0],loc[1],:],data.data[loc[0],loc[1],:,0],\
                                                                data.data[loc[0],loc[1],:,1],True)
            
            phi_m_2D[k,:] = phi_m_loc(nx,ny,Nz_SLayer,z_d,data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,0],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,1],\
                                  data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,12],data.data[loc[0],loc[1],z_start:z_start+Nz_SLayer,15],ustar[k])
                
        phi_xr[:] = np.mean(phi_m_2D,axis=(0))
        aniso_xr[:] = np.mean(yb_twr,axis=(0))
        tke_xr[:] = np.mean(res,axis=(0))
        
        phi_prof[cases[case]] = phi_xr
        aniso_prof[cases[case]] = aniso_xr
        tke_prof[cases[case]] = tke_xr

#---------------------------------------------------------------------------Giometto's UC case----------------------------------------------------------------
    else:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/'
        data = scipy.io.loadmat(path_to_data + cases[case] + '/profiles.mat')
        anisotropy = xr.open_dataarray(path_to_data + cases[case] + '/anisotropy.nc')
        anisotropy_w = xr.open_dataarray(path_to_data + cases[case] + '/anisotropy_w.nc')
        
        lx = data['lx'][0][0]; ly = data['ly'][0][0]; lz = data['lz'][0][0]
        nx = data['nx'][0][0]; ny = data['ny'][0][0]; nz = data['nz'][0][0]
        dx = data['dx'][0][0]; dy = data['dy'][0][0]; dz = data['dz'][0][0]
        z_uvp = data['z'][0]
        z_w = data['zi'][0]
        uscale = 1.23
        canopyH = 15.3
        
        phi_xr = xr.DataArray(np.ones(shape = (nz-8,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['xy','tw']})
        aniso_xr = xr.DataArray(np.ones(shape = (nz-8,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['xy','tw']})
        tke_xr = xr.DataArray(np.ones(shape = (nz-8,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['xy','tw']})
        
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
        
        phi_xr[:,0] = phi_xy
        phi_xr[:,1] = phi_tw
        aniso_xr[:,0] = anisotropy.data[8:,1]
        aniso_xr[:,1] = anisotropy_w.data[8:,1]
        
        R13_xy = data['uw_xy'].squeeze(); R13_tw = data['uw_tw'].squeeze()
        R23_xy = data['vw_xy'].squeeze(); R23_tw = data['vw_tw'].squeeze()

        R13d = data['uw_xy'].squeeze() + data['uwd_xy'].squeeze()
        R23d = data['vw_xy'].squeeze() + data['vwd_xy'].squeeze()

        ustar_xy = ((R13_xy + data['txz_xy'].squeeze())**2 + (R23_xy + data['tyz_xy'].squeeze())**2)**(0.25)
        ustar_tw = ((R13_tw + data['txz_tw'].squeeze())**2 + (R23_tw + data['tyz_tw'].squeeze())**2)**(0.25)
        ustar_d_xy = ((R13d + data['txz_xy'].squeeze())**2 + (R23d + data['tyz_xy'].squeeze())**2)**(0.25)

        tke_nd_xy = canopyH/ustar_d_xy[27]**3
        tke_nd_tw = canopyH/ustar_tw[27]**3

        diss_xy = data['ds_xy'].squeeze(); diss_tw = data['ds_tw'].squeeze()

        prod_xy = data['sp_xy'].squeeze()[8:] + data['spd_xy'].squeeze()[8:] - data['spm2_xy'].squeeze()[7:]; 
        prod_tw = data['sp_tw'].squeeze()

        res_xy = diss_xy[8:] + prod_xy; res_tw = diss_tw[8:] + prod_tw[8:]
        # res_xy[(abs(res_xy)<0.01)] = 0; res_tw[(abs(res_tw)<0.01)] = 0
        res_xy[(abs(res_xy)<0.05*np.max(abs(res_xy)))] = 0; res_tw[(abs(res_tw)<0.05*np.max(abs(res_tw)))] = 0
        norm_xy = res_xy/abs(diss_xy[8:])*100; norm_tw = res_tw/abs(diss_tw[8:])*100
        
        tke_xr[:,0] = norm_xy
        tke_xr[:,1] = norm_tw
        
        phi_prof[cases[case]] = phi_xr
        aniso_prof[cases[case]] = aniso_xr
        tke_prof[cases[case]] = tke_xr


#%%Plot the profiles 

def crossing_points_yb(a, z, ref=0.38):
    """Return list of z positions where a crosses ref."""
    pts = []
    for i in range(len(a)-1):
        if (a[i] - ref) * (a[i+1] - ref) < 0:  # signs differ → crossing
            # linear interpolation
            frac = (ref - a[i]) / (a[i+1] - a[i])
            z_cross = z[i] + frac * (z[i+1] - z[i])
            pts.append(z_cross)
    return pts

def crossing_points(a, z):
    """Crossings at both +0.1 and -0.1."""
    pts = []
    for ref in (0.1, -0.1):
        for i in range(len(a) - 1):
            if (a[i] - ref) * (a[i+1] - ref) < 0:
                frac = (ref - a[i]) / (a[i+1] - a[i])
                z_cross = z[i] + frac * (z[i+1] - z[i])
                pts.append((z_cross))  # keep track of which ref
    return pts

def first_sustained_below(a, z, threshold=0.1):

    a = np.asarray(a)
    z = np.asarray(z)

    below = np.abs(a) < threshold

    for i in range(len(a)):
        if below[i] and np.all(below[i:]):
            # Linear interpolation to find exact crossing point
            if i == 0:
                return z[0]  # already below at start
            
            # Find the crossing between i-1 and i
            a0, a1 = a[i-1], a[i]
            z0, z1 = z[i-1], z[i]
            
            # Interpolate crossing of |a| = threshold
            # Solve |a0 + frac*(a1-a0)| = threshold
            # For simplicity, interpolate on signed value using the correct sign
            sign = np.sign(a1)
            ref = sign * threshold
            frac = (ref - a0) / (a1 - a0)
            z_cross = z0 + frac * (z1 - z0)
            return z_cross

    return None

cases = ['Flat','Sinusoidal','ATTO','Gap_8_9mps','Patch_8_9mps','simulation_G']
col = 'coral'
col2 = 'tomato'
from matplotlib.transforms import ScaledTranslation
# fig,axs = plt.subplots(1,6,tight_layout=True,sharey=True,figsize=(12,4))
layout = [['a)', 'b)', 'c)'],
          ['d)', 'e)', 'f)']]
fig, axs_dict = plt.subplot_mosaic(layout, layout='constrained',figsize=(6,6),sharey=True,sharex=True)

for label, ax in axs_dict.items():

    # ax.text(
    #     0.0, 1.0, label, transform=(
    #         ax.transAxes + ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans)),
    #     fontsize=15, va='bottom', fontfamily='serif')
    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes,
        fontsize=14, va='top', ha='left',
        fontfamily='serif')

from scipy.stats import gaussian_kde
# fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(12,3))
# axs = np.array([axs_dict[label] for label in layout[0]])
axs = np.array([[axs_dict[label] for label in row] for row in layout])

axs = axs.flatten()

for case in range(len(cases)):
    if case>2 and case<5:
        lx = 2*np.pi; ly = 2*np.pi; lz = 1
        nx,ny,nz = 256,256,256
        zi = 1000
        canopyH = 39/zi
        dx = lx/nx; dy = ly/ny; dz = lz/nz
        z_uvp = np.arange(1,nz)*dz + dz/2
        
        axs[case].plot(phi_prof[cases[case]][:Nz_SLayer,0],z_uvp[:Nz_SLayer]/canopyH,c='k',ls=ls[0])
        axs[case].plot(phi_prof[cases[case]][:Nz_SLayer,1],z_uvp[:Nz_SLayer]/canopyH,c='k',ls=ls[1])
        axs[case].axvline(1,c='k',ls=':')
        
        a0 = tke_prof[cases[case]][:100,0]
        a1 = tke_prof[cases[case]][:100,1]
        # cross0 = crossing_points(a0, z_uvp[:Nz_SLayer]/canopyH)
        # cross1 = crossing_points(a1, z_uvp[:Nz_SLayer]/canopyH)
        cr0 = first_sustained_below(a0, z_uvp[:100]/canopyH, 12)
        cr1 = first_sustained_below(a1, z_uvp[:100]/canopyH, 12)
        axs[case].scatter(1, cr0, color='k', s=40, zorder=5)
        axs[case].scatter(1, cr1, s=50, marker='o',facecolors='none', edgecolors='k',linewidths=1.5, zorder=5) 
        # tmp = True
        # for zc in cross0:
        #     if zc>1 and zc<8 and tmp:
        #         axs[case].scatter(1, zc, color='k', s=30, zorder=5)
        #         tmp = False
        # tmp = True
        # for zc in cross1:
        #     if zc>1 and zc<8 and tmp:
        #         axs[case].scatter(1, zc, s=40, marker='o',facecolors='none', edgecolors='k',linewidths=1.5, zorder=5) 
        #         tmp = False
                
        ax2 = axs[case].twiny()
        a0 = aniso_prof[cases[case]][:,0]
        a1 = aniso_prof[cases[case]][:,1]
        ax2.plot(a0, z_uvp[:Nz_SLayer]/canopyH, c=col2, ls=ls[0])
        ax2.plot(a1, z_uvp[:Nz_SLayer]/canopyH, c=col2, ls=ls[1])
        # cross0 = crossing_points_yb(a0, z_uvp[:Nz_SLayer]/canopyH)
        # cross1 = crossing_points_yb(a1, z_uvp[:Nz_SLayer]/canopyH)
        # for zc in cross0:
        #     if zc>1:
        #         ax2.scatter(0.38, zc, color='red', s=30, zorder=5)
        # for zc in cross1:
        #     if zc>1:
        #         ax2.scatter(0.38, zc, s=40, marker='o',facecolors='none', edgecolors='red',linewidths=1.5, zorder=5) 
            
        ax2.axvline(0.38,c=col2,ls=':')
        ax2.axvspan(0.35,0.39,alpha=0.5,color=col)
        ax2.axvspan(0.32,0.42,alpha=0.3,color=col)
        # ax2.set_xlabel(r"$y_B$",c=col,fontsize=16,labelpad=7)
        ax2.set_xlim(0,0.5)
        # ax2.tick_params(axis='x', colors=col)
        # ax2.spines['top'].set_color(col)
        # ax2.tick_params(axis='x', which='major', labelsize=12)
        # ax2.set_xticks([0,0.2,0.4])
        ax2.set_xticklabels([])
        

    elif case>=1 and case<3:
        lx = 2.88; ly = 2.88; lz = 0.96
        nx,ny,nz = 256,256,379
        nz = nz + 5
        zi = 1000
        canopyH = 39/zi
        dx = lx/nx; dy = ly/ny; dz = lz/nz
        z_uvp = np.arange(0,nz)*dz + dz/2
        
        axs[case].plot(phi_prof[cases[case]][:,0],z_uvp[:Nz_SLayer]/canopyH,c='k',ls=ls[0])
        axs[case].plot(phi_prof[cases[case]][:,1],z_uvp[:Nz_SLayer]/canopyH,c='k',ls=ls[1])
        axs[case].axvline(1,c='k',ls=':')
        
        a0 = tke_prof[cases[case]][:,0]
        a1 = tke_prof[cases[case]][:,1]
        cr0 = first_sustained_below(a0, z_uvp[:Nz_SLayer]/canopyH, 10)
        cr1 = first_sustained_below(a1, z_uvp[:Nz_SLayer]/canopyH, 10)
        axs[case].scatter(1, cr0, color='k', s=40, zorder=5)
        axs[case].scatter(1, cr1, s=50, marker='o',facecolors='none', edgecolors='k',linewidths=1.5, zorder=5)
        # cross0 = crossing_points(a0, z_uvp[:Nz_SLayer]/canopyH)
        # cross1 = crossing_points(a1, z_uvp[:Nz_SLayer]/canopyH)
        # tmp = True
        # for zc in cross0:
        #     if zc>1 and zc<9 and tmp:
        #         axs[case].scatter(1, zc, color='k', s=30, zorder=5)
        #         tmp = False
        # tmp = True
        # for zc in cross1:
        #     if zc>1 and zc<9 and tmp:
        #         axs[case].scatter(1, zc, s=40, marker='o',facecolors='none', edgecolors='k',linewidths=1.5, zorder=5) 
        #         tmp = False
        
        ax2 = axs[case].twiny()
        
        a0 = aniso_prof[cases[case]][:,0]
        a1 = aniso_prof[cases[case]][:,1]
        ax2.plot(a0, z_uvp[:Nz_SLayer]/canopyH, c=col2, ls=ls[0])
        ax2.plot(a1, z_uvp[:Nz_SLayer]/canopyH, c=col2, ls=ls[1])
        # cross0 = crossing_points_yb(a0, z_uvp[:Nz_SLayer]/canopyH)
        # cross1 = crossing_points_yb(a1, z_uvp[:Nz_SLayer]/canopyH)
        # for zc in cross0:
        #     if zc>2:
        #         ax2.scatter(0.38, zc, color='red', s=30, zorder=5)
        # for zc in cross1:
        #     if zc>2:
        #         ax2.scatter(0.38, zc, s=40, marker='o',facecolors='none', edgecolors='red',linewidths=1.5, zorder=5) 
                
        ax2.axvline(0.38,c=col2,ls=':')
        ax2.axvspan(0.35,0.39,alpha=0.5,color=col)
        ax2.axvspan(0.32,0.42,alpha=0.3,color=col)
        ax2.set_xlabel(r"$y_B$",c=col2,fontsize=16,labelpad=7)
        ax2.set_xlim(0,0.5)
        ax2.tick_params(axis='x', colors=col2)
        ax2.spines['top'].set_color(col2)
        ax2.tick_params(axis='x', which='major', labelsize=12)
        ax2.set_xticks([0,0.2,0.4])
        
    elif case==0:
        lx = 2.88; ly = 2.88; lz = 0.96
        nx,ny,nz = 256,256,379
        nz = nz + 5
        zi = 1000
        canopyH = 39/zi
        dx = lx/nx; dy = ly/ny; dz = lz/nz
        z_uvp = np.arange(0,nz)*dz + dz/2
        
        axs[case].plot(phi_prof[cases[case]][:],z_uvp[:Nz_SLayer]/canopyH,c='k',ls=ls[0])
        axs[case].axvline(1,c='k',ls=':')
        
        a0 = tke_prof[cases[case]][:]
        cr0 = first_sustained_below(a0, z_uvp[:Nz_SLayer]/canopyH, 10)
        axs[case].scatter(1, cr0, color='k', s=40, zorder=5)
        # cross0 = crossing_points(a0, z_uvp[:Nz_SLayer]/canopyH)
        # tmp = True
        # for zc in cross0:
        #     if zc>1 and zc<9 and tmp:
        #         axs[case].scatter(1, zc, color='k', s=30, zorder=5)
        #         tmp = False
        
        ax2 = axs[case].twiny()
        
        a0 = aniso_prof[cases[case]][:]
        ax2.plot(a0, z_uvp[:Nz_SLayer]/canopyH, c=col2, ls=ls[0])
        # cross0 = crossing_points_yb(a0, z_uvp[:Nz_SLayer]/canopyH)
        # for zc in cross0:
        #     if zc>2:
        #         ax2.scatter(0.38, zc, color='red', s=30, zorder=5)

        ax2.axvline(0.38,c=col2,ls=':')
        ax2.axvspan(0.35,0.39,alpha=0.5,color=col)
        ax2.axvspan(0.32,0.42,alpha=0.3,color=col)
        ax2.set_xlabel(r"$y_B$",c=col2,fontsize=16,labelpad=7)
        ax2.set_xlim(0,0.5)
        ax2.tick_params(axis='x', colors=col2)
        ax2.spines['top'].set_color(col2)
        ax2.tick_params(axis='x', which='major', labelsize=12)
        ax2.set_xticks([0,0.2,0.4])
        
    else:
        lx = data['lx'][0][0]; ly = data['ly'][0][0]; lz = data['lz'][0][0]
        nx = data['nx'][0][0]; ny = data['ny'][0][0]; nz = data['nz'][0][0]
        dx = data['dx'][0][0]; dy = data['dy'][0][0]; dz = data['dz'][0][0]
        z_uvp = data['z'][0]
        canopyH = 15.3
        
        axs[case].plot(phi_prof[cases[case]][:,0],z_uvp[:-8]/canopyH,c='k',ls=ls[0])
        axs[case].plot(phi_prof[cases[case]][:,1],z_uvp[:-8]/canopyH,c='k',ls=ls[1])
        axs[case].axvline(1,c='k',ls=':')
        
        a0 = tke_prof[cases[case]][:,0]
        a1 = tke_prof[cases[case]][:,1]
        cr0 = 4.5#first_sustained_below(a0, z_uvp[:Nz_SLayer]/canopyH, 10)
        cr1 = 4.3#first_sustained_below(a1, z_uvp[:Nz_SLayer]/canopyH, 10)
        axs[case].scatter(1, cr0, color='k', s=40, zorder=5)
        axs[case].scatter(1, cr1, s=50, marker='o',facecolors='none', edgecolors='k',linewidths=1.5, zorder=5)
        # cross0 = crossing_points(a0, z_uvp[:-8]/canopyH)
        # cross1 = crossing_points(a1, z_uvp[:-8]/canopyH)
        # tmp = True
        # for zc in cross0:
        #     if zc>1.5 and zc<9 and tmp:
        #         axs[case].scatter(1, zc, color='k', s=30, zorder=5)
        #         tmp = False
        # tmp = True
        # for zc in cross1:
        #     if zc>1.5 and zc<9 and tmp:
        #         axs[case].scatter(1, zc, s=40, marker='o',facecolors='none', edgecolors='k',linewidths=1.5, zorder=5) 
        #         tmp = False
        
        ax2 = axs[case].twiny()

        a0 = aniso_prof[cases[case]][:,0]
        a1 = aniso_prof[cases[case]][:,1]
        ax2.plot(a0, z_uvp[:-8]/canopyH, c=col2, ls=ls[0])
        ax2.plot(a1, z_uvp[:-8]/canopyH, c=col2, ls=ls[1])
        # cross0 = crossing_points_yb(a0, z_uvp[:-8]/canopyH)
        # cross1 = crossing_points_yb(a1, z_uvp[:-8]/canopyH)
        # for zc in cross0:
        #     if zc>3 and zc<8:
        #         ax2.scatter(0.38, zc, color='red', s=30, zorder=5)
        # tmp = True
        # for zc in cross1:
        #     if zc>3 and zc<8 and tmp:
        #         ax2.scatter(0.38, zc, s=40, marker='o',facecolors='none', edgecolors='red',linewidths=1.5, zorder=5) 
        #         tmp = False
        ax2.axvline(0.38,c=col2,ls=':')
        ax2.axvspan(0.35,0.39,alpha=0.5,color=col)
        ax2.axvspan(0.32,0.42,alpha=0.3,color=col)
        # ax2.set_xlabel(r"$y_B$",c=col,fontsize=16,labelpad=7)
        ax2.set_xlim(0,0.5)
        # ax2.tick_params(axis='x', colors=col)
        # ax2.spines['top'].set_color(col)
        # ax2.tick_params(axis='x', which='major', labelsize=12)
        ax2.set_xticks([0,0.2,0.4])
        ax2.set_xticklabels([])
    
for i in range(len(cases)):
    axs[i].set_xlim(0,2)
    axs[i].set_ylim(0,10)
    # axs[i].set_xlabel(r"$\phi_M$",fontsize=16)
    # axs[i].set_title(titles[i],fontsize=16)
    axs[i].tick_params(axis='y', which='major', labelsize=12)
    axs[i].tick_params(axis='x', which='major', labelsize=12)

axs[3].set_xlabel(r"$\phi_M$",fontsize=16)
axs[4].set_xlabel(r"$\phi_M$",fontsize=16)
axs[5].set_xlabel(r"$\phi_M$",fontsize=16)
axs[0].set_ylabel(r"$z/h_C$",fontsize=16)
axs[3].set_ylabel(r"$z/h_C$",fontsize=16)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'RSL_height_comparison.png',dpi=300,edgecolor='white',facecolor='white')

plt.show() 

#%%














































