#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:27:57 2026

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

# First Comment

#%%

#%%Set path to the profiles
    
cases = ['Flat','Sinusoidal','ATTO']

TKE = dict()

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'

for i in range(len(cases)):
    TKE[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/TKE_terms.nc').data

#%%Simulation parameters

lx = 2.88; ly = 2.88; lz = 0.96
mpiProc = 32
nx,ny,nz = np.shape(TKE[cases[0]][:,:,:,0])
nz = nz + 5
zi = 1000
canopyH = 39/zi
dx = lx/nx; dy = ly/ny; dz = lz/nz
z_w = np.arange(0,nz)*dz
z_uvp = np.arange(0,nz)*dz + dz/2

Ntwr = 100
Nz_SLayer = 200

#%% Select tower coordinates

case = 2

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

# coord = find_coordinates(intf,Ntwr,'flat')
coord = find_coordinates(intf,Ntwr,'max')
# coord = find_coordinates(intf,Ntwr,'min')

z_profile = np.arange(nz) * dz * zi
dist = np.ones((nx, ny, 1)) * z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] > 0  # shape (Nx, Ny, Nz_SLayer)
# mask4D = np.expand_dims(mask, axis=-1)
# TKE[cases[case]][:, :, :, :] = np.where(mask4D, np.nan, TKE[cases[case]][:, :, :, :])
# del mask,mask4D

#%%Plot TKE profiles

Dt = TKE[cases[case]][:,:,:,11]
A = TKE[cases[case]][:,:,:,2]/abs(Dt)
P = TKE[cases[case]][:,:,:,-1]/abs(Dt)
Tt = TKE[cases[case]][:,:,:,5]/abs(Dt)
Pt = TKE[cases[case]][:,:,:,8]/abs(Dt)
Dv = TKE[cases[case]][:,:,:,9]/abs(Dt)
Dc = -TKE[cases[case]][:,:,:,10]/abs(Dt)

# fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(3,5))

# axs.plot(Pt[coord[:,0],coord[:,1],:].mean(axis=(0)),z_uvp[:-5]/canopyH,label='Pt',c='brown')
# axs.plot(A[coord[:,0],coord[:,1],:].mean(axis=(0)),z_uvp[:-5]/canopyH,label='A',c='red')
# axs.plot(Tt[coord[:,0],coord[:,1],:].mean(axis=(0)),z_uvp[:-5]/canopyH,label='Tt',c='purple')
# axs.plot(P[coord[:,0],coord[:,1],:].mean(axis=(0)),z_uvp[:-5]/canopyH,label='P',c='blue')
# axs.plot(Dv[coord[:,0],coord[:,1],:].mean(axis=(0)),z_uvp[:-5]/canopyH,label='Dv',c='orange')
# axs.plot(Dc[coord[:,0],coord[:,1],:].mean(axis=(0)),z_uvp[:-5]/canopyH,label='Dc',c='green')

# axs.set_xlim(-1.1,2)
# axs.set_ylim(0,3)
# axs.axhline(1,c='k',ls='--')
# axs.axhline(2,c='k',ls='--')
# axs.axvline(1,c='grey',ls='--')
# axs.axvline(0,c='grey',ls='--')
# axs.axvline(-1,c='grey',ls='--')

# axs.set_xlabel(r"$\frac{\partial\overline{e}}{\partial t}/|\epsilon_{t}|$",fontsize=14)
# axs.set_ylabel(r"$z/h_c$",fontsize=14)

# axs.tick_params(axis='both',labelsize=12)

# axs.legend()

# plt.show()


#%%

def terrain_following_profile(field_3d, mask, x_idx, y_idx, n_levels=100):
    """
    Average field_3d over x_idx, y_idx points aligned from first point
    above topography (defined by mask) for n_levels levels.
    """
    field_sel = field_3d[x_idx, y_idx, :]   # shape: (n_points, nz)
    mask_sel  = mask[x_idx, y_idx, :]        # shape: (n_points, nz)

    n_points = field_sel.shape[0]
    field_tf = np.full((n_points, n_levels), np.nan)

    first_valid = mask_sel.argmax(axis=1)    # first level above surface

    for k in range(n_levels):
        idx = first_valid + k
        valid = idx < field_sel.shape[1]
        field_tf[valid, k] = field_sel[valid, idx[valid]]

    return np.nanmean(field_tf, axis=0)      # shape: (n_levels,)

n_levels = 100
z_tf = np.arange(n_levels) * dz / canopyH + dz/2/canopyH

fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=(9, 8), sharey=True, sharex=True)

case = 0

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord = find_coordinates(intf,Ntwr,'flat')

x_idx = coord[:, 0]
y_idx = coord[:, 1]

z_profile = np.arange(nz) * dz * zi
dist = np.ones((nx, ny, 1)) * z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] > 0  # shape (Nx, Ny, Nz_SLayer)

Dt = TKE[cases[case]][:,:,:,11]
A = TKE[cases[case]][:,:,:,2]/abs(Dt)
P = TKE[cases[case]][:,:,:,-1]/abs(Dt)
Tt = TKE[cases[case]][:,:,:,5]/abs(Dt)
Pt = TKE[cases[case]][:,:,:,8]/abs(Dt)
Dv = -TKE[cases[case]][:,:,:,9]/abs(Dt)
Dc = -TKE[cases[case]][:,:,:,10]/abs(Dt)

for field, label, color in zip(
    [Pt, A, Tt, P, Dv, Dc],
    ['Pt', 'A', 'Tt', 'P', 'Dv', 'Dc'],
    ['brown', 'red', 'purple', 'blue', 'orange', 'green']
):
    profile = terrain_following_profile(field, mask, x_idx, y_idx, n_levels)
    axs[0,0].plot(profile, z_tf, label=label, c=color)

case = 1

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord = find_coordinates(intf,Ntwr,'max')

x_idx = coord[:, 0]
y_idx = coord[:, 1]

z_profile = np.arange(nz) * dz * zi
dist = np.ones((nx, ny, 1)) * z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] > 0  # shape (Nx, Ny, Nz_SLayer)

Dt = TKE[cases[case]][:,:,:,11]
A = TKE[cases[case]][:,:,:,2]/abs(Dt)
P = TKE[cases[case]][:,:,:,-1]/abs(Dt)
Tt = TKE[cases[case]][:,:,:,5]/abs(Dt)
Pt = TKE[cases[case]][:,:,:,8]/abs(Dt)
Dv = TKE[cases[case]][:,:,:,9]/abs(Dt)
Dc = -TKE[cases[case]][:,:,:,10]/abs(Dt)

for field, label, color in zip(
    [Pt, A, Tt, P, Dv, Dc],
    ['Pt', 'A', 'Tt', 'P', 'Dv', 'Dc'],
    ['brown', 'red', 'purple', 'blue', 'orange', 'green']
):
    profile = terrain_following_profile(field, mask, x_idx, y_idx, n_levels)
    axs[0,1].plot(profile, z_tf, label=label, c=color)
    
case = 1

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord = find_coordinates(intf,Ntwr,'min')

x_idx = coord[:, 0]
y_idx = coord[:, 1]

z_profile = np.arange(nz) * dz * zi
dist = np.ones((nx, ny, 1)) * z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] > 0  # shape (Nx, Ny, Nz_SLayer)

Dt = TKE[cases[case]][:,:,:,11]
A = TKE[cases[case]][:,:,:,2]/abs(Dt)
P = TKE[cases[case]][:,:,:,-1]/abs(Dt)
Tt = TKE[cases[case]][:,:,:,5]/abs(Dt)
Pt = TKE[cases[case]][:,:,:,8]/abs(Dt)
Dv = TKE[cases[case]][:,:,:,9]/abs(Dt)
Dc = -TKE[cases[case]][:,:,:,10]/abs(Dt)

for field, label, color in zip(
    [Pt, A, Tt, P, Dv, Dc],
    ['Pt', 'A', 'Tt', 'P', 'Dv', 'Dc'],
    ['brown', 'red', 'purple', 'blue', 'orange', 'green']
):
    profile = terrain_following_profile(field, mask, x_idx, y_idx, n_levels)
    axs[0,2].plot(profile, z_tf, label=label, c=color)
    
case = 2

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord = find_coordinates(intf,Ntwr,'max')

x_idx = coord[:, 0]
y_idx = coord[:, 1]

z_profile = np.arange(nz) * dz * zi
dist = np.ones((nx, ny, 1)) * z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] > 0  # shape (Nx, Ny, Nz_SLayer)

Dt = TKE[cases[case]][:,:,:,11]
A = TKE[cases[case]][:,:,:,2]/abs(Dt)
P = TKE[cases[case]][:,:,:,-1]/abs(Dt)
Tt = TKE[cases[case]][:,:,:,5]/abs(Dt)
Pt = TKE[cases[case]][:,:,:,8]/abs(Dt)
Dv = TKE[cases[case]][:,:,:,9]/abs(Dt)
Dc = -TKE[cases[case]][:,:,:,10]/abs(Dt)

for field, label, color in zip(
    [Pt, A, Tt, P, Dv, Dc],
    ['Pt', 'A', 'Tt', 'P', 'Dv', 'Dc'],
    ['brown', 'red', 'purple', 'blue', 'orange', 'green']
):
    profile = terrain_following_profile(field, mask, x_idx, y_idx, n_levels)
    axs[1,0].plot(profile, z_tf, label=label, c=color)
    
case = 2

from functions import build_phi, build_intf

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord = find_coordinates(intf,Ntwr,'min')

x_idx = coord[:, 0]
y_idx = coord[:, 1]

z_profile = np.arange(nz) * dz * zi
dist = np.ones((nx, ny, 1)) * z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] > 0  # shape (Nx, Ny, Nz_SLayer)

Dt = TKE[cases[case]][:,:,:,11]
A = TKE[cases[case]][:,:,:,2]/abs(Dt)
P = TKE[cases[case]][:,:,:,-1]/abs(Dt)
Tt = TKE[cases[case]][:,:,:,5]/abs(Dt)
Pt = TKE[cases[case]][:,:,:,8]/abs(Dt)
Dv = TKE[cases[case]][:,:,:,9]/abs(Dt)
Dc = -TKE[cases[case]][:,:,:,10]/abs(Dt)

for field, label, color in zip(
    [Pt, A, Tt, P, Dv, Dc],
    ['Pt', 'A', 'Tt', 'P', 'Dv', 'Dc'],
    ['brown', 'red', 'purple', 'blue', 'orange', 'green']
):
    profile = terrain_following_profile(field, mask, x_idx, y_idx, n_levels)
    axs[1,1].plot(profile, z_tf, label=label, c=color)

axs[1,0].set_xlim(-1.5, 3)
axs[1,1].set_xlim(-1.5, 3)
axs[1,2].set_xlim(-1.5, 3)
axs[0,0].set_ylim(0, 3)
axs[1,0].set_ylim(0, 3)

title = ['flat','idealized crest','idealized trough','real ridge','real trough']
axs_flat = axs.flatten()
for i in range(len(axs_flat)-1):
    axs_flat[i].axhline(1, c='k', ls='--')
    axs_flat[i].axhline(2, c='k', ls='--')
    axs_flat[i].axvline(1,  c='grey', ls='--')
    axs_flat[i].axvline(0,  c='grey', ls='--')
    axs_flat[i].axvline(-1, c='grey', ls='--')
    axs_flat[i].tick_params(axis='both', labelsize=12)
    axs_flat[i].set_title(title[i],fontsize=12)
    
axs[1,0].set_xlabel(r"$\frac{\partial\overline{e}}{\partial t}/|\epsilon_{t}|$", fontsize=14)
axs[1,1].set_xlabel(r"$\frac{\partial\overline{e}}{\partial t}/|\epsilon_{t}|$", fontsize=14)
axs[0,0].set_ylabel(r"$z/h_c$", fontsize=14)
axs[1,0].set_ylabel(r"$z/h_c$", fontsize=14)
# axs_flat[i].tick_params(axis='both', labelsize=12)
# axs.legend()
axs[1,2].axis('off')
plt.show()


#%%

#%%

cases = ['Flat','Sinusoidal','ATTO']

data = dict()

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'

for i in range(len(cases)):
    data[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/dataTKE.nc').data

#%%

case = 0

uu = data[cases[case]][:,:,:,4] - data[cases[case]][:,:,:,0]*data[cases[case]][:,:,:,0]
vv = data[cases[case]][:,:,:,5] - data[cases[case]][:,:,:,1]*data[cases[case]][:,:,:,1]
ww = data[cases[case]][:,:,:,6] - data[cases[case]][:,:,:,2]*data[cases[case]][:,:,:,2]
uv = data[cases[case]][:,:,:,7] - data[cases[case]][:,:,:,0]*data[cases[case]][:,:,:,1]
uw = data[cases[case]][:,:,:,8] - data[cases[case]][:,:,:,0]*data[cases[case]][:,:,:,2]
vw = data[cases[case]][:,:,:,9] - data[cases[case]][:,:,:,1]*data[cases[case]][:,:,:,2]

txx = -data[cases[case]][:,:,:,19]
tyy = -data[cases[case]][:,:,:,20]
tzz = -data[cases[case]][:,:,:,21]
txy = -data[cases[case]][:,:,:,22]
txz = -data[cases[case]][:,:,:,23]
tyz = -data[cases[case]][:,:,:,24]

fig,axs = plt.subplots(2,3,tight_layout=True)

axs[0,0].plot(np.mean(uu,axis=(0,1)),z_uvp[:-5])
axs[0,1].plot(np.mean(vv,axis=(0,1)),z_uvp[:-5])
axs[0,2].plot(np.mean(ww,axis=(0,1)),z_uvp[:-5])
axs[1,0].plot(np.mean(uv,axis=(0,1)),z_uvp[:-5])
axs[1,1].plot(np.mean(uw,axis=(0,1)),z_uvp[:-5])
axs[1,2].plot(np.mean(vw,axis=(0,1)),z_uvp[:-5])

axs[0,0].plot(np.mean(txx,axis=(0,1)),z_uvp[:-5])
axs[0,1].plot(np.mean(tyy,axis=(0,1)),z_uvp[:-5])
axs[0,2].plot(np.mean(tzz,axis=(0,1)),z_uvp[:-5])
axs[1,0].plot(np.mean(txy,axis=(0,1)),z_uvp[:-5])
axs[1,1].plot(np.mean(txz,axis=(0,1)),z_uvp[:-5])
axs[1,2].plot(np.mean(tyz,axis=(0,1)),z_uvp[:-5])

plt.show()

#%%

# Second Comment

#%%

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

cases = ['Gap_8_9mps','Patch_8_9mps','ATTO','Sinusoidal','Flat','simulation_G']
titles = ['g800','i800','ATTO','Sinusoidal','Flat','Urban']

tke_prof = dict()

ls = ['-','--']
Ntwr = 100
Nz_SLayer = 200

for case in range(len(cases)):
#----------------------------------------------------------------Giulia's HC cases--------------------------------------------------------------------------
    if case < 2:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'

        terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc').data
        
        lx = 2*np.pi; ly = 2*np.pi; lz = 1
        nx,ny,nz = np.shape(terms_bdg[:,:,:,0])
        zi = 1000
        canopyH = 39/zi
        dx = lx/nx; dy = ly/ny; dz = lz/nz
        
        sfc = np.load(path_to_data+'../input_txt_files/'+cases[case]+'/sfc.npy')
        
        tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
        tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])
        # tmpRES = np.where(np.abs(tmpRES) < 10, 0, tmpRES)
        tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0

        # Avoid divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
        
        tke_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['forest','patch']})
        
        coord_f = np.argwhere(sfc==1.6)
        coord_p = np.argwhere(sfc==0)
        idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
        idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)
        sel_coord_f = coord_f[idx_f]
        sel_coord_p = coord_p[idx_p]
        del coord_f,coord_p,idx_f,idx_p
        
        tke_xr[:,0] = average_over_selected_coords(tmpNorm[:,:,:Nz_SLayer],sel_coord_f)
        tke_xr[:,1] = average_over_selected_coords(tmpNorm[:,:,:Nz_SLayer],sel_coord_p)

        height = math.ceil(canopyH/dz)
        
        tke_prof[cases[case]] = tke_xr
        
#------------------------------------------------------------Ben's TC cases------------------------------------------------------------------------------        
    elif case >= 2 and case < 4:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
        terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc')
        
        lx = 2.88; ly = 2.88; lz = 0.96
        mpiProc = 32
        nx,ny,nz = np.shape(terms_bdg[:,:,:,0])
        nz = nz + 5
        zi = 1000
        canopyH = 39/zi
        dx = lx/nx; dy = ly/ny; dz = lz/nz
            
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

        tke_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['ridge','valley']})

        idx_P, idx_D = 14, 11

        res = np.zeros((Ntwr,Nz_SLayer))
        
        for k, (ix, iy) in enumerate(coord_p):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
            
            z_end = z_start + Nz_SLayer
            # res[k,:] = (terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D])*canopyH / (ustar[k]**3)
            res[k,:] = tmpNorm[ix,iy,z_start:z_end]
            
        tke_xr[:,0] = np.mean(res,axis=(0))

        res = np.zeros((Ntwr,Nz_SLayer))
        
        for k, (ix, iy) in enumerate(coord_v):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5

            # Slice velocity components over Nz_SLayer starting from z_start
            z_end = z_start + Nz_SLayer
            # res[k,:] = (terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D])*canopyH / (ustar[k]**3)
            res[k,:] = tmpNorm[ix,iy,z_start:z_end]

        tke_xr[:,1] = np.mean(res,axis=(0))
    
        tke_prof[cases[case]] = tke_xr
        
    elif case == 4:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
        terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc')
        
        lx = 2.88; ly = 2.88; lz = 0.96
        mpiProc = 32
        nx,ny,nz = np.shape(terms_bdg[:,:,:,0])
        nz = nz + 5
        zi = 1000
        canopyH = 39/zi
        dx = lx/nx; dy = ly/ny; dz = lz/nz

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

        tke_xr = xr.DataArray(np.ones(shape = (Nz_SLayer,),order='F'),\
                                dims=('z',),)

        idx_P, idx_D = 14, 11

        res = np.zeros((Ntwr,Nz_SLayer))
        
        for k, (ix, iy) in enumerate(coord_f):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5

            # Slice velocity components over Nz_SLayer starting from z_start
            z_end = z_start + Nz_SLayer
            # res[k,:] = (terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D])*canopyH / (ustar[k]**3)
            res[k,:] = tmpNorm[ix,iy,z_start:z_end]

        tke_xr[:] = np.mean(res,axis=(0))
        
        tke_prof[cases[case]] = tke_xr

#---------------------------------------------------------------------------Giometto's UC case----------------------------------------------------------------
    else:
        path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/'
        data = scipy.io.loadmat(path_to_data + cases[case] + '/profiles.mat')
        
        lx = data['lx'][0][0]; ly = data['ly'][0][0]; lz = data['lz'][0][0]
        nx = data['nx'][0][0]; ny = data['ny'][0][0]; nz = data['nz'][0][0]
        dx = data['dx'][0][0]; dy = data['dy'][0][0]; dz = data['dz'][0][0]
        z_uvp = data['z'][0]
        z_w = data['zi'][0]
        uscale = 1.23
        canopyH = 15.3
  
        tke_xr = xr.DataArray(np.ones(shape = (nz-8,2),order='F'),\
                                dims=('z','variable'), coords = {'variable':['xy','tw']})
        
        kappa = 0.4

        diss_xy = data['ds_xy'].squeeze(); diss_tw = data['ds_tw'].squeeze()

        prod_xy = data['sp_xy'].squeeze()[8:] + data['spd_xy'].squeeze()[8:] - data['spm2_xy'].squeeze()[7:]; 
        prod_tw = data['sp_tw'].squeeze()

        res_xy = diss_xy[8:] + prod_xy; res_tw = diss_tw[8:] + prod_tw[8:]
        res_xy[(abs(res_xy)<0.05*np.max(abs(res_xy)))] = 0; res_tw[(abs(res_tw)<0.05*np.max(abs(res_tw)))] = 0
        norm_xy = res_xy/abs(diss_xy[8:])*100; norm_tw = res_tw/abs(diss_tw[8:])*100
        
        tke_xr[:,0] = norm_xy
        tke_xr[:,1] = norm_tw

        tke_prof[cases[case]] = tke_xr


#%%Plot the profiles 

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

thresh = [5,10,12]
cases = ['Flat','Sinusoidal','ATTO','Gap_8_9mps','Patch_8_9mps','simulation_G']
col = ['g','k','r']
from matplotlib.transforms import ScaledTranslation

layout = [['a)', 'b)', 'c)'],
          ['d)', 'e)', 'f)']]
fig, axs_dict = plt.subplot_mosaic(layout, layout='constrained',figsize=(6,6),sharey=True,sharex=True)

for label, ax in axs_dict.items():

    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes,
        fontsize=14, va='top', ha='left',
        fontfamily='serif')

from scipy.stats import gaussian_kde

axs = np.array([[axs_dict[label] for label in row] for row in layout])

axs = axs.flatten()

for i in range(len(thresh)):
    
    for case in range(len(cases)):
        if case>2 and case<5:
            lx = 2*np.pi; ly = 2*np.pi; lz = 1
            nx,ny,nz = 256,256,256
            zi = 1000
            canopyH = 39/zi
            dx = lx/nx; dy = ly/ny; dz = lz/nz
            z_uvp = np.arange(0,nz)*dz + dz/2
            
            a0 = abs(tke_prof[cases[case]][:100,0])
            a1 = abs(tke_prof[cases[case]][:100,1])
            cr0 = first_sustained_below(a0, z_uvp[:100]/canopyH, thresh[i])
            cr1 = first_sustained_below(a1, z_uvp[:100]/canopyH, thresh[i])
            axs[case].scatter(0, cr0, color=col[i], s=40, zorder=5)
            axs[case].scatter(0, cr1, s=50, marker='o',facecolors='none', edgecolors=col[i],linewidths=1.5, zorder=5) 
    
        elif case>=1 and case<3:
            lx = 2.88; ly = 2.88; lz = 0.96
            nx,ny,nz = 256,256,379
            nz = nz + 5
            zi = 1000
            canopyH = 39/zi
            dx = lx/nx; dy = ly/ny; dz = lz/nz
            z_uvp = np.arange(0,nz)*dz + dz/2
            
            a0 = abs(tke_prof[cases[case]][:,0])
            a1 = abs(tke_prof[cases[case]][:,1])
            cr0 = first_sustained_below(a0, z_uvp[:Nz_SLayer]/canopyH, thresh[i])
            cr1 = first_sustained_below(a1, z_uvp[:Nz_SLayer]/canopyH, thresh[i])
            axs[case].scatter(0, cr0, color=col[i], s=40, zorder=5)
            axs[case].scatter(0, cr1, s=50, marker='o',facecolors='none', edgecolors=col[i],linewidths=1.5, zorder=5)
            
        elif case==0:
            lx = 2.88; ly = 2.88; lz = 0.96
            nx,ny,nz = 256,256,379
            nz = nz + 5
            zi = 1000
            canopyH = 39/zi
            dx = lx/nx; dy = ly/ny; dz = lz/nz
            z_uvp = np.arange(0,nz)*dz + dz/2
            
            a0 = abs(tke_prof[cases[case]][:])
            cr0 = first_sustained_below(a0, z_uvp[:Nz_SLayer]/canopyH, thresh[i])
            axs[case].scatter(0, cr0, color=col[i], s=40, zorder=5)
            
        else:
            lx = data['lx'][0][0]; ly = data['ly'][0][0]; lz = data['lz'][0][0]
            nx = data['nx'][0][0]; ny = data['ny'][0][0]; nz = data['nz'][0][0]
            dx = data['dx'][0][0]; dy = data['dy'][0][0]; dz = data['dz'][0][0]
            z_uvp = data['z'][0]
            canopyH = 15.3
            
            a0 = tke_prof[cases[case]][:,0]
            a1 = tke_prof[cases[case]][:,1]
            cr0 = first_sustained_below(a0, z_uvp[:Nz_SLayer]/canopyH, thresh[i])#4.5#
            cr1 = first_sustained_below(a1, z_uvp[:Nz_SLayer]/canopyH, thresh[i])#4.3#
            axs[case].scatter(0, cr0, color=col[i], s=40, zorder=5)
            axs[case].scatter(0, cr1, s=50, marker='o',facecolors='none', edgecolors=col[i],linewidths=1.5, zorder=5)
    
for i in range(len(cases)):
    axs[i].set_xlim(-1,1)
    axs[i].set_ylim(0,10)
    # axs[i].set_xlabel(r"$\phi_M$",fontsize=16)
    # axs[i].set_title(titles[i],fontsize=16)
    axs[i].tick_params(axis='y', which='major', labelsize=12)
    axs[i].tick_params(axis='x', which='major', labelsize=12)
    axs[i].axvline(0,c='k',ls=':')

axs[3].set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$",fontsize=16)
axs[4].set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$",fontsize=16)
axs[5].set_xlabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$",fontsize=16)
axs[0].set_ylabel(r"$z/h_C$",fontsize=16)
axs[3].set_ylabel(r"$z/h_C$",fontsize=16)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'RSL_height_comparison.png',dpi=300,edgecolor='white',facecolor='white')

plt.show() 







































