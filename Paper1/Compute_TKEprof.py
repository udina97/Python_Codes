#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:10:58 2025

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


#%%Set path to the profiles
    
cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat','simulation_G']

case = 9

if case < 6:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc')
    terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc')
elif case >= 6 and case < 9:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc')
    terms_bdg = xr.open_dataarray(path_to_data + cases[case] + '/TKE_terms.nc')
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

#%%Select coordinates and compute ustar for Giulia's data

# sfc = np.load(path_to_data+'../input_txt_files/'+cases[case]+'/sfc.npy')

# coord_f = np.argwhere(sfc==1.6)
# coord_p = np.argwhere(sfc==0)
# idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
# idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)
# sel_coord_f = coord_f[idx_f]
# sel_coord_p = coord_p[idx_p]
# del coord_f,coord_p,idx_f,idx_p

# T_13 = wnode2uvpnode(((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23])))
# T_23 = wnode2uvpnode(((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24])))
# cov_turb = -np.sqrt((T_13)**2 + (T_23)**2)
# ustar = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height
# del T_13,T_23,cov_turb

# TwrAvgProf_f = np.zeros((3,Nz_SLayer),order='F')
# TwrAvgProf_p = np.zeros((3,Nz_SLayer),order='F')

# TwrAvgProf_f[0,:] = average_over_selected_coords(terms_bdg.data[:,:,:Nz_SLayer,14]*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_f)
# TwrAvgProf_p[0,:] = average_over_selected_coords(terms_bdg.data[:,:,:Nz_SLayer,14]*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_p)
# TwrAvgProf_f[1,:] = average_over_selected_coords(terms_bdg.data[:,:,:Nz_SLayer,11]*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_f)
# TwrAvgProf_p[1,:] = average_over_selected_coords(terms_bdg.data[:,:,:Nz_SLayer,11]*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_p)
# TwrAvgProf_f[2,:] = average_over_selected_coords((terms_bdg.data[:,:,:Nz_SLayer,14] + terms_bdg.data[:,:,:Nz_SLayer,11])*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_f)
# TwrAvgProf_p[2,:] = average_over_selected_coords((terms_bdg.data[:,:,:Nz_SLayer,14] + terms_bdg.data[:,:,:Nz_SLayer,11])*canopyH/(ustar[:,:,np.newaxis]**3), sel_coord_p)

# fig,axs = plt.subplots(1,1,figsize=(4,8))

# axs.plot(TwrAvgProf_f[0,:],z_uvp[:Nz_SLayer],c='green',ls='-',label='Prod')
# axs.plot(TwrAvgProf_p[0,:],z_uvp[:Nz_SLayer],c='green',ls='--',label='Prod')
# axs.plot(TwrAvgProf_f[1,:],z_uvp[:Nz_SLayer],c='red',ls='-',label='Diss')
# axs.plot(TwrAvgProf_p[1,:],z_uvp[:Nz_SLayer],c='red',ls='--',label='Diss')
# axs.plot(TwrAvgProf_f[2,:],z_uvp[:Nz_SLayer],c='black',ls='-',label='Res')
# axs.plot(TwrAvgProf_p[2,:],z_uvp[:Nz_SLayer],c='black',ls='--',label='Res')

# axs.vlines(0,0,5,linestyle='--',colors='grey')
# axs.set_ylim(z_uvp[0],z_uvp[Nz_SLayer])
# # plt.xlim(-30,0)
# # plt.title(r'Real')
# axs.set_xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# # plt.xlabel(r'$\left \langle de/dt \right \rangle$')
# axs.set_ylabel(r'$z/z_i$')
# axs.legend(loc='upper right')
# plt.show()


#%%Save profiles

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'RedTKE_' + cases[case] + '_f.npy',TwrAvgProf_f)
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'RedTKE_' + cases[case] + '_p.npy',TwrAvgProf_p)

#%%Compute the profiles with Ben's data

# from functions import build_phi, build_intf

# phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
# intf, iintf = build_intf(phi, dz)

# # coord_f = find_coordinates(intf,Ntwr,'flat')
# coord_p = find_coordinates(intf,Ntwr,'max')
# coord_v = find_coordinates(intf,Ntwr,'min')

# z_profile = np.arange(nz) * dz * zi
# zeds = np.ones((nx, ny, 1)) * z_profile
# dist = copy.deepcopy(zeds)
# del zeds,z_profile
# dist -= intf[:, :, np.newaxis] * zi
# mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
# mask4D = np.expand_dims(mask, axis=-1)
# data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
# terms_bdg.data[:, :, :, :] = np.where(mask4D, np.nan, terms_bdg.data[:, :, :, :])
# del mask,mask4D

# TKE_p = np.zeros((3,Ntwr, Nz_SLayer), dtype='float64', order='F')
# TKE_v = np.zeros((3,Ntwr, Nz_SLayer), dtype='float64', order='F')
# # TKE_f = np.zeros((3,Ntwr, Nz_SLayer), dtype='float64', order='F')
# ustar = np.zeros(Ntwr)

# # Pre-extract needed variable indices
# idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
# idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24
# idx_P, idx_D = 14, 11

# for k, (ix, iy) in enumerate(coord_p):
#     # Find first index where dist > 0
#     z_start = np.argmax(dist[ix, iy, :] > 0) - 5

#     # Extract Ruw and Rvw at z_start + 16
#     z_idx_ustar = z_start + 16
#     Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
#             data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
#             data.data[ix, iy, z_idx_ustar, idx_txz])

#     Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
#             data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
#             data.data[ix, iy, z_idx_ustar, idx_tyz])

#     ustar[k] = (Ruw**2 + Rvw**2)**0.25

#     # Slice velocity components over Nz_SLayer starting from z_start
#     z_end = z_start + Nz_SLayer
#     P = terms_bdg.data[ix, iy, z_start:z_end, idx_P]
#     D = - terms_bdg.data[ix, iy, z_start:z_end, idx_D]
#     R = terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D]

#     TKE_p[0,k, :] = P*canopyH / (ustar[k]**3)
#     TKE_p[1,k, :] = D*canopyH / (ustar[k]**3)
#     TKE_p[2,k, :] = R*canopyH / (ustar[k]**3)
    
# for k, (ix, iy) in enumerate(coord_v):
#     # Find first index where dist > 0
#     z_start = np.argmax(dist[ix, iy, :] > 0) - 5

#     # Extract Ruw and Rvw at z_start + 16
#     z_idx_ustar = z_start + 16
#     Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
#             data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
#             data.data[ix, iy, z_idx_ustar, idx_txz])

#     Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
#             data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
#             data.data[ix, iy, z_idx_ustar, idx_tyz])

#     ustar[k] = (Ruw**2 + Rvw**2)**0.25

#     # Slice velocity components over Nz_SLayer starting from z_start
#     z_end = z_start + Nz_SLayer
#     P = terms_bdg.data[ix, iy, z_start:z_end, idx_P]
#     D = - terms_bdg.data[ix, iy, z_start:z_end, idx_D]
#     R = terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D]

#     TKE_v[0,k, :] = P*canopyH / (ustar[k]**3)
#     TKE_v[1,k, :] = D*canopyH / (ustar[k]**3)
#     TKE_v[2,k, :] = R*canopyH / (ustar[k]**3)
    
# for k, (ix, iy) in enumerate(coord_f):
#     # Find first index where dist > 0
#     z_start = np.argmax(dist[ix, iy, :] > 0) - 5

#     # Extract Ruw and Rvw at z_start + 16
#     z_idx_ustar = z_start + 16
#     Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
#             data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
#             data.data[ix, iy, z_idx_ustar, idx_txz])

#     Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
#             data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
#             data.data[ix, iy, z_idx_ustar, idx_tyz])

#     ustar[k] = (Ruw**2 + Rvw**2)**0.25

#     # Slice velocity components over Nz_SLayer starting from z_start
#     z_end = z_start + Nz_SLayer
#     P = terms_bdg.data[ix, iy, z_start:z_end, idx_P]
#     D = - terms_bdg.data[ix, iy, z_start:z_end, idx_D]
#     R = terms_bdg.data[ix, iy, z_start:z_end, idx_P] - terms_bdg.data[ix, iy, z_start:z_end, idx_D]

#     TKE_f[0,k, :] = P*canopyH / (ustar[k]**3)
#     TKE_f[1,k, :] = D*canopyH / (ustar[k]**3)
#     TKE_f[2,k, :] = R*canopyH / (ustar[k]**3)

# fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(np.mean(TKE_p[0,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='green',ls='-',label='Prod')
# axs.plot(np.mean(TKE_p[1,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='red',ls='-',label='Diss')
# axs.plot(np.mean(TKE_p[2,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='black',ls='-',label='Res')
# axs.plot(np.mean(TKE_v[0,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='green',ls='--',label='Prod')
# axs.plot(np.mean(TKE_v[1,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='red',ls='--',label='Diss')
# axs.plot(np.mean(TKE_v[2,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='black',ls='--',label='Res')

# # axs.plot(np.mean(TKE_f[0,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='green',ls='-',label='Prod')
# # axs.plot(np.mean(TKE_f[1,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='red',ls='-',label='Diss')
# # axs.plot(np.mean(TKE_f[2,:,:],axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c='black',ls='-',label='Res')

# axs.vlines(0,0,5,linestyle='--',colors='grey')
# # axs.set_ylim(z_uvp[0],z_uvp[Nz_SLayer])
# axs.set_xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# axs.set_ylabel(r'$z/z_i$')
# axs.legend(loc='upper right')
# plt.show()

#%%Save profiles

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'RedTKE_' + cases[case] + '_p.npy',np.mean(TKE_p,axis=(1)))
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'RedTKE_' + cases[case] + '_v.npy',np.mean(TKE_v,axis=(1)))
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'RedTKE_' + cases[case] + '.npy',np.mean(TKE_f,axis=(1)))


#%%Compute the profiles with Giometto's data

R13_xy = data['uw_xy'].squeeze(); R13_tw = data['uw_tw'].squeeze()
R23_xy = data['vw_xy'].squeeze(); R23_tw = data['vw_tw'].squeeze()

R13d = data['uw_xy'].squeeze() + data['uwd_xy'].squeeze()
R23d = data['vw_xy'].squeeze() + data['vwd_xy'].squeeze()

ustar_xy = ((R13_xy + data['txz_xy'].squeeze())**2 + (R23_xy + data['tyz_xy'].squeeze())**2)**(0.25)
ustar_tw = ((R13_tw + data['txz_tw'].squeeze())**2 + (R23_tw + data['tyz_tw'].squeeze())**2)**(0.25)
ustar_d_xy = ((R13d + data['txz_xy'].squeeze())**2 + (R23d + data['tyz_xy'].squeeze())**2)**(0.25)

tke_nd_xy = canopyH/ustar_d_xy[27]**3
tke_nd_tw = canopyH/ustar_tw[27]**3

diss_xy = data['ds_xy'].squeeze()*tke_nd_xy; diss_tw = data['ds_tw'].squeeze()*tke_nd_tw

prod_xy = data['sp_xy'].squeeze()[8:]*tke_nd_xy + data['spd_xy'].squeeze()[8:]*tke_nd_xy - data['spm2_xy'].squeeze()[7:]*tke_nd_xy; 
prod_tw = data['sp_tw'].squeeze()*tke_nd_tw

res_xy = diss_xy[8:] + prod_xy; res_tw = diss_tw[8:] + prod_tw[8:]

z_plot = 80

z = z_uvp[:z_plot]

fig,axs = plt.subplots(1,2,tight_layout=True)

axs[0].plot(diss_xy[8:z_plot+8],z,c='r',label='diss')
axs[0].plot(prod_xy[:z_plot],z,c='g',label='prod')
axs[0].plot(res_xy[:z_plot],z,c='k',label='res')
axs[0].axhline(canopyH,c='grey',linestyle='--')
axs[0].axhline(1.28*canopyH,c='grey',linestyle='--')

axs[1].plot(diss_tw[8:z_plot+8],z,c='r',label='diss')
axs[1].plot(prod_tw[8:z_plot+8],z,c='g',label='prod')
axs[1].plot(res_tw[:z_plot],z,c='k',label='res')
axs[1].axhline(canopyH,c='grey',linestyle='--')
axs[1].axhline(1.28*canopyH,c='grey',linestyle='--')

axs[0].set_ylim(0,z_plot)
axs[0].legend()
axs[1].set_ylim(0,z_plot)
axs[1].legend()

plt.show()

#%%Save Profiles

TKE_prof_xy = np.zeros((3,152),order='F'); TKE_prof_tw = np.zeros((3,152),order='F')

TKE_prof_xy[0,:] = prod_xy; TKE_prof_tw[0,:] = prod_tw[8:]
TKE_prof_xy[1,:] = diss_xy[8:]; TKE_prof_tw[1,:] = diss_tw[8:]
TKE_prof_xy[2,:] = res_xy; TKE_prof_tw[2,:] = res_tw

# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'RedTKE_' + cases[case] + '_xy.npy',TKE_prof_xy)
# np.save('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/' + 'RedTKE_' + cases[case] + '_tw.npy',TKE_prof_tw)













































