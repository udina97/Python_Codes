#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:13:37 2026

@author: u1450851
"""

#%%Import libraries
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom, read_checkpoint_tke
from Anisotropy_Functions import ColorAnisotropy, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Simulation parameters

nx = 128
ny = 128
nz = 128

lx = 2000*np.pi
ly = 2000*np.pi
lz = 2000

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
# zbl = 1099
uscale = 1
# wstar = 2.05
Tscale = 290
# qstar = 0.24
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load RAV data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim1 = 'p800_aniso_128'
sim2 = 'p800_noaniso_128'

step = 162000
time_ext = '30min'

cases = ['a','c']

mom3D = dict(); sc3D = dict(); tke3D = dict(); mom2D = dict(); sc2D = dict(); aniso3D = dict()

aniso3D['c'] = xr.open_dataarray(path+sim2+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
mom3D['c'] = xr.open_dataarray(path+sim2+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
sc3D['c'] = xr.open_dataarray(path+sim2+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
tke3D['c'] = xr.open_dataarray(path+sim2+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
mom2D['c'] = xr.open_dataarray(path+sim2+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
sc2D['c'] = xr.open_dataarray(path+sim2+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

aniso3D['a'] = xr.open_dataarray(path+sim1+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
mom3D['a'] = xr.open_dataarray(path+sim1+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
sc3D['a'] = xr.open_dataarray(path+sim1+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
tke3D['a'] = xr.open_dataarray(path+sim1+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
mom2D['a'] = xr.open_dataarray(path+sim1+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
sc2D['a'] = xr.open_dataarray(path+sim1+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

#Variable = ['u','v','w','p','u2','v2','w2','u3','v3','w3','u4','v4','w4',\
#            'uv','uw','vw','txx','tyy','tzz','txy','txz','tyz',\
#            'dudz','dvdz','nut','cs','tke_sgs']

# vardataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
#                         dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
#                         'avgUT','avgVT','avgWT','avgUTT','avgVTT','avgWTT','avgDTDX','avgDTDY','avgDTDZ',\
#                         'avgDTDX2','avgDTDY2','avgDTDZ2','avgTUT_sgs','avgTVT_sgs','avgTWT_sgs',\
#                         'avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
#                         'avg_ds']}) 

# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
#                 dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
#                 'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})

# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesE),order='F'),\
#                     dims=('x', 'y','z','variable'), coords = {'variable':['avgE_w','avgE_uvp',\
#                     'avgRHS','avgProd','avgDiss','avgNut','avgNuh','avgKTX','avgKTY',\
#                     'avgKTZ']})

#%%Boundary layer height

# for i in cases:
#     dTdz = sc3D[i][:,:,:,10]
#     max_dTdz_level = np.argmax(dTdz, axis=2)
#     max_dTdz_height = (max_dTdz_level)*dz 
#     print(np.mean(max_dTdz_height))
    
def get_dphidz_uvp2w(phi, dz):
    import numpy as np
    dphidz = np.zeros(phi.shape)
    # dphidz[:,:,0] = phi[:,:,0]/(0.5*dz)
    dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
    dphidz[:,:,-1] = dphidz[:,:,-2]
    return dphidz

for i in cases:
    dTdz = get_dphidz_uvp2w(sc3D[i][:,:,:,0]*Tscale, dz)
    max_dTdz_level = np.argmax(dTdz, axis=2)
    max_dTdz_height = (max_dTdz_level)*dz 
    print(np.mean(max_dTdz_height))
    
#%%PDF of ustar and wstar

from scipy.stats import gaussian_kde

fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(10, 6))

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'
    
    # Variables displayed in the top row
    u_star = (mom2D[i][:, :, 0] * uscale).T
    w_star = (sc2D[i][:, :, 0] * uscale).T
    
    # Remove NaN/inf values before computing KDEs
    u_values = u_star[np.isfinite(u_star)]
    w_values = w_star[np.isfinite(w_star)]
    
    # Gaussian kernel-density estimates
    u_kde = gaussian_kde(u_values)
    w_kde = gaussian_kde(w_values)
    
    # Coordinates on which to evaluate each PDF
    u_grid = np.linspace(u_values.min(), u_values.max(), 1000)
    w_grid = np.linspace(w_values.min(), w_values.max(), 1000)

    axs[0].plot(u_grid, u_kde(u_grid), color="tab:blue", lw=2, ls=ls, label=i)
    axs[1].plot(w_grid, w_kde(w_grid), color="tab:orange", lw=2, ls=ls, label=i)
    
    axs[0].legend()
    
    # print(f'Mean ustar for ' + sim + f': {np.mean(u_star)}')

    total_shear_force = np.trapz(np.trapz(u_star**2, x=x, axis=1), x=y, axis=0)
    surface_area = lx * ly
    mean_shear_stress = total_shear_force / surface_area
    effective_u_star = np.sqrt(mean_shear_stress)
    
    # print(f'Effective ustar for ' + sim + f': {effective_u_star}')

axs[0].set_title(r"PDF of $u_*$", fontsize=14)
axs[1].set_title(r"PDF of $w_*$", fontsize=14)

axs[0].set_xlabel(r"$u_* \,[ms^{-1}]$", fontsize=12)
axs[1].set_xlabel(r"$w_* \,[ms^{-1}]$", fontsize=12)
axs[0].set_ylabel("Probability density", fontsize=12)

axs[0].set_ylim(0)
axs[1].set_ylim(0)

# fig.suptitle(sim, fontsize=16)
plt.show()

#%%Plot profiles of velocity components

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,6))

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'

    axs.plot(np.mean(mom3D[i][:,:,:,0]*uscale,axis=(0,1)),z_uvp,c='k',label='u - '+i,ls=ls)
    axs.plot(np.mean(mom3D[i][:,:,:,1]*uscale,axis=(0,1)),z_uvp,c='r',label='v - '+i,ls=ls)
    axs.plot(np.mean(mom3D[i][:,:,:,2]*uscale,axis=(0,1)),z_w,c='g',label='w - '+i,ls=ls)

axs.set_ylabel(r'$z$', fontsize=12)
axs.set_xlabel(r'$u_i/w_*$', fontsize=12)

axs.set_ylim(0,z_w[-1])

# fig.suptitle(sim, fontsize=16)
plt.show()

#%%Turbulence intensity

fig,axs = plt.subplots(1,1,tight_layout=True,sharey=True,figsize=(4,6))

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'

    # uu = mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0]
    # vv = mom3D[:,:,:,5] - mom3D[:,:,:,1]*mom3D[:,:,:,1]
    # ww = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2])
    
    uu_disp = (mom3D[i][:,:,:,0] - np.broadcast_to(np.mean(mom3D[i][:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2
    vv_disp = (mom3D[i][:,:,:,1] - np.broadcast_to(np.mean(mom3D[i][:,:,:,1],axis=(0,1)), (nx,ny,nz)))**2
    ww_disp = wnode2uvpnode((mom3D[i][:,:,:,2] - np.broadcast_to(np.mean(mom3D[i][:,:,:,2],axis=(0,1)), (nx,ny,nz)))**2)
    
    # tke_res = 0.5*(uu + vv + ww)
    tke_sgs = tke3D[i][:,:,:,1]
    # tke_disp = 0.5*(uu_disp + vv_disp + ww_disp)
    # tke_tot = tke_res + tke_sgs #+ tke_disp
    
    # U = np.sqrt(mom3D[:,:,:,0]**2 + mom3D[:,:,:,1]**2 + wnode2uvpnode(mom3D[:,:,:,2]**2))
    # I = np.sqrt((2/3)*tke_tot)/U
    
    uu1D = np.mean(mom3D[i][:,:,:,4],axis=(0,1)) - np.mean(mom3D[i][:,:,:,0]**2,axis=(0,1)) + np.mean(uu_disp,axis=(0,1))
    vv1D = np.mean(mom3D[i][:,:,:,5],axis=(0,1)) - np.mean(mom3D[i][:,:,:,1]**2,axis=(0,1)) + np.mean(vv_disp,axis=(0,1))
    ww1D = np.mean(wnode2uvpnode(mom3D[i][:,:,:,6]),axis=(0,1)) - np.mean(wnode2uvpnode(mom3D[i][:,:,:,2])**2,axis=(0,1)) + np.mean(ww_disp,axis=(0,1))
    tke1D = 0.5*(uu1D + vv1D + ww1D) + np.mean(tke_sgs,axis=(0,1))
    # U1D = np.sqrt(np.mean(mom3D[i][:,:,:,0]**2,axis=(0,1)) + np.mean(mom3D[i][:,:,:,1]**2,axis=(0,1)) + np.mean(wnode2uvpnode(mom3D[i][:,:,:,2])**2, axis=(0,1)))
    # U1D = np.sqrt(np.mean(mom3D[i][:,:,:,0],axis=(0,1))**2 + np.mean(mom3D[i][:,:,:,1],axis=(0,1))**2 + np.mean(wnode2uvpnode(mom3D[i][:,:,:,2]), axis=(0,1))**2)
    U1D = np.mean(np.sqrt(mom3D[i][:,:,:,0]**2 + mom3D[i][:,:,:,1]**2 + wnode2uvpnode(mom3D[i][:,:,:,2]**2)),axis=(0,1))
    I1D = np.sqrt((2/3)*tke1D)/U1D



    # axs[0].pcolormesh(x,z_uvp/zbl,I[:,ny//2,:].T,cmap='viridis',vmin=0,vmax=2)
    # axs[1].plot(np.mean(I,axis=(0,1)),z_uvp/zbl,c='k')
    axs.plot(I1D,z_uvp,c='k',ls=ls)

axs.set_ylabel(r'$z/z_i$',fontsize=12)
axs.set_xlabel(r'$I$')
axs.set_xlim(0,3)
axs.set_ylim(0)

# fig.suptitle(sim, fontsize=16)
plt.show()


#%%Plot temperature profile and heatflux

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'

    T = sc3D[i][:,:,:,0]
    T_w = uvpnode2wnode(T)
    T_w[:,:,0] = sc2D[i][:,:,-2]
    # wT_sgs = - sc3D[:,:,:,-2]*sc3D[:,:,:,10]
    # wT_sgs[:,:,0] = sc2D[:,:,-1]
    wT = np.mean(sc3D[i][:,:,:,4],axis=(0,1)) - np.mean(mom3D[i][:,:,:,2]*T_w,axis=(0,1)) #- sc3D[:,:,:,19]
    wT_sgs = - np.mean(sc3D[i][:,:,:,19],axis=(0,1))
    wT_disp = np.mean((mom3D[i][:,:,:,2] - np.broadcast_to(np.mean(mom3D[i][:,:,:,2],axis=(0,1)), (nx,ny,nz))) * (T_w - np.broadcast_to(np.mean(T_w,axis=(0,1)), (nx,ny,nz))),axis=(0,1))

    axs[0].plot(np.mean(T,axis=(0,1))*Tscale,z_uvp,c='k',ls=ls)
    axs[1].plot(wT/np.mean(sc2D[i][:,:,-1]),z_w,c='orange',ls=ls)
    axs[1].plot(wT_sgs/np.mean(sc2D[i][:,:,-1]),z_w,c='r',ls=ls)
    axs[1].plot(wT_disp/np.mean(sc2D[i][:,:,-1]),z_w,c='g',ls=ls)
    axs[1].plot((wT + wT_sgs + wT_disp)/np.mean(sc2D[i][:,:,-1]),z_w,c='k',ls=ls)

    
# theta = np.select([z_uvp <= 974.0, (z_uvp > 974.0) & (z_uvp < 1074.0), z_uvp >= 1074.0], [300.0, 300.0 + (z_uvp - 974.0) * 0.08, 308.0 + (z_uvp - 1074.0) * 0.003])
# axs[0].plot(theta,z_uvp/zbl,c='k',ls='-.')
axs[1].axvline(0,c='k',ls='--')

axs[0].set_ylim(0,1500)
axs[0].set_xlabel(r'$\theta [K]$', fontsize=12)
axs[0].set_ylabel(r'$z/z_i$', fontsize=12)
axs[1].set_xlabel(r"$\overline{w'T'}/Q_*$", fontsize=12)

# fig.suptitle(sim)

plt.show()


#%%Compute the surface integrated heat flux

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,5))

for i,case in enumerate(cases):
    axs[i].pcolormesh(x,y,(sc2D[case][:,:,-1]*Tscale*uscale).T,vmin=-0.1,vmax=0.5)

plt.show()


#%%Compute anisotropy invariants from RAV data

xB = dict(); yB = dict(); l3 = dict()

for i in cases:
    uu = mom3D[i][:,:,:,4] - mom3D[i][:,:,:,0]**2 - mom3D[i][:,:,:,16] + (2/3)*tke3D[i][:,:,:,1]
    vv = mom3D[i][:,:,:,5] - mom3D[i][:,:,:,1]**2 - mom3D[i][:,:,:,17] + (2/3)*tke3D[i][:,:,:,1]
    ww = wnode2uvpnode(mom3D[i][:,:,:,6] - mom3D[i][:,:,:,2]**2) - mom3D[i][:,:,:,18] + (2/3)*tke3D[i][:,:,:,1]
    uv = mom3D[i][:,:,:,13] - mom3D[i][:,:,:,0]*mom3D[i][:,:,:,1] - mom3D[i][:,:,:,19]
    uw = wnode2uvpnode(mom3D[i][:,:,:,14] - uvpnode2wnode(mom3D[i][:,:,:,0])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,20])
    vw = wnode2uvpnode(mom3D[i][:,:,:,15] - uvpnode2wnode(mom3D[i][:,:,:,1])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,21])
    
    [xB[i],yB[i],l3[i]] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%%Plot anisotropy

fig,axs = plt.subplots(2,2,sharey=True,sharex=True,figsize=(11,4))

for i,case in enumerate(cases):
    # p = axs[0,i].pcolormesh(x,z_uvp,np.mean(yB[case],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    # p = axs[1,i].pcolormesh(x,z_uvp,np.mean(aniso3D[case][:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[0,i].pcolormesh(x,z_uvp,yB[case][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[1,i].pcolormesh(x,z_uvp,aniso3D[case][:,ny//2,:,1].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    axs[0,i].set_title(cases[i])


cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
fig.suptitle('yB')

plt.show()






















































