#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:43:22 2026

@author: u1450851
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters 

nx = 128
ny = 128
nz = 128

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.4
Tscale = 290

x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_w = np.arange(0,nz)*dz
z_uvp = np.arange(0,nz)*dz + dz/2

#%%Import data

sim_name = 'Classic_J'
sfc_config = 'Patch'
Ug = 1
avg_time = 30
path_to_sim = '/scratch/general/nfs1/u1450851/LES_Sims/'+sfc_config+'/'+str(nx)+'/'+str(Ug)+'ms/'+sim_name+'/'
path_to_data = path_to_sim+'data/'

# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                     dims=('x', 'y','z','variable'), coords = {'variable':['avgU','avgV',\
#                     'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                     'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                     'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                     'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC','avgL3D']})
# vardataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
#                     dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
#                     'avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
#                     'avg_ds']})
# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
#                 dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})
# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
#                 dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
#                 'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})

mom3D = xr.open_dataarray(path_to_data+'Momentum3D/Data_Momentum_'+str(avg_time)+'min.nc').data
mom2D = xr.open_dataarray(path_to_data+'Momentum2D/Data_Momentum_2D_'+str(avg_time)+'min.nc').data
sc3D = xr.open_dataarray(path_to_data+'Scalar3D/Data_Scalar_'+str(avg_time)+'min.nc').data
sc2D = xr.open_dataarray(path_to_data+'Scalar2D/Data_Scalar_2D_'+str(avg_time)+'min.nc').data

#%%Compute the Reynolds stresses and anisotropy

uu = mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0]
vv = mom3D[:,:,:,5] - mom3D[:,:,:,1]*mom3D[:,:,:,1]
ww = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2])
uv = mom3D[:,:,:,13] - mom3D[:,:,:,0]*mom3D[:,:,:,1]
uw = wnode2uvpnode(mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2])
vw = wnode2uvpnode(mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2])

txx = mom3D[:,:,:,16]
tyy = mom3D[:,:,:,17]
tzz = mom3D[:,:,:,18]
txy = mom3D[:,:,:,19]
txz = wnode2uvpnode(mom3D[:,:,:,20])
tyz = wnode2uvpnode(mom3D[:,:,:,21])

[xB_R,yB_R,L3_R] = Anisotropy(nx, ny, nz, uu, vv, ww, uv, uw, vw)

# [xB_S,yB_S,L3_S] = Anisotropy(nx, ny, nz, -mom3D[:,:,:,16], -mom3D[:,:,:,17], -mom3D[:,:,:,18], -mom3D[:,:,:,19], -wnode2uvpnode(mom3D[:,:,:,20]),
#                               -wnode2uvpnode(mom3D[:,:,:,21]))

[xB_T,yB_T,L3_T] = Anisotropy(nx, ny, nz, uu-txx, vv-tyy, ww-tzz, uv-txy, uw-(txz), vw-(tyz))

#%%Plot PDF of yB for positive values

mask_T = (yB_T[:,:,0] > 0) #& (tyy[:,:,0] < 0) & (tzz[:,:,0] < 0)
yB_R_filt = yB_R[:,:,0][mask_T]
yB_T_filt = yB_T[:,:,0][mask_T]

def compute_PDF(data,n=1000):
    from scipy.stats import gaussian_kde
    x_pdf = np.linspace(min(data.flatten()),max(data.flatten()),n)
    kde = gaussian_kde(data.flatten())
    pdf = kde(x_pdf)
    return x_pdf,pdf

fig,axs = plt.subplots(1,1)
x_pdf,pdf = compute_PDF(yB_R_filt)
axs.plot(x_pdf,pdf,c='k',label=r'$y_{B,r}$')
x_pdf,pdf = compute_PDF(yB_T_filt)
axs.plot(x_pdf,pdf,c='r',label=r'$y_{B,t}$')

axs.set_xlabel(r'$y_B$', fontsize=15)
axs.set_ylabel(r'$PDF$',fontsize=15)
axs.set_title(r'$y_{B,t}$ at sfc > 0'+f' -- {nx}-{Ug}-'+sfc_config)
axs.legend()

plt.show()

#%%

mask_T = yB_T[:,:,10] > 0
yB_R_filt = yB_R[:,:,10][mask_T]
yB_T_filt = yB_T[:,:,10][mask_T]

diff = abs(yB_R_filt - yB_T_filt)

fig,axs = plt.subplots(1,1)
x_pdf,pdf = compute_PDF(diff)
axs.plot(x_pdf,pdf,c='k')

plt.show()

#%%Profiles of the resolved stresses versus the SGS component

fig,axs= plt.subplots(1,6,tight_layout=True,figsize=(10,3))

axs[0].plot(np.mean(uu,axis=(0,1)),z_uvp,c='k')
axs[0].plot(np.mean(txx,axis=(0,1)),z_uvp,c='r')
axs[0].plot(np.mean(uu-txx,axis=(0,1)),z_uvp,c='g')

axs[1].plot(np.mean(vv,axis=(0,1)),z_uvp,c='k')
axs[1].plot(np.mean(tyy,axis=(0,1)),z_uvp,c='r')

axs[2].plot(np.mean(ww,axis=(0,1)),z_uvp,c='k')
axs[2].plot(np.mean(tzz,axis=(0,1)),z_uvp,c='r')

axs[3].plot(np.mean(uv,axis=(0,1)),z_uvp,c='k')
axs[3].plot(np.mean(txy,axis=(0,1)),z_uvp,c='r')

axs[4].plot(np.mean(uw,axis=(0,1)),z_uvp,c='k')
axs[4].plot(np.mean(txz,axis=(0,1)),z_uvp,c='r')

axs[5].plot(np.mean(vw,axis=(0,1)),z_uvp,c='k')
axs[5].plot(np.mean(tyz,axis=(0,1)),z_uvp,c='r')

for i in range(len(axs)):
    axs[i].set_ylim(0,0.1)

plt.show()


#%%

w = mom3D[:,:,:,2]




















































