#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:44:14 2026

@author: u1450851
"""

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import gaussian_kde

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso
from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 256
ny = 256
nz = 256

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
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Import RAV and checkpoint data

sim = 'patch256_aniso_1ms'
# sim = 'homo256_aniso_1ms'

path_to_RAV = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim
step = 130000   

checkpnt = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_checkpoint/',[step],nx,ny,nz)
RAV = xr.open_dataarray(path_to_RAV+'/data/Momentum3D/Data_Momentum_30min.nc')
aniso = xr.open_dataarray(path_to_RAV+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc')

#%%Compute anisotropy from the checkpoint reynolds stresses (on the fly averaging)

uu = checkpnt['uu_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = checkpnt['vv_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = checkpnt['ww_new'][:,:,:nz] - checkpnt['w_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]#- checkpnt['tzz_new'][:,:,:nz]
uv = checkpnt['uv_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = checkpnt['uw_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = checkpnt['vw_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke = uu + vv + ww

[xB_snap,yB_snap,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

#%%Compute anisotropy from the RAV files

uu = RAV[:,:,:,4].data - RAV[:,:,:,0].data*RAV[:,:,:,0].data #- RAV[:,:,:,16].data
vv = RAV[:,:,:,5].data - RAV[:,:,:,1].data*RAV[:,:,:,1].data #- RAV[:,:,:,17].data
ww = wnode2uvpnode(RAV[:,:,:,6].data - RAV[:,:,:,2].data*RAV[:,:,:,2].data) #- RAV[:,:,:,18].data)
uv = RAV[:,:,:,13].data - RAV[:,:,:,0].data*RAV[:,:,:,1].data #- RAV[:,:,:,19].data
uw = wnode2uvpnode(RAV[:,:,:,14].data - uvpnode2wnode(RAV[:,:,:,0].data)*RAV[:,:,:,2].data) #- RAV[:,:,:,20].data)
vw = wnode2uvpnode(RAV[:,:,:,15].data - uvpnode2wnode(RAV[:,:,:,1].data)*RAV[:,:,:,2].data) #- RAV[:,:,:,21].data)

[xB_rav,yB_rav,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

#%%Anisotropy computed with the on-the-fly averaging in the code and outputted

xB_otf = aniso[:,:,:,0].data
yB_otf = aniso[:,:,:,1].data

#%%Plot pcolor of anisotropy at z-level for RAV and on-the-fly

lvl = 10

fig,axs = plt.subplots(1,2,constrained_layout=True,sharey=True,figsize=(10,5))
p1 = axs[0].pcolormesh(x,y,yB_rav[:,:,lvl].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
p2 = axs[1].pcolormesh(x,y,yB_snap[:,:,lvl].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[1].set_xlabel(r"$x/z_i$",fontsize=14)
axs[0].set_xlabel(r"$x/z_i$",fontsize=14)
axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[0].set_title(r"RAV $y_B$",fontsize=14)
axs[1].set_title(r"Snap $y_B$",fontsize=14)
fig.colorbar(p1,ax=axs)
plt.show()

#%%Histogram of surface values of yB

lvl = 4

fig,axs = plt.subplots(1,1,tight_layout=True)
kde = gaussian_kde(yB_snap[:,:,lvl].flatten())
x_pdf = np.linspace(min(yB_snap[:,:,lvl].flatten()), max(yB_snap[:,:,lvl].flatten()), 1000)
pdf = kde(x_pdf)
axs.hist(yB_snap[:,:,lvl].flatten(),bins=50,color='r',alpha=.5,density=True)
axs.plot(x_pdf,pdf,c='r',label='Snap')
kde = gaussian_kde(yB_rav[:,:,lvl].flatten())
x_pdf = np.linspace(min(yB_rav[:,:,lvl].flatten()), max(yB_rav[:,:,lvl].flatten()), 1000)
pdf = kde(x_pdf)
axs.hist(yB_rav[:,:,lvl].flatten(),bins=50,color='g',alpha=.5,density=True)
axs.plot(x_pdf,pdf,c='g',label='RAV')

axs.set_xlabel(r"$y_B$",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.legend()

plt.show()

#%%Vertical slices of yB for RAV and Snap

vslice = 10

fig,axs = plt.subplots(1,2,constrained_layout=True,sharey=True,figsize=(10,5))
p1 = axs[0].pcolormesh(x,z_uvp,yB_rav[:,vslice,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
p2 = axs[1].pcolormesh(x,z_uvp,yB_snap[:,vslice,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[1].set_xlabel(r"$x/z_i$",fontsize=14)
axs[0].set_xlabel(r"$x/z_i$",fontsize=14)
axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
axs[0].set_title(r"RAV $y_B$",fontsize=14)
axs[1].set_title(r"Snap $y_B$",fontsize=14)
fig.colorbar(p1,ax=axs)
plt.show()













































