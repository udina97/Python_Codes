#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:03:59 2026

@author: u1450851
"""

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso
from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 64
ny = 64
nz = 64

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

#%%Import data

sim = 'test_lag_3'
time = 5
path_to_data = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim

mom3D = xr.open_dataarray(path_to_data+'/data/Momentum3D/Data_Momentum_'+str(time)+'min.nc').data
mom2D = xr.open_dataarray(path_to_data+'/data/Momentum2D/Data_Momentum_2D_'+str(time)+'min.nc').data
sc3D = xr.open_dataarray(path_to_data+'/data/Scalar3D/Data_Scalar_'+str(time)+'min.nc').data
sc2D = xr.open_dataarray(path_to_data+'/data/Scalar2D/Data_Scalar_2D_'+str(time)+'min.nc').data

#%%Compute Reynolds stresses and Anisotropy

uu = mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0] - mom3D[:,:,:,16]
vv = mom3D[:,:,:,5] - mom3D[:,:,:,1]*mom3D[:,:,:,1] - mom3D[:,:,:,17]
ww = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2]) - mom3D[:,:,:,18]
uv = mom3D[:,:,:,13] - mom3D[:,:,:,0]*mom3D[:,:,:,1] - mom3D[:,:,:,19]
uw = wnode2uvpnode(mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2] - mom3D[:,:,:,20])
vw = wnode2uvpnode(mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2] - mom3D[:,:,:,21])

tke = uu + vv + ww

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

#%%Check yB and xB are within physical interval

if np.any(yB<0):
    print('Negative values of yB')
if np.any(yB>np.sqrt(3)/2):
    print('yB exceeds max value')
if np.any(xB<0):
    print('Negative values of xB')
if np.any(xB>1):
    print('xB exceeds max value')
    
#%%Mean profile of velocity components

fig,axs = plt.subplots(1,3,tight_layout=True,sharey=True)

axs[0].plot(np.mean(mom3D[:,:,:,0],axis=(0,1)),z_uvp,c='k')
axs[1].plot(np.mean(mom3D[:,:,:,1],axis=(0,1)),z_uvp,c='k')
axs[2].plot(np.mean(mom3D[:,:,:,2],axis=(0,1)),z_w,c='k')

axs[0].set_ylim(0,z_uvp[-1])
axs[0].set_ylabel(r"$z/z_i$",fontsize=14)

axs[0].set_xlabel(r'$u$',fontsize=12)
axs[1].set_xlabel(r'$v$',fontsize=12)
axs[2].set_xlabel(r'$w$',fontsize=12)

fig.suptitle(sim, fontsize=12)
plt.show()

#%%Pcolor of velocity components

vslice = ny//2

fig,axs = plt.subplots(1,3,tight_layout=True,sharey=True,figsize=(10,4))

axs[0].pcolormesh(x,z_uvp,mom3D[:,vslice,:,0].T,cmap='hot_r')
axs[1].pcolormesh(x,z_uvp,mom3D[:,vslice,:,1].T,cmap='hot_r')
axs[2].pcolormesh(x,z_w,mom3D[:,vslice,:,2].T,cmap='hot_r')

axs[0].set_ylabel(r'$z/z_i$',fontsize=12)
axs[0].set_xlabel(r'$x/z_i$',fontsize=12)
axs[1].set_xlabel(r'$x/z_i$',fontsize=12)
axs[2].set_xlabel(r'$x/z_i$',fontsize=12)

axs[0].set_title(r'u',fontsize=12)
axs[1].set_title(r'v',fontsize=12)
axs[2].set_title(r'w',fontsize=12)

fig.suptitle(sim, fontsize=12)
plt.show()


#%%Plot anisotropy for a vertical slice

vslice = ny//2

fig,axs = plt.subplots(2,1,tight_layout=True,figsize=(5,8))
p = axs[0].pcolormesh(x,z_uvp,yB[:,vslice,:].T,cmap= ColorAnisotropy(),vmin=0, vmax=np.sqrt(3)/2)
kde = gaussian_kde((yB[:,vslice,:]).flatten())
x_pdf = np.linspace(min((yB[:,vslice,:]).flatten()), max((yB[:,vslice,:]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((yB[:,vslice,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
axs[0].set_xlabel(r'$x/z_i$', fontsize=14)
axs[0].set_ylabel(r'$z/z_i$', fontsize=14)
axs[0].set_title('yB vertical slice', fontsize=14)
axs[1].set_xlabel(r'$y_B$', fontsize=14)
axs[1].set_ylabel('PDF', fontsize=14)
axs[1].set_xlim(0,np.sqrt(3)/2)
cbar = plt.colorbar(p)
# axs[0].set_ylim(0,0.5)

fig.suptitle(sim, fontsize=12)
plt.show()

#%%Plot yb at the surface

level = 0

fig,axs = plt.subplots(2,1,tight_layout=True,figsize=(5,8))
p = axs[0].pcolormesh(x,y,yB[:,:,level].T,cmap= ColorAnisotropy(),vmin=0, vmax=np.sqrt(3)/2)
kde = gaussian_kde((yB[:,:,level]).flatten())
x_pdf = np.linspace(min((yB[:,:,level]).flatten()), max((yB[:,:,level]).flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist((yB[:,:,level]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
axs[0].set_xlabel(r'$x/z_i$', fontsize=14)
axs[0].set_ylabel(r'$y/z_i$', fontsize=14)
axs[0].set_title('yB at Sfc', fontsize=14)
axs[1].set_xlabel(r'$y_B$', fontsize=14)
axs[1].set_ylabel('PDF', fontsize=14)
# axs[1].set_xlim(0,np.sqrt(3)/2)
cbar = plt.colorbar(p)

fig.suptitle(sim, fontsize=12)
plt.show()

#%%Reynolds stress profiles

fig,axs = plt.subplots(1,6,tight_layout=True,figsize=(12,4))

axs[0].plot(np.mean(uu,axis=(0,1)),z_uvp)
axs[1].plot(np.mean(vv,axis=(0,1)),z_uvp)
axs[2].plot(np.mean(ww,axis=(0,1)),z_uvp)
axs[3].plot(np.mean(uv,axis=(0,1)),z_uvp)
axs[4].plot(np.mean(uw,axis=(0,1)),z_uvp)
axs[5].plot(np.mean(vw,axis=(0,1)),z_uvp)
    
axs[0].set_xlabel(r"$\overline{u'u'}$",fontsize=14)
axs[1].set_xlabel(r"$\overline{v'v'}$",fontsize=14)
axs[2].set_xlabel(r"$\overline{w'w'}$",fontsize=14)
axs[3].set_xlabel(r"$\overline{u'v'}$",fontsize=14)
axs[4].set_xlabel(r"$\overline{u'w'}$",fontsize=14)
axs[5].set_xlabel(r"$\overline{v'w'}$",fontsize=14)
axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
for i in range(len(axs)):
    # axs[i].legend()
    axs[i].set_ylim(0,z_uvp[-1])
    
# axs[-1].legend()
# fig.suptitle(sfc+f" - {nx} - {Ug}m/s", fontsize=12)
fig.suptitle(sim, fontsize=12)
plt.show()

#%%Temperature pcolor

tmp = copy.deepcopy(sc3D[:,:,:,0])

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 40 #int(nz/2)
zslice = 10

fig, axs=plt.subplots(1,2,constrained_layout=True,figsize=(10,5),sharey=True)

plt1 = axs[0].pcolormesh(x_ax,z_ax,tmp[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')

axs[1].plot(np.mean(tmp,axis=(0,1)),z_uvp,c='k')

# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs[0].set_ylabel(r'$z/z_i$',fontsize=12)
axs[0].set_xlabel(r'$x/z_i$',fontsize=12)
axs[1].set_xlabel(r'$\theta$',fontsize=12)
fig.colorbar(plt1,ax=axs[0])
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%%Plot pdf for ustar and heatflux at the surface

ustar = mom2D[:,:,0]
    
heatflux = sc2D[:,:,-1]

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((ustar[:,:]).flatten())
x_pdf = np.linspace(min((ustar[:,:]).flatten()), max((ustar[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,ustar[:,:].T,cmap='bwr')
axs[1,0].hist((ustar[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((heatflux[:,:]).flatten())
x_pdf = np.linspace(min((heatflux[:,:]).flatten()), max((heatflux[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,heatflux[:,:].T,cmap='bwr')
axs[1,1].hist((heatflux[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$u_*$ at sfc', fontsize=14)
axs[1,1].set_xlabel(r'$\overline{wT}$ at sfc', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

# axs[1,0].set_xlim(0,1)
# axs[1,0].set_ylim(0,6)
# axs[1,1].set_xlim(-0.001,0.005)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()












































