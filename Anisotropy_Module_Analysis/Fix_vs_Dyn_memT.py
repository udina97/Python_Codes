#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:34:42 2026

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

nx = 64
ny = 64
nz = 64

lx = 2000*np.pi
ly = 2000*np.pi
lz = 1000

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
# zbl = 1099
uscale = 1
# wstar = 2.05
# Tscale = 290
# qstar = 0.24
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sims = ['neut_9ms_64_Dear_Tfix','neut_9ms_64_Dear_Tdyn']

step = 936000
time_ext = '30min'
bools = [False,False]

cases = ['F','D']

mom3D = dict(); tke3D = dict(); mom2D = dict(); aniso3D = dict()#; sc3D = dict(); sc2D = dict()

for i,case in enumerate(cases):
    aniso3D[case] = xr.open_dataarray(path+sims[i]+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
    mom3D[case] = xr.open_dataarray(path+sims[i]+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
    # sc3D[case] = xr.open_dataarray(path+sims[i]+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
    tke3D[case] = xr.open_dataarray(path+sims[i]+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
    mom2D[case] = xr.open_dataarray(path+sims[i]+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
    # sc2D[case] = xr.open_dataarray(path+sims[i]+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

aniso_C = dict(); tke_C = dict(); mom_C = dict()

for i,case in enumerate(cases):
    aniso_C[case] = read_checkpoint_aniso(path+sims[i]+'/output_checkpoint/',[step],nx,ny,nz)
    tke_C[case] = read_checkpoint_tke(path+sims[i]+'/output_checkpoint/',[step],nx,ny,nz)
    # mom2D[case] = read_checkpoint_sfc_L(path+sims[i]+'/output_checkpoint/',[step],nx,ny)
    mom_C[case] = read_checkpoint_mom(path+sims[i]+'/output_checkpoint/',[step],nx,ny,nz,bools[i])

#%%Compute anisotropy using RAV data

xB = dict(); yB = dict(); l3 = dict()


for i in cases:
    uu = mom3D[i][:,:,:,4] - mom3D[i][:,:,:,0]**2 - mom3D[i][:,:,:,16] + (2/3)*tke3D[i][:,:,:,1]
    vv = mom3D[i][:,:,:,5] - mom3D[i][:,:,:,1]**2 - mom3D[i][:,:,:,17] + (2/3)*tke3D[i][:,:,:,1]
    ww = wnode2uvpnode(mom3D[i][:,:,:,6] - mom3D[i][:,:,:,2]**2) - mom3D[i][:,:,:,18] + (2/3)*tke3D[i][:,:,:,1]
    uv = mom3D[i][:,:,:,13] - mom3D[i][:,:,:,0]*mom3D[i][:,:,:,1] - mom3D[i][:,:,:,19]
    uw = wnode2uvpnode(mom3D[i][:,:,:,14] - uvpnode2wnode(mom3D[i][:,:,:,0])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,20])
    vw = wnode2uvpnode(mom3D[i][:,:,:,15] - uvpnode2wnode(mom3D[i][:,:,:,1])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,21])
    
    [xB[i],yB[i],l3[i]] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%%Plot anisotropy slice

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

#%%Compute and plot shear stress

sgs = dict()

for i,case in enumerate(cases):
    # sgs[case] = - mom3D[case][:,:,:,16]- mom3D[case][:,:,:,17]- mom3D[case][:,:,:,18]
    sgs[case] = - aniso_C[case]['txx_new']- aniso_C[case]['tyy_new']- aniso_C[case]['tzz_new']


#%%Plot PDF of yB at the surface

fig,axs = plt.subplots(1,1,tight_layout=True)

y_pdf = gaussian_kde(yB['D'][:,:,0].flatten())
x_pdf = np.linspace(yB['D'][:,:,0].flatten().min(),yB['D'][:,:,0].flatten().max(),1000)
axs.plot(x_pdf,y_pdf(x_pdf),c='k',ls='-')

y_pdf = gaussian_kde(aniso3D['D'][:,:,0,1].flatten())
x_pdf = np.linspace(aniso3D['D'][:,:,0,1].flatten().min(),aniso3D['D'][:,:,0,1].flatten().max(),1000)
axs.plot(x_pdf,y_pdf(x_pdf),c='k',ls='--')

plt.show()


#%%Plot the dynamic memory time scale

from matplotlib.colors import LogNorm
T = ((3/aniso_C['D']['S_uvp'][:,:,:nz])*(zi/uscale))

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

p = axs[0].pcolormesh(x,z_uvp,T[:,ny//2,:].T,cmap='viridis',norm=LogNorm(vmin=T[T > 0].min(), vmax=T.max()))
cbar = plt.colorbar(p)
axs[0].set_xlabel(r'x [m]',fontsize=12)
axs[0].set_ylabel(r'z [m]',fontsize=12)
axs[0].set_title(r'T', fontsize=12)

z_lvl = 0
axs[0].axhline(z_lvl*dz + dz/2,c='r',ls='--')
y_pdf = gaussian_kde(T[:,:,z_lvl].flatten())
x_pdf = np.linspace(T[:,:,z_lvl].flatten().min(),T[:,:,z_lvl].flatten().max(),1000)
axs[1].plot(x_pdf,y_pdf(x_pdf),c='k',ls='-')
axs[1].set_xlabel(r'T [min]',fontsize=12)
axs[1].set_title(f'uvp level: {z_lvl}', fontsize=12)
axs[1].set_ylim(0)

plt.show()

#%%Plot velocity profiles

fig,axs = plt.subplots(1,1,tight_layout=True)
ls = ['-','--']
for i,case in enumerate(cases):
    axs.plot(np.mean(mom3D[case][:,:,:,0],axis=(0,1))*uscale,z_uvp,c='k',ls=ls[i])
    axs.plot(np.mean(mom3D[case][:,:,:,1],axis=(0,1))*uscale,z_uvp,c='r',ls=ls[i])
    axs.plot(np.mean(mom3D[case][:,:,:,2],axis=(0,1))*uscale,z_w,c='g',ls=ls[i])

axs.set_ylim(0,z_uvp[-1])
axs.set_ylabel(r'z [m]', fontsize=12)
axs.set_xlabel(r'$u_i$ [m/s]', fontsize=12)
plt.show()

#%%Compare Reynolds stresses

uu = dict(); vv = dict(); ww = dict(); uv = dict(); uw = dict(); vw = dict()
uu_otf = dict(); vv_otf = dict(); ww_otf = dict(); uv_otf = dict(); uw_otf = dict(); vw_otf = dict()
tke_rav = dict(); tke_otf = dict()

for i in cases:
    uu[i] = mom3D[i][:,:,:,4] - mom3D[i][:,:,:,0]**2 - mom3D[i][:,:,:,16] + (2/3)*tke3D[i][:,:,:,1]
    vv[i] = mom3D[i][:,:,:,5] - mom3D[i][:,:,:,1]**2 - mom3D[i][:,:,:,17] + (2/3)*tke3D[i][:,:,:,1]
    ww[i] = wnode2uvpnode(mom3D[i][:,:,:,6] - mom3D[i][:,:,:,2]**2) - mom3D[i][:,:,:,18] + (2/3)*tke3D[i][:,:,:,1]
    uv[i] = mom3D[i][:,:,:,13] - mom3D[i][:,:,:,0]*mom3D[i][:,:,:,1] - mom3D[i][:,:,:,19]
    uw[i] = wnode2uvpnode(mom3D[i][:,:,:,14] - uvpnode2wnode(mom3D[i][:,:,:,0])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,20])
    vw[i] = wnode2uvpnode(mom3D[i][:,:,:,15] - uvpnode2wnode(mom3D[i][:,:,:,1])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,21])
    
    uu_otf[i] = aniso_C[i]['uu_new'][:,:,:nz] - aniso_C[i]['u_new'][:,:,:nz]**2 - aniso_C[i]['txx_new'][:,:,:nz] + (2/3)*aniso_C[i]['tke_sgs_new'][:,:,:nz]
    vv_otf[i] = aniso_C[i]['vv_new'][:,:,:nz] - aniso_C[i]['v_new'][:,:,:nz]**2 - aniso_C[i]['tyy_new'][:,:,:nz] + (2/3)*aniso_C[i]['tke_sgs_new'][:,:,:nz]
    ww_otf[i] = aniso_C[i]['ww_new'][:,:,:nz] - aniso_C[i]['w_new'][:,:,:nz]**2 - aniso_C[i]['tzz_new'][:,:,:nz] + (2/3)*aniso_C[i]['tke_sgs_new'][:,:,:nz]
    uv_otf[i] = aniso_C[i]['uv_new'][:,:,:nz] - aniso_C[i]['u_new'][:,:,:nz]*aniso_C[i]['v_new'][:,:,:nz] - aniso_C[i]['txy_new'][:,:,:nz]
    uw_otf[i] = aniso_C[i]['uw_new'][:,:,:nz] - aniso_C[i]['u_new'][:,:,:nz]*aniso_C[i]['w_new'][:,:,:nz] - aniso_C[i]['txz_new'][:,:,:nz]
    vw_otf[i] = aniso_C[i]['vw_new'][:,:,:nz] - aniso_C[i]['v_new'][:,:,:nz]*aniso_C[i]['w_new'][:,:,:nz] - aniso_C[i]['tyz_new'][:,:,:nz]
    
    tke_rav[i] = uu[i] + vv[i] + ww[i]
    tke_otf[i] = uu_otf[i] + vv_otf[i] + ww_otf[i]


#%%Plot Reynolds stress

fig,axs = plt.subplots(2,2,sharey=True,sharex=True,figsize=(11,4))

for i,case in enumerate(cases):
    # p = axs[0,i].pcolormesh(x,z_uvp,np.mean(yB[case],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    # p = axs[1,i].pcolormesh(x,z_uvp,np.mean(aniso3D[case][:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[0,i].pcolormesh(x,z_uvp,tke_rav[case][:,ny//2,:].T,cmap='viridis',vmin=0,vmax=2)
    p = axs[1,i].pcolormesh(x,z_uvp,tke_otf[case][:,ny//2,:].T,cmap='viridis',vmin=0,vmax=2)
    axs[0,i].set_title(cases[i])


cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
fig.suptitle('yB')

plt.show()

#%%Compute yB approx

yBa_rav = dict(); yBa_otf = dict()

for i,case in enumerate(cases):
    yBa_rav[case] = (ww[case]/tke_rav[case])#*(3*np.sqrt(3)/4)
    yBa_otf[case] = (ww_otf[case]/tke_otf[case])#*(3*np.sqrt(3)/4)

fig,axs = plt.subplots(2,2,figsize=(10,5))

for i,case in enumerate(cases):
    axs[0,i].pcolormesh(x,z_uvp,yBa_rav[case][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=1)
    p = axs[1,i].pcolormesh(x,z_uvp,yBa_otf[case][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=1)

cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
plt.show()

#%%









































