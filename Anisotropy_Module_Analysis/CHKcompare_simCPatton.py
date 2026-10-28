#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:11:26 2026

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

lx = 2000*np.pi #5120 #2000*np.pi
ly = 2000*np.pi #5120 #2000*np.pi
lz = 2000 #2048 #2000

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000 #1024 #1000
# zbl = 1099
uscale = 1
# wstar = 2.05
Tscale = 290 #300 #290
# qstar = 0.24
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load surface checkpoint/istantaneous fields 

# sim = ['simC_Patton2011','simC_Patton2011_aniso_noiter_ainit']#,'simC_Patton2011_aniso_noiter','simC_Patton2011_aniso_noiter_ainit']
sim = ['p800_noaniso_128','p800_aniso_128']
path = '/scratch/general/nfs1/u1450851/LES_Sims/'
mom3D = dict(); mom2D = dict(); aniso = dict(); tke = dict();

cases = ['c','a']

step = 162000

aniso['c'] = read_checkpoint_aniso(path+sim[0]+'/output_checkpoint/',[step],nx,ny,nz)
tke['c'] = read_checkpoint_tke(path+sim[0]+'/output_checkpoint/',[step],nx,ny,nz)
mom2D['c'] = read_checkpoint_sfc_L(path+sim[0]+'/output_checkpoint/',[step],nx,ny)
mom3D['c'] = read_checkpoint(path+sim[0]+'/output_checkpoint/',[step],nx,ny,nz,False,False)

aniso['a'] = read_checkpoint_aniso(path+sim[1]+'/output_checkpoint/',[step],nx,ny,nz)
tke['a'] = read_checkpoint_tke(path+sim[1]+'/output_checkpoint/',[step],nx,ny,nz)
mom2D['a'] = read_checkpoint_sfc_L(path+sim[1]+'/output_checkpoint/',[step],nx,ny)
mom3D['a'] = read_checkpoint(path+sim[1]+'/output_checkpoint/',[step],nx,ny,nz,False,False)

#%%Plot anisotropy vertical slice

fig,axs = plt.subplots(1,2,sharey=True,figsize=(11,4))

# p = axs[0].pcolormesh(x,z_uvp,np.mean(aniso['c']['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x,z_uvp,np.mean(aniso['a']['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
p = axs[0].pcolormesh(x,z_uvp,aniso['c']['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[1].pcolormesh(x,z_uvp,aniso['a']['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[0].set_title(sim[0])
axs[1].set_title(sim[1])

cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
fig.suptitle('yB')

plt.show()

#%%Plot xB vertical slice

fig,axs = plt.subplots(1,2,sharey=True,figsize=(11,4))

p = axs[0].pcolormesh(x,z_uvp,np.mean(aniso['c']['xB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=1)
axs[1].pcolormesh(x,z_uvp,np.mean(aniso['a']['xB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=1)
axs[0].set_title(sim[0])
axs[1].set_title(sim[1])

cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
fig.suptitle('xB')

plt.show()

#%%Compute anisotropy 

xB = dict(); yB = dict(); l3 = dict()

for i in cases:
    uu = aniso[i]['uu_new'][:,:,:nz] - aniso[i]['u_new'][:,:,:nz]**2 #- aniso[i]['txx_new'][:,:,:nz] + (2/3)*aniso[i]['tke_sgs_new'][:,:,:nz]
    vv = aniso[i]['vv_new'][:,:,:nz] - aniso[i]['v_new'][:,:,:nz]**2 #- aniso[i]['tyy_new'][:,:,:nz] + (2/3)*aniso[i]['tke_sgs_new'][:,:,:nz]
    ww = aniso[i]['ww_new'][:,:,:nz] - aniso[i]['w_new'][:,:,:nz]**2 #- aniso[i]['tzz_new'][:,:,:nz] + (2/3)*aniso[i]['tke_sgs_new'][:,:,:nz]
    uv = aniso[i]['uv_new'][:,:,:nz] - aniso[i]['u_new'][:,:,:nz]*aniso[i]['v_new'][:,:,:nz] #- aniso[i]['txy_new'][:,:,:nz]
    uw = aniso[i]['uw_new'][:,:,:nz] - aniso[i]['u_new'][:,:,:nz]*aniso[i]['w_new'][:,:,:nz] #- aniso[i]['txz_new'][:,:,:nz]
    vw = aniso[i]['vw_new'][:,:,:nz] - aniso[i]['v_new'][:,:,:nz]*aniso[i]['w_new'][:,:,:nz] #- aniso[i]['tyz_new'][:,:,:nz]
    
    [xB[i],yB[i],l3[i]] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)


#%%Plot anisotropy vertical slice

fig,axs = plt.subplots(2,2,sharey=True,sharex=True,figsize=(11,4))

for i,case in enumerate(cases):
    # p = axs[0,i].pcolormesh(x,z_uvp,np.mean(yB[case],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    # p = axs[1,i].pcolormesh(x,z_uvp,np.mean(aniso3D[case][:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[0,i].pcolormesh(x,z_uvp,yB[case][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[1,i].pcolormesh(x,z_uvp,aniso[case]['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    # p = axs[0,i].pcolormesh(x,y,yB[case][:,:,0].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    # p = axs[1,i].pcolormesh(x,y,aniso[case]['yB'][:,:,0].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    axs[0,i].set_title(cases[i])


cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
fig.suptitle('yB')

plt.show()

#%%Plot PDF of yB at the surface

fig,axs = plt.subplots(1,1,tight_layout=True)

for i,case in enumerate(cases):
    if case == 'a':
        ls = '-'
    else:
        ls = '--'
    y_pdf = gaussian_kde(aniso[case]['yB'][:,:,0].flatten())
    x_pdf = np.linspace(aniso[case]['yB'][:,:,0].flatten().min(),aniso[case]['yB'][:,:,0].flatten().max(),1000)
    axs.plot(x_pdf,y_pdf(x_pdf),c='k',ls=ls)

plt.show()

#%%Surface temperature

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,mom2D['a']['sfcVAL'].T,cmap='hot_r')
cbar = plt.colorbar(p)
plt.show()

#%%Plot vertical velocity horizontal plane

fig,axs = plt.subplots(1,2,tight_layout=True)

for i,case in enumerate(cases):
    axs[i].pcolormesh(x,y,mom3D[case]['w'][:,:,4].T,cmap='bwr',vmin=-1,vmax=1)

plt.show()

#%%

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))

w_levels = [0.5,1]  # adjust to your velocity scale

for i, case in enumerate(cases):
    for row, yB_field in enumerate([xB[case], aniso[case]['xB']]):
        ax = axs[row, i]

        # anisotropy at the surface
        p = ax.pcolormesh(x, y, yB_field[:, :, 0].T,cmap=cmap, vmin=0, vmax=1,shading="auto")

        # vertical velocity contours at the matching vertical location
        w_surface = mom3D[case]['w'][:, :, 4].T
        cs = ax.contour(x, y, w_surface,levels=w_levels,colors="k",linewidths=0.9)

        # Negative = dashed; positive = solid
        for level, collection in zip(cs.levels, cs.collections):
            collection.set_linestyle("--" if level < 0 else "-")

        ax.clabel(cs, fmt="%0.1f", fontsize=7)
        # ax.set_aspect("equal")
        ax.set_title(case)

axs[0, 0].set_ylabel("y")
axs[1, 0].set_ylabel("y")
for ax in axs[1]:
    ax.set_xlabel("x")

cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, label="yB")
fig.suptitle("Surface anisotropy with vertical-velocity contours")
plt.show()

































