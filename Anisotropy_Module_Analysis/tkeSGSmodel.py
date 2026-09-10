#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:32:04 2026

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
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Simulation parameters

nx = 32
ny = 32
nz = 32

lx = 2*np.pi
ly = 2*np.pi
lz = 1

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.45
Tscale = 290
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load surface checkpoint/istantaneous fields 
# sim = 'yb_test_v2'

sim = 'test_planavg'

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

step = 50000   

checkpnt = read_checkpoint_mom(path+sim+'/output_checkpoint/',[step],nx,ny,nz)
# checkpnt_sfc = read_checkpoint_sfc_L(path+sim+'/output_checkpoint/',[step],nx,ny)

# NumVariables = 27
# NumIt = 1
# NumFiles = 1
# StartTime = 1300
# anisotropy_flag = False
# data = xr.DataArray(np.ones(shape = (nx,ny,nz,NumVariables),order='F'),\
#                         dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
#                         'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                         'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                         'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                         'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']}) 
    
# vardata = xr.DataArray(np.ones(shape = (nx,ny,nz,NumVariables),order='F'),\
#                     dims=('x', 'y','z','variable'), coords = {'variable':['avgU','avgV',\
#                     'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                     'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                     'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                     'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']})

# data = load_momentum(nx,ny,nz,NumVariables,NumIt,NumFiles,StartTime,path+sim+'/output_RAV/',vardata,anisotropy_flag)

#%%Plot anisotropy invariant

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x,z_uvp,np.median(checkpnt['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs.pcolormesh(x,z_uvp,checkpnt['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs.pcolormesh(x,z_w,checkpnt['yB'][nx//2,:,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs.pcolormesh(x,y,checkpnt['yB'][:,:,15].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

plt.show()

#%%Plot subgrid scale tke

k_sgs = checkpnt["k_sgs"][:,:,:nz]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x,z_w,np.median(k_sgs,axis=(1)).T,cmap=cmap)
# axs.pcolormesh(x,z_w,k_sgs[nx//2,:,:].T,cmap=cmap)
# axs.pcolormesh(x,y,k_sgs[:,:,5].T,cmap=cmap)

plt.show()

#%%PLot velocity components

fig,axs = plt.subplots(1,3,tight_layout=True,figsize = (10,3))

axs[0].pcolormesh(x,z_uvp,checkpnt['u'][:,ny//2,:nz].T,cmap='viridis')
axs[1].pcolormesh(x,z_uvp,checkpnt['v'][:,ny//2,:nz].T,cmap='viridis')
axs[2].pcolormesh(x,z_w,checkpnt['w'][:,ny//2,:nz].T,cmap='viridis')

# axs[0].pcolormesh(y,z_uvp,checkpnt['u'][nx//2,:,:nz].T,cmap='viridis')
# axs[1].pcolormesh(y,z_uvp,checkpnt['v'][nx//2,:,:nz].T,cmap='viridis')
# axs[2].pcolormesh(y,z_w,checkpnt['w'][nx//2,:,:nz].T,cmap='viridis')

# axs[0].pcolormesh(x,y,checkpnt['u'][:,:,5].T,cmap='viridis')
# axs[1].pcolormesh(x,y,checkpnt['v'][:,:,5].T,cmap='viridis')
# axs[2].pcolormesh(x,y,checkpnt['w'][:,:,5].T,cmap='viridis')

axs[0].set_title('u')
axs[1].set_title('v')
axs[2].set_title('w')

plt.show()

#%%

L_neg = checkpnt['L'][(checkpnt['L']<0)]
L_pos = checkpnt['L'][(checkpnt['L']>0)]

print(len(L_neg))
print(len(L_pos))

fig,axs = plt.subplots(1,2,tight_layout=True)
kde = gaussian_kde((L_neg).flatten())
x_pdf = np.linspace(min(L_neg.flatten()), max(L_neg.flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist(L_neg.flatten(), bins=100, density=True, alpha=0.4, label="Histogram")
axs[0].plot(x_pdf, pdf, 'r-', label="KDE PDF")

kde = gaussian_kde((L_pos).flatten())
x_pdf = np.linspace(min(L_pos.flatten()), max(L_pos.flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist(L_pos.flatten(), bins=100, density=True, alpha=0.4, label="Histogram")
axs[1].plot(x_pdf, pdf, 'r-', label="KDE PDF")

axs[0].set_xlim(-1,0)
axs[1].set_xlim(0,1)

plt.show()

#%%

txx = data[:,:,:,16].data
tyy = data[:,:,:,17].data
tzz = data[:,:,:,18].data
txy = data[:,:,:,19].data
txz = data[:,:,:,20].data
tyz = data[:,:,:,21].data

#%%Compute Reynolds stresses using the on-the-fly averages

uu = checkpnt['uu_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['u_new'][:,:,:nz] #+ (2/3)*checkpnt['tke_sgs']
vv = checkpnt['vv_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz] #+ (2/3)*checkpnt['tke_sgs']
ww = checkpnt['ww_new'][:,:,:nz] - checkpnt['w_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz] #+ (2/3)*checkpnt['tke_sgs']
uv = checkpnt['uv_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz] #- checkpnt['txy'][:,:,:nz]
uw = checkpnt['uw_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz] #- checkpnt['txz'][:,:,:nz]
vw = checkpnt['vw_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz] #- checkpnt['tyz'][:,:,:nz]

txx = checkpnt['txx'][:,:,:nz]
tyy = checkpnt['tyy'][:,:,:nz]
tzz = checkpnt['tzz'][:,:,:nz]
txy = checkpnt['txy'][:,:,:nz]
txz = checkpnt['txz'][:,:,:nz]
tyz = checkpnt['tyz'][:,:,:nz]

tke_res = 0.5*(uu + vv + ww)
tke_sgs = checkpnt['tke_sgs']
tke_tot = tke_res + tke_sgs

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu + (2/3)*tke_sgs,vv + (2/3)*tke_sgs,ww + (2/3)*tke_sgs,uv + txy,uw + txz,vw + tyz)

# b11 = uu/(2*tke)-1/3
# b22 = vv/(2*tke)-1/3
# b33 = ww/(2*tke)-1/3

#%%Check yB and xB are within physical interval

if np.any(yB<0):
    print('Negative values of yB')
if np.any(yB>np.sqrt(3)/2):
    print('yB exceeds max value')
if np.any(xB<0):
    print('Negative values of xB')
if np.any(xB>1):
    print('xB exceeds max value')

#%%Tke SGS model terms PDFs

L_trace = checkpnt['trace_L_sgs'][:,:,:nz]
denom = checkpnt['denom_tke_sgs'][:,:,:nz]

fig,axs = plt.subplots(1,2,tight_layout=True)
kde = gaussian_kde((L_trace).flatten())
x_pdf = np.linspace(min(L_trace.flatten()), max(L_trace.flatten()), 1000)
pdf = kde(x_pdf)
axs[0].hist(L_trace.flatten(), bins=100, density=True, alpha=0.4, label="Histogram")
axs[0].plot(x_pdf, pdf, 'r-', label="KDE PDF")

kde = gaussian_kde((denom).flatten())
x_pdf = np.linspace(min(denom.flatten()), max(denom.flatten()), 1000)
pdf = kde(x_pdf)
axs[1].hist(denom.flatten(), bins=100, density=True, alpha=0.4, label="Histogram")
axs[1].plot(x_pdf, pdf, 'r-', label="KDE PDF")


plt.show()

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(y,z_w,denom[nx//2,:,:].T,cmap='viridis')

plt.show()

#%%

tmp = wnode2uvpnode(checkpnt['trace_L_sgs']/checkpnt['denom_tke_sgs'])[:,:,:nz]

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.mean(tke_sgs,axis=(0,1)),z_uvp)
axs.set_xlim(0,2)
plt.show()














































