#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:40:31 2024

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
import os
import sys
import copy
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from read_checkpoint_SC import read_checkpoint

from read_checkpoint_SC import read_checkpoint_sfc

#%%#inputs

sim = 'diurnal_c_aniso_L3D'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/'

nx = 128
ny = 128
nz = 384
lx = 1*np.pi
ly = 1*np.pi
lz = 3
dx = lx/nx
dy = ly/ny
dz = lz/nz

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.075 #05; %0.000005;
zi = 3000.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = path

#%%Import variables


var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']
    
varS = ['avgT','avgT2','avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus','avg_ds']
    
dataM = xr.open_dataarray(path+'Data_Momentum.nc')

data_mom = dict()
data_sc = dict()

for i in range(len(var)):
    data_mom[var[i]] = dataM.data[:,:,:,i]
    
# for i in range(len(varS)):
#     data_sc[varS[i]] = dataS.data[:,:,:,i]
    
#%%

checkpnt = read_checkpoint('/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_checkpoint/',[1080000],nx,ny,nz)
# checkpnt_sfc = read_checkpoint_sfc('/scratch/general/nfs1/u1450851/LES_Sims/anisotropy/output_checkpoint/',[180000],nx,ny)

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_h[:, :, -1]
  return phi_c

#%%Plot profiles of txz for RAV and checkpoint

Rxz = wnode2uvpnode(data_mom['avgUW'] - data_mom['avgW']*uvpnode2wnode(data_mom['avgU']))
Rxz_c = checkpnt['uw_new'] - checkpnt['u_new']*checkpnt['w_new']

fig,axs = plt.subplots(1,1)
axs.plot(np.mean(-data_mom['avgtxz'],axis=(0,1)),np.arange(0,nz)*dz,c='k')
axs.plot(np.mean(-checkpnt['txz_new'][:,:,:-1],axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='g')
axs.plot(np.mean(Rxz,axis=(0,1)),np.arange(0,nz)*dz,c='k')
axs.plot(np.mean(Rxz_c[:,:,:-1],axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='g')
plt.show()

#%% Calculate the Reynolds Stresses:
    
items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'prod_dudz', 'canopy', 'dissip', 'sum')

# Initialization
terms_ptb = dict()
terms_drv = dict()
terms_bdg = dict.fromkeys(items_bdg, None)

wn_x = 2*np.pi*np.fft.rfftfreq(nx, dx)
wn_y = 2*np.pi*np.fft.rfftfreq(ny, dy)

# Interpolate u and v to w node
u_h = uvpnode2wnode(data_mom['avgU'])
v_h = uvpnode2wnode(data_mom['avgV'])


terms_ptb['u2_t'] = data_mom['avgU2'] - data_mom['avgU']**2
terms_ptb['uv_t'] = data_mom['avgUV'] - data_mom['avgU']*data_mom['avgV']
# terms_ptb['uw_t'] = data_mom['avgUW'] - u_h*data_mom['avgW']
terms_ptb['uw_t'] = wnode2uvpnode(data_mom['avgUW']) - data_mom['avgU']*wnode2uvpnode(data_mom['avgW'])
# terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t'])

terms_ptb['v2_t'] = data_mom['avgV2'] - data_mom['avgV']**2
terms_ptb['vw_t'] = data_mom['avgVW'] - v_h*data_mom['avgW']
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t'])

terms_ptb['w2_t'] = data_mom['avgW2'] -data_mom['avgW']**2
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t'])

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2
terms_ptb['tke_SGS'] = (data_mom['avgtxx']+data_mom['avgtyy']+data_mom['avgtzz']) / 2

#%%#%%Compute Reynolds stresses using the on-the-fly averages

uu = checkpnt['uu_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['u_new'][:,:,:nz]
vv = checkpnt['vv_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]
ww = checkpnt['ww_new'][:,:,:nz] - checkpnt['w_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]
uv = checkpnt['uv_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]
uw = checkpnt['uw_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]
vw = checkpnt['vw_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]

#%%
# tmp = terms_ptb['uw_t'] - (uw)
tmp = (data_mom['avgU']) - (checkpnt['u_new'][:,:,:nz])
# tmp = (wnode2uvpnode(data_mom['avgUW']) - (data_mom['avgU'])*wnode2uvpnode(data_mom['avgW'])) - (checkpnt['uw_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz])
# tmp = wnode2uvpnode(data_mom['avgW2'])

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz-1)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 20 #int(nz/2)
zslice = 30

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:-1].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()


#%%

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,terms_ptb['u2_t'],terms_ptb['v2_t'],terms_ptb['w2_t'],terms_ptb['uv_t'],terms_ptb['uw_t'],terms_ptb['vw_t'])
[xB_c,yB_c,lamba3_c] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%% Pcolor of Anisotropy

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz-1)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 20 #int(nz/2)
zslice = 20

# tmp = data_mom['avgYB'] - yB_c
tmp = yB - yB_c
# tmp = yB_c

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

# plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:-1].T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()


















































