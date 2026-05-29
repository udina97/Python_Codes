#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:31:58 2024

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
import scipy.io
import h5py

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

#%% Import Giulia's data

save_path = '/uufs/chpc.utah.edu/common/home/u1450851/Giulia_yB/'
pathOUT= save_path+'Figures/'

import pickle

# Loading the dictionary from the pickle file
with open(save_path+'g800_9.pickle', 'rb') as file:
    g800_9 = pickle.load(file)

with open(save_path+'g800_i_9.pickle', 'rb') as file:
    g800_i_9 = pickle.load(file)

with open(save_path+'empty.pickle', 'rb') as file:
    empty = pickle.load(file)
    
#%%Some parametes
Nx = 256
Ny = 256
Nz = 256

Lx = 2000*np.pi
Ly = 2000*np.pi
Lz = 1000

zi = 1000

dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

#%%Pcolor plot test - yslice

yslice = 100

fig,axs = plt.subplots(1,3,figsize=(12,4),tight_layout=True)
p1 = axs[0].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Nz)*dz/(39),empty['u'][:,yslice,:].T,cmap='coolwarm',shading='gouraud',vmin=0,vmax=20)
p2 = axs[1].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Nz)*dz/(39),empty['v'][:,yslice,:].T,cmap='coolwarm',shading='gouraud',vmin=0,vmax=20)
p3 = axs[2].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Nz)*dz/(39),empty['w'][:,yslice,:].T,cmap='coolwarm',shading='gouraud',vmin=-0.3,vmax=0.3)
# axs[0].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs[1].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs[2].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs[0].axhline(canopyH/canopyH,ls='--',c='k'),axs[1].axhline(canopyH/canopyH,ls='--',c='k')
axs[0].set_xlabel(r'$x/z_i$'), axs[1].set_xlabel(r'$x/z_i$'), axs[2].set_xlabel(r'$x/z_i$')
axs[0].set_ylabel(r'$z/h_c$')
axs[0].set_title(r'U/$u_{scale}$ (yslice)'),axs[1].set_title(r'V/$u_{scale}$ (yslice)'), axs[2].set_title(r'W/$u_{scale}$ (yslice)')
cbar1 = plt.colorbar(p1); cbar2 = plt.colorbar(p2); cbar3 = plt.colorbar(p3)
fig.suptitle(f'yslice = {yslice*dy}m')

# plt.savefig(pathOUT+'empty_uvw_yslice.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%Pcolor test plot z-slice

zslice = 25

fig,axs = plt.subplots(1,3,figsize=(12,4),tight_layout=True)
p1 = axs[0].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Ny)*dy/zi,empty['u'][:,:,zslice].T,cmap='coolwarm',shading='gouraud')
p2 = axs[1].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Ny)*dy/zi,empty['v'][:,:,zslice].T,cmap='coolwarm',shading='gouraud')
p3 = axs[2].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Ny)*dy/zi,empty['w'][:,:,zslice].T,cmap='coolwarm',shading='gouraud')

axs[0].set_xlabel(r'$x/z_i$'), axs[1].set_xlabel(r'$x/z_i$'), axs[2].set_xlabel(r'$x/z_i$')
axs[0].set_ylabel(r'$y/z_i$'), axs[1].set_ylabel(r'$y/z_i$'), axs[2].set_ylabel(r'$y/z_i$')
axs[0].set_title(r'U/$u_{scale}$ (zslice)'),axs[1].set_title(r'V/$u_{scale}$ (zslice)'), axs[2].set_title(r'W/$u_{scale}$ (zslice)')
cbar1 = plt.colorbar(p1); cbar2 = plt.colorbar(p2); cbar3 = plt.colorbar(p3)
fig.suptitle(f'zslice = {zslice*dz}m')

# plt.savefig(pathOUT+'empty_uvw_zslice.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%Pcolor zslice plot of velocity magnitude

# tmpUsqrt = np.sqrt(g800_9['u']**2 + g800_9['v']**2)
# tmpUsqrt = np.sqrt(g800_i_9['u']**2 + g800_i_9['v']**2)
tmpUsqrt = np.sqrt(empty['u']**2 + empty['v']**2)

zslice = 26

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
p1 = axs.pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Ny)*dy/zi,tmpUsqrt[:,:,zslice].T,cmap='coolwarm',shading='gouraud')
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$y/z_i$')
axs.set_title(r'U/$u_{scale}$ (zslice)')
cbar1 = plt.colorbar(p1)
fig.suptitle(f'zslice = {zslice*dz}m')

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Umag100m_empty.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()

#%%Interpolation functions

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_c[:, :, -2]
  return phi_c

#%% Compute Anisotropy yB

# Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
#                         dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
#                         'Rzz','Rxy','Rxz','Ryz']})

# Rstress = ReynoldsStressUVP(Nx,Ny,Nz,empty['u'],empty['v'],empty['w'],empty['uu'],\
#                           empty['vv'],empty['ww'],empty['uv'],empty['uw'],empty['vw'])#,\
#                               # data_tavg['txx'],data_tavg['tyy'],data_tavg['tzz'],data_tavg['txy'],data_tavg['txz'],data_tavg['tyz'])

        
# [xB,yB,AnisType_1D,lambda3] = Anisotropy_Clustering(Nx,Ny,Nz,Rstress)

#%%Compute Anisotropy Diag

# Rstress_d = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
#                         dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
#                         'Rzz','Rxy','Rxz','Ryz']})

# Rstress_d = ReynoldsStressUVP(Nx,Ny,Nz,empty['u'],empty['v'],empty['w'],empty['uu'],\
#                           empty['vv'],empty['ww'],empty['uv'],empty['uw'],empty['vw'])

# Rstress_d.data[:,:,:,3] = np.zeros((Nx,Ny,Nz),'d',order='F')
# Rstress_d.data[:,:,:,4] = np.zeros((Nx,Ny,Nz),'d',order='F')
# Rstress_d.data[:,:,:,5] = np.zeros((Nx,Ny,Nz),'d',order='F')

# [xB_d,yB_d,AnisType_1D_d,lambda3_d] = Anisotropy_Clustering(Nx,Ny,Nz,Rstress_d)

#%%
# aniso_full = dict()
# aniso_diag = dict()

# aniso_full['xB'] = xB
# aniso_full['yB'] = yB
# aniso_full['type'] = AnisType_1D
# aniso_full['lambda3'] = lambda3

# aniso_diag['xB'] = xB_d
# aniso_diag['yB'] = yB_d
# aniso_diag['type'] = AnisType_1D_d
# aniso_diag['lambda3'] = lambda3_d

# with open(save_path+'empty_aniso_full.pickle', 'wb') as file:
#     pickle.dump(aniso_full,file)
    
# with open(save_path+'empty_aniso_diag.pickle', 'wb') as file:
#     pickle.dump(aniso_diag,file)

#%%Load anisotropy data

import pickle

# Saving the dictionary to a pickle file
with open(save_path+'g800_9_aniso_full.pickle', 'rb') as file:
    g800_9_aniso_full = pickle.load(file)
    
with open(save_path+'g800_9_aniso_diag.pickle', 'rb') as file:
    g800_9_aniso_diag = pickle.load(file)
    
with open(save_path+'g800_i_9_aniso_full.pickle', 'rb') as file:
    g800_i_9_aniso_full = pickle.load(file)
    
with open(save_path+'g800_i_9_aniso_diag.pickle', 'rb') as file:
    g800_i_9_aniso_diag = pickle.load(file)
    
with open(save_path+'empty_aniso_full.pickle', 'rb') as file:
    empty_aniso_full = pickle.load(file)
    
with open(save_path+'empty_aniso_diag.pickle', 'rb') as file:
    empty_aniso_diag = pickle.load(file)
    
#%%Pcolor of YB at yslice

tmpYB = copy.deepcopy(np.reshape(g800_9_aniso_diag['yB'],(Nx,Ny,Nz)))
cmap = ColorAnisotropy()

yslice = 125

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
p1 = axs.pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Nz)*(dz/zi)/(39/zi),tmpYB[:,yslice,:].T,cmap=cmap,shading='gouraud',vmin=0,vmax=np.sqrt(3)/2)
axs.axhline(1,c='k',ls='--')
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$z/h_C$')
axs.set_title(r'$y_{B,d}$ (yslice)')
cbar1 = plt.colorbar(p1)
fig.suptitle(f'yslice = {yslice*dy}m')

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'yB_d_g800.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()
    
#%%
yslice = 100
cmap = ColorAnisotropy() 
tmb_yB = empty_aniso_full['yB'][:,yslice,:]
tmb_yB_d = empty_aniso_diag['yB'][:,yslice,:]

fig, axs = plt.subplots(1,2,figsize=(10,5),tight_layout=True)

p1 = axs[0].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Nz)*dz/(39), tmb_yB.T,cmap = cmap,vmin = 0, vmax = np.sqrt(3)/2)
p2 = axs[1].pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Nz)*dz/(39), tmb_yB_d.T,cmap = cmap,vmin = 0, vmax = np.sqrt(3)/2)

axs[0].set_xlabel(r'$x/z_i$'), axs[1].set_xlabel(r'$x/z_i$')
axs[0].set_ylabel(r'$z/h_c$')
axs[0].set_title(r'yB'),axs[1].set_title(r'yB_d')
cbar1 = plt.colorbar(p1); cbar2 = plt.colorbar(p2)
fig.suptitle(f'yslice = {yslice*dy}m')

# plt.savefig(pathOUT+'empty_yB_vs_yBdiag_yslice.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%Ratio plot and histogram

tmb_yB = empty_aniso_full['yB']
tmb_yB_d = empty_aniso_diag['yB']
ratio = tmb_yB_d/tmb_yB
# # ratio = yB_DIAG - yB

yslice = 100

fig,axs = plt.subplots(1,1,tight_layout=True)
p = axs.pcolormesh(np.arange(0,Nx)*dx/zi,np.arange(0,Nz)*dz/(39),ratio[:,yslice,:].T,cmap=cmap,vmin=0,vmax=3)
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$z/h_C$')
axs.set_title(f'$y = {yslice*dy}m$')
cb = plt.colorbar(p,label=r'$\frac{y_{B,DIAG}}{y_{B}}$')

# plt.savefig(pathOUT+'empty_yBratio_yslice.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%

lv1 = 10
lv2 = 200

ratio_subset = ratio[:,:,lv1:lv2].flatten()

fig, axs = plt.subplots(1,1,figsize=(10,5),tight_layout=True)

axs.hist(ratio_subset,bins=100,density=False,stacked=False,cumulative=False)
axs.axvline(np.nanmean(ratio_subset),c='k',ls='--',label='Mean')
axs.axvline(np.nanmedian(ratio_subset),c='k',ls='-.',label='Median')
axs.axvline(np.nanmean(ratio_subset)-np.std(ratio_subset),c='k',ls=':',label='+/- STD')
axs.axvline(np.nanmean(ratio_subset)+np.std(ratio_subset),c='k',ls=':')
axs.plot([],[],' ',label=f'Mean = {round(np.nanmean(ratio_subset),2)}')
axs.plot([],[],' ',label=f'Median = {round(np.nanmedian(ratio_subset),2)}')
# axs.axhline(1,c='k',ls='--')
axs.set_xlabel(r'$\frac{y_{B,DIAG}}{y_{B}}$',fontsize=12)
axs.set_ylabel(r'Counts',fontsize=12)
axs.set_title(f'All points between {lv1*dz}m and {lv2*dz}m')
axs.legend()
# axs.set_xlim(0,0.3)

# plt.savefig(pathOUT+'empty_yBratio_hist.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()








