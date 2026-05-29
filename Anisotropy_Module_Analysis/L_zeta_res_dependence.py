#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:59:23 2026

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from scipy.stats import gaussian_kde

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso
from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L
from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = [64,128,256]
ny = [64,128,256]
nz = [64,128,256]

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = [lx/nx[0],lx/nx[1],lx/nx[2]]
dy = [ly/ny[0],ly/ny[1],ly/ny[2]]
dz = [lz/nz[0],lz/nz[1],lz/nz[2]]
dt = 0.1

zi = 1000
uscale = 0.4
Tscale = 290
# x = np.arange(0,nx)*dx
# y = np.arange(0,ny)*dy
# z_uvp = np.arange(0,nz)*dz + dz/2
# z_w = np.arange(0,nz)*dz

Ug = 1
path = '/scratch/general/nfs1/u1450851/LES_Sims/RandTurbStats/'
sim = 'Aniso_1ms_A'

#%%Load data

cases = ['64','128','256']
RAV_times = ['63000','66000','69000','72000','75000','78000','81000','84000','87000','90000','93000','96000']
RAV = 10
mom3D = dict(); mom2D = dict(); sc3D = dict(); sc2D = dict()

for i in range(len(cases)):
    
    mom3D[cases[i]] = xr.open_dataarray(path+cases[i]+'/'+sim+'/data/Momentum3D/Data_Momentum_'+'30'+'min.nc').data
    mom2D[cases[i]] = xr.open_dataarray(path+cases[i]+'/'+sim+'/data/Momentum2D/Data_Momentum_2D_'+'30'+'min.nc').data
    sc3D[cases[i]] = xr.open_dataarray(path+cases[i]+'/'+sim+'/data/Scalar3D/Data_Scalar_'+'30'+'min.nc').data
    sc2D[cases[i]] = xr.open_dataarray(path+cases[i]+'/'+sim+'/data/Scalar2D/Data_Scalar_2D_'+'30'+'min.nc').data

#%%Compute the mean wind and heatflux

U = dict()
wT = dict()

for i in range(len(cases)):
    U[cases[i]] = np.sqrt(mom3D[cases[i]][:,:,:,0]**2 + mom3D[cases[i]][:,:,:,1]**2)

for i in range(len(cases)):
    wT[cases[i]] = sc3D[cases[i]][:,:,:,4] - uvpnode2wnode(sc3D[cases[i]][:,:,:,0])*mom3D[cases[i]][:,:,:,2] - sc3D[cases[i]][:,:,:,7]

#%%Profiles of T,U,wT

fig,axs = plt.subplots(1,3,tight_layout=True)

for i in range(len(cases)):
    axs[0].plot(np.mean(sc3D[cases[i]][:,:,:,0],axis=(0,1))*Tscale,np.arange(0,nz[i])*dz[i],label=cases[i])
for i in range(len(cases)):
    axs[1].plot(np.mean(U[cases[i]],axis=(0,1))*uscale,np.arange(0,nz[i])*dz[i],label=cases[i])
for i in range(len(cases)):
    axs[2].plot(np.mean(wT[cases[i]],axis=(0,1))*uscale*Tscale*1000,np.arange(0,nz[i])*dz[i],label=cases[i])

axs[0].set_xlabel(r"$\theta$ [K]",fontsize=14)
axs[1].set_xlabel(r"$\sqrt{u^2 + v^2}$ [m/s]",fontsize=14)
axs[2].set_xlabel(r"$\overline{w'\theta'}$ [W]",fontsize=14)
axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
fig.suptitle(RAV_times[RAV],fontsize=12)
for i in range(len(axs)):
    axs[i].legend()
plt.show()


#%%Plot L and zeta

L = dict()
zeta = dict()

for i in range(len(cases)):
    L[cases[i]] = sc2D[cases[i]][:,:,1]
    zeta[cases[i]] = 0.5*dz[i]/L[cases[i]]

#%%Plot PDFs

def compute_PDF(data,n=1000):
    from scipy.stats import gaussian_kde
    std = data.flatten().std()
    x_pdf = np.linspace(data.flatten().min() - 3*std, data.flatten().max() + 3*std, n)
    # x_pdf = np.linspace(min(data.flatten()),max(data.flatten()),n)
    kde = gaussian_kde(data.flatten())
    pdf = kde(x_pdf)
    return x_pdf,pdf

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(len(cases)):
    tmp = sc2D[cases[i]][:,:,-1]
    [x_pdf,pdf] = compute_PDF(tmp,10000)
    axs.plot((x_pdf),pdf,label=cases[i])
    # print(f'Expected value: {np.trapz(pdf*x_pdf,x_pdf)}')
    # print(f'Expected value: {np.trapz(pdf,x_pdf)}')
    print(f'Min: {tmp.min()}, Max: {tmp.max()}')
    print(f'Mean: {tmp.mean()}')

# axs.set_xscale('log')
# axs.invert_xaxis()
axs.legend()
plt.show()

#%%

for i in range(len(cases)):
    print(f'{np.std(sc2D[cases[i]][:,:,-1]*uscale*Tscale*1000)}')













































































