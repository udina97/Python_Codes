#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:42:35 2025

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf

#%%Paths to data

# pathFig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/'
directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'

sim = 'simflat_256x256x384_out5hr_v2'
# sim = 'simATTO_256x256x384_full_out5hr'
# sim = 'simbicheng_hill_256x256x384_out5hr'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/' 

path_sim = pathOUT + sim + '/'


#%%Loading Topo data

zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py

tpath_sim = directory + sim + '/output/'
# tpath_sin = directory + sin + '/output/'
# tpath_atto = directory + atto + '/output/'

# Read parameter file for ta1_field
with open(tpath_sim + 'ta1_field/parameters.txt', 'r') as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

Nx = int(param[0])
Ny = int(param[1])
Nz = int(param[2])
Lx = param[3]
Ly = param[4]
Lz = param[5]
dx = param[6]
dy = param[7]
dz = param[8]
mpiProc = int(param[11])
nzTot = Nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

# Assuming you have a function build_phi to load phi data from a file
phi_flat = build_phi(tpath_sim + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf, iintf = build_intf(phi_flat, dz)

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
Nz_SLayer = 300 #int(Nz/2)

zeds = np.zeros((Nx,Ny,Nz))

for i in range(0,Nx):
    for j in range(0,Ny):
        zeds[i,j,:] = np.arange(0,Nz)*(dz*zi)
    
dist = copy.deepcopy(zeds);

for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi

#%%Load TKE data

TKE = np.load(path_sim + 'TKE_terms.npy',allow_pickle='TRUE').item()
TKE_v2 = np.load(path_sim + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()

#%%Plot TKE profiles

# var_bdg = ['adv_h', 'adv_v', 'advSGS_h', 'advSGS_v', 'uturb_h', 'uturb_v', 'pturb_h', 'pturb_v', 'dissip', 'canopy', 
#             'prod_h', 'prod_v','adv','res','ptrans','ttrans','totdis','prod']#,'prod_dudz','prod_dvdz','prod_dwdz']

fig,axs = plt.subplots(1,1)
axs.plot(np.mean(TKE['ttrans'],axis=(0,1)),(np.arange(0,nzTot)*dz+dz/2)/(39/zi),c='k')
axs.plot(np.mean(TKE_v2['ttrans'],axis=(0,1)),(np.arange(0,nzTot)*dz+dz/2)/(39/zi),c='r')
# axs.plot(np.mean(TKE_v2['res']-TKE['res'],axis=(0,1)),(np.arange(0,nzTot)*dz+dz/2)/(39/zi),c='r')
axs.axhline(1,c='k',ls='--')
axs.set_xlabel('TKE term')
axs.set_ylabel('z')
axs.set_ylim(0,nzTot*dz/(39/zi))

plt.show()

#%%Pcolor slices

slc = 125
var = 'ttrans'

fig,axs = plt.subplots(3,1,figsize=(6,12))
p1=axs[0].pcolormesh(np.arange(0,Nx)*dx,(np.arange(0,nzTot-5)*dz+dz/2)/(39/zi),TKE[var][:,slc,5:].T,cmap='jet',vmin=-20,vmax=20)
p2=axs[1].pcolormesh(np.arange(0,Nx)*dx,(np.arange(0,nzTot-5)*dz+dz/2)/(39/zi),TKE_v2[var][:,slc,5:].T,cmap='jet',vmin=-20,vmax=20)
p3=axs[2].pcolormesh(np.arange(0,Nx)*dx,(np.arange(0,nzTot-5)*dz+dz/2)/(39/zi),(TKE_v2[var]-TKE[var])[:,slc,5:].T,cmap='jet')

axs[2].set_xlabel('x')
for i in range(len(axs)):
    axs[i].set_ylabel('z')
    axs[i].plot(np.arange(0,Nx)*dx,(intf[:,slc]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    axs[i].plot(np.arange(0,Nx)*dx,(intf[:,slc]-z_shift)/(39/zi),ls='--',c='k')

plt.colorbar(p1)
plt.colorbar(p2)
plt.colorbar(p3)
    
plt.show()


































































