#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 06:04:17 2025

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

#%%

pathFig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/'
directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'

flat = 'simflat_256x256x384_out5hr_v2'
atto = 'simATTO_256x256x384_full_out5hr'
sin = 'simbicheng_hill_256x256x384_out5hr'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/' 

path_flat = pathOUT + flat + '/'
path_sin = pathOUT + sin + '/'
path_atto = pathOUT + atto + '/'

#%%Loading Topo data

zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py

tpath_flat = directory + flat + '/output/'
tpath_sin = directory + sin + '/output/'
tpath_atto = directory + atto + '/output/'

# Read parameter file for ta1_field
with open(tpath_flat + 'ta1_field/parameters.txt', 'r') as param_file:
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
phi_flat = build_phi(tpath_flat + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_flat, iintf_flat = build_intf(phi_flat, dz)
phi_sin = build_phi(tpath_sin + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_sin, iintf_sin = build_intf(phi_sin, dz)
phi_atto = build_phi(tpath_atto + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_atto, iintf_atto = build_intf(phi_atto, dz)

intf = {
        'flat' : intf_flat,
        'sin' : intf_sin,
        'atto' : intf_atto
        }

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
Nz_SLayer = 300 #int(Nz/2)

zeds = np.zeros((Nx,Ny,Nz))

for i in range(0,Nx):
    for j in range(0,Ny):
        zeds[i,j,:] = np.arange(0,Nz)*(dz*zi)
    
dist_flat = copy.deepcopy(zeds); dist_sin = copy.deepcopy(zeds); dist_atto = copy.deepcopy(zeds)

for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz):
            dist_flat[i,j,k] = dist_flat[i,j,k] - intf_flat[i,j]*zi
            dist_sin[i,j,k] = dist_sin[i,j,k] - intf_sin[i,j]*zi
            dist_atto[i,j,k] = dist_atto[i,j,k] - intf_atto[i,j]*zi

dist = {
        'flat' : dist_flat,
        'sin_min' : dist_sin,
        'atto_min' : dist_atto,
        'sin_max' : dist_sin,
        'atto_max' : dist_atto
        }

#%%Import data

import pickle

var = ['u','v','w','uw','vw']

with open(path_flat+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_flat = dict()
for i in range(len(var)):
    data_flat[var[i]] = np.mean(data[var[i]],axis=3)

with open(path_sin+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_sin = dict()
for i in range(len(var)):
    data_sin[var[i]] = np.mean(data[var[i]],axis=3)

with open(path_atto+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_atto = dict()
for i in range(len(var)):
    data_atto[var[i]] = np.mean(data[var[i]],axis=3)

velocity = {
    'flat':data_flat,
    'sin_min':data_sin,
    'sin_max':data_sin,
    'atto_min':data_atto,
    'atto_max':data_atto
    }

   
#%%Select coordinates
import random
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex

N_twrs = 100
Nz_SLayer = 300

coord_atto_max = find_coordinates(intf['atto'],N_twrs,'max') #min,max
coord_sin_max = find_coordinates(intf['sin'],N_twrs,'max') #min,max
coord_atto_min = find_coordinates(intf['atto'],N_twrs,'min') #min,max
coord_sin_min = find_coordinates(intf['sin'],N_twrs,'min') #min,max
coord_flat = []
for i in range(0,N_twrs):
    coord_flat.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))

coord = {
    'flat' : coord_flat,
    'sin_min' : coord_sin_min,
    'atto_min' : coord_atto_min,
    'sin_max' : coord_sin_max,
    'atto_max' : coord_atto_max
    }

#%%Plot

cases = ['flat','sin_min','sin_max','atto_min','atto_max']
SH_terms = ['uw','vw']

for j in range(0,1):
    
    fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,6))
    
    axs.axvline(0,c='k')
    for i in range(len(SH_terms)):
        tmpSH = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
        Ruw = (velocity[cases[j]]['uw'] - velocity[cases[j]]['u']*velocity[cases[j]]['w'])*(u_scale**2)
        Rvw = (velocity[cases[j]]['vw'] - velocity[cases[j]]['v']*velocity[cases[j]]['w'])*(u_scale**2)
        shear = {
            'uw' : Ruw,
            'vw' : Rvw}
    
        for k in range(len(coord[cases[j]])):
            loc = coord[cases[j]][k]
            ustar = ((Ruw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2 + \
                            (Rvw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2)**(1/4)
            tmpSH[k,:] = abs((shear[SH_terms[i]][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                              int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]))/(ustar**2)
    
        axs.plot(np.nanmean(tmpSH,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label=SH_terms[i],linewidth=2)
        
    
    axs.set_xlabel(r"$\frac{\overline{u'w'}}{u_{*}^{2}}$",fontsize=17)
    axs.set_ylabel(r'$z/h_C$',fontsize=15)
    axs.set_ylim(0,2)
    # axs.set_xlim(0,20)
    axs.grid()
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.axhline(1,ls='--',c='k')
    axs.legend(loc='upper right',fontsize=15)
    axs.set_title(f'{cases[j]}', fontsize=15)
    # plt.savefig(pathFig+f'{cases[j]}_SHEARprofTWR_ustar.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()










































