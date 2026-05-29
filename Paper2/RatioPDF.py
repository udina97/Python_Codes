#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:12:17 2024

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

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf

#%%Import Giulia's yB data

save_path = '/uufs/chpc.utah.edu/common/home/u1450851/Giulia_yB/'
pathOUT= save_path+'Figures/'

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
    
aniso_g = {
    'g800' : g800_9_aniso_full,
    'g800_i' : g800_i_9_aniso_full,
    'empty' : empty_aniso_full
    }

aniso_d_g = {
    'g800' : g800_9_aniso_diag,
    'g800_i' : g800_i_9_aniso_diag,
    'empty' : empty_aniso_diag
    }
    
#%%Paths to data

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
        'sin' : dist_sin,
        'atto' : dist_atto
        }

#%% Import my yB data
            
Aniso_data_flat = xr.open_dataarray(path_flat + 'Anisotropy_clustering_'+flat+'.nc')
Aniso_data_sin = xr.open_dataarray(path_sin + 'Anisotropy_clustering_'+sin+'.nc')
Aniso_data_atto = xr.open_dataarray(path_atto + 'Anisotropy_clustering_'+atto+'.nc')

aniso_flat = dict(); aniso_sin = dict(); aniso_atto = dict()
aniso_var = ['xB','yB','type']

for i in range(len(aniso_var)):
    aniso_flat[aniso_var[i]] = np.reshape(np.copy(Aniso_data_flat.data[:,i]),(Nx,Ny,Nz)) 
    aniso_sin[aniso_var[i]] = np.reshape(np.copy(Aniso_data_sin.data[:,i]),(Nx,Ny,Nz)) 
    aniso_atto[aniso_var[i]] = np.reshape(np.copy(Aniso_data_atto.data[:,i]),(Nx,Ny,Nz)) 

for i in range(len(aniso_var)):
    aniso_flat[aniso_var[i]][(dist_flat[:,:,:]<0)] = float('nan')
    aniso_sin[aniso_var[i]][(dist_sin[:,:,:]<0)] = float('nan')
    aniso_atto[aniso_var[i]][(dist_atto[:,:,:]<0)] = float('nan')

aniso = {
    'flat' : aniso_flat,
    'sin' : aniso_sin,
    'atto' : aniso_atto
    }

#%%Import my anisotropy data DIAG

Aniso_d_data_flat = xr.open_dataarray(path_flat + 'Anisotropy_clustering_'+flat+'_DIAG.nc')
Aniso_d_data_sin = xr.open_dataarray(path_sin + 'Anisotropy_clustering_'+sin+'_DIAG.nc')
Aniso_d_data_atto = xr.open_dataarray(path_atto + 'Anisotropy_clustering_'+atto+'_DIAG.nc')

aniso_d_flat = dict(); aniso_d_sin = dict(); aniso_d_atto = dict()
aniso_var = ['xB','yB','type']

for i in range(len(aniso_var)):
    aniso_d_flat[aniso_var[i]] = np.reshape(np.copy(Aniso_d_data_flat.data[:,i]),(Nx,Ny,Nz)) 
    aniso_d_sin[aniso_var[i]] = np.reshape(np.copy(Aniso_d_data_sin.data[:,i]),(Nx,Ny,Nz)) 
    aniso_d_atto[aniso_var[i]] = np.reshape(np.copy(Aniso_d_data_atto.data[:,i]),(Nx,Ny,Nz)) 

for i in range(len(aniso_var)):
    aniso_d_flat[aniso_var[i]][(dist_flat[:,:,:]<0)] = float('nan')
    aniso_d_sin[aniso_var[i]][(dist_sin[:,:,:]<0)] = float('nan')
    aniso_d_atto[aniso_var[i]][(dist_atto[:,:,:]<0)] = float('nan')

aniso_d = {
    'flat' : aniso_d_flat,
    'sin' : aniso_d_sin,
    'atto' : aniso_d_atto
    }


#%% 

c=['k','r','g','y','b','pink']
cases = ['flat','sin','atto']
cases_g = ['g800','g800_i','empty']

fig, axs = plt.subplots(1,1,figsize=(10,6),tight_layout=True)

for i in range(len(cases)):
    lvl1 = 39
    lvl2 = 39*20
    tmb_yB = aniso[cases[i]]['yB']
    tmb_yB_d = aniso_d[cases[i]]['yB']
    ratio = tmb_yB_d/tmb_yB
    ratio_subset = ratio[(dist[cases[i]]>lvl1) & (dist[cases[i]]<lvl2)].flatten()

    pdf,bin_edge = np.histogram(ratio_subset,bins=100)
    bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    
    axs.plot(bin_center,pdf,c=c[i],label=f'{cases[i]}')
    # axs.hist(ratio_subset,bins=100,density=False,stacked=False,cumulative=False)
    # axs.axvline(np.nanmean(ratio_subset),c=c[i],ls='--',label=f'Mean = {round(np.nanmean(ratio_subset),2)}')
    axs.axvline(np.nanmedian(ratio_subset),c=c[i],ls='-.',label=f'Median = {round(np.nanmedian(ratio_subset),2)}')
    # axs.axvline(np.nanmean(ratio_subset)-np.std(ratio_subset),c=c[i],ls=':',label='+/- STD')
    # axs.axvline(np.nanmean(ratio_subset)+np.std(ratio_subset),c=c[i],ls=':')

for i in range(len(cases_g)):
    lvl1 = 10
    lvl2 = 200
    tmb_yB = aniso_g[cases_g[i]]['yB']
    tmb_yB_d = aniso_d_g[cases_g[i]]['yB']
    ratio = tmb_yB_d/tmb_yB
    ratio_subset = ratio[:,:,lvl1:lvl2].flatten()

    pdf,bin_edge = np.histogram(ratio_subset,bins=100)
    bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    
    axs.plot(bin_center,pdf,c=c[i+3],label=f'{cases_g[i]}')
    # axs.hist(ratio_subset,bins=100,density=False,stacked=False,cumulative=False)
    # axs.axvline(np.nanmean(ratio_subset),c=c[i+3],ls='--',label=f'Mean = {round(np.nanmean(ratio_subset),2)}')
    axs.axvline(np.nanmedian(ratio_subset),c=c[i+3],ls='-.',label=f'Median = {round(np.nanmedian(ratio_subset),2)}')
    # axs.axvline(np.nanmean(ratio_subset)-np.std(ratio_subset),c=c[i],ls=':',label='+/- STD')
    # axs.axvline(np.nanmean(ratio_subset)+np.std(ratio_subset),c=c[i],ls=':')
    
axs.set_xlabel(r'$\frac{y_{B,DIAG}}{y_{B}}$',fontsize=15)
axs.set_ylabel(r'Counts',fontsize=12)
axs.set_title(f'All points between {lvl1*3.9}m and {lvl2*3.9}m',fontsize=12)
axs.legend()
axs.set_ylim(0)
axs.set_xlim(1,2)

plt.show()

#%%Box plots instead of PDF

cases = ['flat','sin','atto']
cases_g = ['g800','g800_i','empty']
ratio_dict = dict()
ratio_list = []

for i in range(len(cases)):
    lvl1 = 39
    lvl2 = 39*20
    tmb_yB = aniso[cases[i]]['yB']
    tmb_yB_d = aniso_d[cases[i]]['yB']
    ratio = tmb_yB_d/tmb_yB
    ratio_subset = ratio[(dist[cases[i]]>lvl1) & (dist[cases[i]]<lvl2)].flatten()

    ratio_dict[cases[i]] = ratio_subset
    ratio_list.append(ratio_subset)

for i in range(len(cases_g)):
    lvl1 = 10
    lvl2 = 200
    tmb_yB = aniso_g[cases_g[i]]['yB']
    tmb_yB_d = aniso_d_g[cases_g[i]]['yB']
    ratio = tmb_yB_d/tmb_yB
    ratio_subset = ratio[:,:,lvl1:lvl2].flatten()

    ratio_dict[cases_g[i]] = ratio_subset
    ratio_list.append(ratio_subset)

mean = [np.mean(d) for d in ratio_list]
std_devs = [np.std(d) for d in ratio_list]

cases_tot = ['flat','sin','atto','g800','g800_i','empty']

fig,axs = plt.subplots(1,1,figsize=(10,5),tight_layout=True)

axs.boxplot(ratio_list,labels=cases_tot, showfliers=False)
for i in range(len(mean)):
   axs.plot(i + 1, mean[i], 'ro')
# Adds standard deviations as error bars
for i in range(len(std_devs)):
   axs.errorbar(i + 1, mean[i], yerr=std_devs[i], fmt='o', color='red')
axs.set_xlabel('Cases',fontsize=12)
axs.set_ylabel(r'$\frac{y_{B,DIAG}}{y_{B}}$',fontsize=15)
# axs.set_ylabel(r'$\frac{y_{B}}{y_{B,DIAG}}$',fontsize=15)
axs.set_title(f'All points between {lvl1*3.9}m and {lvl2*3.9}m',fontsize=12)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'yB_boxplot_ratio_i.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()





































