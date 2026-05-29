#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:33:41 2024

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

#%% Import Giulia's data

Nx = 256
Ny = 256
Nz = 256

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

cases_g = ['g800','g800_i','empty']

data_g = {
    'g800' : g800_9,
    'g800_i' : g800_i_9,
    'empty' : empty
    }

Rstress_g = dict()

for i in range(len(cases_g)):
    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                            'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStressUVP(Nx,Ny,Nz,data_g[cases_g[i]]['u'],data_g[cases_g[i]]['v'],data_g[cases_g[i]]['w'],data_g[cases_g[i]]['uu'],\
                              data_g[cases_g[i]]['vv'],data_g[cases_g[i]]['ww'],data_g[cases_g[i]]['uv'],data_g[cases_g[i]]['uw'],data_g[cases_g[i]]['vw'])
        
    Rstress_g[cases_g[i]] = Rstress
        
    
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

#%% Load data

import pickle

var = ['u','v','w','p','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','txx','txy','txz','tyy','tyz',\
        'tzz','dxx','dxy','dxz','dyy','dyz','dzz','fdx','fdy','fdz','dpdx','dpdy','dpdz','pdudx','pdvdy','pdwdz','pu','pv','pw','dudx','dudy',\
        'dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2',\
        'utxx','utxy','utxz','vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','ufdx','vfdy','wfdz','utyy','utzz','vtxx','vtzz','wtxx','wtyy']

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

dataNAN_flat = copy.deepcopy(data_flat); dataNAN_sin = copy.deepcopy(data_sin); dataNAN_atto = copy.deepcopy(data_atto)

for i in range(len(var)):
    dataNAN_flat[var[i]][(dist_flat<0)] = float("nan")
    dataNAN_sin[var[i]][(dist_sin<0)] = float("nan")
    dataNAN_atto[var[i]][(dist_atto<0)] = float("nan")

cases = ['flat','sin','atto']

data_b = {
    'flat' : dataNAN_flat,
    'sin' : dataNAN_sin,
    'atto' : dataNAN_atto
    }

Rstress_b = dict()

for i in range(len(cases)):
    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                            'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStressUVP(Nx,Ny,Nz,data_b[cases[i]]['u'],data_b[cases[i]]['v'],data_b[cases[i]]['w'],data_b[cases[i]]['uu'],\
                              data_b[cases[i]]['vv'],data_b[cases[i]]['ww'],data_b[cases[i]]['uv'],data_b[cases[i]]['uw'],data_b[cases[i]]['vw'])
        
    Rstress_b[cases[i]] = Rstress

#%%Plot pdf of sigma_u,v,w for a specific horizontal slice

zslice = 100
cases_tot = cases_g + cases
Rstress_tot = {**Rstress_g, **Rstress_b}

fig,axs = plt.subplots(1,6,figsize=(10,5),tight_layout=True)

c=['k','r','g']
s=['sU','sV','sW']

for i in range(0,3):
    lvl1 = 10
    lvl2 = 100
    
    tmb_sU = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,lvl1:lvl2,0]).flatten()
    tmb_sV = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,lvl1:lvl2,1]).flatten()
    tmb_sW = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,lvl1:lvl2,2]).flatten()
    
    sigma = {
        'sU' : tmb_sU,
        'sV' : tmb_sV,
        'sW' : tmb_sW}
    
    for j in range(len(sigma)):
        pdf,bin_edge = np.histogram(sigma[s[j]],bins=100)
        bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    
        axs[i].plot(bin_center,pdf,c=c[j],label=f'{s[j]}')
    
    axs[i].set_xlabel(r'$\sigma_{u,v,w}$',fontsize=15)
    axs[i].set_title(f'{cases_tot[i]}',fontsize=12)
    axs[i].set_ylim(0)
    axs[i].legend()
    
for i in range(3,6):
    lvl1 = 39
    lvl2 = 39*10
    
    tmb_sU = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,:,0][(dist[cases_tot[i]]>lvl1) & (dist[cases_tot[i]]<lvl2)]).flatten()
    tmb_sV = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,:,1][(dist[cases_tot[i]]>lvl1) & (dist[cases_tot[i]]<lvl2)]).flatten()
    tmb_sW = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,:,2][(dist[cases_tot[i]]>lvl1) & (dist[cases_tot[i]]<lvl2)]).flatten()
    
    sigma = {
        'sU' : tmb_sU,
        'sV' : tmb_sV,
        'sW' : tmb_sW}
    
    for j in range(len(sigma)):
        pdf,bin_edge = np.histogram(sigma[s[j]],bins=100)
        bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    
        axs[i].plot(bin_center,pdf,c=c[j],label=f'{s[j]}')
    
    axs[i].set_xlabel(r'$\sigma_{u,v,w}$',fontsize=15)
    axs[i].set_title(f'{cases_tot[i]}',fontsize=12)
    axs[i].set_ylim(0)
    axs[i].legend()
    
axs[0].set_ylabel(r'Counts',fontsize=12)
# axs.set_title(f'All points between {lvl1*3.9}m and {lvl2*3.9}m',fontsize=12)

plt.show()

#%%Check whether sigma_w is always the smallest or not

def compare_vectors_or(vec1, vec2, vec3):
   
    indices = []

    for i in range(len(vec1)):
        if vec1[i] > vec2[i] or vec1[i] > vec3[i]:
            indices.append(i)
    
    return indices

indices = dict()

for i in range(0,3):
    lvl1 = 10
    lvl2 = 200
    
    tmb_sU = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,lvl1:lvl2,0]).flatten()
    tmb_sV = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,lvl1:lvl2,1]).flatten()
    tmb_sW = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,lvl1:lvl2,2]).flatten()
    
    indices[cases_tot[i]] = compare_vectors_or(tmb_sW, tmb_sV, tmb_sU)

for i in range(3,6):
    lvl1 = 39
    lvl2 = 39*20
    
    tmb_sU = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,:,0][(dist[cases_tot[i]]>lvl1) & (dist[cases_tot[i]]<lvl2)]).flatten()
    tmb_sV = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,:,1][(dist[cases_tot[i]]>lvl1) & (dist[cases_tot[i]]<lvl2)]).flatten()
    tmb_sW = np.sqrt(Rstress_tot[cases_tot[i]].data[:,:,:,2][(dist[cases_tot[i]]>lvl1) & (dist[cases_tot[i]]<lvl2)]).flatten()
    
    indices[cases_tot[i]] = compare_vectors_or(tmb_sW, tmb_sV, tmb_sU)












































