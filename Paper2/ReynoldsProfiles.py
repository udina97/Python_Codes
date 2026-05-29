#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:42:08 2024

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
    
#%% Compute Reynolds stresses

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

#%%Plot Reynolds stresses Ben cases

# cases_tot = cases_g + cases
# Rstress_tot = {**Rstress_g, **Rstress_b}

stress = ['Rxx','Ryy','Rzz','Rxy','Rxz','Ryz']

fig,axs = plt.subplots(1,6,figsize=(14,6),tight_layout=True)

for i in range(len(cases)):
    for j in range(len(axs)):
        axs[j].plot(np.nanmean(Rstress_b[cases[i]].data[:,:,5:,j],axis=(0,1)),np.arange(0,nzTot-5)*dz/(39/zi) + ((dz/2)/(39/zi)),label=cases[i])
        axs[j].set_xlabel(r'$\frac{'+stress[j]+'}{u_{s}^{2}}$',fontsize=15)
        axs[j].set_ylim(0,25)
        axs[j].grid()
        axs[j].tick_params(axis='x', which='major', labelsize=12)
        
axs[0].set_ylabel(f'$z/h_C$',fontsize=15)
axs[0].tick_params(axis='y', which='major', labelsize=12)
axs[5].legend(fontsize=15)
fig.suptitle(f'Domain average',fontsize=15)

for j in range(1,6):
    axs[j].tick_params(axis='y', which='both', labelbottom=False, labelleft=False)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Rstress_prof_B.png',dpi=300,facecolor='None', edgecolor='None')
plt.show()

# %%Plot Reynolds stresses Giulia cases

# cases_tot = cases_g + cases
# Rstress_tot = {**Rstress_g, **Rstress_b}

stress = ['Rxx','Ryy','Rzz','Rxy','Rxz','Ryz']

fig,axs = plt.subplots(1,6,figsize=(14,6),tight_layout=True)

for i in range(len(cases)):
    for j in range(len(axs)):
        axs[j].plot(np.nanmean(Rstress_g[cases_g[i]].data[:,:,:,j],axis=(0,1))*(0.313**2)/(0.4**2),np.arange(0,Nz)*(1/Nz)/(39/zi) + (((1/Nz)/2)/(39/zi)),label=cases_g[i])
        axs[j].set_xlabel(r'$\frac{'+stress[j]+'}{u_{s}^{2}}$',fontsize=15)
        axs[j].set_ylim(0,25)
        axs[j].grid()
        axs[j].tick_params(axis='x', which='major', labelsize=12)
        axs[j].axhline(1,ls='--',c='k')
        
axs[0].set_ylabel(f'$z/h_C$',fontsize=15)
axs[0].tick_params(axis='y', which='major', labelsize=12)
axs[5].legend(fontsize=15)
fig.suptitle(f'Domain average',fontsize=15)

for j in range(1,6):
    axs[j].tick_params(axis='y', which='both', labelbottom=False, labelleft=False)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Rstress_prof_G.png',dpi=300,facecolor='None', edgecolor='None')
plt.show()

#%% Compute profiles using virtual tower averaging
import random
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex

N_twrs = 100
Nz_SLayer = 300

coord_atto = find_coordinates(intf['atto'],N_twrs,'min') #min,max
coord_sin = find_coordinates(intf['sin'],N_twrs,'min') #min,max
coord_flat = []
for i in range(0,N_twrs):
    coord_flat.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))

coord = {
    'flat' : coord_flat,
    'sin' : coord_sin,
    'atto' : coord_atto
    }

stress = ['Rxx','Ryy','Rzz','Rxy','Rxz','Ryz']

fig,axs = plt.subplots(1,6,figsize=(14,6),tight_layout=True)

for i in range(len(cases)):
    for j in range(len(axs)):
        Rij = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

        for k in range(len(coord[cases[i]])):
            loc = coord[cases[i]][k]
            Rij[k,:] = Rstress_b[cases[i]].data[:,:,5:,j][loc[0],loc[1],int(np.where(dist[cases[i]][loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[cases[i]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]

        axs[j].plot(np.nanmean(Rij,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label=cases[i])
        axs[j].set_xlabel(r'$\frac{'+stress[j]+'}{u_{s}^{2}}$',fontsize=15)
        axs[j].set_ylim(0,20)
        axs[j].grid()
        axs[j].tick_params(axis='x', which='major', labelsize=12)
        axs[j].axhline(1,ls='--',c='k')
        
axs[0].set_ylabel(f'$z/h_C$',fontsize=15)
axs[0].tick_params(axis='y', which='major', labelsize=12)
fig.suptitle(f'Average of {N_twrs} virtual towers',fontsize=15)
axs[5].legend(fontsize=15)

for j in range(1,6):
    axs[j].tick_params(axis='y', which='both', labelbottom=False, labelleft=False)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Rstress_prof_B_twr_v.png',dpi=300,facecolor='None', edgecolor='None')
plt.show()

#%%
import random

N_twrs = 100
Nz_SLayer = 190

coord_empty = []; coord_g800 = []; coord_g800i = []
for i in range(0,N_twrs):
    coord_empty.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))
    coord_g800.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))
    coord_g800i.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))

coord_g = {
    'empty' : coord_empty,
    'g800' : coord_g800,
    'g800_i' : coord_g800i
    }

stress = ['Rxx','Ryy','Rzz','Rxy','Rxz','Ryz']

fig,axs = plt.subplots(1,6,figsize=(14,6),tight_layout=True)

for i in range(len(cases_g)):
    for j in range(len(axs)):
        Rij = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

        for k in range(len(coord_g[cases_g[i]])):
            loc = coord_g[cases_g[i]][k]
            Rij[k,:] = Rstress_g[cases_g[i]].data[loc[0],loc[1],:Nz_SLayer,j]

        axs[j].plot(np.nanmean(Rij,axis=(0))*(0.313**2)/(0.4**2),np.arange(0,Nz_SLayer)*(1/Nz)/(39/zi) + (((1/Nz)/2)/(39/zi)),label=cases_g[i])
        axs[j].set_xlabel(r'$\frac{'+stress[j]+'}{u_{s}^{2}}$',fontsize=15)
        axs[j].set_ylim(0,20)
        axs[j].grid()
        axs[j].tick_params(axis='x', which='major', labelsize=12)
        axs[j].axhline(1,ls='--',c='k')
        
axs[0].set_ylabel(f'$z/h_C$',fontsize=15)
axs[0].tick_params(axis='y', which='major', labelsize=12)
fig.suptitle(f'Average of {N_twrs} virtual towers',fontsize=15)
axs[5].legend(fontsize=15)

for j in range(1,6):
    axs[j].tick_params(axis='y', which='both', labelbottom=False, labelleft=False)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Rstress_prof_G_twr.png',dpi=300,facecolor='None', edgecolor='None')
plt.show()
















































    