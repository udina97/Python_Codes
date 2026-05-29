#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:55:12 2024

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

# import pickle

# var = ['u','v','w','p','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','txx','txy','txz','tyy','tyz',\
#        'tzz','dxx','dxy','dxz','dyy','dyz','dzz','fdx','fdy','fdz','dpdx','dpdy','dpdz','pdudx','pdvdy','pdwdz','pu','pv','pw','dudx','dudy',\
#        'dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2',\
#        'utxx','utxy','utxz','vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','ufdx','vfdy','wfdz','utyy','utzz','vtxx','vtzz','wtxx','wtyy']

# with open(path_flat+'data.pkl', 'rb') as f:
# # Load the data from the pickle file
#     data = pickle.load(f)
    
# data_flat = dict()
# for i in range(len(var)):
#     data_flat[var[i]] = np.mean(data[var[i]],axis=3)

# with open(path_sin+'data.pkl', 'rb') as f:
# # Load the data from the pickle file
#     data = pickle.load(f)
    
# data_sin = dict()
# for i in range(len(var)):
#     data_sin[var[i]] = np.mean(data[var[i]],axis=3)

# with open(path_atto+'data.pkl', 'rb') as f:
# # Load the data from the pickle file
#     data = pickle.load(f)
    
# data_atto = dict()
# for i in range(len(var)):
#     data_atto[var[i]] = np.mean(data[var[i]],axis=3)

# dataNAN_flat = copy.deepcopy(data_flat); dataNAN_sin = copy.deepcopy(data_sin); dataNAN_atto = copy.deepcopy(data_atto)

# for i in range(len(var)):
#     dataNAN_flat[var[i]][(dist_flat<0)] = float("nan")
#     dataNAN_sin[var[i]][(dist_sin<0)] = float("nan")
#     dataNAN_atto[var[i]][(dist_atto<0)] = float("nan")
    

#%%Load anisotropy data
            
Aniso_data_flat = xr.open_dataarray(path_flat + 'Anisotropy_clustering_'+flat+'_v2.nc')
Aniso_data_sin = xr.open_dataarray(path_sin + 'Anisotropy_clustering_'+sin+'_v2.nc')
Aniso_data_atto = xr.open_dataarray(path_atto + 'Anisotropy_clustering_'+atto+'_v2.nc')

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

#%%Load TKE data

TKE_flat = np.load(path_flat + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_sin = np.load(path_sin + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_atto = np.load(path_atto + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()

TKE = {
       'flat' : TKE_flat,
       'sin' : TKE_sin,
       'atto' : TKE_atto
       }

#%%Scatter plot of yB,xB and TKE term
from scipy.stats import gaussian_kde
from operator import add
from operator import sub

cases = ['flat','sin','atto']
labels = ["(a)", "(b)", "(c)"]  # Subplot labels
l = 0
level = 5

fig, axs = plt.subplots(1,3,figsize=(16,6))#,tight_layout=True)

for k in range(len(cases)):
    tmpDIS = copy.deepcopy(TKE[cases[k]]['totdis'])
    tmpRES = copy.deepcopy((TKE[cases[k]]['prod']) - (TKE[cases[k]]['totdis']))
    # tmpRES[(abs(tmpRES)<1)] = 0
    tmpNorm = tmpRES/abs(tmpDIS)*100
    tmpNorm[(dist[cases[k]]<0)] = float('nan')
    x_yB = aniso[cases[k]]['yB'][(dist[cases[k]]>39) & (dist[cases[k]]<level*39)]
    y_TKE = tmpNorm[(dist[cases[k]]>39) & (dist[cases[k]]<level*39)]
    
    TKE_median = []
    TKE_std = []
    yB_mean = []
    j=0.1
    for i in range(0,28):
        if i == 0:
            TKE_median.append(np.nanmedian(y_TKE[x_yB<j]))
            TKE_std.append(np.std(y_TKE[x_yB<j]))
            yB_mean.append(0.05)
        else:
            TKE_median.append(np.nanmedian(y_TKE[(x_yB>j) & (x_yB<j+0.025)]))
            TKE_std.append(np.std(y_TKE[(x_yB>j) & (x_yB<j+0.025)]))
            yB_mean.append((j+j+0.025)/2)
        j+=0.025

    axs[k].plot(yB_mean,TKE_median,c='k',marker='o')
    axs[k].fill_between(yB_mean,list(map(sub,TKE_median,TKE_std)),list(map(add,TKE_median,TKE_std)),alpha=0.5)
    axs[k].axhline(0,0,1,c='k',ls='-.')
    # axs[k].axvline(0.36,-2000,20000,c='k',ls='-.')
    axs[k].axvline(0.38,-2000,20000,c='k',ls='-.')
    axs[k].set_xlabel(r'$y_B$',fontsize=18)
    # axs[k].set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=18)
    axs[k].text(0.1, 0.1, f'r = {round(np.corrcoef(x_yB.flatten(),y_TKE.flatten())[0,1],2)}',transform=axs[k].transAxes, fontsize=18,verticalalignment='top')
    # axs[k].set_title(cases[k]+r' - $h_C$ to' + fr' {level}$h_C$', fontsize=18)
    # axs[k].tick_params(axis='y', which='major', labelsize=12)
    axs[k].tick_params(axis='x', which='major', labelsize=12)
    axs[k].set_ylim(-70,250)
    axs[k].set_xlim(0.15,0.6)
    
for j in range(1,3):
    axs[j].tick_params(axis='y', which='both', labelbottom=False, labelleft=False)
axs[0].tick_params(axis='y', which='major', labelsize=12)
axs[0].set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=21)

plt.subplots_adjust(left=0.07,
                    bottom=0.1, 
                    right=0.97, 
                    top=0.94, 
                    wspace=0.09, 
                    hspace=0.1)

for m in range(0,3):
    axs[m].text(0.1, 0.98, labels[l], transform=axs[m].transAxes, fontsize=14, va="top", ha="right")
    l += 1
# plt.savefig(pathFig + 'ResyBCorr_tij.png',dpi=300,facecolor='None', edgecolor='None')
plt.show()
# plt.savefig(pathOUT+'/Figures/PD_yB_y' + str(yslice) + '_ATTO.png',dpi=300,facecolor='white', edgecolor='white')
# plt.savefig(pathOUT+'/Figures/PD_yB_3D_3_DIAG.png',dpi=300,facecolor='white', edgecolor='white')




























































