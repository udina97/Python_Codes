#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:59:18 2024

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

#%%Load TKE data

TKE_flat = np.load(path_flat + 'TKE_terms.npy',allow_pickle='TRUE').item()
TKE_sin = np.load(path_sin + 'TKE_terms.npy',allow_pickle='TRUE').item()
TKE_atto = np.load(path_atto + 'TKE_terms.npy',allow_pickle='TRUE').item()

TKE = {
       'flat' : TKE_flat,
       'sin' : TKE_sin,
       'atto' : TKE_atto
       }

#%%
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap

case = 'atto'

lvl1 = 39
lvl2 = 39*20
tmb_yB = aniso[case]['yB']
tmb_yB_d = aniso_d[case]['yB']
ratio = tmb_yB/tmb_yB_d
ratio_subset = ratio[(dist[case]>lvl1) & (dist[case]<lvl2)].flatten()
ratio_prime = ((ratio_subset - np.nanmedian(ratio_subset))/np.nanmedian(ratio_subset))*100

tmpRES = copy.deepcopy((TKE[case]['prod'] - TKE[case]['totdis']))
tmpDIS = copy.deepcopy((TKE[case]['totdis']))
tmpRES[(abs(tmpRES)<0)] = 0
tmpNorm = ((tmpRES)/abs(tmpDIS))*100
tmpNorm_subset = tmpNorm[(dist[case]>lvl1) & (dist[case]<lvl2)].flatten()

nbins = 20
x = tmpNorm_subset
y = ratio_prime

k = gaussian_kde([x,y])
xi, yi = np.mgrid[
    x.min():x.max():nbins*1j,
    y.min():y.max():nbins*1j
]
z_i = k(np.vstack([
    xi.flatten(),
    yi.flatten()
])).reshape(xi.shape)

#%% Compute the 99% contour

density_flat = z_i.ravel()
sorted_density = np.sort(density_flat)
cumulative_density = np.cumsum(sorted_density) / np.sum(sorted_density)
level_90 = sorted_density[np.searchsorted(cumulative_density, 0.01)]

#%%Subset the vector to 500000 randomly sleected values

indices = np.random.choice(len(ratio_prime), size=500000, replace=False)

subset1 = ratio_prime[indices]
subset2 = tmpNorm_subset[indices]

#%%Plot

x = subset2
y = subset1

# Perform Total Least Squares using SVD
X = np.vstack((x, y)).T
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
direction_vector = Vt[-1, :] 

slope = -direction_vector[0] / direction_vector[1]
intercept = y.mean() - slope * x.mean()

x_fit = np.linspace(min(x), max(x), 100)
y_fit = slope * x_fit + intercept

fig,axs = plt.subplots(1,1,figsize=(6,6),tight_layout=True)

axs.scatter(subset2,subset1,s=1,c='grey',alpha=0.4)
# axs.pcolormesh(xi, yi, z_i, cmap='Blues')
axs.contour(xi, yi, z_i,levels=10,cmap='hot_r')
axs.contour(xi,yi,z_i,levels=[level_90],colors='green',linewidth=2)
axs.plot(x_fit, y_fit, 'r-', label=f'TLS Regression Line\ny = {slope:.2f}x + {intercept:.2f}')

axs.set_xlabel(r'$\frac{P - D}{|D|}$',fontsize=15)
axs.set_ylabel(r"$\frac{C'}{C_{m}}$",fontsize=15)
# axs.set_ylabel(r"$C'$",fontsize=15)
axs.set_title(f'Points between {lvl1}m and {lvl2}m',fontsize=15)
axs.set_xlim(-250,250)
axs.set_ylim(-40,40)
axs.legend(fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.grid()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Ratio_vs_Res_atto_2.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()


#%%

x = subset2
y = subset1

# Perform Total Least Squares using SVD
X = np.vstack((x, y)).T
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
direction_vector = Vt[-1, :] 

slope = -direction_vector[0] / direction_vector[1]
intercept = y.mean() - slope * x.mean()

x_fit = np.linspace(min(x), max(x), 100)
y_fit = slope * x_fit + intercept

plt.figure(figsize=(8, 6))
plt.scatter(x, y,s=1,c='grey',alpha=0.4)
plt.plot(x_fit, y_fit, 'r-', label=f'TLS Regression Line\ny = {slope:.2f}x + {intercept:.2f}')
plt.xlabel(r'$\frac{P - D}{|D|}$',fontsize=15)
plt.ylabel(r"$\frac{C'}{C_{m}}$",fontsize=15)
plt.title('Scatter Plot with Total Least Squares Regression')
plt.legend()
plt.grid()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Ratio_vs_Res_flat_Reg.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()














































