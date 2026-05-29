#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:08:52 2024

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

#%% Select tower coords and plot
import random
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
terrain = plt.get_cmap('terrain')
terrain_truncated = LinearSegmentedColormap.from_list(
    'terrain_truncated', terrain(np.linspace(0.25, 1, 100)))

N_twrs = 100

coord_atto = find_coordinates(intf['atto'],N_twrs,'max') #min,max
coord_sin = find_coordinates(intf['sin'],N_twrs,'max') #min,max
coord_atto_v = find_coordinates(intf['atto'],N_twrs,'min') #min,max
coord_sin_v = find_coordinates(intf['sin'],N_twrs,'min') #min,max
coord_flat = []
for i in range(0,N_twrs):
    coord_flat.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))

coord = {
    'flat' : coord_flat,
    'sin' : coord_sin,
    'atto' : coord_atto,
    'sin_v' : coord_sin_v,
    'atto_v' : coord_atto_v
    }

cases = coord.keys()

#%%

from matplotlib.transforms import ScaledTranslation
labels = ['a)','b)','c)']

fig, ax = plt.subplots(1,3,figsize=(17,6))
contour1 = ax[0].contourf(x * zi, y * zi, (intf['flat']*zi - (4.5*dz*zi)).T, levels=np.arange(0,100,1),cmap=terrain_truncated)
for i in range(len(coord['flat'])):
    ax[0].plot(x[coord['flat'][i][0]]*zi,y[coord['flat'][i][1]]*zi,'ok')
contour2 = ax[1].contourf(x * zi, y * zi, (intf['sin']*zi - (4.5*dz*zi)).T, levels=np.arange(0,100,1),cmap=terrain_truncated)
for i in range(len(coord['sin'])):
    ax[1].plot(x[coord['sin'][i][0]]*zi,y[coord['sin'][i][1]]*zi,'ok')
# contour3 = ax[2].contourf(x * zi, y * zi, (intf['sin']*zi - (4.5*dz*zi)).T, levels=np.arange(0,100,1),cmap=terrain_truncated)
for i in range(len(coord['sin_v'])):
    ax[1].plot(x[coord['sin_v'][i][0]]*zi,y[coord['sin_v'][i][1]]*zi,'vr')
contour4 = ax[2].contourf(x * zi, y * zi, (intf['atto']*zi - (4.5*dz*zi)).T, levels=np.arange(0,100,1),cmap=terrain_truncated)
for i in range(len(coord['atto'])):
    ax[2].plot(x[coord['atto'][i][0]]*zi,y[coord['atto'][i][1]]*zi,'ok')
# contour5 = ax[4].contourf(x * zi, y * zi, (intf['atto']*zi - (4.5*dz*zi)).T, levels=np.arange(0,100,1),cmap=terrain_truncated)
for i in range(len(coord['atto_v'])):
    ax[2].plot(x[coord['atto_v'][i][0]]*zi,y[coord['atto_v'][i][1]]*zi,'vr')

for j in range(len(ax)):
    ax[j].tick_params(axis='x', which='major', labelsize=16)
    ax[j].set_xlabel(r'$x [m]$', fontsize=18)
    ax[j].set_xlim(0,x[-1]*zi)
    ax[j].set_ylim(0,y[-1]*zi)
    
for j in range(1,3):
    ax[j].tick_params(axis='y', which='both', labelbottom=False, labelleft=False)
# for j in range(len(ax)):
#     ax[j].text(0.1, 0.98, labels[j], transform=ax[j].transAxes, fontsize=14, va="top", ha="right",fontweight='bold')
for j in range(len(ax)):
    ax[j].text(
        0.0, 1.0, labels[j],
        transform=ax[j].transAxes +
                 ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans),
        fontsize=18, va="bottom", ha="left", fontfamily='serif'
    )
    
ax[0].tick_params(axis='y', which='major', labelsize=16)
ax[0].set_ylabel(r'$y [m]$', fontsize=18)

cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.79])  # Position and size of the colorbar
cbar1 = fig.colorbar(contour4, cax=cbar_ax)
cbar1.set_label(r"$z [m]$ ", fontsize=18, labelpad=8)  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=14)  # Set colorbar tick label size

plt.subplots_adjust(left=0.07,
                    bottom=0.15, 
                    right=0.93, 
                    top=0.94, 
                    wspace=0.1, 
                    hspace=0.2)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/Topo&Twrs_v3.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()

















































