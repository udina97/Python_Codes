#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:32:14 2024

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import os
import sys
import copy

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

#%%
#inputs

sim_flat = 'flat_256x256x384_out5hr_v2'
sim_ATTO = 'ATTO_256x256x384_full_out5hr'
sim_hill = 'bicheng_hill_256x256x384_out5hr'

completed_sim = True
infinite_geom = True
slice_profiles = False
shifting_z = False
tke_budget_flag = True
derivatives_flag = True
PCON_flag = False
PCON_TEMP_flag = False
v_fracflag = False
SAVEAS_flag = False
clear_var_flag = False

zi = 1000.0
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'

#%% Import simulation parameters and build topography

opath_flat = simPath + f'sim{sim_flat}/output/'
opath_hill = simPath + f'sim{sim_hill}/output/'
opath_ATTO = simPath + f'sim{sim_ATTO}/output/'

nx = 256
ny = 256
nz = 384
lx = 2.88
ly = 2.88
lz = 0.96
dx = 1.125e-2
dy = 1.125e-2
dz = 2.5e-3
mpiProc = 32
ibm = 1
nzTot = nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, nx) * dx
y = np.arange(0, ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

topo = dict()

for i in range(0,3):
    if i == 0:
        opath = opath_flat
        phi = build_phi(opath + 'phi_functions/', nx, ny, nzTot, mpiProc)
        intf, iintf = build_intf(phi, dz)
        topo['flat'] = intf*zi - (4.5*dz*zi)
    elif i == 1:
        opath = opath_hill
        phi = build_phi(opath + 'phi_functions/', nx, ny, nzTot, mpiProc)
        intf, iintf = build_intf(phi, dz)
        topo['hill'] = intf*zi - (4.5*dz*zi)
    else:
        opath = opath_ATTO
        phi = build_phi(opath + 'phi_functions/', nx, ny, nzTot, mpiProc)
        intf, iintf = build_intf(phi, dz)
        topo['ATTO'] = intf*zi - (4.5*dz*zi)

#%% plot the topography or IBM

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
terrain = plt.get_cmap('terrain')
terrain_truncated = LinearSegmentedColormap.from_list(
    'terrain_truncated', terrain(np.linspace(0.25, 1, 100)))

keys = ['flat','hill','ATTO']

zi = 1000

fig, ax = plt.subplots(1,3,tight_layout = True,figsize=(12,4))
for i in range(0,3):
    contour = ax[i].contourf(x * zi, y * zi,topo[keys[i]].T, levels=np.arange(0,100,1),cmap=terrain_truncated)
    ax[i].set_xlabel('$x [m]$', usetex=True, fontsize=15)
    ax[i].set_aspect('auto')
# for i in range(len(coord)):
    # ax.plot(x[coord[i][0]]*zi,y[coord[i][1]]*zi,'ok')
# contour = ax.contourf(x, y, intf.T, 30, cmap='viridis')
# plt.title('Geometry (top-down view)', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
ax[0].set_ylabel('$y [m]$', usetex=True, fontsize=15)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(contour, cax=cax,label='z [m]')
ax[0].text(0.1, 0.95, 'a', transform=ax[0].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
ax[1].text(0.1, 0.95, 'b', transform=ax[1].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
ax[2].text(0.1, 0.95, 'c', transform=ax[2].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.gca().set_facecolor('white')

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/topo4paper1.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

 #%%Compute the slopes
slope = np.zeros((nx))
for i in range(0,nx-1):
    slope[i] = math.atan(abs((intf[i+1,1]-intf[i,1])/dx))
    
#%% Loading the topography data:

zeds = np.zeros((nx,ny,nz))

for i in range(0,nx):
    for j in range(0,ny):
        zeds[i,j,:] = np.arange(0,nz)*(dz*zi)
    
dist = copy.deepcopy(zeds)

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nz):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi
            