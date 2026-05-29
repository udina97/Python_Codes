#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:50:25 2024

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

from functions import build_phi, build_intf, get_var, get_snapvar

#%%
#inputs

sim = 'flat_inst_test'

simPath = '/scratch/general/nfs1/u1450851/LES_Sims/'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/sim'+sim+'/'

if os.path.exists(pathOUT):
    print(f"The forlder '{pathOUT}' aready exists.")
else:
    os.makedirs(pathOUT)
    print(f"Created folder '{pathOUT}'.")

pcnt1 = 50 #18000; %Important! used in get_var 
avg_time = pcnt1*1 #total timesteps averaged  Important! used in get_var
startavg = 1
endavg = 1 # max is avg_time/pcnt3
incskip = 0 #should be 0 unless simulation incomplete (# of avgs incompleted)

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

T_STC = 305 #320; %298.15; %[K], temperature scale
dt = 0.1 #05; %0.000005;
zi = 1000.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0


#%% Plot ke.txt for convergence checking

parentpath = simPath + f'sim{sim}/'
file_name = 'ke.txt'

# Load and import data
ke = np.genfromtxt(parentpath + file_name)
ts = np.linspace(0, len(ke) * wbase, len(ke))

# Create the plot
plt.figure()
plt.plot(ts, ke, 'k-', linewidth=2.5)
plt.xlabel('timesteps')
plt.ylabel('MKE')
plt.gca().set_facecolor('white')  # Set background color to white

# plt.savefig(figPath+'ke.png',dpi=300,facecolor='white', edgecolor='white')
# Show the plot
plt.show()

#%% Import simulation parameters and build topography

# Input and output paths
ipath = simPath + f'sim{sim}/output/ts1_field/'
opath = simPath + f'sim{sim}/output/'

# Read parameter file for ta1_field
with open(ipath + 'parameters.txt', 'r') as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

nx = int(param[0])
ny = int(param[1])
nz = int(param[2])
dx = param[3]
dy = param[4]
dz = param[5]
actLaunch = int(param[7])
mpiProc = int(param[9])
Re = param[10]
alpha = param[11]
ibm = param[12]
rot = param[13]
nzTot = nz*mpiProc  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, nx) * dx
y = np.arange(0, ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

if shifting_z:
    # Modify axes to move 0 to IBM surface
    z_uvp = z_uvp[5:]
    z_uvp = z_uvp - z_uvp[0]
    z_w = z_w[5:]
    z_w = z_w - z_w[0]

# Build interface and solidity of IBM
if not completed_sim:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))
    nt = len(ke) // pcnt1
elif ibm == 1:
    # Assuming you have a function build_phi to load phi data from a file
    phi = build_phi(opath + 'phi_functions/', nx, ny, nzTot, mpiProc)
    intf, iintf = build_intf(phi, dz)
else:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))

if ibm:
    phi_uv = np.zeros_like(phi)
    for k in range(nz - 1):
        phi_uv[:, :, k] = (phi[:, :, k] + phi[:, :, k + 1]) / 2.0

if PCON_flag and v_fracflag:
    v_frac = get_var(opath + 'cut_cell/', 'v_frac', nx, ny, nzTot, 1, iintf, mpiProc)
    a_cut = get_var(opath + 'cut_cell/', 'a_cut', nx, ny, nzTot, 1, iintf, mpiProc)

#%% plot the topography or IBM

zi = 1000
fig, ax = plt.subplots()
contour = ax.contourf(x * zi, y * zi, intf.T * zi, 30, cmap='viridis')
# for i in range(len(coord)):
    # ax.plot(x[coord[i][0]]*zi,y[coord[i][1]]*zi,'ok')
# contour = ax.contourf(x, y, intf.T, 30, cmap='viridis')
plt.title('Geometry (top-down view)', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
plt.xlabel('$x$', usetex=True)
plt.ylabel('$y$', usetex=True)
plt.colorbar(contour)
ax.set_aspect('auto')
plt.gca().set_facecolor('white')

# plt.savefig(pathOUT+'Figures/topo.png',dpi=300,facecolor='white', edgecolor='white')

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
            

#%%Import variables

import pickle

save_data = False
load_data = False

jt = 1800046

var = ['u','v','w','p']
    
if load_data:
    with open(pathOUT+'data_snapshots.pkl', 'rb') as f:
    # Load the data from the pickle file
        data = pickle.load(f)
else:
    data = dict()

    for i in range(len(var)):
        data[var[i]] = get_snapvar(ipath,var[i],nx,ny,nzTot,iintf,mpiProc,jt)

if save_data:
    with open(pathOUT+'data_snapshots.pkl', 'wb') as f:
        pickle.dump(data, f)
        
#%%

var = 'w'
# tmp = copy.deepcopy(data['p']-(data['uu']+data['vv']+data['ww']))
tmp = copy.deepcopy(data[var])
# tmp[dist<0] = float('nan')

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nzTot)*dz
X, Y = np.meshgrid(x_ax,z_ax)
xslice = 150
yslice = 63 #int(nz/2)
zslice = 50

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,z_ax,tmp[xslice,:,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')
# axs.set_ylim(z_ax[0],z_ax[-1])
axs.plot(np.arange(0,nx)*dx,(intf[:,yslice]),ls='-',c='k')
axs.plot(np.arange(0,nx)*dx,(intf[:,yslice]+(39/zi)),ls='--',c='k')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var)
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_c[:, :, -2]
  return phi_c


