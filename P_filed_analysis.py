#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:06:56 2024

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import math 
import pickle
import copy

#%%Import the data sets for ATTO1,2,12

path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/'

cases = ['simATTO1_256x256x384_v2_out5hr','simATTO2_256x256x384_out5hr','simATTO12_256x256x384_out5hr']

names = ['A1','A2','A12']

var = ['u','v','w','p','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','txx','txy','txz','tyy','tyz',\
       'tzz','dxx','dxy','dxz','dyy','dyz','dzz','fdx','fdy','fdz','dpdx','dpdy','dpdz','pdudx','pdvdy','pdwdz','pu','pv','pw','dudx','dudy',\
       'dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2',\
       'utxx','utxy','utxz','vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','ufdx','vfdy','wfdz','utyy','utzz','vtxx','vtzz','wtxx','wtyy']
    
for i in range(len(cases)):
    with open(path+cases[i]+'/data.pkl', 'rb') as f:
        # Load the data from the pickle file
        data = pickle.load(f)
    exec(f"data_{names[i]} = data")
    for j in range(len(var)):
        exec(f"data_{names[i]}[var[j]] = np.mean(data_{names[i]}[var[j]],axis=3)")
    
#%%Compute the topography to place nan below the surface

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

simpath = '/scratch/general/nfs1/u1450851/LES_Sims/'

T_STC = 305 #320; %298.15; %[K], temperature scale
zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

intf = dict()
iintf = dict()

for i in range(len(cases)):
    
    # Read parameter file for ta1_field
    with open(simpath + cases[i] + '/output/ta1_field/parameters.txt', 'r') as param_file:
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
    nt = int(param[9])
    actLaunch = int(param[10])
    mpiProc = int(param[11])
    Re = param[12]
    Ri = param[13]
    Pr = param[14]
    alpha = param[15]
    SGS = param[16]
    S_FLAG = param[17]
    ibm = param[18]
    rot = param[19]
    nzTot = Nz  # nz * mpiProc
    z_shift = 4.5 * dz  # IBM surface vertical shift
    canopyH = (39+z_shift*zi)/zi
    
    # Build normalized axes
    
    # Build interface and solidity of IBM
    # Assuming you have a function build_phi to load phi data from a file
    phi = build_phi(simpath + cases[i] + '/output/phi_functions/', Nx, Ny, nzTot, mpiProc)
    intf[cases[i]], iintf[cases[i]] = build_intf(phi, dz)

x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz
height = math.ceil(canopyH/dz) # Canopy height in grid points.
kappa = 0.4
Nz_SLayer = int(Nz/2)

zeds = np.zeros((Nx,Ny,Nz))
for i in range(0,Nx):
    for j in range(0,Ny):
        zeds[i,j,:] = np.arange(0,Nz)*(dz*zi)
        
dist = dict()

for i in range(len(cases)):
    dist[cases[i]] = copy.deepcopy(zeds)

for case in range(len(cases)):
    for i in range(0,Nx):
        for j in range(0,Ny):
            for k in range(0,Nz):
                dist[cases[case]][i,j,k] = dist[cases[case]][i,j,k] - intf[cases[case]][i,j]*zi
            
#%%
pathFig = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/Pres_Study/'

n = 2

fig, ax = plt.subplots()
contour = ax.contourf(x * zi, y * zi, intf[cases[n]].T * zi, 30, cmap='viridis')
# for i in range(len(coord)):
    # ax.plot(x[coord[i][0]]*zi,y[coord[i][1]]*zi,'ok')
# contour = ax.contourf(x, y, intf.T, 30, cmap='viridis')
plt.title('Geometry (top-down view)', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
plt.xlabel('$x$', usetex=True)
plt.ylabel('$y$', usetex=True)
plt.colorbar(contour)
ax.set_aspect('auto')
plt.gca().set_facecolor('white')
# plt.savefig(pathFig+'topo_A12.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%Place NaNs below the IBM surface

for i in range(len(var)):
    data_A1[var[i]][(dist[cases[0]]<0)] = float("nan")
    data_A2[var[i]][(dist[cases[1]]<0)] = float("nan")
    data_A12[var[i]][(dist[cases[2]]<0)] = float("nan")

#%% pcolor test plots

var = 'p'
# tmp = copy.deepcopy(data_tavg['p']-(data_tavg['uu']+data_tavg['vv']+data_tavg['ww']))
tmp = copy.deepcopy(data_A1[var])
# tmp[dist<0] = float('nan')

x_ax = np.arange(0,Nx)*dx
y_ax = np.arange(0,Ny)*dy
z_ax = np.arange(0,Nz)*dz

yslice = 150 #int(nz/2)
zslice = 50

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
# divnorm = colors.TwoSlopeNorm(vmin=-5,vcenter=0,vmax=10)

# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.streamplot(X,Y,tmp[:,yslice,:].T,tmp2[:,yslice,:].T)
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')
# axs.plot((x_ax),(np.mean(topodata['intf'],axis=1) + h_canopy )/dm.zi,'--k')
# axs.set_ylim(z_ax[0],z_ax[-1])
# axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
axs.plot(x_ax,canopyH-z_shift+intf[cases[0]][:,yslice],color='black')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
axs.plot(x_ax,intf[cases[0]][:,yslice],color='black')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var)
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
# fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%%Remove the planar average from the local value of pressure

pathFig = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/Pres_Study/'

P_A1_lin = np.zeros((Nx,Ny,Nz),'d',order='F')
P_A2_lin = np.zeros((Nx,Ny,Nz),'d',order='F')
P_A12_lin = np.zeros((Nx,Ny,Nz),'d',order='F')

for k in range(0,Nz):
    P_A1_lin[:,:,k] = data_A1['p'][:,:,k] - np.nanmean(data_A1['p'][:,:,k],axis=(0,1))
    P_A2_lin[:,:,k] = data_A2['p'][:,:,k] - np.nanmean(data_A2['p'][:,:,k],axis=(0,1))
    P_A12_lin[:,:,k] = data_A12['p'][:,:,k] - np.nanmean(data_A12['p'][:,:,k],axis=(0,1))
    
zslice = 35

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,8))

# plt1 = axs.pcolormesh(x_ax,z_ax,P_A2_lin[:,yslice,:].T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,P_A1_lin[:,yslice,:].T+P_A2_lin[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,P_A12_lin[:,:,zslice].T-(P_A1_lin[:,:,zslice].T+P_A2_lin[:,:,zslice].T),cmap= 'coolwarm',vmin=-1,vmax=1)
# plt1 = axs.pcolormesh(x_ax,y_ax,P_A1_lin[:,:,zslice].T+P_A2_lin[:,:,zslice].T,cmap= 'coolwarm',vmin=-2,vmax=2)
# plt1 = axs.pcolormesh(x_ax,y_ax,P_A12_lin[:,:,zslice].T,cmap= 'coolwarm',vmin=-2,vmax=2)

# axs.set_ylim(z_ax[0],z_ax[-1])
# axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,canopyH-z_shift+intf[cases[1]][:,yslice],color='black')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,intf[cases[1]][:,yslice],color='black')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var)
axs.set_ylabel(r'$y/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)

# plt.savefig(pathFig+'P_A12minusA1A2_'+str(zslice)+'.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%Compute pressure field with hydrostatic pressure

p_hydro = dict()
g = 9.81

for i in range(len(cases)):
    p_hydro[cases[i]] = 9.81*dist[cases[i]]
    



fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,8))

# plt1 = axs.pcolormesh(x_ax,z_ax,P_A12_lin[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,z_ax,P_A1_lin[:,yslice,:].T+P_A2_lin[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,P_A12_lin[:,:,zslice].T-(P_A1_lin[:,:,zslice].T+P_A2_lin[:,:,zslice].T),cmap= 'coolwarm',vmin=-1,vmax=1)
plt1 = axs.pcolormesh(x_ax,y_ax,p_hydro[cases[0]][:,:,zslice].T+p_hydro[cases[1]][:,:,zslice].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,p_hydro[cases[2]][:,:,zslice].T,cmap= 'coolwarm')

# axs.set_ylim(z_ax[0],z_ax[-1])
# axs.plot(x_ax,canopyH-z_shift+np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,canopyH-z_shift+intf[cases[1]][:,yslice],color='black')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.plot(x_ax,intf[cases[1]][:,yslice],color='black')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var)
axs.set_ylabel(r'$y/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)

# plt.savefig(pathFig+'P_A12minusA1A2_'+str(zslice)+'.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()


































































