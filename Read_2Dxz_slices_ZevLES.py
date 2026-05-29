#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:11:09 2024

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

from functions import build_phi, build_intf, get_var, get_snapvar, get_2Dxz_slice

#%%
#inputs

sim = 'bicheng_hill_slices_v2'

simPath = '/scratch/general/nfs1/u1450851/LES_Sims/'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/sim'+sim+'/'

if os.path.exists(pathOUT):
    print(f"The forlder '{pathOUT}' aready exists.")
else:
    os.makedirs(pathOUT)
    print(f"Created folder '{pathOUT}'.")

pcnt1 = 50 #18000; %Important! used in get_var 

completed_sim = True

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
ipath = simPath + f'sim{sim}/output/ts1_slices/'
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

zeds = np.zeros((nx,ny,nzTot))

for i in range(0,nx):
    for j in range(0,ny):
        zeds[i,j,:] = np.arange(0,nzTot)*(dz*zi)
    
dist = copy.deepcopy(zeds)

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nzTot):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi
            

#%%Import variables

import pickle

save_data = False
load_data = False

jt_start = 1260011
nsteps = 4000

var = ['u','v','w']
data = dict()
for i in range(len(var)):
    data[var[i]] = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
    
    
for i in range(0,nsteps):
    jt = jt_start + i
    print(i)
    for j in range(len(var)):
        data[var[j]][:,:,i] = get_2Dxz_slice(ipath,var[j],nx,nzTot,iintf,mpiProc,jt)
        



# if load_data:
#     with open(pathOUT+'data_snapshots.pkl', 'rb') as f:
#     # Load the data from the pickle file
#         data = pickle.load(f)
# else:
#     data = dict()

#     for i in range(len(var)):
#         data[var[i]] = get_2Dxz_slice(ipath,var[i],nx,nzTot,iintf,mpiProc,jt)

# if save_data:
#     with open(pathOUT+'data_snapshots.pkl', 'wb') as f:
#         pickle.dump(data, f)
        
#%%

var = 'w'
tmp = copy.deepcopy(data[var])
tmp[dist[:,100,:]<0] = float('nan')

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nzTot)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 100

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))

# plt1 = axs.pcolormesh(x_ax,z_ax,np.mean(tmp,axis=1).T,cmap= 'coolwarm')
plt1 = axs.pcolormesh(x_ax,z_ax,tmp.T,cmap= 'coolwarm')
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
        
#%%

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib

# %%
### VIDEO ATTEMPT ####

data_vel = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
for i in range(0,nsteps):
    data_vel[:,:,i] = np.sqrt(data['u'][:,:,i]**2 + data['v'][:,:,i]**2 + data['w'][:,:,i]**2)
    data_vel[:,:,i][dist[:,100,:]<0] = float('nan')

fig,axs = plt.subplots(1,1,figsize=(8,4),dpi=200)
x_ax = np.arange(0,nx)*dx
z_ax = np.arange(0,nzTot)*dz
yslice = 100

def animate(i):
    print(i)
    data_u=data_vel[:,:,i]
    axs.clear()
    #ax1.imshow(data,origin='lower',cmap='terrain',vmax=vmax,vmin=0)
    axs.pcolormesh(x_ax,z_ax,data_u.T,cmap= 'bwr',vmin=-20,vmax=20)
    axs.plot(np.arange(0,nx)*dx,(intf[:,yslice]),ls='-',c='k')
    axs.plot(np.arange(0,nx)*dx,(intf[:,yslice]+(39/zi)),ls='--',c='k')
    axs.axis('off')
    axs.set_title('Wind Flow')
    # axs.set_ylabel(r'$z/z_i$')
    # axs.set_xlabel(r'$x/z_i$')
    
    #fig.subplots_adjust(wspace=.45, hspace=.45)
    
    return fig
    
ani=FuncAnimation(fig,animate,frames=900,interval=10,repeat=True)
# FFwriter = matplotlib.animation.FFMpegWriter(fps=10)
# ani.save(pathOUT+'ani_2.mp4',writer=FFwriter)
HTML(ani.to_jshtml())
# HTML(ani.to_html5_video())

#%%Compute anisotropy

def Anisotropy2D(Nx,Nz,R11,R22,R33,R12,R13,R23):
    "Calling the Anisotorpy() function"
    
    import numpy as np
    
    # Calculate the TKE
    e = R11[:,:] + R22[:,:] + R33[:,:]

    # Define identity matrix
    Id = np.eye(3,3) 

    # xB = np.zeros((Nx,Nz),order='F')
    # yB = np.zeros((Nx,Nz),order = 'F')
    
    xB = np.zeros((Nx,Nz), dtype=np.float64)
    yB = np.zeros((Nx,Nz), dtype=np.float64)


    #... Loop over each point of the LES domain.
    for k in range(0,Nz):
        # print(f"iteration = {k}")
        for i in range(0,Nx):
           
                # ... Reynolds Matrix
                R = np.matrix([[R11[i,k], R12[i,k], R13[i,k]],
                               [R12[i,k], R22[i,k], R23[i,k]],
                               [R13[i,k], R23[i,k], R33[i,k]]])
       
       

                # .. calculate the anisotropy tensor
                B = R/e[i,k] -1/3*Id
            
                # .. calculate the eigenvalues values
                [eigenVal,eigenVec] = np.linalg.eig(B)
       
                #Sort Eigenvalues in decreasing order
                SortedeigenVal = np.sort(eigenVal)[::-1] 
           
       
                # .. compute the C coefficients for Barycentric map
                C1c = SortedeigenVal[0] - SortedeigenVal[1]
                C2c = 2*(SortedeigenVal[1] - SortedeigenVal[2])
                C3c = 3*SortedeigenVal[2] + 1
       
                # .. compute the barycentric invariants
                xB[i,k] = C1c + C3c*1/2
                yB[i,k] = C3c*np.sqrt(3)/2
       
    
    
    return(xB,yB)

# xB = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
# yB = np.zeros((nx,nzTot,nsteps), dtype=np.float64)

uu = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
uv = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
uw = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
vv = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
vw = np.zeros((nx,nzTot,nsteps), dtype=np.float64)
ww = np.zeros((nx,nzTot,nsteps), dtype=np.float64)

for i in range(0,nsteps):
    print(i)
    uu[:,:,i] = (data['u'][:,:,i] - np.mean(data['u'],axis=(2)))*(data['u'][:,:,i] - np.mean(data['u'],axis=(2)))
    uv[:,:,i] = (data['u'][:,:,i] - np.mean(data['u'],axis=(2)))*(data['v'][:,:,i] - np.mean(data['v'],axis=(2)))
    uw[:,:,i] = (data['u'][:,:,i] - np.mean(data['u'],axis=(2)))*(data['w'][:,:,i] - np.mean(data['w'],axis=(2)))
    vv[:,:,i] = (data['v'][:,:,i] - np.mean(data['v'],axis=(2)))*(data['v'][:,:,i] - np.mean(data['v'],axis=(2)))
    vw[:,:,i] = (data['v'][:,:,i] - np.mean(data['v'],axis=(2)))*(data['w'][:,:,i] - np.mean(data['w'],axis=(2)))
    ww[:,:,i] = (data['w'][:,:,i] - np.mean(data['w'],axis=(2)))*(data['w'][:,:,i] - np.mean(data['w'],axis=(2)))
    
#     xB[:,:,i],yB[:,:,i] = Anisotropy2D(nx, nzTot, uu[:,:,i], vv[:,:,i], ww[:,:,i], uv[:,:,i], uw[:,:,i], vw[:,:,i])

#%%

xB,yB = Anisotropy2D(nx, nzTot, uu[:,:,4], vv[:,:,4], ww[:,:,4], uv[:,:,4], uw[:,:,4], vw[:,:,4])




































