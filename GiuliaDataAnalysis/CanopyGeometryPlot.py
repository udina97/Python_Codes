# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 07:23:47 2025

@author: udina
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

#%%Set case and path to data


# Path to the data files:
path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/'
os.chdir(path)  

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','Hom_Amazon_9mps']
case = 0

#%% Defining the main parameters of the simulations

Nx = 256
Ny = 256
Nz = 256
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 1
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

x = np.arange(0,Nx)*dx
y = np.arange(0,Ny)*dy
z_w = np.arange(0,Nz)*dz
z_uvp = np.arange(0,Nz)*dz + dz/2

zi = 1000
uscale = 0.313
Hcanopy = 39/zi
kappa = 0.4  # von Karman constant
LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
height = math.ceil(Hcanopy/dz)

Ntwr = 100
Nz_Slayer = 126

#%%Import sfc data

sfc = np.load(path+'input_txt_files/'+cases[case]+'/sfc.npy')

#%%Plot sfc data

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
terrain = plt.get_cmap('terrain')
terrain_truncated = LinearSegmentedColormap.from_list(
    'terrain_truncated', terrain(np.linspace(0.25, 1, 100)))

labels=['(a)','(b)','(c)','(d)','(e)','(f)']

fig,axs = plt.subplots(4,3,tight_layout=True,sharey=True,sharex=True,figsize=(11,6))
axs = axs.flatten()
for i in range(len(cases)-1):
    sfc = np.load(path+'input_txt_files\\'+cases[i]+'\\sfc.npy')
    axs[i].pcolormesh(x*zi,y*zi,sfc.T,cmap=terrain_truncated.reversed())
    
axs[0].set_ylabel(r"$y[m]$",fontsize=15)
axs[3].set_ylabel(r"$y[m]$",fontsize=15)

for i in range(3,6):
    axs[i].set_xlabel(r"$x[m]$",fontsize=15)
    
for i in range(len(axs)):
    axs[i].text(0.01,0.9,labels[i],fontsize=15,transform=axs[i].transAxes)

plt.show()



#%%Creeate a 3D LAD field

sfc[(sfc > 0)] = 1

sfc3D = sfc[:,:,np.newaxis] * LAD

#%%Plot a xz profile

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(11,6),sharex=True,sharey=True)
axs = axs.flatten()
for i in range(len(cases)-1):
    sfc = np.load(path+'input_txt_files\\'+cases[i]+'\\sfc.npy')
    sfc[(sfc > 0)] = 1
    sfc3D = sfc[:,:,np.newaxis] * LAD
    axs[i].pcolormesh(x,z_uvp[:10]/(39/zi),sfc3D[:,50,:10].T,cmap='Greens',shading='auto')

axs[0].set_ylabel(r"$z/h_C$",fontsize=15)
axs[3].set_ylabel(r"$z/h_C$",fontsize=15)

for i in range(3,6):
    axs[i].set_xlabel(r"$x[m]$",fontsize=15)
    
for i in range(len(axs)):
    axs[i].text(0.01,0.9,labels[i],fontsize=15,transform=axs[i].transAxes)
    axs[i].set_ylim(0,1)

plt.show()

#%%Composite plot

from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
terrain = plt.get_cmap('terrain')
green_cmap = LinearSegmentedColormap.from_list("mygreens", ["#FFFFFF", "#006d2c"])
terrain_truncated = LinearSegmentedColormap.from_list(
    'terrain_truncated', terrain(np.linspace(0.32, 1, 100)))

yslice = [70,90,115]
labels=['a)','b)','c)','d)','e)','f)']

fig = plt.figure(figsize=(11, 11))
gs = gridspec.GridSpec(5, 3, height_ratios=[2,1,0.3,2,1], figure=fig)

ax = [[None]*3 for _ in range(5)]

# Row 0 (top terrain plots)
for i in range(3):
    ax[0][i] = fig.add_subplot(gs[0, i], sharey=ax[0][0] if i > 0 else None)

# Row 1 (LAD plots below each terrain)
for i in range(3):
    ax[1][i] = fig.add_subplot(gs[1, i])

# Row 2 (bottom terrain plots)
for i in range(3):
    ax[3][i] = fig.add_subplot(gs[3, i], sharey=ax[3][0] if i > 0 else None)

# Row 3 (LAD plots below each terrain)
for i in range(3):
    ax[4][i] = fig.add_subplot(gs[4, i])


for i in range(0,3):
    sfc = np.load(path+'input_txt_files/'+cases[i]+'/sfc.npy')
    sfc[(sfc > 0)] = 1
    contour = ax[0][i].pcolormesh(x*zi,y*zi,sfc.T,cmap=green_cmap)
    ax[0][i].tick_params(axis='x', which='major', labelsize=12)
    ax[0][i].axhline(yslice[i]*dy*zi,ls='--',c='k')
    sfc[(sfc == 0)] = np.nan
    sfc3D = sfc[:,:,np.newaxis] * LAD
    ax[1][i].pcolormesh(x*zi,z_uvp[:10]/(39/zi),sfc3D[:,yslice[i],:10].T,cmap='Greens',shading='auto')
    ax[1][i].spines['top'].set_visible(False)
    ax[1][i].spines['right'].set_visible(False)

for i in range(0,3):
    sfc = np.load(path+'input_txt_files/'+cases[i+3]+'/sfc.npy')
    sfc[(sfc > 0)] = 1
    contour = ax[3][i].pcolormesh(x*zi,y*zi,sfc.T,cmap=green_cmap)
    ax[3][i].tick_params(axis='x', which='major', labelsize=12)
    ax[3][i].axhline(yslice[i]*dy*zi,ls='--',c='k')
    sfc[(sfc == 0)] = np.nan
    sfc3D = sfc[:,:,np.newaxis] * LAD
    ax[4][i].pcolormesh(x*zi,z_uvp[:10]/(39/zi),sfc3D[:,yslice[i],:10].T,cmap='Greens',shading='auto')
    ax[4][i].spines['top'].set_visible(False)
    ax[4][i].spines['right'].set_visible(False)
    
for i in range(0,3):
    ax[0][i].tick_params(labelbottom=False)
    # ax[1][i].tick_params(labelbottom=False)
    ax[3][i].tick_params(labelbottom=False)
    ax[1][i].tick_params(axis='x', which='major', labelsize=12)
    ax[1][i].set_xlabel(r"$x[m]$",fontsize=15)
    ax[4][i].tick_params(axis='x', which='major', labelsize=12)
    ax[4][i].set_xlabel(r"$x[m]$",fontsize=15)
    # ax[0][i].text(0.05,0.9,labels[i],fontsize=15,weight='bold',transform=ax[0][i].transAxes)
    # ax[2][i].text(0.05,0.9,labels[i+3],fontsize=15,weight='bold',transform=ax[2][i].transAxes)
    ax[0][i].text(
        0.0, 1.0, labels[i],
        transform=ax[0][i].transAxes +
                 ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans),
        fontsize=15, va='bottom', fontfamily='serif'
    )

    # --- Labels (d), (e), (f) on the second terrain row ---
    ax[3][i].text(
        0.0, 1.0, labels[i+3],
        transform=ax[3][i].transAxes +
                 ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans),
        fontsize=15, va='bottom', fontfamily='serif'
    )

for i in range(1,3):
    ax[0][i].tick_params(labelleft=False)
    ax[1][i].tick_params(labelleft=False)
    ax[3][i].tick_params(labelleft=False)
    ax[4][i].tick_params(labelleft=False)
    
ax[0][0].set_ylabel(r"$y[m]$",fontsize=15)
ax[1][0].set_ylabel(r"$z/h_C$",fontsize=15)
ax[3][0].set_ylabel(r"$y[m]$",fontsize=15)
ax[4][0].set_ylabel(r"$z/h_C$",fontsize=15)

ax[0][0].tick_params(axis='y', which='major', labelsize=12)
ax[1][0].tick_params(axis='y', which='major', labelsize=12)
ax[3][0].tick_params(axis='y', which='major', labelsize=12)
ax[4][0].tick_params(axis='y', which='major', labelsize=12)

plt.subplots_adjust(left=0.08,
                    bottom=0.06, 
                    right=0.97, 
                    top=0.95, 
                    wspace=0.06, 
                    hspace=0.2)
# for i in range(3):
#     ax[1][i].set_ylim(0, 1)
#     ax[3][i].set_ylim(0, 1)

plt.gca().set_facecolor('white')
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'Topo_HC_v2.png',dpi=300,edgecolor='white',facecolor='white')
plt.show()



































