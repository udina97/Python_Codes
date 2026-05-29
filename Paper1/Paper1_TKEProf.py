#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:34:43 2025

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.transforms import ScaledTranslation

#%%Set path to the profiles

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/'
# cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat','simulation_G']
cases = ['Flat','Sinusoidal','ATTO','Gap_8_9mps','Patch_8_9mps','simulation_G']
Nz_SLayer = 200

#%%Plot

layout = [['a)', 'b)', 'c)'],
          ['d)', 'e)', 'f)']]

fig, axs_dict = plt.subplot_mosaic(layout, layout='constrained',figsize=(10,8),sharey=True)

for label, ax in axs_dict.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans)),
        fontsize=15, va='bottom', fontfamily='serif')

# fig,axs = plt.subplots(5,4,tight_layout=True,sharey=True,sharex='col',figsize=(10,8))

axs = np.array([[axs_dict[label] for label in row] for row in layout])

for j in range(axs.shape[1]):        # loop over columns
    for i in range(1, axs.shape[0]): # loop over rows in that column
        axs[i, j].sharex(axs[0, j])  # share with the top axis in that column
        
# Hide inner labels
for ax in axs.ravel():
    ax.label_outer()

for i in range(len(cases)):
    if i == 0:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000; canopyH = 39
        prof = np.load(path_to_data + 'RedTKE_' + cases[i] + '.npy')
        axs[0,0].plot(prof[0,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#0072B2')
        axs[0,0].plot(prof[1,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#E69F00')
        axs[0,0].plot(prof[2,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#CC79A7')
    elif i > 0 and i < 3:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000; canopyH = 39
        prof_p = np.load(path_to_data + 'RedTKE_' + cases[i] + '_p.npy')
        prof_v = np.load(path_to_data + 'RedTKE_' + cases[i] + '_v.npy')
        axs[0,i].plot(prof_p[0,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#0072B2',ls='-')
        axs[0,i].plot(prof_p[1,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#E69F00',ls='-')
        axs[0,i].plot(prof_p[2,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#CC79A7',ls='-')
        axs[0,i].plot(prof_v[0,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#0072B2',ls='--')
        axs[0,i].plot(prof_v[1,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#E69F00',ls='--')
        axs[0,i].plot(prof_v[2,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#CC79A7',ls='--')
    elif i >= 3 and i < 5:
        lz = 1; nz = 256; dz = lz/nz; zi = 1000; canopyH = 39
        prof_p = np.load(path_to_data + 'RedTKE_' + cases[i] + '_p.npy')
        prof_f = np.load(path_to_data + 'RedTKE_' + cases[i] + '_f.npy')
        axs[1,i-3].plot(prof_f[0,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#0072B2',ls='-')
        axs[1,i-3].plot(prof_f[1,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#E69F00',ls='-')
        axs[1,i-3].plot(prof_f[2,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#CC79A7',ls='-')
        axs[1,i-3].plot(prof_p[0,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#0072B2',ls='--')
        axs[1,i-3].plot(prof_p[1,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#E69F00',ls='--')
        axs[1,i-3].plot(prof_p[2,:],(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='#CC79A7',ls='--')
    else:
        z = np.arange(0.5,152.5,1)/15.3
        prof_xy = np.load(path_to_data + 'RedTKE_' + cases[i] + '_xy.npy')
        prof_tw = np.load(path_to_data + 'RedTKE_' + cases[i] + '_tw.npy')
        axs[1,i-3].plot(prof_xy[0,:],z,c='#0072B2',ls='-')
        axs[1,i-3].plot(prof_xy[1,:],z,c='#E69F00',ls='-')
        axs[1,i-3].plot(prof_xy[2,:],z,c='#CC79A7',ls='-')
        axs[1,i-3].plot(prof_tw[0,:],z,c='#0072B2',ls='--')
        axs[1,i-3].plot(prof_tw[1,:],z,c='#E69F00',ls='--')
        axs[1,i-3].plot(prof_tw[2,:],z,c='#CC79A7',ls='--')
        
for i in range(0,len(axs)):
    axs[i,0].set_ylim(0,10)
    axs[i,0].set_ylabel(r"$z/h_C$",fontsize=16)
    
for i in range(len(axs)):
    for j in range(len(axs[0])):
        axs[i,j].axhline(1,c='k',ls='--')
        axs[i,j].axvline(0,c='k',ls='-')
        axs[i,j].grid(alpha=0.2)
        axs[i,j].set_xlim(-9,15)
    
axs[-1,0].set_xlabel(r'$\left \langle \frac{\partial e}{\partial t} \right \rangle \frac{h_C}{u_*^3}$',fontsize=16)
axs[-1,1].set_xlabel(r'$\left \langle \frac{\partial e}{\partial t} \right \rangle \frac{h_C}{u_*^3}$',fontsize=16)
axs[-1,2].set_xlabel(r'$\left \langle \frac{\partial e}{\partial t} \right \rangle \frac{h_C}{u_*^3}$',fontsize=16)

axs[0,0].tick_params(axis='y', which='major', labelsize=12)
axs[1,0].tick_params(axis='y', which='major', labelsize=12)

for i in range(3):
    axs[-1,i].tick_params(axis='x', which='major', labelsize=12)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'TKE_TwrProf_AllCases_colorblind.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()