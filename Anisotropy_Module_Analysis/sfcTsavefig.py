#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:51:23 2026

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile

def create_patches(T_mu, T_sig, nx, ny, nbr_pth_x, nbr_pth_y, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Number of grid points per patches in x and y
    pth_sz_x = nx // nbr_pth_x
    pth_sz_y = ny // nbr_pth_y

    # Surface variables
    T_s_unshfd = np.zeros((nx, ny))
    T_s = np.zeros((nx, ny))
    T_s_pth = np.zeros(nbr_pth_x * nbr_pth_y)

    # Random number generation
    pth_ind = np.random.permutation(nbr_pth_x * nbr_pth_y)
    T_lst = np.random.normal(T_mu, T_sig, nbr_pth_x * nbr_pth_y)

    # Random patch temperature
    for kk in range(nbr_pth_x * nbr_pth_y):
        T_s_pth[pth_ind[kk]] = T_lst[kk]
    meanT_s = np.mean(T_s_pth)
    stdT_s = np.std(T_s_pth)

    # Reshape T_s_pth to 2D array
    T_s_pth = np.reshape(T_s_pth, (nbr_pth_x, nbr_pth_y))

    # Add patches into full surface variable
    for ii in range(nbr_pth_x):
        for jj in range(nbr_pth_y):
            T_s_unshfd[(ii*pth_sz_x):((ii+1)*pth_sz_x), (jj*pth_sz_y):((jj+1)*pth_sz_y)] = T_s_pth[ii, jj]

    # Shifting surface distribution to make it periodic
    T_s[(pth_sz_x//2):nx, (pth_sz_y//2):ny] = T_s_unshfd[0:(nx-pth_sz_x//2), 0:(ny-pth_sz_y//2)]
    T_s[0:(pth_sz_x//2), (pth_sz_y//2):ny] = T_s_unshfd[(nx-pth_sz_x//2):nx, 0:(ny-pth_sz_y//2)]
    T_s[(pth_sz_x//2):nx, 0:(pth_sz_y//2)] = T_s_unshfd[0:(nx-pth_sz_x//2), (ny-pth_sz_y//2):ny]
    T_s[0:(pth_sz_x//2), 0:(pth_sz_y//2)] = T_s_unshfd[(nx-pth_sz_x//2):nx, (ny-pth_sz_y//2):ny]

    # Check mean surface temperature before and after shifting
    mean_unshfd = np.mean(np.mean(T_s_unshfd))
    mean_shfd = np.mean(np.mean(T_s))
    print(f"# mean before/after shifting {mean_unshfd} | {mean_shfd}")

    return T_s, T_s_unshfd, T_lst   

#%%
    
nx=256
ny=256

lx = 2*np.pi*1000
ly = 2*np.pi*1000

dx = lx/nx
dy = ly/ny

x=np.arange(0,nx)*dx
y=np.arange(0,ny)*dy

# Set default figure background color to white
plt.rcParams['figure.facecolor'] = 'w'

# Seed for the RNG
# seed = 1  # -v1
seed = 4  # -v2
# seed = 11  # -v3
# seed = 25

# # Input parameters
T_mu = 290
T_sig = 5

# # Number of patches in x and y directions
nbr_pth_x = 8
nbr_pth_y = 8

T_s, T_s_unshfd, T_lst = create_patches(T_mu, T_sig, nx, ny, nbr_pth_x, nbr_pth_y, seed)

meanT_s = np.mean(T_lst)
stdT_s = np.std(T_lst)

fig,axs = plt.subplots(2,1,tight_layout=True,sharex=True,figsize=(5,8))

im = axs[0].pcolormesh(x,y,np.ones((nx,ny))*T_mu, shading='gouraud',cmap='hot_r',vmin=285,vmax=295)
axs[1].pcolormesh(x,y,T_s,shading='gouraud',cmap='hot_r',vmin=285,vmax=295)

axs[1].set_xlabel(r"$x$ [m]", fontsize=15)
axs[0].set_ylabel(r"$y$ [m]", fontsize=15)
axs[1].set_ylabel(r"$y$ [m]", fontsize=15)

cb = fig.colorbar(im, ax=axs[0], orientation='horizontal', location='top', label='T [K]')
cb.ax.xaxis.set_label_position('top')
cb.set_label('T [K]', fontsize=15)
cb.ax.tick_params(labelsize=12)
axs[0].tick_params(labelsize=12)
axs[1].tick_params(labelsize=12)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Proposal_sfcT.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()


















































