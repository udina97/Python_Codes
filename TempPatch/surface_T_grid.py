#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:29:54 2023

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
    
    
nx=256
ny=256

x=np.arange(0,nx)
y=np.arange(0,ny)

# Set default font size
plt.rc('font', size=10)

# Set default text interpreter to LaTeX
plt.rc('text', usetex=True)

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

# str_case = f'sfcdistrib_T_{T_mu}_sigT_{T_sig}_{nx}x{ny}_{nbr_pth_x}x{nbr_pth_y}_seed{seed}'
str_case = f'/uufs/chpc.utah.edu/common/home/u1450851/LES_code/LES-2Ddecomp/build/input/surface_SC1'

T_s, T_s_unshfd, T_lst = create_patches(T_mu, T_sig, nx, ny, nbr_pth_x, nbr_pth_y, seed)

meanT_s = np.mean(T_lst)
stdT_s = np.std(T_lst)

# Writing the surface distribution into a file
# with open(str_case + '.dat', 'wb') as f:
#     for jj in range(ny):
#         np.array(T_s[:, jj]).tofile(f)

# T_s = np.full((nx,ny),290)

# T_s[:,:int(nx/2)] = 280

# with open(str_case + '.dat', 'wb') as f:
#     for jj in range(ny):
#         np.array(T_s[:, jj]).tofile(f)

plt.pcolormesh(x,y,T_s,shading='gouraud',cmap='coolwarm',vmin=285,vmax=295)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    