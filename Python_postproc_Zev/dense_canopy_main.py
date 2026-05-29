#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:14:21 2023

@author: u1450851

Dense Forest Preparation

"""

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.ndimage import convolve1d

def calc_LAI(dz, H_canopy_grid, zi):
    # Load LAI data
    LAI_data = np.loadtxt('/uufs/chpc.utah.edu/common/home/u0851921/DOE-ARM/LAI_Amazon.csv', delimiter=',')
    LAI_data[8, :] = (LAI_data[7, :] + LAI_data[9, :]) / 2
    LAI_data[:, 1] = LAI_data[:, 1] / zi

    LAI = np.zeros(H_canopy_grid)
    h = np.linspace(dz, H_canopy_grid * dz, H_canopy_grid)
    
    if h[0] < LAI_data[0, 1]:
        LAI[0] = LAI_data[0, 1]
        print('dz < LAI data height CHECK 2ND LEVEL')
    
    for i in range(len(LAI)):
        for j in range(len(LAI_data) - 1):
            if h[i] > LAI_data[j, 1] and h[i] <= LAI_data[j + 1, 1]:
                LAI[i] = LAI_data[j, 0]
                break
    
    LAI[2] = 0.182
    LAI[-1] = LAI_data[-1, 0]
    LAI[2:] = convolve(LAI[2:], np.array([1, 1, 1]) / 3, mode='same')
    
    return LAI


wall_correct = 0
save_ = True
outpath = '/uufs/chpc.utah.edu/common/home/u1450851/LES_code/UtahIBMParticle/dense_forest'

nx = 320
ny = 16
nz = 270

lx_dim = 2000
ly_dim = 100
lz_dim = 540
zi = 1000

lx = lx_dim/zi
ly = ly_dim/zi
lz = lz_dim/zi

dx = lx/nx
dy = ly/ny
dz = lz/nz

H_canopy_m = 39.0/zi;
H_canopy_grid = round(H_canopy_m/dz);
h = np.linspace(1,H_canopy_grid,H_canopy_grid);

LAI = calc_LAI(dz,H_canopy_grid,zi);
Cd = .2 ; # This follows the Cd*Pi in Chamecki Paper

LAI = np.append(LAI, H_canopy_m)

if save_:
    np.savetxt(outpath+'/test_1_Ben_python.dat', LAI, fmt='%.6f', delimiter=' ')
 