#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:37:29 2025

@author: u1450851
"""

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso

from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 128
ny = 128
nz = 128

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.4
Tscale = 290
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load the data

sim1 = 'homo_unstable_aniso_9ms'
sim2 = 'homo_unstable_classic_9ms'

data1 = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+sim1+'/output_checkpoint/',[110000],nx,ny)
data2 = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+sim2+'/output_checkpoint/',[110000],nx,ny)

#%%Compute difference between ustar

ustar1 = (data1['ustar']*0.4)**2
ustar2 = (data2['ustar']*0.4)**2

diff = ustar1 - ustar2

area = ustar1.size*dx*dy*(zi**2)

diff_tot = np.sum(diff)*dx*dy*(zi**2)

diff_mean = diff_tot/area

#%%compute the difference in heat flux

wT1 = data1['sfcFLUX']*0.4*290*1000
wT2 = data2['sfcFLUX']*0.4*290*1000

diff = (wT1 - wT2)

area = wT1.size*dx*dy*(zi**2)

diff_tot = np.sum(diff)*dx*dy*(zi**2)

diff_mean = diff_tot/area















































