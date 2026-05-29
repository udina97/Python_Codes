#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 16:20:54 2026

@author: u1450851
"""

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso
from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 64
ny = 64
nz = 64

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

#%%Load surface checkpoint/istantaneous fields 
# sim = 'yb_test_v2'

sim = 'test_'

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

step = 30000

A = read_checkpoint(path+sim+'A/output_checkpoint/',[step],nx,ny,nz)
A_sfc = read_checkpoint_sfc_L(path+sim+'A/output_checkpoint/',[step],nx,ny)
B = read_checkpoint(path+sim+'B/output_checkpoint/',[step],nx,ny,nz)
B_sfc = read_checkpoint_sfc_L(path+sim+'B/output_checkpoint/',[step],nx,ny)

#%%

keys_3D = list(A.keys())
keys_2D = list(A_sfc.keys())

# for i in range(len(keys_3D)):
#     same = np.array_equal(A[keys_3D[i]],B[keys_3D[i]])
#     print(same)


for i in range(len(keys_2D)):
    same = np.array_equal(A_sfc[keys_2D[i]],B_sfc[keys_2D[i]])
    print(same)

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.mean(A['w'][:,:,:-1],axis=(0,1)),z_uvp,c='k')
axs.plot(np.mean(B['w'][:,:,:-1],axis=(0,1)),z_uvp,c='r')

plt.show()






















































