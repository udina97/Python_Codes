#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 11:07:45 2025

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
kvonk = 0.4

#%%Load the data

# sim = 'patch128_aniso_1ms'
sim = 'homo_unstable_aniso_1ms'

data = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_checkpoint/',[110000],nx,ny)
data3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_checkpoint/',[110000],nx,ny,nz)

#%%ustar,L and zeta

ustar = (data['ustar'].flatten())
wT = data['sfcFLUX'].flatten()
L = (data['L'].flatten())
zeta = 0.5*dz/L

ustar_u = np.median(ustar[(L<0)])
zeta_u = np.median(zeta[(L<0)])
wT_u = np.median(wT[(L<0)])

#%%Compute Reynolds stresses using the on-the-fly averages and anisotropy

uu = data3D['uu_new'][:,:,:nz] - data3D['u_new'][:,:,:nz]*data3D['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = data3D['vv_new'][:,:,:nz] - data3D['v_new'][:,:,:nz]*data3D['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = data3D['ww_new'][:,:,:nz] - data3D['w_new'][:,:,:nz]*data3D['w_new'][:,:,:nz]#+ checkpnt['tzz_new'][:,:,:nz]
uv = data3D['uv_new'][:,:,:nz] - data3D['u_new'][:,:,:nz]*data3D['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = data3D['uw_new'][:,:,:nz] - data3D['u_new'][:,:,:nz]*data3D['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = data3D['vw_new'][:,:,:nz] - data3D['v_new'][:,:,:nz]*data3D['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke = uu + vv + ww

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

#%%

dudz = (ustar/(kvonk*0.5*dz)*data['phi_m'].flatten())
dudz_u = np.median(dudz[(L<0)])

#%%

yB_u = np.median(yB[:,:,0].flatten()[(L<0)])

b = 0.061
deltaDuDz = abs(ustar_u/(kvonk*0.5*dz)*((-0.939)*abs(zeta_u)**(-0.12+6.4*yB_u)*((0.24-0.38*yB_u)*np.log(abs(zeta_u))*6.4+0.38)/(0.24-0.38*yB_u + \
                                abs(zeta_u)**(-0.12+6.4*yB_u))**2 - 0.53*abs(zeta_u)**(1/3)))*0.4

ratio = deltaDuDz/dudz_u

#%%

deltaWT = abs(wT_u/(kvonk*0.5*dz*ustar_u)*1.8*((3-2.5*zeta_u)/(1-10*zeta_u+50*zeta_u**2))**(1/3))*0.1
dTdz = np.median((wT*data['phi_h'].flatten()/(kvonk*0.5*dz*ustar)).flatten()[(L<0)])

ratio = deltaWT/dTdz
















































