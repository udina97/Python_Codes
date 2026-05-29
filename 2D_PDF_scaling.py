#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:30:12 2025

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
import os
import sys
import copy
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from read_checkpoint_SC import read_checkpoint

from read_checkpoint_SC import read_checkpoint_sfc

#%%#inputs

sim = '64x3_1hr_2wnode_tzzP'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/'

nx = 64
ny = 64
nz = 64
lx = 0.5*np.pi
ly = 0.5*np.pi
lz = 0.5
dx = lx/nx
dy = ly/ny
dz = lz/nz

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.1 #05; %0.000005;
zi = 500.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = path

#%%Import variables


var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']
    
var2D = ['avgUstar']
    
varS = ['avgT','avgT2','avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus','avg_ds']

varS2D = ['avgWstar','avgL','avgPHIm','avgPSIm','avgPHIh','avgPSIh','avgSFCval','avgSFCflux']

varA = ['avgXB','avgYB']#,'avgPHIM','avgPHIH','avgPSIM','avgPSIH','avgL3D','avgustar3D','avgSCF3D']
    
dataM = xr.open_dataarray(path+'Data_Momentum.nc')
dataM2D = xr.open_dataarray(path+'Data_Momentum_2D.nc')
dataS = xr.open_dataarray(path+'Data_Scalar.nc')
dataS2D = xr.open_dataarray(path+'Data_Scalar_2D.nc')
dataA = xr.open_dataarray(path+'Data_Anisotropy.nc')

data_mom = dict()
data_mom_2D = dict()
data_sc = dict()
data_sc_2D = dict()
data_aniso = dict()

for i in range(len(var)):
    data_mom[var[i]] = dataM.data[:,:,:,i]
    
for i in range(len(var2D)):
    data_mom_2D[var2D[i]] = dataM2D.data[:,:,i]

for i in range(len(varS)):
    data_sc[varS[i]] = dataS.data[:,:,:,i]

for i in range(len(varS2D)):
    data_sc_2D[varS2D[i]] = dataS2D.data[:,:,i]
    
for i in range(len(varA)):
    data_aniso[varA[i]] = dataA.data[:,:,:,i]
    
#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_h[:, :, -1]
  return phi_c

#%%Stiperski 2023 scaling relations for \phi_u, \phi_v, \phi_w (See supplemental material for the 2023 paper)

zeta = (dz*zi/2)/(data_sc_2D['avgL']*zi)

a_n = [0.784,-2.582]

tmp_a = np.zeros_like(data_aniso['avgYB'][:,:,0])
a = np.zeros_like(data_aniso['avgYB'][:,:,0])

for i in range(len(a_n)):
    tmp_a = a_n[i]*(np.log10(data_aniso['avgYB'][:,:,1]))**(i)
    a =  a + tmp_a

phi_u = a*((1 - 3*zeta)**(1/3))

#%%Compute sigma_u from the data and normalize by ustar to get phi_u

phi_u_data = (np.sqrt(data_mom['avgU2'][:,:,0] - data_mom['avgU'][:,:,0]*data_mom['avgU'][:,:,0])*uscale)/((data_mom_2D['avgUstar']*uscale))
phi_v_data = (np.sqrt(data_mom['avgV2'][:,:,0] - data_mom['avgV'][:,:,0]*data_mom['avgV'][:,:,0])*uscale)/(data_mom_2D['avgUstar']*uscale)
phi_w_data = (np.sqrt(wnode2uvpnode(data_mom['avgW2'])[:,:,0] - wnode2uvpnode(data_mom['avgW'])[:,:,0]*wnode2uvpnode(data_mom['avgW'])[:,:,0])*uscale)/(data_mom_2D['avgUstar']*uscale)

#%%

zeta_vec = np.array([-1e2,-1e1,-1e0,-1e-1,-1e-2,-1e-3,-1e-4])
y_b_vec = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
phi_u_fit = np.zeros((len(zeta_vec),len(y_b_vec)))
phi_v_fit = np.zeros((len(zeta_vec),len(y_b_vec)))
phi_w_fit = np.zeros((len(zeta_vec),len(y_b_vec)))
coef_u = [0.784,-2.582]
coef_v = [0.725,-2.702]
coef_w = [1.119,-0.019,-0.065,0.028]

for i in range(len(y_b_vec)):
    a_u=0
    a_v=0
    a_w=0
    for j in range(len(coef_u)):
        tmp_a_u = coef_u[j]*(np.log10(y_b_vec[i]))**(j)
        a_u = tmp_a_u + a_u
        tmp_a_v = coef_v[j]*(np.log10(y_b_vec[i]))**(j)
        a_v = tmp_a_v + a_v
    phi_u_fit[:,i] = a_u*((1 - 3*zeta_vec)**(1/3))
    phi_v_fit[:,i] = a_v*((1 - 3*zeta_vec)**(1/3))
    for k in range(len(coef_w)):
        tmp_a_w =coef_w[k]*(y_b_vec[i])**(k)
        a_w = a_w + tmp_a_w
    phi_w_fit[:,i] = a_w*((1 - 3*zeta_vec)**(1/3))

#%%Plot the scaling as a function of zeta as a 2D PDF
os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
nbins = 20

# pdf, xedges, yedges = np.histogram2d(zeta.flatten(),phi_u.flatten(),bins=nbins,density=True)

# xcenters = 0.5 * (xedges[1:] + xedges[:-1])
# ycenters = 0.5 * (yedges[1:] + yedges[:-1])
# X, Y = np.meshgrid(xcenters, ycenters)

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
for i in range(len(y_b_vec)):
    axs.semilogx(-zeta_vec,phi_u_fit[:,i],c=cmap(y_b_vec[i]))
# axs.contourf(X,Y,pdf.T,levels=10,cmap='Greens')
axs.scatter(-zeta, phi_u_data)
# axs.contour(X,Y,pdf.T,levels=10,c='k')
# axs.plot([-1e-6], [0], alpha=0)
# axs.set_xscale('symlog',linthresh=0.1)
axs.set_xlim(-1e2,-1e-6)
axs.invert_xaxis()
axs.set_ylim(0,10)
axs.autoscale(False)
axs.set_xlabel(r'$\zeta$',fontsize=15)
axs.set_ylabel(r'$\phi$',fontsize=15)
# axs.set_xticks([-1e2, -1e1, -1e0, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5, -1e-6])
# axs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

plt.show()















































