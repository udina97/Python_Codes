#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:01:23 2025

@author: u1450851
"""

#%%Import Libraries

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import xarray as xr

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 256
ny = 256
nz = 256

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

aniso = 'patch256_aniso_9ms'
classic = 'patch256_classic_9ms'
# aniso = 'homo_unstable_aniso_9ms'
# classic = 'homo_unstable_classic_9ms'

mom3D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_a = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

mom3D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Momentum3D/Data_Momentum_30min.nc')
mom2D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Momentum2D/Data_Momentum_2D_30min.nc')
sc3D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Scalar3D/Data_Scalar_30min.nc')
sc2D_c = xr.open_dataarray('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/data/Scalar2D/Data_Scalar_2D_30min.nc')

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

#%%

sfcT = sc2D_a.data[:,:,-2]*Tscale

fig,axs = plt.subplots(1,1)

p = axs.pcolormesh(x,y,sfcT.T,cmap='hot_r')
cbar = plt.colorbar(p)

plt.show()

#%%Compute ustar

uw_a = wnode2uvpnode(mom3D_a.data[:,:,:,14] - uvpnode2wnode(mom3D_a.data[:,:,:,0])*mom3D_a.data[:,:,:,2] - mom3D_a.data[:,:,:,20])
vw_a = wnode2uvpnode(mom3D_a.data[:,:,:,15] - uvpnode2wnode(mom3D_a.data[:,:,:,1])*mom3D_a.data[:,:,:,2] - mom3D_a.data[:,:,:,21])

ustar_a = (uw_a**2 + vw_a**2)**0.25

uw_c = wnode2uvpnode(mom3D_c.data[:,:,:,14] - uvpnode2wnode(mom3D_c.data[:,:,:,0])*mom3D_c.data[:,:,:,2] - mom3D_c.data[:,:,:,20])
vw_c = wnode2uvpnode(mom3D_c.data[:,:,:,15] - uvpnode2wnode(mom3D_c.data[:,:,:,1])*mom3D_c.data[:,:,:,2] - mom3D_c.data[:,:,:,21])

ustar_c = (uw_c**2 + vw_c**2)**0.25

#%%Compute theta_star

wT_a = wnode2uvpnode(sc3D_a.data[:,:,:,4] - uvpnode2wnode(sc3D_a.data[:,:,:,0])*mom3D_a.data[:,:,:,2] - sc3D_a.data[:,:,:,7])

theta_s_a = -wT_a/ustar_a

wT_c = wnode2uvpnode(sc3D_c.data[:,:,:,4] - uvpnode2wnode(sc3D_c.data[:,:,:,0])*mom3D_c.data[:,:,:,2] - sc3D_c.data[:,:,:,7])

theta_s_c = -wT_c/ustar_c

#%%meanU/ustar and T/theta

U_a = np.sqrt(mom3D_a.data[:,:,:,0]**2 + mom3D_a.data[:,:,:,1]**2 + wnode2uvpnode(mom3D_a.data[:,:,:,2])**2)/ustar_a
U_c = np.sqrt(mom3D_c.data[:,:,:,0]**2 + mom3D_c.data[:,:,:,1]**2 + wnode2uvpnode(mom3D_c.data[:,:,:,2])**2)/ustar_c

T_a = sc3D_a.data[:,:,:,0]/theta_s_a
T_c = sc3D_c.data[:,:,:,0]/theta_s_c

#%%Stability
z3D = np.ones((nx,ny,nz),'d',order='F')*z_uvp
L_a = -sc3D_a.data[:,:,:,0]*ustar_a**3/(0.4*(9.8*zi/uscale**2)*wT_a)
L_c = -sc3D_c.data[:,:,:,0]*ustar_c**3/(0.4*(9.8*zi/uscale**2)*wT_c)

zoverL_a = z3D/L_a
zoverL_c = z3D/L_c

#%%Plot the PDFs of velocity magnitude for unstable grid points

levels = [1,5,10]
ls = ['-','--',':']

fig,axs = plt.subplots(1,1,figsize=(5,2.5))
# fig.subplots_adjust(right=0.3)
# box = axs.get_position()
# axs.set_position([box.x0, box.y0, box.width * 0.72, box.height])

for i in range(len(levels)):
    tmp_U = U_a[:,:,levels[i]].flatten()[(zoverL_a[:,:,levels[i]].flatten() < 0)]
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_a = kde(x_pdf)
    int_a = np.trapz(pdf_a*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_a,c='g',ls=ls[i])
    
    tmp_U = U_c[:,:,levels[i]].flatten()[(zoverL_c[:,:,levels[i]].flatten() < 0)]
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_c = kde(x_pdf)
    int_c = np.trapz(pdf_c*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_c,c='k',ls=ls[i])
    
    print((int_a/int_c - 1)*100)
    
axs.set_xlim(0,30)
axs.set_ylim(0,)
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xlabel(r"$\sqrt{\overline{u}^2 + \overline{v}^2 + \overline{w}^2}/u_*$",fontsize=14)
axs.tick_params(axis='both',which='major',labelsize=11)

from matplotlib.lines import Line2D

# line-style legend (levels)
# level_handles = [
#     Line2D([0], [0], color='k', ls=ls[i], lw=2,
#             label=f'{levels[i]*dz*zi} m')
#     for i in range(len(levels))
# ]

# # color legend (datasets)
# dataset_handles = [
#     Line2D([0], [0], color='g', lw=2, label='Aniso'),
#     Line2D([0], [0], color='k', lw=2, label='Ref')
# ]

# leg1 = axs.legend(
#     handles=level_handles,
#     loc='upper left',
#     bbox_to_anchor=(1.02, 1.0),
#     frameon=True,
#     fontsize=10,
# )

# leg2 = axs.legend(
#     handles=dataset_handles,
#     loc='upper left',
#     bbox_to_anchor=(1.02, 0.45),
#     frameon=True,
#     fontsize=10,
# )

# axs.add_artist(leg1)  # keep first legend
# fig.subplots_adjust(right=0.7)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/legend.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()
    
#%%Plot the PDFs of velocity magnitude for stable grid points
    
levels = [1,5,10]
ls = ['-','--',':']

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(5,2.5))

for i in range(len(levels)):
    tmp_U = U_a[:,:,levels[i]].flatten()[(zoverL_a[:,:,levels[i]].flatten() > 0)]
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_a = kde(x_pdf)
    int_a = np.trapz(pdf_a*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_a,c='g',ls=ls[i])
    
    tmp_U = U_c[:,:,levels[i]].flatten()[(zoverL_c[:,:,levels[i]].flatten() > 0)]
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_c = kde(x_pdf)
    int_c = np.trapz(pdf_c*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_c,c='k',ls=ls[i])
    
    print((int_a/int_c - 1)*100)
    
axs.set_xlim(0,30)
axs.set_ylim(0,)
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xlabel(r"$\sqrt{\overline{u}^2 + \overline{v}^2 + \overline{w}^2}/u_*$",fontsize=14)
axs.tick_params(axis='both',which='major',labelsize=11)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/U_s.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%average surface temp

sfcT = np.mean(Tscale*sc2D_c.data[:,:,-2])

#%%Plot the PDFs of temperature for unstable grid points

levels = [1,5,10]
ls = ['-','--',':']

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(5,2.5))

for i in range(len(levels)):
    tmp_U = abs(sc3D_a.data[:,:,levels[i],0].flatten()[(zoverL_a[:,:,levels[i]].flatten() < 0)])
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_a = kde(x_pdf)
    int_a = np.trapz(pdf_a*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_a,c='g',ls=ls[i])
    
    tmp_U = abs(sc3D_c.data[:,:,levels[i],0].flatten()[(zoverL_c[:,:,levels[i]].flatten() < 0)])
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_c = kde(x_pdf)
    int_c = np.trapz(pdf_c*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_c,c='k',ls=ls[i])
    
    print((int_a/int_c - 1)*100)
    
axs.set_xlim(0.984,0.99)
axs.set_ylim(0,)
# axs.set_xscale('log')
# axs.set_yscale('log')
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xlabel(r"$\theta/<\theta_s>$",fontsize=14)
axs.tick_params(axis='both',which='major',labelsize=11)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/T_u.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()
    
#%%Plot the PDFs of temperature for stable grid points
    
levels = [1,5,10]
ls = ['-','--',':']

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(5,2.5))

for i in range(len(levels)):
    tmp_U = abs(sc3D_a.data[:,:,levels[i],0].flatten()[(zoverL_a[:,:,levels[i]].flatten() > 0)])
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_a = kde(x_pdf)
    int_a = np.trapz(pdf_a*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_a,c='g',ls=ls[i])
    
    tmp_U = abs(sc3D_c.data[:,:,levels[i],0].flatten()[(zoverL_c[:,:,levels[i]].flatten() > 0)])
    kde = gaussian_kde(tmp_U)
    x_pdf = np.linspace(min(tmp_U), max(tmp_U), 1000)
    pdf_c = kde(x_pdf)
    int_c = np.trapz(pdf_c*x_pdf,x_pdf)
    axs.plot(x_pdf,pdf_c,c='k',ls=ls[i])
    
    print((int_a/int_c - 1)*100)
    
axs.set_xlim(0.984,0.988)
axs.set_ylim(0,)
# axs.set_xscale('log')
# axs.set_yscale('log')
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xlabel(r"$\theta/<\theta_s>$",fontsize=14)
axs.tick_params(axis='both',which='major',labelsize=11)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/T_s.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()


#%%

fig,axs = plt.subplots(1,1)

axs.plot(np.mean(T_a,axis=(0,1)),z_uvp,c='k')

plt.show()

















































