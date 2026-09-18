#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:37:32 2025

@author: u1450851
"""
#%%
#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")
from functions import build_phi, build_intf

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from Anisotropy_Functions import ColorAnisotropy
cmap = ColorAnisotropy()

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

#%%Paths to data

pathFig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/'
directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'

#%%Plot the composite plot of TKE res, Yb 

from matplotlib.transforms import ScaledTranslation
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

levels=[-100,-1,1,100]
levels_2 = [0.37]
# levels_3 = [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11]
levels_3 = [-0.0005,-0.0003,-0.0001,0.0001,0.0003,0.0005]
levels_yb = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
colors=['blue','white','red']
labels = ["a1)", "a2)", "a3)", "b1)", "b2)",'b3)','c1)','c2)','c3)','d1)','d2)','d3)','e1)','e2)','e3)']  # Subplot labels
yslice = 50

fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(4, 3, height_ratios=[1,1,1,1], figure=fig)

axs = [[None]*3 for _ in range(4)]

# Row 1 gap 12
for i in range(3):
    axs[0][i] = fig.add_subplot(gs[0, i], sharey=axs[0][0] if i > 0 else None)

# Row 2 gap 4
for i in range(3):
    axs[1][i] = fig.add_subplot(gs[1, i], sharey=axs[1][0] if i > 0 else None)

# Row 3 patch 12
for i in range(3):
    axs[2][i] = fig.add_subplot(gs[2, i], sharey=axs[2][0] if i > 0 else None)

# Row 4 patch 4
for i in range(3):
    axs[3][i] = fig.add_subplot(gs[3, i], sharey=axs[3][0] if i > 0 else None)

cases = ['Gap_12_9mps','Gap_4_9mps','Patch_12_9mps','Patch_4_9mps']
Nx_G = 256
Nz_G = 256
Lx_G = 2*np.pi
Lz_G = 1
dx_G = Lx_G/Nx_G
dz_G = Lz_G/Nz_G
x_G = np.arange(0,Nx_G)*dx_G
z_w_G = np.arange(0,Nz_G)*dz_G
z_uvp_G = np.arange(0,Nz_G)*dz_G + dz_G/2
zi_G = 1000
Hcanopy_G = 39/zi_G

for i in range(len(cases)):
    path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path+cases[i]+'/Data_Momentum_4TKE.nc')
    terms_ptb = xr.open_dataarray(path+cases[i]+'/terms_ptb.nc')
    terms_bdg = xr.open_dataarray(path+cases[i] + '/TKE_terms.nc')
    anisotropy = xr.open_dataarray(path+cases[i] + '/anisotropy.nc')
    Res = terms_bdg.data[:,:,:,-1] + terms_bdg.data[:,:,:,11]
    # ResNorm[(abs(ResNorm) < 50)] = 0

    Ug_G = np.nanmean(np.sqrt(data.data[:, :, -1, 0]**2 + data.data[:, :, -1, 1]**2))
    if not np.isfinite(Ug_G) or Ug_G == 0:
        raise ValueError(f'Invalid Ug for {cases[i]}: {Ug_G}')

    ResNorm = Res*(Hcanopy_G)/(Ug_G**3)
    
    p1 = axs[i][0].contourf(x_G,z_uvp_G/Hcanopy_G,ResNorm[:,yslice,:].T,cmap='bwr',levels=levels_3,extend='both')
    p1.cmap.set_under('blue')
    p1.cmap.set_over('red')
    axs[i][0].axhline(0,ls='-',color='k')
    axs[i][0].axhline(1,ls='--',color='k')
    
    axs[i][1].axhline(0,ls='-',color='k')
    axs[i][1].axhline(1,ls='--',color='k')
    sc1 = axs[i][1].contour(x_G,z_uvp_G/Hcanopy_G,anisotropy[:,yslice,:,1].T,levels=levels_2,colors=['black'])
    sc2 = axs[i][1].contourf(x_G,z_uvp_G/Hcanopy_G,anisotropy[:,yslice,:,1].T,cmap = cmap,  levels=levels_yb, vmin = 0, vmax = np.sqrt(3)/2)
    
    tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
    tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])
    # tmpRES = np.where(np.abs(ResNorm_v2) < 0.14, 0, tmpRES)
    tmpRES = np.where(np.abs(Res) < 20, 0, tmpRES)

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
    
    p2 = axs[i][2].contourf(x_G,z_uvp_G/Hcanopy_G,tmpNorm[:,yslice,:].T,colors=colors,alpha=0.5,levels=levels,extend='both')#vmin=-10,vmax=10)
    p2.cmap.set_under('blue')
    p2.cmap.set_over('red')
    sc = axs[i][2].contour(x_G,z_uvp_G/Hcanopy_G,anisotropy[:,yslice,:,1].T,levels=levels_2,colors=['black'])
    axs[i][2].axhline(0,ls='-',color='k')
    axs[i][2].axhline(1,ls='--',color='k')
    
cbar_ax = fig.add_axes([0.083, 0.07, 0.28, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p1, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$R\cdot\frac{h_C}{U_g^{3}}$ ", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size
cbar1.set_ticks([-0.0005, -0.0003, -0.0001, 0.0001, 0.0003, 0.0005])
cbar1.formatter = ScalarFormatter(useMathText=True)
cbar1.formatter.set_scientific(True)
cbar1.formatter.set_powerlimits((-4, -4))
cbar1.update_ticks()

cbar_ax = fig.add_axes([0.39, 0.07, 0.28, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(sc2, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$yB$", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size

cbar_ax = fig.add_axes([0.70, 0.07, 0.28, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p2, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$\frac{P-D}{|D|}$", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size

for row in range(4):
    for col in range(1, 3):  # columns 1 and 2
        axs[row][col].tick_params(labelleft=False)  # hide tick labels
        axs[row][col].set_ylabel("")  
  
for i in range(3):
    axs[0][i].tick_params(labelbottom=False)  # hide tick labels
    axs[1][i].tick_params(labelbottom=False)  # hide tick labels
    axs[2][i].tick_params(labelbottom=False)  # hide tick labels
    axs[0][i].set_ylim(0,20)
    axs[1][i].set_ylim(0,20)
    axs[2][i].set_ylim(0,20)
    axs[3][i].set_ylim(0,20)

for i in range(4):
    axs[i][0].set_ylabel(r"$z/h_C$", fontsize=14)

for j in range(3):
    axs[3][j].set_xlabel(r"$x/z_i$", fontsize=14)

plt.subplots_adjust(left=0.08,bottom=0.15,right=0.98,top=0.96,wspace=0.08,hspace=0.4)

l = 0
for m in range(4):
    for n in range(3):
        axs[m][n].text(0.0, 1.0, labels[l],transform=axs[m][n].transAxes + ScaledTranslation(-5/72, +5/72, fig.dpi_scale_trans), fontsize=12, \
                       va="bottom", ha="left", fontfamily="serif")
        l += 1
        
plt.savefig(pathFig + 'ResYB_Combo_Other_Ug.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()

# %%
