#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:20:25 2026

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy

#%%Set up cases, paths, names

B_path = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
G_path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'

B_cases = ['Flat','Sinusoidal','ATTO']
G_cases = ['Empty_9mps','Gap_8_9mps','Patch_8_9mps']

SGS = False

if SGS:
    Full = '/anisotropy.nc'
    Diag = '/anisotropy_D.nc'
else:
    Full = '/anisotropy_NoSGS.nc'
    Diag = '/anisotropy_D_NoSGS.nc'
    
#%%

from matplotlib.transforms import ScaledTranslation
import matplotlib.gridspec as gridspec

levels=[-100,-1,1,100]
colors=['blue','white','red']
levels_yb = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
labels = ["a1)", "a2)", "a3)", "a4)", "b1)", "b2)",'b3)', 'b4)','c1)','c2)','c3)', 'c4)','d1)','d2)','d3)', 'd4)','e1)','e2)','e3)', 'e4)',
          'f1)','f2)','f3)', 'f4)']  # Subplot labels

fig = plt.figure(figsize=(9, 10))
gs = gridspec.GridSpec(7, 4, height_ratios=[1,1,1,0.1,1,1,1], figure=fig)

axs = [[None]*4 for _ in range(7)]

# Row 1 Empty
for i in range(4):
    axs[0][i] = fig.add_subplot(gs[0, i], sharey=axs[0][0] if i > 0 else None)

# Row 2 g800
for i in range(4):
    axs[1][i] = fig.add_subplot(gs[1, i], sharey=axs[1][0] if i > 0 else None)

# Row 3 i800
for i in range(4):
    axs[2][i] = fig.add_subplot(gs[2, i], sharey=axs[2][0] if i > 0 else None)

# Row 4 Flat
for i in range(4):
    axs[4][i] = fig.add_subplot(gs[4, i], sharey=axs[4][0] if i > 0 else None)
    
# Row 5 Sinusoidal
for i in range(4):
    axs[5][i] = fig.add_subplot(gs[5, i], sharey=axs[5][0] if i > 0 else None)
    
# Row 6 ATTO
for i in range(4):
    axs[6][i] = fig.add_subplot(gs[6, i], sharey=axs[6][0] if i > 0 else None)
    
for i in range(0,3):
    Nx_G = 256
    Ny_G = 256
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
    
    aniso = xr.open_dataarray(G_path + G_cases[i] + Full)
    anisoD = xr.open_dataarray(G_path + G_cases[i] + Diag)
    TKE_terms = xr.open_dataarray(G_path + G_cases[i] + '/TKE_terms.nc')
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    # tmpDIS[abs(tmpDIS) < 0.05*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data + TKE_terms[:, :, :, 11].data)
    tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
    
    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, 0)
    
    axs[i][0].contourf(x_G,z_uvp_G,aniso[:,Ny_G//2,:,1].T,cmap = ColorAnisotropy(),levels=levels_yb,vmin=0,vmax=np.sqrt(3)/2)
    axs[i][1].contourf(x_G,z_uvp_G,anisoD[:,Ny_G//2,:,1].T,cmap = ColorAnisotropy(),levels=levels_yb,vmin=0,vmax=np.sqrt(3)/2,extend='max')
    axs[i][2].contourf(x_G,z_uvp_G,(aniso[:,Ny_G//2,:,1]/anisoD[:,Ny_G//2,:,1]).T,cmap = 'jet',levels=[0.6,0.7,0.8,0.9,1],vmin=0,extend='both')
    p = axs[i][3].contourf(x_G,z_uvp_G,tmpNorm[:,Ny_G//2,:].T,colors=colors,alpha=0.5,levels=levels,extend='both')
    p.cmap.set_under('blue')
    p.cmap.set_over('red')
    
for i in range(4,7):
    Nx_B = 256
    Ny_B = 256
    Nz_B = 384
    Lx_B = 2.880
    Lz_B = 0.960
    dx_B = Lx_B/Nx_B
    dz_B = Lz_B/Nz_B
    x_B = np.arange(0,Nx_B)*dx_B
    z_w_B = np.arange(0,Nz_B)*dz_B
    z_uvp_B = np.arange(0,Nz_B)*dz_B + dz_B/2
    zi_B = 1000
    Hcanopy_B = 39/zi_B
    
    aniso = xr.open_dataarray(B_path + B_cases[i-4] + Full)
    anisoD = xr.open_dataarray(B_path + B_cases[i-4] + Diag)
    dist = xr.open_dataarray(B_path + B_cases[i-4] + '/dist.nc')[:,:,:,0].data
    TKE_terms = xr.open_dataarray(B_path + B_cases[i-4] + '/TKE_terms.nc')
    mask = dist < 0
    for j in range(0,3):
        aniso[:,:,:, j] = np.where(mask, np.nan, aniso[:,:,:, j])
        anisoD[:,:,:, j] = np.where(mask, np.nan, anisoD[:,:,:, j])
        
    tmpDIS = copy.deepcopy(TKE_terms[:, :, :, 11].data)
    tmpDIS[dist < 0] = np.nan
    # tmpDIS[abs(tmpDIS) < 0.05*np.max(np.nanmedian(abs(tmpDIS),axis=(0,1)))] = np.nan
    tmpRES = copy.deepcopy(TKE_terms[:, :, :, -1].data - TKE_terms[:, :, :, 11].data)
    tmpRES[dist < 0] = np.nan
    val = np.nanmedian(tmpRES[(dist>38) & (dist<40)])
    tmpRES[abs(tmpRES) < 0.05*val] = 0
    # tmpRES[abs(tmpRES) < 0.05*np.max(np.nanmedian(tmpRES,axis=(0,1)))] = 0
    
    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
    
    # tmpNorm[dist[:,:,:,0].data < 0] = np.nan
    
    p1 = axs[i][0].contourf(x_B,z_uvp_B[:-5],aniso[:,Ny_B//2,:,1].T,cmap = ColorAnisotropy(),levels=levels_yb,vmin=0,vmax=np.sqrt(3)/2)
    p2 = axs[i][1].contourf(x_B,z_uvp_B[:-5],anisoD[:,Ny_B//2,:,1].T,cmap = ColorAnisotropy(),levels=levels_yb,vmin=0,vmax=np.sqrt(3)/2,extend='max')
    p3 = axs[i][2].contourf(x_B,z_uvp_B[:-5],(aniso[:,Ny_B//2,:,1]/anisoD[:,Ny_B//2,:,1]).T,cmap = 'jet',levels=[0.6,0.7,0.8,0.9,1],vmin=0,extend='both')
    p4 = axs[i][3].contourf(x_B,z_uvp_B[:-5],tmpNorm[:,Ny_B//2,:].T,colors=colors,alpha=0.5,levels=levels,extend='both')
    p4.cmap.set_under('blue')
    p4.cmap.set_over('red')
    
cbar_ax = fig.add_axes([0.08, 0.07, 0.211, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p1, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$y_B$ ", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size
cbar1.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
cbar1.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8'])

cbar_ax = fig.add_axes([0.31, 0.07, 0.211, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p2, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$y_{B,d}$", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size
cbar1.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
cbar1.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8'])

cbar_ax = fig.add_axes([0.54, 0.07, 0.211, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p3, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$y_B/y_{B,d}$", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size

cbar_ax = fig.add_axes([0.77, 0.07, 0.211, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p4, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size

for row in range(3):
    for col in range(1, 4):  # columns 1 and 2
        axs[row][col].tick_params(labelleft=False)  # hide tick labels
        axs[row][col].set_ylabel("")  
for row in range(4,7):
    for col in range(1, 4):  # columns 1 and 2
        axs[row][col].tick_params(labelleft=False)  # hide tick labels
        axs[row][col].set_ylabel("")  
    
for i in range(4):
    axs[0][i].tick_params(labelbottom=False)  # hide tick labels
    axs[1][i].tick_params(labelbottom=False)  # hide tick labels
    axs[4][i].tick_params(labelbottom=False)  # hide tick labels
    axs[5][i].tick_params(labelbottom=False)  # hide tick labels
    # axs[0][i].set_ylim(0,20)
    # axs[1][i].set_ylim(0,20)
    # axs[2][i].set_ylim(0,20)
    # axs[4][i].set_ylim(0,20)
    # axs[5][i].set_ylim(0,20)
    # axs[6][i].set_ylim(0,20)

for i in range(3):
    axs[i][0].set_ylabel(r"$z/z_i$", fontsize=14)
for i in range(4,7):
    axs[i][0].set_ylabel(r"$z/z_i$", fontsize=14)

for j in range(4):
    axs[2][j].set_xlabel(r"$x/z_i$", fontsize=14)
    axs[6][j].set_xlabel(r"$x/z_i$", fontsize=14)

plt.subplots_adjust(left=0.08,bottom=0.15,right=0.98,top=0.96,wspace=0.08,hspace=0.4)

l = 0
for m in range(3):
    for n in range(4):
        axs[m][n].text(0.0, 1.0, labels[l],transform=axs[m][n].transAxes + ScaledTranslation(-5/72, +5/72, fig.dpi_scale_trans), fontsize=12, \
                       va="bottom", ha="left", fontfamily="serif")
        l += 1
for m in range(4,7):
    for n in range(4):
        axs[m][n].text(0.0, 1.0, labels[l],transform=axs[m][n].transAxes + ScaledTranslation(-5/72, +5/72, fig.dpi_scale_trans), fontsize=12, \
                       va="bottom", ha="left", fontfamily="serif")
        l += 1
        
# plt.savefig(pathFig + 'ResYB_Combo_All.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()