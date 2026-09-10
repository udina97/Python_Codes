#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 13:45:47 2025

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.transforms import ScaledTranslation
import scipy.io

#%%Set path to the profiles

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/'
# cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat']
cases = ['Flat','Sinusoidal','ATTO','Gap_8_9mps','Patch_8_9mps','simulation_G']
Nz_SLayer = 200
canopyH = 39

#%%Plot

layout = [['a1)', 'a2)', 'a3)', 'a4)'],
          ['b1)', 'b2)', 'b3)', 'b4)'],
          ['c1)', 'c2)', 'c3)', 'c4)'],
          ['d1)', 'd2)', 'd3)', 'd4)'],
          ['e1)', 'e2)', 'e3)', 'e4)'],
          ['f1)', 'f2)', 'f3)', 'f4)']]

fig, axs_dict = plt.subplot_mosaic(layout,figsize=(9,9),sharey=True)

for label, ax in axs_dict.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    # ax.text(
    #     0.0, 1.0, label, transform=(
    #         ax.transAxes + ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans)),
    #     fontsize=12, va='bottom', fontfamily='serif')
    ax.text(
        0.02, 0.96, label,
        transform=ax.transAxes,
        fontsize=12, va='top', ha='left',
        fontfamily='serif')

# fig,axs = plt.subplots(5,4,tight_layout=True,sharey=True,sharex='col',figsize=(10,8))

axs = np.array([[axs_dict[label] for label in row] for row in layout])

for j in range(axs.shape[1]):        # loop over columns
    if j == axs.shape[1]:
        for i in range(1, axs.shape[0]-1): # loop over rows in that column
            axs[i, j].sharex(axs[0, j])
    else:
        for i in range(1, axs.shape[0]): # loop over rows in that column
            axs[i, j].sharex(axs[0, j])  # share with the top axis in that column
        
# Hide inner labels
# for ax in axs.ravel():
#     ax.label_outer()
    
for j in range(axs.shape[1]):  # loop over columns
    # find last visible axis in this column
    visible_axes = [axs[i, j] for i in range(axs.shape[0]) if axs[i, j].get_visible()]
    for ax in visible_axes[:-1]:  # all but last visible: hide x labels
        ax.tick_params(labelbottom=False)
    # keep labels for last visible one

for i in range(len(cases)):
    if i == 0:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof = np.load(path_to_data + 'U_' + cases[i] + '_Ug.npy')
        axs[i,0].plot(prof,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k')
    elif i > 0 and i < 3:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'U_' + cases[i] + '_p_Ug.npy')
        prof_v = np.load(path_to_data + 'U_' + cases[i] + '_v_Ug.npy')
        axs[i,0].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,0].plot(prof_v,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    elif i >= 3 and i < 5:
        lz = 1; nz = 256; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'U_' + cases[i] + '_p_Ug.npy')
        prof_f = np.load(path_to_data + 'U_' + cases[i] + '_f_Ug.npy')
        axs[i,0].plot(prof_f,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,0].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    else:
        z = np.arange(0.5,160.5,1)/15.3
        prof_xy = np.load(path_to_data + 'U_' + cases[i] + '_xy_Ug.npy')
        prof_tw = np.load(path_to_data + 'U_' + cases[i] + '_tw_Ug.npy')
        axs[i,0].plot(prof_xy[8:],z[:-8],c='k',ls='-')
        axs[i,0].plot(prof_tw[8:],z[:-8],c='k',ls='--')
        
for i in range(len(cases)):
    if i == 0:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof = np.load(path_to_data + 'Shear_' + cases[i] + '_Ug.npy')
        axs[i,1].plot(prof,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k')
    elif i > 0 and i < 3:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'Shear_' + cases[i] + '_p_Ug.npy')
        prof_v = np.load(path_to_data + 'Shear_' + cases[i] + '_v_Ug.npy')
        axs[i,1].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,1].plot(prof_v,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    elif i >= 3 and i < 5:
        lz = 1; nz = 256; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'Shear_' + cases[i] + '_p_Ug.npy')
        prof_f = np.load(path_to_data + 'Shear_' + cases[i] + '_f_Ug.npy')
        axs[i,1].plot(prof_f,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,1].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    else:
        z = np.arange(0.5,160.5,1)/15.3
        prof_xy = np.load(path_to_data + 'Shear_' + cases[i] + '_xy_Ug.npy')
        prof_tw = np.load(path_to_data + 'Shear_' + cases[i] + '_tw_Ug.npy')
        axs[i,1].plot(prof_xy[8:],z[:-8],c='k',ls='-')
        axs[i,1].plot(prof_tw[8:],z[:-8],c='k',ls='--')
    

for i in range(len(cases)):
    if i == 0:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof = np.load(path_to_data + 'PHI_' + cases[i] + '.npy')
        axs[i,2].plot(prof,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k')
    elif i > 0 and i < 3:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'PHI_' + cases[i] + '_p.npy')
        prof_v = np.load(path_to_data + 'PHI_' + cases[i] + '_v.npy')
        axs[i,2].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,2].plot(prof_v,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    elif i >= 3 and i < 5:
        lz = 1; nz = 256; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'PHI_' + cases[i] + '_p.npy')
        prof_f = np.load(path_to_data + 'PHI_' + cases[i] + '_f.npy')
        axs[i,2].plot(prof_f,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,2].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    else:
        z = np.arange(0.5,160.5,1)/15.3
        prof_xy = np.load(path_to_data + 'PHI_' + cases[i] + '_xy.npy')
        prof_tw = np.load(path_to_data + 'PHI_' + cases[i] + '_tw.npy')
        axs[i,2].plot(prof_xy[:],z[:-8],c='k',ls='-')
        axs[i,2].plot(prof_tw[:],z[:-8],c='k',ls='--')
        
for i in range(len(cases)):
    if i == 0:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof = np.load(path_to_data + 'Skew_' + cases[i] + '.npy')
        axs[i,3].plot(prof,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k')
    elif i > 0 and i < 3:
        lz = 0.96; nz = 384; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'Skew_' + cases[i] + '_p.npy')
        prof_v = np.load(path_to_data + 'Skew_' + cases[i] + '_v.npy')
        axs[i,3].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,3].plot(prof_v,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    elif i >= 3 and i < 5:
        lz = 1; nz = 256; dz = lz/nz; zi = 1000
        prof_p = np.load(path_to_data + 'Skew_' + cases[i] + '_p.npy')
        prof_f = np.load(path_to_data + 'Skew_' + cases[i] + '_f.npy')
        axs[i,3].plot(prof_f,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='-')
        axs[i,3].plot(prof_p,(np.arange(0,Nz_SLayer)*dz + dz/2)/(canopyH/zi),c='k',ls='--')
    else:
        axs[i,3].set_visible(False)
        axs[i,3].set_in_layout(False)
        
# axs[-1, 3].get_shared_x_axes().remove(axs[-1, 3])
        
for i in range(0,len(cases)):
    axs[i,0].set_xlim(0,1)
    axs[i,0].set_ylim(0,10)
    # axs[i,0].set_ylabel(r"hi"+"\n"+r"$z/h_C$",fontsize=15)
    axs[i,1].set_xlim(-0.001,0.01)
    axs[i,2].set_xlim(-0.4,2.5)
    axs[i,0].tick_params(axis='y',which='major',labelsize=12)
axs[-2,3].set_xlim(-0.75,0.75)
axs[-2, 3].tick_params(labelbottom=True)
    
for i in range(len(axs)):
    for j in range(len(axs[0])):
        axs[i,j].axhline(1,c='k',ls='--',alpha=0.5)
        axs[i,j].grid(alpha=0.5)
        # axs[i,j].set_yscale('log')
    
axs[-1,0].set_xlabel(r"$\overline{U}/U_G$",fontsize=13)
axs[-1,1].set_xlabel(r"$\sqrt{\overline{u'w'}^2 + \overline{v'w'}^2}/U_G^2$",fontsize=13)
axs[-1,2].set_xlabel(r"$\phi_M$",fontsize=13)
axs[-2,3].set_xlabel(r"$Sk_w$",fontsize=13)

axs[-1,0].tick_params(axis='x',which='major',labelsize=12)
axs[-1,1].tick_params(axis='x',which='major',labelsize=12)
axs[-1,2].tick_params(axis='x',which='major',labelsize=12)
axs[-2,3].tick_params(axis='x',which='major',labelsize=12)

axs[0,0].set_ylabel(r"Flat"+"\n"+r"$z/h_C$",fontsize=15,labelpad=0.1)
axs[1,0].set_ylabel(r"Sinusoidal"+"\n"+r"$z/h_C$",fontsize=15,labelpad=0.1)
axs[2,0].set_ylabel(r"ATTO"+"\n"+r"$z/h_C$",fontsize=15,labelpad=0.1)
axs[3,0].set_ylabel(r"g800"+"\n"+r"$z/h_C$",fontsize=15,labelpad=0.1)
axs[4,0].set_ylabel(r"i800"+"\n"+r"$z/h_C$",fontsize=15,labelpad=0.1)
axs[5,0].set_ylabel(r"Urban"+"\n"+r"$z/h_C$",fontsize=15,labelpad=0.1)

plt.subplots_adjust(hspace=0.2, wspace=0.25)

fig.subplots_adjust(
    left=0.1,   # space on left side   (fraction of figure width)
    right=0.96, # space on right side
    bottom=0.08, # space at bottom
    top=0.97,    # space at top
    wspace=0.1
)

from matplotlib.ticker import ScalarFormatter


class ZeroScalarFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        if np.isclose(x, 0):
            return "0"
        return super().__call__(x, pos)
    
for ax in axs.ravel():
    if ax.get_visible():
        ax.xaxis.set_major_formatter(ZeroScalarFormatter())

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'Mom_TwrProf_AllCases_labels_Ug.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()