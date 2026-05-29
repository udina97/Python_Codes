#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:26:41 2025

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.transforms import ScaledTranslation

#%%Set path to the profiles

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/'
cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat']
# cases = ['Flat','Sinusoidal','ATTO','Gap_8_9mps','Patch_8_9mps']
Nz_SLayer = 200
canopyH = 39

#%%Plot

colors = ['green','limegreen','lightgreen','beige','khaki','gold','peachpuff','sandybrown','saddlebrown']

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(len(cases)):
    
    prof = np.load(path_to_data + 'ResTKEvsYB_' + cases[i] + '.npy')
    
    axs.plot(prof[2,:], prof[0,:], c=colors[i])
    axs.fill_between(
        prof[2,:],
        np.array(prof[0,:]) - np.array(prof[1,:]),
        np.array(prof[0,:]) + np.array(prof[1,:]),
        alpha=0.1,color=colors[i]
    )

axs.axhline(0, color='k', linestyle='-.')
# axs.axvline(0.38, color='k', linestyle='-.')
# axs.text(0.38 + 0.01, 200, 'yB = 0.38', rotation=90, va='center', ha='left', color='black')
# axs.axvline(0.36, color='k', linestyle='-.')
# axs.text(0.36 - 0.02, 200, 'yB = 0.36', rotation=90, va='center', ha='left', color='black')

axs.set_xlabel(r'$y_B$', fontsize=18)
axs.set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=21)
axs.set_xlim(0.15, 0.6)
axs.set_ylim(-70, 250)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)

# Correlation text
# x_yB_flat = x_yB.values[mask_dist]
# y_TKE_flat = y_TKE[mask_dist]

# Compute correlation, avoiding NaNs
# valid = ~np.isnan(x_yB_flat) & ~np.isnan(y_TKE_flat)
# corr = np.corrcoef(x_yB_flat[valid], y_TKE_flat[valid])[0, 1]
# axs.text(0.1, 0.9, f'r = {round(corr, 2)}', transform=axs.transAxes, fontsize=18)

plt.tight_layout()

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'TKEres_vs_YB_AllCases.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()



























































