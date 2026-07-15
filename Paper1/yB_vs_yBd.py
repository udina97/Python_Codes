#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:34:08 2026

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from functions import get_dphidx,get_dphidy,get_dphidz,uvpnode2wnode,wnode2uvpnode,compute_d_twr,find_coordinates,average_over_selected_coords

#%%Import data

cases = ['ATTO','Sinusoidal','Flat']
# cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps']
# cases = ['Patch_12_9mps','Patch_8_9mps','Patch_4_9mps']
data = dict()
anisotropy = dict()
dist = dict()
TKE = dict()
anisotropy_D = dict()

for i in range(len(cases)):
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    # path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
    data[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/dataTKE.nc').data
    # data[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/Data_Momentum_4TKE.nc').data
    anisotropy[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/anisotropy.nc').data
    anisotropy_D[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/anisotropy_D.nc').data
    dist[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/dist.nc').data
    TKE[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/TKE_terms.nc').data

#%%compute REynolds stresses

uu = dict(); vv = dict(); ww = dict(); uv = dict(); uw = dict(); vw = dict()

tke = dict();

for i in range(len(cases)):
    
    uu[cases[i]] = data[cases[i]][:,:,:,4] - data[cases[i]][:,:,:,0]*data[cases[i]][:,:,:,0] - data[cases[i]][:,:,:,19]
    vv[cases[i]] = data[cases[i]][:,:,:,5] - data[cases[i]][:,:,:,1]*data[cases[i]][:,:,:,1] - data[cases[i]][:,:,:,20]
    ww[cases[i]] = data[cases[i]][:,:,:,6] - data[cases[i]][:,:,:,2]*data[cases[i]][:,:,:,2] - data[cases[i]][:,:,:,21]
    uv[cases[i]] = data[cases[i]][:,:,:,7] - data[cases[i]][:,:,:,0]*data[cases[i]][:,:,:,1] - data[cases[i]][:,:,:,22]
    uw[cases[i]] = data[cases[i]][:,:,:,8] - data[cases[i]][:,:,:,0]*data[cases[i]][:,:,:,2] - data[cases[i]][:,:,:,23]
    vw[cases[i]] = data[cases[i]][:,:,:,9] - data[cases[i]][:,:,:,1]*data[cases[i]][:,:,:,2] - data[cases[i]][:,:,:,24]
    
    # uu[cases[i]] = data[cases[i]][:,:,:,4] - data[cases[i]][:,:,:,0]*data[cases[i]][:,:,:,0] - data[cases[i]][:,:,:,19]
    # vv[cases[i]] = data[cases[i]][:,:,:,5] - data[cases[i]][:,:,:,1]*data[cases[i]][:,:,:,1] - data[cases[i]][:,:,:,20]
    # ww[cases[i]] = wnode2uvpnode(data[cases[i]][:,:,:,6] - data[cases[i]][:,:,:,2]*data[cases[i]][:,:,:,2]) - data[cases[i]][:,:,:,21]
    # uv[cases[i]] = data[cases[i]][:,:,:,7] - data[cases[i]][:,:,:,0]*data[cases[i]][:,:,:,1] - data[cases[i]][:,:,:,22]
    # uw[cases[i]] = wnode2uvpnode(data[cases[i]][:,:,:,8] - uvpnode2wnode(data[cases[i]][:,:,:,0])*data[cases[i]][:,:,:,2] - data[cases[i]][:,:,:,23])
    # vw[cases[i]] = wnode2uvpnode(data[cases[i]][:,:,:,9] - uvpnode2wnode(data[cases[i]][:,:,:,1])*data[cases[i]][:,:,:,2] - data[cases[i]][:,:,:,24])    
    
    tke[cases[i]] = 0.5*(uu[cases[i]] + vv[cases[i]] + ww[cases[i]])

#%%Compute the residual of the TKE

ResNorm = dict()

for i in range(len(cases)):
    tmpRES = copy.deepcopy((TKE[cases[i]][:,:,:,14] - TKE[cases[i]][:,:,:,11]))
    tmpDIS = copy.deepcopy((TKE[cases[i]][:,:,:,11]))
    val = np.nanmedian(tmpRES[(dist[cases[i]][:,:,:,0]>38) & (dist[cases[i]][:,:,:,0]<44)])
    tmpRES[abs(tmpRES) < 0.05*val] = 0
    
    # tmpDIS = copy.deepcopy(TKE[cases[i]][:, :, :, 11])
    # tmpRES = copy.deepcopy(TKE[cases[i]][:, :, :, -1] + TKE[cases[i]][:, :, :, 11])
    # threshold = 0.05 * np.nanmax(np.nanmedian(tmpRES, axis=(0, 1)))
    # tmpRES[abs(tmpRES)<threshold] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        ResNorm[cases[i]] = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

#%%Simulation parameters

lx = 2.88; ly = 2.88; lz = 0.96
mpiProc = 32
nx,ny,nz = np.shape(data[cases[0]][:,:,:,0])
nz = nz + 5
zi = 1000
canopyH = 39/zi
dx = lx/nx; dy = ly/ny; dz = lz/nz
z_w = np.arange(0,nz)*dz

# lx = 2*np.pi; ly = 2*np.pi; lz = 1
# nx,ny,nz = np.shape(data[cases[0]][:,:,:,0])
# zi = 1000
# canopyH = 39/zi
# dx = lx/nx; dy = ly/ny; dz = lz/nz
# z_w = np.arange(0,nz)*dz
# z_uvp = np.arange(0,nz)*dz + dz/2
# LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
# uscale = 0.313

#%%PDF of the ratio betweeen yB and yBd

ratio = dict()

for i in range(len(cases)):
    ratio[cases[i]] = anisotropy_D[cases[i]][:,:,:,1]/anisotropy[cases[i]][:,:,:,1]

ratio_filt = dict()

for i in range(len(cases)):
    ratio_filt[cases[i]] = ratio[cases[i]][(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]

from matplotlib.transforms import ScaledTranslation

colors = ['#0072B2','#E69F00','#CC79A7']

fig, axs = plt.subplots(1,1, layout='constrained',figsize=(6,5))

from scipy.stats import gaussian_kde

for i in range(len(cases)):
    
    nsamples = 200000
    
    tmp_samp = np.random.choice(ratio_filt[cases[i]],size=nsamples,replace=False)

    kde = gaussian_kde(tmp_samp)
    x_pdf = np.linspace(min(tmp_samp),max(tmp_samp),1000)
    pdf = kde(x_pdf)
    axs.plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(tmp_samp)
    median_val = np.median(tmp_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    axs.plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    print(f"Median yB for {cases[i]} is: {median_val}")

axs.set_xlabel(r"$y_B/y_{B,d}$",fontsize=16)
axs.set_ylabel(r"PDF",fontsize=15)
# axs.set_xlim(0.4,1)
axs.set_ylim(0)
axs.tick_params(axis='both', which='major', labelsize=12)
    
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'PDF_yB_1to5.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()

#%%

tmp_filt = dict()

for i in range(len(cases)):
    tmp_filt[cases[i]] = anisotropy[cases[i]][:,:,:,1][(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]

from matplotlib.transforms import ScaledTranslation

colors = ['#0072B2','#E69F00','#CC79A7']

fig, axs = plt.subplots(1,1, layout='constrained',figsize=(6,5))

from scipy.stats import gaussian_kde

for i in range(len(cases)):
    
    nsamples = 200000
    
    tmp_samp = np.random.choice(tmp_filt[cases[i]],size=nsamples,replace=False)

    kde = gaussian_kde(tmp_samp)
    x_pdf = np.linspace(min(tmp_samp),max(tmp_samp),1000)
    pdf = kde(x_pdf)
    axs.plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(tmp_samp)
    median_val = np.median(tmp_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    axs.plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    print(f"Median yB for {cases[i]} is: {median_val}")

# axs.set_xlabel(r"$y_B/y_{B,d}$",fontsize=16)
axs.set_ylabel(r"PDF",fontsize=15)
# axs.set_xlim(0.4,1)
axs.set_ylim(0)
axs.tick_params(axis='both', which='major', labelsize=12)
    
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'PDF_yB_1to5.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()


#%% Filter the vertical variance and compute yBd

yB_filt = {}

for i in range(len(cases)):

    b33 = ww[cases[i]]/(2*tke[cases[i]]) - 1/3
    
    b33_filt = b33[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
    
    npts = len(b33_filt)

    bij_filt = np.zeros((npts, 3, 3))

    bij_filt[:, 0, 0] = np.zeros((npts))
    bij_filt[:, 1, 1] = np.zeros((npts))
    bij_filt[:, 2, 2] = b33_filt

    bij_filt[:, 0, 1] = np.zeros((npts))
    bij_filt[:, 1, 0] = np.zeros((npts))

    bij_filt[:, 0, 2] = np.zeros((npts))
    bij_filt[:, 2, 0] = np.zeros((npts))

    bij_filt[:, 1, 2] = np.zeros((npts))
    bij_filt[:, 2, 1] = np.zeros((npts))

    eigvals = np.linalg.eigvalsh(bij_filt)

    lambda3 = eigvals[:, 0]

    yB_filt[cases[i]] = (np.sqrt(3) / 2) * (3 * lambda3 + 1)

from matplotlib.transforms import ScaledTranslation

colors = ['#0072B2','#E69F00','#CC79A7']

fig, axs = plt.subplots(1,1, layout='constrained',figsize=(6,5))

from scipy.stats import gaussian_kde

for i in range(len(cases)):
    
    nsamples = 200000
    
    yB_samp = np.random.choice(yB_filt[cases[i]],size=nsamples,replace=False)

    kde = gaussian_kde(yB_samp)
    x_pdf = np.linspace(min(yB_samp),max(yB_samp),1000)
    pdf = kde(x_pdf)
    axs.plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(yB_samp)
    median_val = np.median(yB_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    axs.plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    print(f"Median yB for {cases[i]} is: {median_val}")

axs.set_xlabel(r"$y_B$",fontsize=16)
axs.set_ylabel(r"PDF",fontsize=15)
# axs.set_xlim(0.2,0.6)
axs.set_ylim(0)
axs.tick_params(axis='both', which='major', labelsize=12)
    
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'PDF_yB_1to5.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()



















































