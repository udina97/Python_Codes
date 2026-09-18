#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:57:21 2026

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

for i in range(len(cases)):
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    # path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
    data[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/dataTKE.nc').data
    # data[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/Data_Momentum_4TKE.nc').data
    anisotropy[cases[i]] = xr.open_dataarray(path_to_data + cases[i] + '/anisotropy.nc').data
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

#%%SGS

txx = dict(); tyy = dict(); tzz = dict(); txy = dict(); txz = dict(); tyz = dict()
tke_sgs = dict()
for i in range(len(cases)):
    txx[cases[i]] = - data[cases[i]][:,:,:,19]
    tyy[cases[i]] = - data[cases[i]][:,:,:,20]
    tzz[cases[i]] = - data[cases[i]][:,:,:,21]
    txy[cases[i]] = - data[cases[i]][:,:,:,22]
    txz[cases[i]] = - data[cases[i]][:,:,:,23]
    tyz[cases[i]] = - data[cases[i]][:,:,:,24]
    
    tke_sgs[cases[i]] = 0.5*(txx[cases[i]] + tyy[cases[i]] + tzz[cases[i]])

#%%

ResNorm = dict()

for i in range(len(cases)):
    tmpRES = copy.deepcopy((TKE[cases[i]][:,:,:,14] - TKE[cases[i]][:,:,:,11]))
    tmpDIS = copy.deepcopy((TKE[cases[i]][:,:,:,11]))
    val = np.nanmedian(tmpRES[(dist[cases[i]][:,:,:,0]>38) & (dist[cases[i]][:,:,:,0]<44)])
    tmpRES[abs(tmpRES) < 0.01*val] = 0
    
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

#%%Plot verical profile of domain average normalized variance

fig,axs = plt.subplots(1,3,tight_layout=True)

for i in range(len(cases)):
    uu_e = uu[cases[i]]/tke[cases[i]]
    axs[0].plot(np.nanmedian(uu_e,axis=(0,1)),np.arange(0,nz)*dz+dz/2)

for i in range(len(cases)):
    vv_e = vv[cases[i]]/tke[cases[i]]
    axs[1].plot(np.nanmedian(vv_e,axis=(0,1)),np.arange(0,nz)*dz+dz/2)

for i in range(len(cases)):
    ww_e = ww[cases[i]]/tke[cases[i]]
    axs[2].plot(np.nanmedian(ww_e,axis=(0,1)),np.arange(0,nz)*dz+dz/2)

plt.show()

#%%Plot pcolor of the residual

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,4))

for i in range(len(cases)):
    axs[i].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,ResNorm[cases[i]][:,ny//2,:].T,cmap='bwr',vmin=-100,vmax=100)

plt.show()

#%%Plot the normalized PDF

from matplotlib.transforms import ScaledTranslation

# colors = ['#0072B2','#E69F00','#CC79A7']
colors = ['#41049D','#CC4778','#FCA636']
ls = ['-','--','-.']
tols = [10,5,15]
layout = [['a)', 'b)', 'c)'],
          ['d)', 'e)', 'f)']]

fig, axs_dict = plt.subplot_mosaic(layout, layout='constrained',figsize=(8,6),sharey=False)

for label, ax in axs_dict.items():
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans)),
        fontsize=15, va='bottom', fontfamily='serif')

from scipy.stats import gaussian_kde
axs = np.array([[axs_dict[label] for label in row] for row in layout])

for i in range(len(cases)):

    uu_e = uu[cases[i]]/tke[cases[i]]
    vv_e = vv[cases[i]]/tke[cases[i]]
    ww_e = ww[cases[i]]/tke[cases[i]]
    uw_e = abs(uw[cases[i]]/tke[cases[i]])
    
    # uu_filt = uu_e[(anisotropy[cases[i]][:,:,:,1]<1) & (anisotropy[cases[i]][:,:,:,1]>0) & (dist[cases[i]][:,:,:,0]>40) & (dist[cases[i]][:,:,:,0]<40*15)]
    # vv_filt = vv_e[(anisotropy[cases[i]][:,:,:,1]<1) & (anisotropy[cases[i]][:,:,:,1]>0) & (dist[cases[i]][:,:,:,0]>40) & (dist[cases[i]][:,:,:,0]<40*15)]
    # ww_filt = ww_e[(anisotropy[cases[i]][:,:,:,1]<1) & (anisotropy[cases[i]][:,:,:,1]>0) & (dist[cases[i]][:,:,:,0]>40) & (dist[cases[i]][:,:,:,0]<40*15)]
    # uw_filt = uw_e[(anisotropy[cases[i]][:,:,:,1]<1) & (anisotropy[cases[i]][:,:,:,1]>0) & (dist[cases[i]][:,:,:,0]>40) & (dist[cases[i]][:,:,:,0]<40*15)]
    for j in range(len(tols)):
        uu_filt = uu_e[(abs(ResNorm[cases[i]])<tols[j]) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
        vv_filt = vv_e[(abs(ResNorm[cases[i]])<tols[j]) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
        ww_filt = ww_e[(abs(ResNorm[cases[i]])<tols[j]) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
        uw_filt = uw_e[(abs(ResNorm[cases[i]])<tols[j]) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
        
        # uu_filt = uu_e[:,:,10:][(abs(ResNorm[cases[i]])[:,:,10:]<10)]
        # vv_filt = vv_e[:,:,10:][(abs(ResNorm[cases[i]])[:,:,10:]<10)]
        # ww_filt = ww_e[:,:,10:][(abs(ResNorm[cases[i]])[:,:,10:]<10)]
        # uw_filt = uw_e[:,:,10:][(abs(ResNorm[cases[i]])[:,:,10:]<10)]
        
        nsamples = 200000
        
        uu_samp = np.random.choice(uu_filt,size=nsamples,replace=False)
        vv_samp = np.random.choice(vv_filt,size=nsamples,replace=False)
        ww_samp = np.random.choice(ww_filt,size=nsamples,replace=False)
        uw_samp = np.random.choice(uw_filt,size=nsamples,replace=False)
        
        cR = 1/(1 - (3/2)*ww_samp) # for vv and ww
        yB = (np.sqrt(3)/2)*(1-1/cR) # for vv and ww

        kde = gaussian_kde(uu_samp)
        x_pdf = np.linspace(min(uu_samp),max(uu_samp),1000)
        pdf = kde(x_pdf)
        axs[0,0].plot(x_pdf,pdf,c=colors[i],linestyle=ls[j])
        mean_val = np.mean(uu_samp)
        median_val = np.median(uu_samp)
        y_mean = np.interp(mean_val, x_pdf, pdf)
        y_median = np.interp(median_val, x_pdf, pdf)
        axs[0,0].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
        print(f"Median uu/e for {cases[i]} is: {median_val}")
        
        kde = gaussian_kde(vv_samp)
        x_pdf = np.linspace(min(vv_samp),max(vv_samp),1000)
        pdf = kde(x_pdf)
        axs[0,1].plot(x_pdf,pdf,c=colors[i],linestyle=ls[j])
        mean_val = np.mean(vv_samp)
        median_val = np.median(vv_samp)
        y_mean = np.interp(mean_val, x_pdf, pdf)
        y_median = np.interp(median_val, x_pdf, pdf)
        axs[0,1].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
        print(f"Median vv/e for {cases[i]} is: {median_val}")
        
        kde = gaussian_kde(ww_samp)
        x_pdf = np.linspace(min(ww_samp),max(ww_samp),1000)
        pdf = kde(x_pdf)
        axs[0,2].plot(x_pdf,pdf,c=colors[i],linestyle=ls[j])
        mean_val = np.mean(ww_samp)
        median_val = np.median(ww_samp)
        y_mean = np.interp(mean_val, x_pdf, pdf)
        y_median = np.interp(median_val, x_pdf, pdf)
        axs[0,2].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
        print(f"Median ww/e for {cases[i]} is: {median_val}")
        
        kde = gaussian_kde(uw_samp)
        x_pdf = np.linspace(min(uw_samp),max(uw_samp),1000)
        pdf = kde(x_pdf)
        axs[1,0].plot(x_pdf,pdf,c=colors[i],linestyle=ls[j])
        mean_val = np.mean(uw_samp)
        median_val = np.median(uw_samp)
        y_mean = np.interp(mean_val, x_pdf, pdf)
        y_median = np.interp(median_val, x_pdf, pdf)
        axs[1,0].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
        
        kde = gaussian_kde(cR)
        x_pdf = np.linspace(min(cR),max(cR),1000)
        pdf = kde(x_pdf)
        axs[1,1].plot(x_pdf,pdf,c=colors[i],linestyle=ls[j])
        mean_val = np.mean(cR)
        median_val = np.median(cR)
        y_mean = np.interp(mean_val, x_pdf, pdf)
        y_median = np.interp(median_val, x_pdf, pdf)
        axs[1,1].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
        print(f"Median cR for {cases[i]} is: {median_val}")
        
        kde = gaussian_kde(yB)
        x_pdf = np.linspace(min(yB),max(yB),1000)
        pdf = kde(x_pdf)
        axs[1,2].plot(x_pdf,pdf,c=colors[i],linestyle=ls[j])
        mean_val = np.mean(yB)
        median_val = np.median(yB)
        y_mean = np.interp(mean_val, x_pdf, pdf)
        y_median = np.interp(median_val, x_pdf, pdf)
        axs[1,2].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
        print(f"Median yB for {cases[i]} is: {median_val}")
    
# cr = [1.8]
# cr_ls = ['-.']

# for i in range(len(cr)):
#     uu_r = 2/3*(2/cr[i] + 1)
#     vv_r = 2/3*(-1/(cr[i]) + 1)
#     ww_r = 2/3*(-1/(cr[i]) + 1)
    
    # axs[0,0].axvline(uu_r,c='k',ls=cr_ls[i])
    # axs[0,1].axvline(vv_r,c='k',ls=cr_ls[i])
    # axs[1,0].axvline(ww_r,c='k',ls=cr_ls[i])
    
axs[0,0].set_xlabel(r"$\frac{\overline{u'u'}}{\overline{e}}$",fontsize=16)
axs[0,1].set_xlabel(r"$\frac{\overline{v'v'}}{\overline{e}}$",fontsize=16)
axs[0,2].set_xlabel(r"$\frac{\overline{w'w'}}{\overline{e}}$",fontsize=16)
axs[1,0].set_xlabel(r"$\frac{|\overline{u'w'}|}{\overline{e}}$",fontsize=16)
axs[1,1].set_xlabel(r"$c_R$",fontsize=16)
axs[1,2].set_xlabel(r"$y_B^{approx}$",fontsize=16)

axs[0,0].set_ylabel(r"PDF",fontsize=15)
axs[1,0].set_ylabel(r"PDF",fontsize=15)

axs[0,0].set_xlim(0.65,1.25)
axs[0,1].set_xlim(0.35,1)
axs[0,2].set_xlim(0.25,0.65)
axs[1,0].set_xlim(0.1,0.4)
axs[1,1].set_xlim(1,5)
axs[1,2].set_xlim(0.25,0.8)

    
for i in range(len(axs[0])):
    axs[0,i].set_ylim(0)
    axs[1,i].set_ylim(0)
    axs[0,i].tick_params(axis='both', which='major', labelsize=12)
    axs[1,i].tick_params(axis='both', which='major', labelsize=12)
    
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'Variances_RottaModel_cRyB.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()

#%%COmpute variance PDF, then cR PDF, finally yB PDF

from matplotlib.transforms import ScaledTranslation
from scipy.stats import gaussian_kde
colors = ['#0072B2','#E69F00','#CC79A7']
fig, axs = plt.subplots(1,3, layout='constrained',figsize=(10,4))

for i in range(len(cases)):

    # var_e = uu[cases[i]]/tke[cases[i]]
    # var_e = vv[cases[i]]/tke[cases[i]]
    var_e = ww[cases[i]]/tke[cases[i]]
    
    var_filt = var_e[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
    
    nsamples = 200000
    
    var_samp = np.random.choice(var_filt,size=nsamples,replace=False)
    
    # cR = 1/((3/4)*var_samp - 0.5) # for uu
    cR = 1/(1 - (3/2)*var_samp) # for vv and ww
    
    # yB = (np.sqrt(3)/2)*(1+2/cR) # for uu
    yB = (np.sqrt(3)/2)*(1-1/cR) # for vv and ww
    
    kde = gaussian_kde(var_samp)
    x_pdf = np.linspace(min(var_samp),max(var_samp),1000)
    pdf = kde(x_pdf)
    axs[0].plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(var_samp)
    median_val = np.median(var_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    axs[0].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    print(f"Median variance for {cases[i]} is: {median_val}")
    
    kde = gaussian_kde(cR)
    x_pdf = np.linspace(min(cR),max(cR),1000)
    pdf = kde(x_pdf)
    axs[1].plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(cR)
    median_val = np.median(cR)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    axs[1].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    print(f"Median cR for {cases[i]} is: {median_val}")
    
    kde = gaussian_kde(yB)
    x_pdf = np.linspace(min(yB),max(yB),1000)
    pdf = kde(x_pdf)
    axs[2].plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(yB)
    median_val = np.median(yB)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    axs[2].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    print(f"Median yB for {cases[i]} is: {median_val}")
    
axs[0].set_xlabel(r"$\frac{\overline{u_i'u_i'}}{\overline{e}}$",fontsize=16)
axs[1].set_xlabel(r"$c_R$",fontsize=16)
axs[2].set_xlabel(r"$y_B^{approx}$",fontsize=16)

axs[0].set_ylabel(r"PDF",fontsize=15)

# axs[0].set_xlim(0.1,0.6)
axs[1].set_xlim(1,5)
# axs[2].set_xlim(0.25,0.75)

    
for i in range(len(axs)):
    axs[i].set_ylim(0)
    axs[i].tick_params(axis='both', which='major', labelsize=12)
    
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'Variances_RottaModel_Res.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()
    
#%% Filter all the terms in the REynolds stress tensor and compute yB full

yB_filt = {}

for i in range(len(cases)):

    b11 = uu[cases[i]]/(2*tke[cases[i]]) - 1/3
    b22 = vv[cases[i]]/(2*tke[cases[i]]) - 1/3
    b33 = ww[cases[i]]/(2*tke[cases[i]]) - 1/3
    b12 = uv[cases[i]]/(2*tke[cases[i]])
    b13 = uw[cases[i]]/(2*tke[cases[i]])
    b23 = vw[cases[i]]/(2*tke[cases[i]])
    
    b11_filt = b11[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*5)]
    b22_filt = b22[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*5)]
    b33_filt = b33[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*5)]
    b12_filt = b12[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*5)]
    b13_filt = b13[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*5)]
    b23_filt = b23[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*5)]
    
    npts = len(b11_filt)

    bij_filt = np.zeros((npts, 3, 3))

    bij_filt[:, 0, 0] = b11_filt
    bij_filt[:, 1, 1] = b22_filt
    bij_filt[:, 2, 2] = b33_filt

    bij_filt[:, 0, 1] = b12_filt
    bij_filt[:, 1, 0] = b12_filt

    bij_filt[:, 0, 2] = b13_filt
    bij_filt[:, 2, 0] = b13_filt

    bij_filt[:, 1, 2] = b23_filt
    bij_filt[:, 2, 1] = b23_filt

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

axs.axvline(0.384,c='k',ls='-')
axs.axvline(0.52,c='k',ls='--')
axs.set_xlabel(r"$y_B$",fontsize=16)
axs.set_ylabel(r"PDF",fontsize=15)
axs.set_xlim(0.2,0.6)
axs.set_ylim(0)
axs.tick_params(axis='both', which='major', labelsize=12)
    
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'PDF_yB_1to5.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()

#%%

import numpy as np
from scipy.optimize import minimize


def rotta_predictions(params):
    """
    Modified linear Rotta model.

    params = [c, a_u, a_v, a_w]
    """

    c, a_u, a_v, a_w = params

    Ru_pred = (2/3) * (2/c + a_u)
    Rv_pred = (2/3) * (a_v - 1/c)
    Rw_pred = (2/3) * (a_w - 1/c)

    return np.array([Ru_pred, Rv_pred, Rw_pred])


def objective(params, R, weights=None):
    """
    Sum of squared residuals between measured and predicted variance ratios.

    R should have shape (N, 3), with columns:
        R[:, 0] = u'^2 / e
        R[:, 1] = v'^2 / e
        R[:, 2] = w'^2 / e

    weights is optional and should also have shape (N, 3).
    """

    pred = rotta_predictions(params)

    residuals = R - pred[None, :]

    if weights is not None:
        residuals = residuals * weights

    return np.sum(residuals**2)


def fit_rotta_constrained(R, weights=None):
    """
    Fit c, a_u, a_v, a_w subject to:

        1 < c < 10
        a_u > 0
        a_v > 0
        a_w > 0
        a_u + a_v + a_w = 3
    """

    R = np.asarray(R, dtype=float)

    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError("R must have shape (N, 3), with columns [Ru, Rv, Rw].")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != R.shape:
            raise ValueError("weights must have the same shape as R.")

    eps = 1e-8

    # Initial guess
    c0 = 2.0
    a0 = np.array([1.0, 1.0, 1.0])
    x0 = np.array([c0, a0[0], a0[1], a0[2]])

    # Bounds:
    # 1 < c < 10, a_i > 0
    bounds = [
        (1.0 + eps, 10.0 - eps),  # c
        (eps, None),              # a_u
        (eps, None),              # a_v
        (eps, None),              # a_w
    ]

    # Equality constraint: a_u + a_v + a_w = 3
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: x[1] + x[2] + x[3] - 3.0
        }
    ]

    result = minimize(
        objective,
        x0,
        args=(R, weights),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
            "disp": False
        }
    )

    return result


#%%

for i in range(len(cases)):

    uu_e = uu[cases[i]]/tke[cases[i]]
    vv_e = vv[cases[i]]/tke[cases[i]]
    ww_e = ww[cases[i]]/tke[cases[i]]
    uw_e = abs(uw[cases[i]]/tke[cases[i]])
    
    uu_filt = uu_e[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
    vv_filt = vv_e[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
    ww_filt = ww_e[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
    uw_filt = uw_e[(abs(ResNorm[cases[i]])<10) & (dist[cases[i]][:,:,:,0]>1*40) & (dist[cases[i]][:,:,:,0]<40*15)]
    
    nsamples = 200000
    
    uu_samp = np.random.choice(uu_filt,size=nsamples,replace=False)
    vv_samp = np.random.choice(vv_filt,size=nsamples,replace=False)
    ww_samp = np.random.choice(ww_filt,size=nsamples,replace=False)
    uw_samp = np.random.choice(uw_filt,size=nsamples,replace=False)
    
    R = np.column_stack((uu_samp, vv_samp, ww_samp))
    
    result = fit_rotta_constrained(R)

    if not result.success:
        print("Optimization failed:")
        print(result.message)
    
    c, a_u, a_v, a_w = result.x
    
    print("c   =", c)
    print("a_u =", a_u)
    print("a_v =", a_v)
    print("a_w =", a_w)
    
    print("sum of a's =", a_u + a_v + a_w)
    print("objective =", result.fun)



















































