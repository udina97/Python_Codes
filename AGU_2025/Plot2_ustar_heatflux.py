#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 08:36:40 2025

@author: u1450851
"""

#%%Import Libraries

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

#%%Load the chackpoint data
aniso = 'patch128_aniso_9ms'
classic = 'patch128_classic_9ms'
# aniso = 'homo_unstable_aniso_9ms'
# classic = 'homo_unstable_classic_9ms'

step = 130000

aniso3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/output_checkpoint/',[step],nx,ny,nz)
anisoSFC = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/output_checkpoint/',[step],nx,ny)
classic3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/output_checkpoint/',[step],nx,ny,nz)
classicSFC = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/output_checkpoint/',[step],nx,ny)

#%%ustar

L_a = anisoSFC['L'].flatten()
zoverL_a = 0.5*dz/L_a
ustar_a = anisoSFC['ustar'].flatten()*uscale
ustar_a_u = ustar_a[(L_a<0)]
ustar_a_s = ustar_a[(L_a>0)]

L_c = classicSFC['L'].flatten()
zoverL_c = 0.5*dz/L_c
ustar_c = classicSFC['ustar'].flatten()*uscale
ustar_c_u = ustar_c[(L_c<0)]
ustar_c_s = ustar_c[(L_c>0)]

#%%Plot ustar unstable pdf and ratio value

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,2]}   # bigger top panel
)

kde = gaussian_kde(ustar_a_u)
x_pdf = np.linspace(min(ustar_a_u), max(ustar_a_u), 1000)
pdf_a = kde(x_pdf)
int_a = np.trapz(pdf_a*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_a,c='g')

kde = gaussian_kde(ustar_c_u)
x_pdf = np.linspace(min(ustar_c_u), max(ustar_c_u), 1000)
pdf_c = kde(x_pdf)
int_c = np.trapz(pdf_c*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_c,c='r')

ax_top.set_xlabel(r"$u_*$ at SFC",fontsize=14)
ax_top.set_ylabel(r"$PDF$",fontsize=14)
ax_top.tick_params(axis='both',which='major',labelsize=12)
ax_top.set_xlim(0,)
ax_top.set_ylim(0,)

print((int_a/int_c - 1)*100)

nbins = 10

bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
bin_centers = np.sqrt(bins[:-1]*bins[1:])

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins-1):
    tmp1 = ustar_a[(zoverL_a<-bins[i]) & (zoverL_a>-bins[i+1])]
    tmp2 = ustar_c[(zoverL_c<-bins[i]) & (zoverL_c>-bins[i+1])]
    
    mean_diff[i] = np.median(tmp1)-np.median(tmp2)
    med_ref[i] = np.median(tmp2)
    
    ratio[i] = (mean_diff[i]/med_ref[i])

ax_bot.scatter(bin_centers, ratio[:-1]*100, s=10, c='b')
ax_bot.set_ylabel(r"$\frac{<u_{*}>_{a} - <u_{*}>_{r}}{<u_{*}>_{r}}$ [%]", fontsize=14)
ax_bot.set_xlabel(r"$-\zeta$", fontsize=14)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
ax_bot.invert_xaxis()
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')
ax_bot.set_xlim(10000,0.001)
ax_bot.tick_params(axis='both',which='major',labelsize=12)
ax_bot.set_ylim(-40,40)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/ustar_u.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%Plot ustar stable pdf and ratio value

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,2]}   # bigger top panel
)

kde = gaussian_kde(ustar_a_s)
x_pdf = np.linspace(min(ustar_a_s), max(ustar_a_s), 1000)
pdf_a = kde(x_pdf)
int_a = np.trapz(pdf_a*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_a,c='g')

kde = gaussian_kde(ustar_c_s)
x_pdf = np.linspace(min(ustar_c_s), max(ustar_c_s), 1000)
pdf_c = kde(x_pdf)
int_c = np.trapz(pdf_c*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_c,c='r')

ax_top.set_xlabel(r"$u_*$ at SFC",fontsize=14)
ax_top.set_ylabel(r"$PDF$",fontsize=14)
ax_top.tick_params(axis='both',which='major',labelsize=12)
ax_top.set_xlim(0,)
ax_top.set_ylim(0,)

print((int_a/int_c - 1)*100)

nbins = 10

bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
bin_centers = np.sqrt(bins[:-1]*bins[1:])

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins-1):
    tmp1 = ustar_a[(zoverL_a>bins[i]) & (zoverL_a<bins[i+1])]
    tmp2 = ustar_c[(zoverL_c>bins[i]) & (zoverL_c<bins[i+1])]
    
    mean_diff[i] = np.median(tmp1)-np.median(tmp2)
    med_ref[i] = np.median(tmp2)
    
    ratio[i] = (mean_diff[i]/med_ref[i])

ax_bot.scatter(bin_centers, ratio[:-1]*100, s=10, c='b')
ax_bot.set_ylabel(r"$\frac{<u_{*}>_{a} - <u_{*}>_{r}}{<u_{*}>_{r}}$ [%]", fontsize=14)
ax_bot.set_xlabel(r"$\zeta$", fontsize=14)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')
ax_bot.set_xlim(0.001,10000)
ax_bot.tick_params(axis='both',which='major',labelsize=12)
ax_bot.set_ylim(-40,40)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/ustar_s.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%--------------------------------------------------------------------------------------------------------------------------------------

#               HEATFLUX

#%%-------------------------------------------------------------------------------------------------------------------------------------

L_a = anisoSFC['L'].flatten()
zoverL_a = 0.5*dz/L_a
wT_a = anisoSFC['sfcFLUX'].flatten()*uscale*Tscale*1000
wT_a_u = wT_a[(L_a<0)]
wT_a_s = wT_a[(L_a>0)]

L_c = classicSFC['L'].flatten()
zoverL_c = 0.5*dz/L_c
wT_c = classicSFC['sfcFLUX'].flatten()*uscale*Tscale*1000
wT_c_u = wT_c[(L_c<0)]
wT_c_s = wT_c[(L_c>0)]

#%%heatflux unstable

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[1,1]}   # bigger top panel
)

kde = gaussian_kde(wT_a_u)
x_pdf = np.linspace(min(wT_a_u), max(wT_a_u), 1000)
pdf_a = kde(x_pdf)
int_a = np.trapz(pdf_a*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_a,c='g')

kde = gaussian_kde(wT_c_u)
x_pdf = np.linspace(min(wT_c_u), max(wT_c_u), 1000)
pdf_c = kde(x_pdf)
int_c = np.trapz(pdf_c*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_c,c='r')

ax_top.set_xlabel(r"H [W/$m^2$] at SFC",fontsize=14)
ax_top.set_ylabel(r"$PDF$",fontsize=14)
ax_top.tick_params(axis='both',which='major',labelsize=12)
ax_top.set_xlim(0,)
ax_top.set_ylim(0,)

print((int_a/int_c - 1)*100)

nbins = 10

bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
bin_centers = np.sqrt(bins[:-1]*bins[1:])

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins-1):
    tmp1 = wT_a[(zoverL_a<-bins[i]) & (zoverL_a>-bins[i+1])]
    tmp2 = wT_c[(zoverL_c<-bins[i]) & (zoverL_c>-bins[i+1])]
    
    mean_diff[i] = np.median(tmp1)-np.median(tmp2)
    med_ref[i] = np.median(tmp2)
    
    ratio[i] = (mean_diff[i]/med_ref[i])

ax_bot.scatter(bin_centers, ratio[:-1]*100, s=10, c='b')
ax_bot.set_ylabel(r"$\frac{<H>_{a} - <H>_{r}}{<H>_{r}}$ [%]", fontsize=14)
ax_bot.set_xlabel(r"$-\zeta$", fontsize=14)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
ax_bot.invert_xaxis()
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')
ax_bot.set_xlim(10000,0.001)
ax_bot.tick_params(axis='both',which='major',labelsize=12)
ax_bot.set_ylim(-40,40)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/HF_u.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()


#%%heatflux stable
import matplotlib.ticker as ticker

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[1,1]}   # bigger top panel
)

kde = gaussian_kde(wT_a_s)
x_pdf = np.linspace(min(wT_a_s), max(wT_a_s), 1000)
pdf_a = kde(x_pdf)
int_a = np.trapz(pdf_a*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_a,c='g')

kde = gaussian_kde(wT_c_s)
x_pdf = np.linspace(min(wT_c_s), max(wT_c_s), 1000)
pdf_c = kde(x_pdf)
int_c = np.trapz(pdf_c*x_pdf,x_pdf)
ax_top.plot(x_pdf,pdf_c,c='r')

ax_top.set_xlabel(r"H [W/$m^2$] at SFC",fontsize=14)
ax_top.set_ylabel(r"$PDF$",fontsize=14)
ax_top.tick_params(axis='both',which='major',labelsize=12)
ax_top.set_ylim(0,)
# ax_top.set_yscale('log')

print((int_a/int_c - 1)*100)

nbins = 10

bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
bin_centers = np.sqrt(bins[:-1]*bins[1:])

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins-1):
    tmp1 = wT_a[(zoverL_a>bins[i]) & (zoverL_a<bins[i+1])]
    tmp2 = wT_c[(zoverL_c>bins[i]) & (zoverL_c<bins[i+1])]
    
    mean_diff[i] = np.median(tmp1)-np.median(tmp2)
    med_ref[i] = np.median(tmp2)
    
    ratio[i] = (mean_diff[i]/med_ref[i])

ax_bot.scatter(bin_centers, ratio[:-1]*100, s=10, c='b')
ax_bot.set_ylabel(r"$\frac{<H>_{a} - <H>_{r}}{<H>_{r}}$ [%]", fontsize=14)
ax_bot.set_xlabel(r"$\zeta$", fontsize=14)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

# ax_top.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# ax_top.xaxis.get_offset_text().set_fontsize(12)  # optional styling
ax_bot.set_xlim(0.001,10000)
ax_bot.tick_params(axis='both',which='major',labelsize=12)
ax_bot.set_ylim(-40,40)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/HF_s.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()











































































# %%
