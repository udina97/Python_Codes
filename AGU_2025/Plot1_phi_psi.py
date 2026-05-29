#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 16:16:33 2025

@author: u1450851
"""

#%%Import libraries

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

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
# aniso = 'patch128_aniso_9ms'
# classic = 'patch128_classic_9ms'
aniso = 'Homog/128/homo_unstable_aniso_1ms'
classic = 'Homog/128/homo_unstable_classic_1ms'

step = 100000

aniso3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/output_checkpoint/',[step],nx,ny,nz)
anisoSFC = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/output_checkpoint/',[step],nx,ny)
classic3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/output_checkpoint/',[step],nx,ny,nz)
classicSFC = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/output_checkpoint/',[step],nx,ny)

#%%functions

def phiM_u_ref(zeta):
    phi_ref = np.zeros_like(zeta)
    for i in range(len(zeta)):
        if (abs(zeta[i])>0.41**(-3)):
            phi_ref[i] = 1
        else:
            phi_ref[i] = (0.33 + 0.41*abs(zeta[i])**(4/3))/(0.33 + abs(zeta[i]))
            
    return phi_ref

def phiH_u_ref(zeta):
    phi_ref = np.zeros_like(zeta)
    for i in range(len(zeta)):
        if zeta[i]>30:
            zeta[i] = 30
    phi_ref = (0.33 + 0.057*abs(zeta)**0.78)/(0.33 + abs(zeta)**0.78)
            
    return phi_ref

def psiM_u_ref(zeta):
    psi_ref = np.zeros_like(zeta)
    a = 0.33
    b = 0.41
    for i in range(len(zeta)):
        if (abs(zeta[i])>0.41**(-3)):
            yy = b**(-3)                   
            xx = (yy / a)**(1/3)           
            psi_ref[i] = (np.log(a + yy) - 3*b*(yy**(1/3)) + ((b*(a**(1/3)))/2) * np.log(((1 + xx)**2) / (1 - xx + xx**2)) 
                + np.sqrt(3)*b*(a**(1/3))*np.arctan(((2*xx) - 1) / np.sqrt(3)) - np.log(a) + (np.sqrt(3)*b*(a**(1/3))*np.pi/6))
        else:
            xx = (abs(zeta[i]) / a)**(1/3)
            yy = abs(zeta[i])
            psi_ref[i] = (np.log(a + yy) - 3*b*(yy**(1/3)) + ((b*(a**(1/3)))/2) * np.log(((1 + xx)**2) / (1 - xx + xx**2)) 
                + np.sqrt(3)*b*(a**(1/3))*np.arctan(((2*xx) - 1) / np.sqrt(3)) - np.log(a) + (np.sqrt(3)*b*(a**(1/3))*np.pi/6))
            
    return psi_ref

def psiH_u_ref(zeta):
    psi_ref = np.zeros_like(zeta)
    for i in range(len(zeta)):
        if abs(zeta[i])>30:
            psi_ref[i] = ((1-0.057)/0.78)*np.log((0.33 + abs(30)**(0.78))/0.33)
        else:
            psi_ref[i] = ((1-0.057)/0.78)*np.log((0.33 + abs(zeta[i])**(0.78))/0.33)
            
    return psi_ref

def phiM_s_ref(zeta):
    phi_ref = np.zeros_like(zeta)
    for i in range(len(zeta)):
        if (abs(zeta[i])>10):
            phi_ref[i] = 1 + 5.3*10
        else:
            phi_ref[i] = 1 + 5.3*zeta[i]
            
    return phi_ref

def phiH_s_ref(zeta):
    phi_ref = np.zeros_like(zeta)
    for i in range(len(zeta)):
        if abs(zeta[i])>10:
            phi_ref[i] = 1 + 8*10
        else:
            phi_ref[i] = 1 + 8*zeta[i]
            
    return phi_ref

def psiM_s_ref(zeta):
    psi_ref = np.zeros_like(zeta)
    for i in range(len(zeta)):
        if (abs(zeta[i])>10):         
            psi_ref[i] = -5.3*10
        else:
            psi_ref[i] = -5.3*zeta[i]
            
    return psi_ref

def psiH_s_ref(zeta):
    psi_ref = np.zeros_like(zeta)
    for i in range(len(zeta)):
        if abs(zeta[i])>10:
            psi_ref[i] = -8*10
        else:
            psi_ref[i] = -8*zeta[i]
            
    return psi_ref

#%%

L = anisoSFC['L'].flatten()
zoverL = 0.5*dz/L
phiM = anisoSFC['phi_m'].flatten()
phiH = anisoSFC['phi_h'].flatten()
psiM = anisoSFC['psi_m'].flatten()
psiH = anisoSFC['psi_h'].flatten()

uu = aniso3D['uu_new'][:,:,:nz] - aniso3D['u_new'][:,:,:nz]*aniso3D['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = aniso3D['vv_new'][:,:,:nz] - aniso3D['v_new'][:,:,:nz]*aniso3D['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = aniso3D['ww_new'][:,:,:nz] - aniso3D['w_new'][:,:,:nz]*aniso3D['w_new'][:,:,:nz]#+ checkpnt['tzz_new'][:,:,:nz]
uv = aniso3D['uv_new'][:,:,:nz] - aniso3D['u_new'][:,:,:nz]*aniso3D['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = aniso3D['uw_new'][:,:,:nz] - aniso3D['u_new'][:,:,:nz]*aniso3D['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = aniso3D['vw_new'][:,:,:nz] - aniso3D['v_new'][:,:,:nz]*aniso3D['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke = uu + vv + ww

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

yB_sfc = yB[:,:,0].flatten()

#%%
L_c = classicSFC['L'].flatten()
zoverL_c = 0.5*dz/L_c

#%%phi_M unstable

from matplotlib.colors import Normalize
norm_aniso = Normalize(vmin=0, vmax=0.3)

nbins = 10

zeta_u = abs(zoverL[(L<0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = phiM_u_ref(zeta_u)

# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-2):
#     tmp1 = phiM[(zoverL<-bins[i]) & (zoverL>-bins[i+1])]
#     tmp2 = classicSFC['phi_m'].flatten()[(zoverL_c<-bins[i]) & (zoverL_c>-bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = phiM[(L<0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin - phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = -np.logspace(-3,4,1000)
phi_ax = phiM_u_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = -np.logspace(-3, 4, 1000)
zeta_ax = -np.logspace(-3, 4, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]<-30,-30,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
yb_scale = (yb*np.sqrt(3)/2)/np.max(yb)
phim = np.zeros((len(zeta), len(yb)))
for j in range(len(yb)):
    a = np.where(yb[j] > 0.6, 0.012, 0.24 - 0.38 * yb[j])
    b = 0.061
    c = 0.45 - 0.53 * yb[j]
    n = -0.12 + 6.4 * yb[j]
    d = 0.48 + 1.8 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func(z):
        z_abs = np.abs(z)
        phi = (a + b * z_abs**n) / (a + z_abs**n) + c * z_abs**(1/3)
        phi[(z_abs>0.41**(-3))] = 1
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta)
    
for j in range(len(yb)):
    ax_top.plot(-zeta_ax, phim[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L<0)]),phiM[(L<0)],s=1,c=yB_sfc[(L<0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
ax_top.invert_xaxis()
ax_top.axvline(14.513,c='k',ls='--')
ax_top.axhline(1,c='k',ls='--')
# ax_top.set_ylabel(r"$\phi_M$",fontsize=15)
# cbar = plt.colorbar(p,location='left')
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\phi_a - \phi_r|>}{<|\phi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$-\zeta$", fontsize=15)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(10000,0.001)
ax_bot.set_ylim(0,40)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/phiM_u.png', dpi = 300, facecolor='None', edgecolor='None')

plt.show()

#%%phiM stable
from matplotlib.colors import Normalize
norm_aniso = Normalize(vmin=0, vmax=0.3)

nbins = 10

zeta_u = abs(zoverL[(L>0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = phiM_s_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = phiM[(L>0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin-phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])


# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-1):
#     tmp1 = phiM[(zoverL>bins[i]) & (zoverL<bins[i+1])]
#     tmp2 = classicSFC['phi_m'].flatten()[(zoverL_c>bins[i]) & (zoverL_c<bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = np.logspace(-3,4,1000)
phi_ax = phiM_s_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = np.logspace(-3, 5, 1000)
zeta_ax = np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]>10,10,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
phim = np.zeros((len(zeta), len(yb)))

for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func(z):
        z_abs = np.abs(z)
        phi = (a + b * z)
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta)
    
for j in range(len(yb)):
    ax_top.plot(zeta_ax, phim[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L>0)]),phiM[(L>0)],s=1,c=yB_sfc[(L>0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
# ax_top.set_yscale('log')
ax_top.axhline(1,c='k',ls='--')
ax_top.set_ylabel(r"$\phi_M$",fontsize=15)
# cbar = plt.colorbar(p)
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\phi_a - \phi_r|>}{<|\phi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$\zeta$", fontsize=15)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(0.001,10000)
ax_bot.set_ylim(0,40)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/phiM_s.png', dpi = 300, facecolor='None', edgecolor='None')

plt.show()

#%%phi_H unstable
from matplotlib.colors import Normalize
norm_aniso = Normalize(vmin=0, vmax=0.3)

nbins = 10

zeta_u = abs(zoverL[(L<0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = phiH_u_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
ratio = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = phiH[(L<0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin-phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])

# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-2):
#     tmp1 = phiH[(zoverL<-bins[i]) & (zoverL>-bins[i+1])]
#     tmp2 = classicSFC['phi_h'].flatten()[(zoverL_c<-bins[i]) & (zoverL_c>-bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = -np.logspace(-3,4,1000)
phi_ax = phiH_u_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = -np.logspace(-3, 4, 1000)
zeta_ax = -np.logspace(-3, 4, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]<-30,-30,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
phim = np.zeros((len(zeta), len(yb)))
for j in range(len(yb)):
    a = np.where(yb[j] > 0.6, 0.012, 0.24 - 0.38 * yb[j])
    b = 0.061
    c = 0.45 - 0.53 * yb[j]
    n = -0.12 + 6.4 * yb[j]
    d = 0.48 + 1.8 * yb[j]

    # Define phi functions for this yb[j]
    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiH_func(zeta)
    
for j in range(len(yb)):
    ax_top.plot(-zeta_ax, phim[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L<0)]),phiH[(L<0)],s=1,c=yB_sfc[(L<0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
ax_top.invert_xaxis()
ax_top.axhline(1,c='k',ls='--')
ax_top.set_ylabel(r"$\phi_H$",fontsize=15)
# cbar = plt.colorbar(p)
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\phi_a - \phi_r|>}{<|\phi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$-\zeta$", fontsize=15)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(10000,0.001)
ax_bot.set_ylim(0,40)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/phiH_u.png', dpi = 300, facecolor='None', edgecolor='None')

plt.show()

#%%phiH stable
from matplotlib.colors import Normalize
norm_aniso = Normalize(vmin=0, vmax=0.3)

nbins = 10

zeta_u = abs(zoverL[(L>0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = phiH_s_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = phiH[(L>0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin-phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])

# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-1):
#     tmp1 = phiH[(zoverL>bins[i]) & (zoverL<bins[i+1])]
#     tmp2 = classicSFC['phi_h'].flatten()[(zoverL_c>bins[i]) & (zoverL_c<bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = np.logspace(-3,4,1000)
phi_ax = phiH_s_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = np.logspace(-3, 5, 1000)
zeta_ax = np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]>10,10,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
phim = np.zeros((len(zeta), len(yb)))

for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    # Define phi functions for this yb[j]
    def phiH_func(z):
        phi = (c + d * z)
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiH_func(zeta)
    
for j in range(len(yb)):
    ax_top.plot(zeta_ax, phim[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L>0)]),phiH[(L>0)],s=1,c=yB_sfc[(L>0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
ax_top.axhline(1,c='k',ls='--')
ax_top.set_ylabel(r"$\phi_H$",fontsize=15)
# cbar = plt.colorbar(p)
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\phi_a - \phi_r|>}{<|\phi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$\zeta$", fontsize=15)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(0.001,10000)
ax_bot.set_ylim(0,40)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/phiH_s.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%---------------------------------------------------------------------------------------------------------------------------------------------

#                       PSI FUNCTIONS

#%%---------------------------------------------------------------------------------------------------------------------------------------------
#Psi integration

def psiM_int(phi_func, zeta_in, z0, dz, npts=100):
    
    zeta_in[(zeta_in<-30)] = -30
    zeta_in[(zeta_in>10)] = 10
    
    psi_val = np.zeros((len(zeta_in)))
    
    L = (dz/2)/zeta_in
    zeta0 = z0/L
    zeta_fine = np.zeros((npts))
    
    for j in range(len(zeta_in)):
        for i in range(1,npts+1):
            zeta_fine[i-1] = zeta0[j] + (zeta_in[j]-zeta0[j])*(i-1)/(npts-1)
        vertical_profile = np.zeros_like(zeta_fine)
        vertical_profile = phi_func(zeta_fine)

        for i in range(1,npts):
            psi_val[j] = psi_val[j] + 0.5*(zeta_fine[i] - zeta_fine[i-1])*((1-vertical_profile[i])/zeta_fine[i] + (1-vertical_profile[i-1])/zeta_fine[i-1])
    a = 0.33
    b = 0.41
    yy = b**(-3)                   
    xx = (yy / a)**(1/3)           
    psi_val[(abs(zeta_in)>0.41**(-3))] = (np.log(a + yy) - 3*b*(yy**(1/3)) + ((b*(a**(1/3)))/2) * np.log(((1 + xx)**2) / (1 - xx + xx**2)) 
        + np.sqrt(3)*b*(a**(1/3))*np.arctan(((2*xx) - 1) / np.sqrt(3)) - np.log(a) + (np.sqrt(3)*b*(a**(1/3))*np.pi/6))
    psi_val = np.minimum(psi_val,(np.log(a + yy) - 3*b*(yy**(1/3)) + ((b*(a**(1/3)))/2) * np.log(((1 + xx)**2) / (1 - xx + xx**2)) 
        + np.sqrt(3)*b*(a**(1/3))*np.arctan(((2*xx) - 1) / np.sqrt(3)) - np.log(a) + (np.sqrt(3)*b*(a**(1/3))*np.pi/6)))
    return psi_val

def psiH_int(phi_func, zeta_in, z0, dz, npts=100):
    
    zeta_in[(zeta_in<-30)] = -30
    zeta_in[(zeta_in>10)] = 10
    
    psi_val = np.zeros((len(zeta_in)))
    
    L = (dz/2)/zeta_in
    zeta0 = z0/L
    zeta_fine = np.zeros((npts))
    
    for j in range(len(zeta_in)):
        for i in range(1,npts+1):
            zeta_fine[i-1] = zeta0[j] + (zeta_in[j]-zeta0[j])*(i-1)/(npts-1)
        vertical_profile = np.zeros_like(zeta_fine)
        vertical_profile = phi_func(zeta_fine)

        for i in range(1,npts):
            psi_val[j] = psi_val[j] + 0.5*(zeta_fine[i] - zeta_fine[i-1])*((1-vertical_profile[i])/zeta_fine[i] + (1-vertical_profile[i-1])/zeta_fine[i-1])

    return psi_val

#%%psi_M unstable
from matplotlib.colors import Normalize
norm_aniso = Normalize(vmin=0, vmax=0.3)

nbins = 10

zeta_u = abs(zoverL[(L<0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = psiM_u_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = psiM[(L<0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin-phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])

# ratio = np.where(abs(ratio)<10e-6,0,ratio)

# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-2):
#     tmp1 = psiM[(zoverL<-bins[i]) & (zoverL>-bins[i+1])]
#     tmp2 = classicSFC['psi_m'].flatten()[(zoverL_c<-bins[i]) & (zoverL_c>-bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = -np.logspace(-3,4,1000)
phi_ax = psiM_u_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = -np.logspace(-3, 4, 1000)
zeta_ax = -np.logspace(-3, 4, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]<-30,-30,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
yb_scale = (yb*np.sqrt(3)/2)/np.max(yb)
psim = np.zeros((len(zeta), len(yb)))
for j in range(len(yb)):
    a = np.where(yb[j] > 0.6, 0.012, 0.24 - 0.38 * yb[j])
    b = 0.061
    c = 0.45 - 0.53 * yb[j]
    n = -0.12 + 6.4 * yb[j]
    d = 0.48 + 1.8 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func(z):
        z_abs = np.abs(z)
        phi = (a + b * z_abs**n) / (a + z_abs**n) + c * z_abs**(1/3)
        phi[(z_abs>0.41**(-3))] = 1
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    psim[:, j] = psiM_int(phiM_func, zeta, 0.1/zi, dz, npts=100)
    
for j in range(len(yb)):
    ax_top.plot(-zeta_ax, psim[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L<0)]),psiM[(L<0)],s=1,c=yB_sfc[(L<0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
ax_top.invert_xaxis()
ax_top.axvline(14.513,c='k',ls='--')
ax_top.axhline(0,c='k',ls='--')
ax_top.set_ylabel(r"$\psi_M$",fontsize=15)
# cbar = plt.colorbar(p)
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\psi_a - \psi_r|>}{<|\psi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$-\zeta$", fontsize=15)
ax_bot.set_xscale('log')
# ax_bot.set_yscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(10000,0.001)
ax_bot.set_ylim(0,40)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/psiM_u.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%psiM stable

nbins = 10

zeta_u = abs(zoverL[(L>0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = psiM_s_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = psiM[(L>0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin-phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])
    
# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-1):
#     tmp1 = psiM[(zoverL>bins[i]) & (zoverL<bins[i+1])]
#     tmp2 = classicSFC['psi_m'].flatten()[(zoverL_c>bins[i]) & (zoverL_c<bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = np.logspace(-3,4,1000)
phi_ax = psiM_s_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = np.logspace(-3, 5, 1000)
zeta_ax = np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]>10,10,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
psim = np.zeros((len(zeta), len(yb)))

for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func(z):
        z_abs = np.abs(z)
        phi = (a + b * z)
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    psim[:, j] = psiM_int(phiM_func, zeta, 0.1/zi, dz, npts=100)
    
for j in range(len(yb)):
    ax_top.plot(zeta_ax, psim[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L>0)]),psiM[(L>0)],s=1,c=yB_sfc[(L>0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
# ax_top.set_yscale('log')
ax_top.axhline(0,c='k',ls='--')
ax_top.set_ylabel(r"$\psi_M$",fontsize=15)
# cbar = plt.colorbar(p)
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\psi_a - \psi_r|>}{<|\psi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$\zeta$", fontsize=15)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(0.001,10000)
ax_bot.set_ylim(0,40)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/psiM_s.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%psi_H unstable

nbins = 10

zeta_u = abs(zoverL[(L<0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = psiH_u_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = psiH[(L<0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin-phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])
    
# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-2):
#     tmp1 = psiH[(zoverL<-bins[i]) & (zoverL>-bins[i+1])]
#     tmp2 = classicSFC['psi_h'].flatten()[(zoverL_c<-bins[i]) & (zoverL_c>-bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = -np.logspace(-3,4,1000)
phi_ax = psiH_u_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = -np.logspace(-3, 4, 1000)
zeta_ax = -np.logspace(-3, 4, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]<-30,-30,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
psih = np.zeros((len(zeta), len(yb)))
for j in range(len(yb)):
    a = np.where(yb[j] > 0.6, 0.012, 0.24 - 0.38 * yb[j])
    b = 0.061
    c = 0.45 - 0.53 * yb[j]
    n = -0.12 + 6.4 * yb[j]
    d = 0.48 + 1.8 * yb[j]

    # Define phi functions for this yb[j]
    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    psih[:, j] = psiH_int(phiH_func, zeta, 0.01/zi, dz, npts=100)
    
for j in range(len(yb)):
    ax_top.plot(-zeta_ax, psih[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L<0)]),psiH[(L<0)],s=1,c=yB_sfc[(L<0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
ax_top.invert_xaxis()
ax_top.axhline(0,c='k',ls='--')
ax_top.set_ylabel(r"$\psi_H$",fontsize=15)
# cbar = plt.colorbar(p)
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\psi_a - \psi_r|>}{<|\psi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$-\zeta$", fontsize=15)
ax_bot.set_xscale('log')
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(10000,0.001)
ax_bot.set_ylim(0,40)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/psiH_u.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%psiH stable

nbins = 10

zeta_u = abs(zoverL[(L>0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = psiH_s_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = psiH[(L>0)][mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(abs(phi_bin-phi_ref_bin))
    med_ref[i] = np.median(abs(phi_ref_bin))
    
    ratio[i] = (mean_diff[i]/med_ref[i])

# bins = np.logspace(np.log10(0.0001),np.log10(100000),nbins)
# bin_centers = np.sqrt(bins[:-1]*bins[1:])

# ratio = np.zeros((nbins))
# med_ref = np.zeros((nbins))
# mean_diff = np.zeros((nbins))
# for i in range(nbins-1):
#     tmp1 = psiM[(zoverL>bins[i]) & (zoverL<bins[i+1])]
#     tmp2 = classicSFC['psi_m'].flatten()[(zoverL_c>bins[i]) & (zoverL_c<bins[i+1])]
    
#     mean_diff[i] = np.median(tmp1)-np.median(tmp2)
#     med_ref[i] = np.median(tmp2)
    
#     ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ref = np.logspace(-3,4,1000)
phi_ax = psiH_s_ref(zeta_ref)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,5), tight_layout=True,
    gridspec_kw={'height_ratios':[2,1]}   # bigger top panel
)

zeta = np.logspace(-3, 5, 1000)
zeta_ax = np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]>10,10,zeta[i])
yb = np.linspace(0.1, 0.3, 6)
psih = np.zeros((len(zeta), len(yb)))

for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    # Define phi functions for this yb[j]
    def phiH_func(z):
        phi = (c + d * z)
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    psih[:, j] = psiH_int(phiH_func, zeta, 0.01/zi, dz, npts=100)
    
for j in range(len(yb)):
    ax_top.plot(zeta_ax, psih[:, j], c=cmap(yb_scale[j]))

p = ax_top.scatter(abs(zoverL[(L>0)]),psiH[(L>0)],s=1,c=yB_sfc[(L>0)],cmap=ColorAnisotropy(),norm=norm_aniso)
# ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=20, c='lime')
ax_top.plot(abs(zeta_ref), phi_ax, c='k')
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_top.axvline(bins[i], c='k', ls=':')
ax_top.set_xscale('log')
ax_top.axhline(1,c='k',ls='--')
ax_top.set_ylabel(r"$\psi_H$",fontsize=15)
# cbar = plt.colorbar(p)
ax_bot.scatter(bin_centers, (ratio[:-1]*100), s=20, c='b')
ax_bot.set_ylabel(r"$\frac{<|\psi_a - \psi_r|>}{<|\psi_r|>}$ [%]", fontsize=15)
ax_bot.set_xlabel(r"$\zeta$", fontsize=15)
ax_bot.set_xscale('log')
# ax_bot.set_yscale('log')
ax_bot.set_ylim(-100,100)
ax_bot.axhline(0,c='k',ls='--')
for i in range(nbins):
    ax_bot.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.tick_params(axis='y',which='major',labelsize=11)
ax_bot.tick_params(axis='both',which='major',labelsize=11)
ax_bot.set_xlim(0.001,10000)
ax_bot.set_ylim(0,40)
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/AGU_2025/psiH_s.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()














































# %%
