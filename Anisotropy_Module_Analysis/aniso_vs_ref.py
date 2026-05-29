#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:16:20 2025

@author: u1450851
"""

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

#%%Load the data

sim = 'patch128_aniso_1ms'

data = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_checkpoint/',[110000],nx,ny)
data3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_checkpoint/',[110000],nx,ny,nz)

#%%

L = data['L'].flatten()
phiM = data['phi_m'].flatten()
phiH = data['phi_h'].flatten()
psiM = data['psi_m'].flatten()
psiH = data['psi_h'].flatten()

#%%

phiM_u = phiM[(L<0)]
phiM_s = phiM[(L>0)]
phiH_u = phiH[(L<0)]
phiH_s = phiH[(L>0)]
psiM_u = psiM[(L<0)]
psiM_s = psiM[(L>0)]
psiH_u = psiH[(L<0)]
psiH_s = psiH[(L>0)]

zeta = 0.5*dz/L

fig,axs = plt.subplots(1,1,tight_layout=True)
axs.scatter(abs(zeta[(L<0)]),psiH_u,s=2,c='k')
axs.set_xscale('log')
plt.show()

#%%

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
        if zeta[i]>30:
            zeta[i] = 30
    psi_ref = ((1-0.057)/0.78)*np.log((0.33 + abs(zeta)**(0.78))/0.33)
            
    return psi_ref

#%%

nbins = 10

zeta_u = abs(zeta[(L<0)])
zeta_min = np.min(zeta_u)
zeta_max = np.max(zeta_u)
bins = np.logspace(np.log10(zeta_min),np.log10(zeta_max),nbins)
bin_index = np.digitize(zeta_u, bins) -1
bin_centers = np.sqrt(bins[:-1]*bins[1:])

phi_ref = phiM_u_ref(zeta_u)

ratio = np.zeros((nbins))
med_ref = np.zeros((nbins))
mean_diff = np.zeros((nbins))
ratio = np.zeros((nbins))
for i in range(nbins):
    mask = bin_index == i
    if not np.any(mask):
        ratio[i] = np.nan
        continue
    phi_bin = phiM_u[mask]
    phi_ref_bin = phi_ref[mask]
    
    mean_diff[i] = np.median(phi_bin-phi_ref_bin)
    med_ref[i] = np.median(phi_ref_bin)
    
    ratio[i] = (mean_diff[i]/med_ref[i])

zeta_ax = np.logspace(-3,3,1000)
phi_ax = phiM_u_ref(zeta_ax)

#%%

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6,6), tight_layout=True,
    gridspec_kw={'height_ratios':[3,1]}   # bigger top panel
)

# ---- TOP PANEL ----
ax_top.plot(zeta_ax, phi_ax, c='k')
ax_top.scatter(zeta_u, phiM_u, s=2, c='g')
ax_top.scatter(bin_centers, (med_ref + mean_diff)[:-1], s=10, c='r')
# ax_top.scatter(bin_centers, med_ref[:-1], s=10, c='magenta')

# vertical bin lines
for i in range(nbins):
    ax_top.axvline(bins[i], c='k', ls=':')
    ax_bot.axvline(bins[i], c='k', ls=':')

ax_top.set_ylabel(r"$\phi$",fontsize=14)
ax_top.set_xscale('log')


# ---- BOTTOM PANEL (ratio) ----
ax_bot.scatter(bin_centers, ratio[:-1]*100, s=10, c='b')
ax_bot.set_ylabel(r"$\frac{<\phi_a - \phi_r>}{<\phi_r>}$", fontsize=14)
ax_bot.set_xlabel(r"$\zeta$", fontsize=14)
ax_bot.set_xscale('log')

# share x settings like inversion
ax_bot.invert_xaxis()

plt.show()

#%%compute the mean absotute difference between the values of phi and the corresponing value of phi for the same \zeta from classical scaling

diff_m = np.zeros_like(L)
diff_h = np.zeros_like(L)

for i in range(len(phiM)):
    if L[i]>0:
        zeta = 0.5*dz/L[i]
        if zeta>10:
            zeta = 10
        phi_m = 1 + 5.3*zeta
        phi_h = 1 + 8*zeta
        diff_m[i] = abs(phiM[i] - phi_m)
        diff_h[i] = abs(phiH[i] - phi_h)
    else:
        zeta = 0.5*dz/L[i]
        if zeta<-30:
            zeta = -30
        if (abs(zeta)>0.41**(-3)):
            phi_m = 1
        else:
            phi_m = (0.33 + 0.41*abs(zeta)**(4/3))/(0.33 + abs(zeta))
        phi_h = (0.33 + 0.057*abs(zeta)**0.78)/(0.33 + abs(zeta)**0.78)
        diff_m[i] = abs(phiM[i] - phi_m)
        diff_h[i] = abs(phiH[i] - phi_h)


print(f"the meadian absolute difference for momentum is: {np.median(diff_m)}")
print(f"the meadian absolute difference for heat is: {np.median(diff_h)}")



#%%

diff_m = np.zeros_like(L)
diff_h = np.zeros_like(L)

for i in range(len(psiM)):
    if L[i]>0:
        zeta = 0.5*dz/L[i]
        if zeta>10:
            zeta = 10
        psi_m = -5.3*zeta
        psi_h = -8*zeta
        diff_m[i] = abs(psiM[i] - psi_m)
        diff_h[i] = abs(psiH[i] - psi_h)
    else:
        zeta = 0.5*dz/L[i]
        a = 0.33
        b = 0.41
        if zeta<-30:
            zeta = -30
        if (abs(zeta)>0.41**(-3)):
            yy = b**(-3)                   
            xx = (yy / a)**(1/3)           
            psi_m = (np.log(a + yy) - 3*b*(yy**(1/3)) + ((b*(a**(1/3)))/2) * np.log(((1 + xx)**2) / (1 - xx + xx**2)) 
                + np.sqrt(3)*b*(a**(1/3))*np.arctan(((2*xx) - 1) / np.sqrt(3)) - np.log(a) + (np.sqrt(3)*b*(a**(1/3))*np.pi/6))
        else:
            xx = (abs(zeta) / a)**(1/3)
            yy = abs(zeta)
            psi_m = (np.log(a + yy) - 3*b*(yy**(1/3)) + ((b*(a**(1/3)))/2) * np.log(((1 + xx)**2) / (1 - xx + xx**2)) 
                + np.sqrt(3)*b*(a**(1/3))*np.arctan(((2*xx) - 1) / np.sqrt(3)) - np.log(a) + (np.sqrt(3)*b*(a**(1/3))*np.pi/6))
        psi_h = ((1-0.057)/0.78)*np.log((0.33 + abs(zeta)**(0.78))/0.33)
        diff_m[i] = abs(psiM[i] - psi_m)
        diff_h[i] = abs(psiH[i] - psi_h)


print(f"the meadian absolute difference for momentum is: {np.median(diff_m)}")
print(f"the meadian absolute difference for heat is: {np.median(diff_h)}")

#%%

L = data['L'].flatten()
u = np.median(data3D['u'][:,:,0].flatten())
psi = np.median(data['psi_m'].flatten())
log = np.log(0.5*dz/(0.1/zi))
ustar = np.median(data['ustar'].flatten())
deltaU = -u*0.4/((log-psi)**2)*0.2/ustar*100

























































