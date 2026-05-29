#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 11:36:16 2025

@author: u1450851
"""

#%% Import modules

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy
cmap = ColorAnisotropy()

#%%Mosso stable

zeta = np.linspace(1e-3, 1e2, 500)
yb = np.linspace(0.1, 0.6, 10)

phiM = np.zeros((len(zeta), len(yb)))
phiH = np.zeros((len(zeta), len(yb)))
psiM = np.zeros((len(zeta), len(yb)))
psiH = np.zeros((len(zeta), len(yb)))
    
def fit_phi_get_psi_refined(phi_func, zeta_in, npts=100):

    zeta_fine = np.linspace(0, zeta_in, npts)
    vertical_profile = np.zeros_like(zeta_fine)

    for k in range(1, npts):
        phi_val = phi_func(zeta_fine[k])
        vertical_profile[k] = (1.0 - phi_val) / zeta_fine[k]

    # Trapezoidal integration
    psi_val = 0.5 * np.sum((zeta_fine[1:] - zeta_fine[:-1]) *
                           (vertical_profile[1:] + vertical_profile[:-1]))
    return psi_val

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

    def phiH_func(z):
        phi = (c + d * z)
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phiM[:, j] = phiM_func(zeta)
    phiH[:, j] = phiH_func(zeta)

    for i in range(len(zeta)):
        psiM[i, j] = fit_phi_get_psi_refined(phiM_func, zeta[i], npts=100)
        psiH[i, j] = fit_phi_get_psi_refined(phiH_func, zeta[i], npts=100)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

# φM
for j in range(len(yb)):
    axs[0, 0].plot(zeta, phiM[:, j], c=cmap(yb[j]))
axs[0, 0].set_xscale("log")
axs[0, 0].set_yscale("log")
axs[0, 0].set_xlabel(r'$\zeta$')
axs[0, 0].set_ylabel(r"$\phi_m(\zeta)$")
axs[0, 0].grid(True, which="both", ls="--", lw=0.5)
axs[0, 0].axhline(1,c='k',ls='--')

# φH
for j in range(len(yb)):
    axs[0, 1].plot(zeta, phiH[:, j], c=cmap(yb[j]))
axs[0, 1].set_xscale("log")
axs[0, 1].set_yscale("log")
axs[0, 1].set_xlabel(r'$\zeta$')
axs[0, 1].set_ylabel(r"$\phi_h(\zeta)$")
axs[0, 1].grid(True, which="both", ls="--", lw=0.5)
axs[0, 1].axhline(1,c='k',ls='--')

# ψM
for j in range(len(yb)):
    axs[1, 0].plot(zeta, psiM[:, j], c=cmap(yb[j]))
axs[1, 0].set_xlabel(r'$\zeta$')
axs[1, 0].set_ylabel(r"$\psi_m(\zeta)$")
axs[1, 0].set_xscale("log")
# axs[1, 0].set_xlim(0, 5)
# axs[1, 0].set_ylim(-20, 5)075790
axs[1, 0].grid(True, ls="--", lw=0.5)
axs[1, 0].axhline(0,c='k',ls='--')

# ψH
for j in range(len(yb)):
    axs[1, 1].plot(zeta, psiH[:, j], c=cmap(yb[j]))
axs[1, 1].set_xlabel(r'$\zeta$')
axs[1, 1].set_ylabel(r"$\psi_h(\zeta)$")
axs[1, 1].set_xscale("log")
# axs[1, 1].set_xlim(0, 5)
# axs[1, 1].set_ylim(-20, 5)
axs[1, 1].grid(True, ls="--", lw=0.5)
axs[1, 1].axhline(0,c='k',ls='--')

plt.show()

#%%Mosso unstable

zeta = -np.logspace(-3, 2, 500)
yb = np.linspace(0.1, 0.6, 6)
nz = 256

# Allocate arrays
phiM = np.zeros((len(zeta), len(yb)))
phiH = np.zeros((len(zeta), len(yb)))
psiM = np.zeros((len(zeta), len(yb)))
psiH = np.zeros((len(zeta), len(yb)))

def fit_phi_get_psi_refined(phi_func, zeta_in, z0, dz, npts=100):
    
    # if zeta_in < -30:
    #     zeta_in = -30
    # elif zeta_in > 10:
    #     zeta_in = 10
    
    L = (dz/2)/zeta_in
    zeta0 = z0/L
    zeta_fine = np.zeros((npts))
    
    for i in range(1,npts+1):
        zeta_fine[i-1] = zeta0 + (zeta_in-zeta0)*(i-1)/(npts-1)

    vertical_profile = np.zeros_like(zeta_fine)

    for k in range(0, npts):
        vertical_profile[k] = phi_func(zeta_fine[k])

    psi_val = 0
    for i in range(1,npts):
        if i == 1:
            psi_val = psi_val + 0.5*(zeta_fine[i])*((1-vertical_profile[i])/zeta_fine[i])
        else:
            psi_val = psi_val + 0.5*(zeta_fine[i] - zeta_fine[i-1])*((1-vertical_profile[i])/zeta_fine[i] + (1-vertical_profile[i-1])/zeta_fine[i-1])
    
    return psi_val

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
        # phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        # phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phiM[:, j] = phiM_func(zeta)
    phiH[:, j] = phiH_func(zeta)

    for i in range(len(zeta)):
        psiM[i, j] = fit_phi_get_psi_refined(phiM_func, zeta[i], 0., 2000/nz, npts=100)
        psiH[i, j] = fit_phi_get_psi_refined(phiH_func, zeta[i], 0., 2000/nz, npts=100)

def phiM_B_U(z):
    a = 0.33
    b = 0.41
    ycrit = b**(-3)
    phi = np.where(z>ycrit,1,(a + b * (z**(4.0/3.0))) / (a + z))
    return phi

def phiH_B_U(z):
    c = 0.33
    d = 0.057
    n = 0.78
    phi = (c + d * (z**n)) / (c + (z**n))
    return phi

# Constants
a = 0.33
b = 0.41
c = 0.33
d = 0.057
n = 0.78

# Range of y (zeta)
y = np.logspace(-3, 2, 500)  # 10^-3 to 10^2
ycrit = b**(-3)

phim = phiM_B_U(y)
phih = phiH_B_U(y)

psim = np.zeros_like(y)
# Precompute psi_zero (same expression used in both cases)
psi_zero = -np.log(a) + (np.sqrt(3.0) * b * (a**(1.0/3.0)) * (np.pi / 6.0))

# Case 1: y > b^(-3)
mask1 = y > ycrit
yy = np.full_like(y[mask1], ycrit)
xx = (yy / a)**(1.0 / 3.0)
psim[mask1] = (np.log(a + yy) - 3.0 * b * (yy**(1.0/3.0)) + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0)) + psi_zero)

# Case 2: y <= b^(-3)
mask2 = ~mask1
yy = y[mask2]
xx = (yy / a)**(1.0 / 3.0)
psim[mask2] = (np.log(a + yy) - 3.0 * b * (yy**(1.0/3.0)) + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0)) + psi_zero)
    
psih = ((1 - d) / n) * np.log((c + (y**n)) / c)

# === Plot ===
fig, axs = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True, sharex='col')

# φ plots
for j in range(len(yb)):
    axs[0, 0].plot(-zeta, phiM[:, j], c=cmap(yb[j]))
    axs[0, 1].plot(-zeta, phiH[:, j], c=cmap(yb[j]))

axs[0, 0].plot(y,phim,c='k',linewidth=2.5)
axs[0, 1].plot(y,phih,c='k',linewidth=2.5)
# axs[0, 0].set_yscale('log')
# axs[0, 1].set_yscale('log')
axs[0, 0].set_ylabel(r"$\phi_m(\zeta,y_B)$",fontsize=15)
axs[0, 1].set_ylabel(r"$\phi_h(\zeta,y_B)$",fontsize=15)
# axs[0, 0].axhline(1,c='k',ls='--')
# axs[0, 1].axhline(1,c='k',ls='--')
# axs[0, 0].axvline(14.513,c='k',ls='--')

# ψ plots
for j in range(len(yb)):
    axs[1, 0].plot(-zeta, psiM[:, j], c=cmap(yb[j]))
    axs[1, 1].plot(-zeta, psiH[:, j], c=cmap(yb[j]))

axs[1, 0].plot(y,psim,c='k',linewidth=2.5)
axs[1, 1].plot(y,psih,c='k',linewidth=2.5)
axs[1, 0].set_xscale('log')
axs[1, 1].set_xscale('log')
# axs[1, 0].set_yscale('log')
# axs[1, 1].set_yscale('log')
axs[1, 0].invert_xaxis()
axs[1, 1].invert_xaxis()
axs[1, 0].set_ylabel(r"$\psi_m(\zeta,y_B)$",fontsize=15)
axs[1, 1].set_ylabel(r"$\psi_h(\zeta,y_B)$",fontsize=15)
axs[1, 0].set_xlabel(r'$-\zeta$',fontsize=15)
axs[1, 1].set_xlabel(r'$-\zeta$',fontsize=15)
axs[1, 0].set_ylim(-0.5,4)
axs[1, 1].set_ylim(-10,6)
# axs[1, 0].axhline(0,c='k',ls='--')
# axs[1, 1].axhline(0,c='k',ls='--')
# axs[1, 0].axhline(1.8,c='k',ls='--')
# axs[1, 0].axvline(14.513,c='k',ls='--')

for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=12)

plt.show()


#%%Numerical integration of Brutaseart Unstable functions

def fit_phi_get_psi_refined(phi_func, zeta_in, z0, dz, npts=100):
    
    # if zeta_in < -30:
    #     zeta_in = -30
    # elif zeta_in > 10:
    #     zeta_in = 10
    
    L = (dz/2)/zeta_in
    zeta0 = z0/L
    zeta_fine = np.zeros((npts))
    
    for i in range(1,npts+1):
        zeta_fine[i-1] = zeta0 + (zeta_in-zeta0)*(i-1)/(npts-1)

    vertical_profile = np.zeros_like(zeta_fine)

    for k in range(0, npts):
        vertical_profile[k] = phi_func(zeta_fine[k])

    psi_val = 0
    for i in range(1,npts):
        # if i == 1:
        #     psi_val = psi_val + 0.5*(zeta_fine[i])*((1-vertical_profile[i])/zeta_fine[i])
        # else:
            psi_val = psi_val + 0.5*(zeta_fine[i] - zeta_fine[i-1])*((1-vertical_profile[i])/zeta_fine[i] + (1-vertical_profile[i-1])/zeta_fine[i-1])
    
    return psi_val

def phiM_B_U(z):
    a = 0.33
    b = 0.41
    ycrit = b**(-3)
    phi = np.where(z>ycrit,1,(a + b * (z**(4.0/3.0))) / (a + z))
    return phi

def phiH_B_U(z):
    c = 0.33
    d = 0.057
    n = 0.78
    phi = (c + d * (z**n)) / (c + (z**n))
    return phi

# Constants
a = 0.33
b = 0.41
c = 0.33
d = 0.057
n = 0.78

# Range of y (zeta)
y = np.logspace(-3, 2, 500)  # 10^-3 to 10^2
ycrit = b**(-3)

phim = phiM_B_U(y)
phih = phiH_B_U(y)

psim_v1 = np.zeros_like(y)
psim_v2 = np.zeros_like(y)
psim_v3 = np.zeros_like(y)
psih_v1 = np.zeros_like(y)
psih_v2 = np.zeros_like(y)
psih_v3 = np.zeros_like(y) 

for i in range(len(y)):
    psim_v1[i] = fit_phi_get_psi_refined(phiM_B_U, y[i], 0.01, 2000/128,2)
    psim_v2[i] = fit_phi_get_psi_refined(phiM_B_U, y[i], 0.01, 2000/128,11)
    psim_v3[i] = fit_phi_get_psi_refined(phiM_B_U, y[i], 0.01, 2000/128,101)
    psih_v1[i] = fit_phi_get_psi_refined(phiH_B_U, y[i], 0.01, 2000/128,2)
    psih_v2[i] = fit_phi_get_psi_refined(phiH_B_U, y[i], 0.01, 2000/128,11)
    psih_v3[i] = fit_phi_get_psi_refined(phiH_B_U, y[i], 0.01, 2000/128,101)

psim = np.zeros_like(y)
# Precompute psi_zero (same expression used in both cases)
psi_zero = -np.log(a) + (np.sqrt(3.0) * b * (a**(1.0/3.0)) * (np.pi / 6.0))

# Case 1: y > b^(-3)
mask1 = y > ycrit
yy = np.full_like(y[mask1], ycrit)
xx = (yy / a)**(1.0 / 3.0)
psim[mask1] = (np.log(a + yy) - 3.0 * b * (yy**(1.0/3.0)) + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0)) + psi_zero)

# Case 2: y <= b^(-3)
mask2 = ~mask1
yy = y[mask2]
xx = (yy / a)**(1.0 / 3.0)
psim[mask2] = (np.log(a + yy) - 3.0 * b * (yy**(1.0/3.0)) + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0)) + psi_zero)
    
psih = ((1 - d) / n) * np.log((c + (y**n)) / c)

# === Plot results ===
fig,axs = plt.subplots(2,2,figsize=(10, 6),sharex='col',tight_layout=True)

axs[0,0].semilogx(y, phim, label=r'$\phi_m(\zeta)$',c='k')
axs[0,0].set_ylabel(r'$\phi_m(\zeta)$',fontsize=15)

axs[1,0].loglog(y, psim, label=r'Analytical', color='k',linewidth=2.5)
axs[1,0].loglog(y, psim_v1, color='g', label='N=1',linewidth=2.5)
axs[1,0].loglog(y, psim_v2, color='b',label='N=10',linewidth=2.5)
axs[1,0].loglog(y, psim_v3, color='r',label='N=100',ls=':',linewidth=2.5)
axs[1,0].set_ylabel(r'$\psi_m(\zeta)$',fontsize=15)
axs[1,0].legend()
axs[1,0].invert_xaxis()
axs[1,0].set_xlabel(r'$-\zeta$',fontsize=15)

axs[0,1].semilogx(y, phih, label=r'$\phi_h$',c='k')
axs[0,1].set_ylabel(r'$\phi_h(\zeta)$',fontsize=15)

axs[1,1].loglog(y, psih, label=r'Analytical', color='k',linewidth=2.5)
axs[1,1].loglog(y, psih_v1,c='g',label='N=1',linewidth=2.5)
axs[1,1].loglog(y, psih_v2,c='b',label='N=10',linewidth=2.5)
axs[1,1].loglog(y, psih_v3, color='r',label='N=100',ls=':',linewidth=2.5)
axs[1,1].set_xlabel(r'$-\zeta$',fontsize=15)
axs[1,1].set_ylabel(r'$\psi_h(\zeta)$',fontsize=15)
axs[1,1].legend()
axs[1,1].invert_xaxis()

for ax in axs.flat:
    ax.grid(True, which='major')
    ax.tick_params(labelsize=12)

plt.show()


# %% Plot Brutsaert Unstable and Hogstrom Stable functions for Phi and Psi

# Range of y (zeta)
y = np.logspace(-4, 2, 500)

fig,axs = plt.subplots(2,4,figsize=(12,4),tight_layout=True,sharex='col')

#Phi_M unstable------------------------------------------------------------------------------------
# Constants
a = 0.33
b = 0.41
c = 0.33
d = 0.057
n = 0.78

# Initialize arrays
phim = np.zeros_like(y)

# Threshold value
ycrit = b**(-3)

# Case 1: y > b^(-3)
mask1 = y > ycrit
yy = np.full_like(y[mask1], ycrit)
xx = (yy / a)**(1.0 / 3.0)
phim[mask1] = 1.0

# Case 2: y <= b^(-3)
mask2 = ~mask1
yy = y[mask2]
xx = (yy / a)**(1.0 / 3.0)
phim[mask2] = (a + b * (yy**(4.0/3.0))) / (a + yy)

axs[0,0].semilogx(y, phim, label=r'$\phi_m$', color='k')
axs[0,0].invert_xaxis()
axs[0,0].set_ylabel(r'$\phi_M$',fontsize=15)
# axs[0,0].set_yticks(np.linspace(0, 1.5, 7))

#Phi_H unstable--------------------------------------------------------------------------------------
phih = (c + d * (y**n)) / (c + (y**n))
axs[0,2].semilogx(y, phih, label=r'$\phi_h$', color='k')
axs[0,2].invert_xaxis()
axs[0,2].set_ylabel(r'$\phi_H$',fontsize=15)

#Phi_M stable----------------------------------------------------------------------------------------
phim = 1 + 5.3*y
axs[0,1].semilogx(y, phim, label=r'$\phi_m$', color='k')
axs[0,1].set_yscale('log')

#Phi_H stable----------------------------------------------------------------------------------------
phih = 1 + 8*y
axs[0,3].semilogx(y, phih, label=r'$\phi_h$', color='k')
axs[0,3].set_yscale('log')

#Psi_M unstable--------------------------------------------------------------------------------------
psim = np.zeros_like(y)
# Precompute psi_zero (same expression used in both cases)
psi_zero = -np.log(a) + (np.sqrt(3.0) * b * (a**(1.0/3.0)) * (np.pi / 6.0))
yy = np.full_like(y[mask1], ycrit)
xx = (yy / a)**(1.0 / 3.0)
psim[mask1] = (
    np.log(a + yy)
    - 3.0 * b * (yy**(1.0/3.0))
    + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0))
    + psi_zero
)
yy = y[mask2]
xx = (yy / a)**(1.0 / 3.0)
psim[mask2] = (
    np.log(a + yy)
    - 3.0 * b * (yy**(1.0/3.0))
    + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0))
    + psi_zero
)
axs[1,0].semilogx(y, psim, label=r'$\psi_m$', color='k')
axs[1,0].set_ylabel(r'$\psi_M$',fontsize=15)
axs[1,0].set_xlabel(r'$-\zeta$',fontsize=15)

#Psi_H unstable-----------------------------------------------------------------------------------------
psih = ((1 - d) / n) * np.log((c + (y**n)) / c)
axs[1,2].semilogx(y, psih, label=r'$\psi_h$', color='k')
axs[1,2].set_ylabel(r'$\psi_H$',fontsize=15)
axs[1,2].set_xlabel(r'$-\zeta$',fontsize=15)

#Psi_M stable--------------------------------------------------------------------------------------------
psim = -5.3*y
axs[1,1].semilogx(y, psim, label=r'$\psi_m$', color='k')
axs[1,1].set_xlabel(r'$\zeta$',fontsize=15)

#Psi_H stable--------------------------------------------------------------------------------------------
psih = -8*y
axs[1,3].semilogx(y, psih, label=r'$\psi_h$', color='k')
axs[1,3].set_xlabel(r'$\zeta$',fontsize=15)

# axs[1,1].set_yscale('symlog', linthresh=1e-4)
# axs[1,3].set_yscale('symlog', linthresh=1e-4)

axs_flat = axs.flatten()
for i in range(len(axs_flat)):
    axs_flat[i].tick_params(labelsize=12)

plt.show()


#%%Plot the Mosso Phi functions with the Reference scalings for Unstable and Stable

zeta = -np.logspace(-5, 2, 1000)
zeta_ax = -np.logspace(-5, 2, 1000)
y = np.logspace(-5, 2, 500)
yb = np.linspace(0., 0.8, 9)

import matplotlib as mpl

cmap_trunc = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc_cmap',
    cmap(np.linspace(0., 0.8, 256))
)

sm = mpl.cm.ScalarMappable(cmap=cmap_trunc)
sm.set_array([])

# Allocate arrays
phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))

# === Plot ===
fig, axs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

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
        # phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        # phi = np.where(z<-0.41**-3,1,phi)
        return phi

    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        # phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta)
    phih[:, j] = phiH_func(zeta)
    
# φ plots
for j in range(len(yb)):
    axs[0].plot(-zeta_ax, phim[:, j], c=cmap(yb[j]))
    axs[2].plot(-zeta_ax, phih[:, j], c=cmap(yb[j]))

for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func(z):
        z_abs = np.abs(z)
        phi = (a + b * z)
        # phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    def phiH_func(z):
        phi = (c + d * z)
        # phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(abs(zeta))
    phih[:, j] = phiH_func(abs(zeta))

for j in range(len(yb)):
    axs[1].plot(abs(zeta_ax), phim[:, j], c=cmap(yb[j]))
    axs[3].plot(abs(zeta_ax), phih[:, j], c=cmap(yb[j]))

#----------------------------------------------------------------------------------
a = 0.33
b = 0.41
c = 0.33
d = 0.057
n = 0.78
phim = np.zeros_like(y)
ycrit = b**(-3)

mask1 = y > ycrit
yy = np.full_like(y[mask1], ycrit)
xx = (yy / a)**(1.0 / 3.0)
phim[mask1] = 1.0

mask2 = ~mask1
yy = y[mask2]
xx = (yy / a)**(1.0 / 3.0)
phim[mask2] = (a + b * (yy**(4.0/3.0))) / (a + yy)

axs[0].plot(y, phim, label=r'$\phi_m$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------
phih = np.zeros_like(y)
phih = (c + d * (y**n)) / (c + (y**n))
axs[2].plot(y, phih, label=r'$\phi_h$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------
phim = np.zeros_like(y)
phim = 1 + 5.3*y
axs[1].plot(y, phim, label=r'$\phi_m$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------
phih = np.zeros_like(y)
phih = 1 + 8*y
axs[3].plot(y, phih, label=r'$\phi_h$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------

axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[2].set_xscale('log')
axs[3].set_xscale('log')
axs[1].set_yscale('log')
axs[3].set_yscale('log')
axs[0].invert_xaxis()
axs[2].invert_xaxis()
axs[0].set_ylabel(r"$\phi_m(\zeta,y_B)$",fontsize=15)
axs[2].set_ylabel(r"$\phi_h(\zeta,y_B)$",fontsize=15)
# axs[0, 0].axhline(1,c='k',ls='--')
# axs[0, 1].axhline(1,c='k',ls='--')
# axs[0, 0].axvline(14.513,c='k',ls='--')

axs[0].set_xlabel(r'$-\zeta$',fontsize=15)
axs[1].set_xlabel(r'$\zeta$',fontsize=15)
axs[2].set_xlabel(r'$-\zeta$',fontsize=15)
axs[3].set_xlabel(r'$\zeta$',fontsize=15)

cbar = fig.colorbar(sm, ax=axs, location='right', fraction=0.02, pad=0.02)
cbar.set_label(r'$y_B$', fontsize=13)
ticks = (yb - 0.) / (0.8 - 0.)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{v:.1f}' for v in yb])

for ax in axs.flat:
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=12)

plt.show()

#%%Plotting the Mosso Phi functions and the integrated functions with Brutsaert and Hogstrom for all stability

zeta = -np.logspace(-4, 2, 500)
zeta_ax = -np.logspace(-4, 2, 500)
y = np.logspace(-4, 2, 500)
yb = np.linspace(0.1, 0.6, 6)

import matplotlib as mpl

cmap_trunc = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc_cmap',
    cmap(np.linspace(0.1, 0.6, 256))
)

sm = mpl.cm.ScalarMappable(cmap=cmap_trunc)
sm.set_array([])

# Allocate arrays
phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))

# === Plot ===
fig, axs = plt.subplots(2, 4, figsize=(12, 4), constrained_layout=True, sharex='col')

# PHI FUCNTIONS -------------------------------------------------------------------------------------------------------------------

for j in range(len(yb)):
    a = np.where(yb[j] > 0.6, 0.012, 0.24 - 0.38 * yb[j])
    b = 0.061
    c = 0.45 - 0.53 * yb[j]
    n = -0.12 + 6.4 * yb[j]
    d = 0.48 + 1.8 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func_U(z):
        
        z_abs = np.abs(z)
        phi = (a + b * z_abs**n) / (a + z_abs**n) + c * z_abs**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        phi = np.where(z<-0.41**-3,1,phi)
        return phi

    def phiH_func_U(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func_U(zeta)
    phih[:, j] = phiH_func_U(zeta)
    
# φ plots
for j in range(len(yb)):
    axs[0,0].plot(-zeta_ax, phim[:, j], c=cmap(yb[j]))
    axs[0,2].plot(-zeta_ax, phih[:, j], c=cmap(yb[j]))

for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func_S(z):
        z = np.where(z>10,10,z)
        phi = (a + b * z)
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    def phiH_func_S(z):
        z = np.where(z>10,10,z)
        phi = (c + d * z)
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func_S(abs(zeta))
    phih[:, j] = phiH_func_S(abs(zeta))

for j in range(len(yb)):
    axs[0,1].plot(abs(zeta_ax), phim[:, j], c=cmap(yb[j]))
    axs[0,3].plot(abs(zeta_ax), phih[:, j], c=cmap(yb[j]))

#----------------------------------------------------------------------------------
a = 0.33
b = 0.41
c = 0.33
d = 0.057
n = 0.78
phim = np.zeros_like(y)
ycrit = b**(-3)

mask1 = y > ycrit
yy = np.full_like(y[mask1], ycrit)
xx = (yy / a)**(1.0 / 3.0)
phim[mask1] = 1.0

mask2 = ~mask1
yy = y[mask2]
xx = (yy / a)**(1.0 / 3.0)
phim[mask2] = (a + b * (yy**(4.0/3.0))) / (a + yy)

axs[0,0].plot(y, phim, label=r'$\phi_m$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------
phih = np.zeros_like(y)
phih = (c + d * (y**n)) / (c + (y**n))
axs[0,2].plot(y, phih, label=r'$\phi_h$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------
phim = np.zeros_like(y)
phim = np.where(y>10,1 + 5.3*10,1 + 5.3*y)
axs[0,1].plot(y, phim, label=r'$\phi_m$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------
phih = np.zeros_like(y)
phih = np.where(y>10,1 + 8*10,1 + 8*y)
axs[0,3].plot(y, phih, label=r'$\phi_h$', color='k',linewidth=2.5)
#----------------------------------------------------------------------------------

# PSI FUNCTIONS -----------------------------------------------------------------------------------------------------------------------

def fit_phi_get_psi_refined(phi_func, zeta_in, z0, dz, npts=100):
    
    if zeta_in < -30:
        zeta_in = -30
    elif zeta_in > 10:
        zeta_in = 10
    
    L = (dz/2)/zeta_in
    zeta0 = z0/L
    zeta_fine = np.zeros((npts))
    
    for i in range(1,npts+1):
        zeta_fine[i-1] = zeta0 + (zeta_in-zeta0)*(i-1)/(npts-1)

    vertical_profile = np.zeros_like(zeta_fine)

    for k in range(0, npts):
        vertical_profile[k] = phi_func(zeta_fine[k])

    psi_val = 0
    for i in range(1,npts):
        if i == 1:
            psi_val = psi_val + 0.5*(zeta_fine[i])*((1-vertical_profile[i])/zeta_fine[i])
        else:
            psi_val = psi_val + 0.5*(zeta_fine[i] - zeta_fine[i-1])*((1-vertical_profile[i])/zeta_fine[i] + (1-vertical_profile[i-1])/zeta_fine[i-1])
    
    return psi_val

psiM = np.zeros((len(zeta), len(yb)))
psiH = np.zeros((len(zeta), len(yb)))

for j in range(len(yb)):
    a = np.where(yb[j] > 0.6, 0.012, 0.24 - 0.38 * yb[j])
    b = 0.061
    c = 0.45 - 0.53 * yb[j]
    n = -0.12 + 6.4 * yb[j]
    d = 0.48 + 1.8 * yb[j]

    # Evaluate phi and psi for all zeta
    phiM[:, j] = phiM_func(zeta)
    phiH[:, j] = phiH_func(zeta)

    for i in range(len(zeta)):
        psiM[i, j] = fit_phi_get_psi_refined(phiM_func_U, zeta[i], 0., 2000/nz, npts=100)
        yy = 0.41**-3
        xx = (yy/0.33)**(1/3)
        psi_zero = -np.log(0.33) + ((3**0.5)*0.41*(0.33**(1/3))*(np.pi/6))
        psiM[i, j] = np.minimum(psiM[i,j],np.log(0.33+yy) - (3*0.41*(yy**(1/3))) + (0.41*(0.33**(1/3)))/2*np.log((1+xx)**2/(1-xx+xx**2)) + 
                            (3**0.5*0.41*(0.33**(1/3)))*np.arctan((2*xx-1)/(3**0.5)) + psi_zero)
        psiM[i,j] = np.where(zeta[i]<-0.41**-3,np.log(0.33+yy) - (3*0.41*(yy**(1/3))) + (0.41*(0.33**(1/3)))/2*np.log((1+xx)**2/(1-xx+xx**2)) + 
                            (3**0.5*0.41*(0.33**(1/3)))*np.arctan((2*xx-1)/(3**0.5)) + psi_zero, psiM[i, j])
        psiH[i, j] = fit_phi_get_psi_refined(phiH_func_U, zeta[i], 0., 2000/nz, npts=100)
        
for j in range(len(yb)):
    axs[1, 0].plot(-zeta, psiM[:, j], c=cmap(yb[j]))
    axs[1, 2].plot(-zeta, psiH[:, j], c=cmap(yb[j]))

psiM = np.zeros((len(zeta), len(yb)))
psiH = np.zeros((len(zeta), len(yb)))

for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    for i in range(len(zeta)):
        psiM[i, j] = fit_phi_get_psi_refined(phiM_func_S, np.where(abs(zeta[i])>10,10,abs(zeta[i])), 0., 2000/nz, npts=100)
        psiH[i, j] = fit_phi_get_psi_refined(phiH_func_S, np.where(abs(zeta[i])>10,10,abs(zeta[i])), 0., 2000/nz,npts=100)
        
for j in range(len(yb)):
    axs[1, 1].plot(abs(zeta), psiM[:, j], c=cmap(yb[j]))
    axs[1, 3].plot(abs(zeta), psiH[:, j], c=cmap(yb[j]))

a = 0.33
b = 0.41
c = 0.33
d = 0.057
n = 0.78

psim = np.zeros_like(y)
# Precompute psi_zero (same expression used in both cases)
psi_zero = -np.log(a) + (np.sqrt(3.0) * b * (a**(1.0/3.0)) * (np.pi / 6.0))
yy = np.full_like(y[mask1], ycrit)
xx = (yy / a)**(1.0 / 3.0)
psim[mask1] = (
    np.log(a + yy)
    - 3.0 * b * (yy**(1.0/3.0))
    + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0))
    + psi_zero
)
yy = y[mask2]
xx = (yy / a)**(1.0 / 3.0)
psim[mask2] = (
    np.log(a + yy)
    - 3.0 * b * (yy**(1.0/3.0))
    + ((b * (a**(1.0/3.0))) / 2.0) * np.log(((1 + xx)**2) / (1 - xx + (xx**2)))
    + (np.sqrt(3.0) * b * (a**(1.0/3.0))) * np.arctan(((2.0 * xx) - 1.0) / np.sqrt(3.0))
    + psi_zero
)

axs[1,0].plot(y, psim, label=r'$\psi_m$', color='k',linewidth=2.5)

#Psi_H unstable-----------------------------------------------------------------------------------------
psih = np.where(y>30,((1 - d) / n) * np.log((c + (30**n)) / c),((1 - d) / n) * np.log((c + (y**n)) / c))
axs[1,2].plot(y, psih, label=r'$\psi_h$', color='k',linewidth=2.5)

#Psi_M stable--------------------------------------------------------------------------------------------
psim = np.where(y>10,-5.3*10,-5.3*y)
axs[1,1].plot(y, psim, label=r'$\psi_m$', color='k',linewidth=2.5)

#Psi_H stable--------------------------------------------------------------------------------------------
psih = np.where(y>10,-8*10,-8*y)
axs[1,3].plot(y, psih, label=r'$\psi_h$', color='k',linewidth=2.5)
    
#--------------------------------------------------------------------------------------------------------------------------------------

axs[1,0].set_xscale('log')
axs[1,1].set_xscale('log')
axs[1,2].set_xscale('log')
axs[1,3].set_xscale('log')
axs[0,1].set_yscale('log')
axs[0,3].set_yscale('log')
# axs[1,1].set_yscale('log')
# axs[1,3].set_yscale('log')
axs[0,0].invert_xaxis()
axs[0,2].invert_xaxis()
axs[0,0].set_ylabel(r"$\phi_m(\zeta,y_B)$",fontsize=15)
axs[0,2].set_ylabel(r"$\phi_h(\zeta,y_B)$",fontsize=15)
axs[1,0].set_ylabel(r"$\psi_m(\zeta,y_B)$",fontsize=15)
axs[1,2].set_ylabel(r"$\psi_h(\zeta,y_B)$",fontsize=15)
axs[0, 0].axhline(1,c='k',ls='--')
axs[0, 1].axhline(1,c='k',ls='--')
axs[0, 2].axhline(1,c='k',ls='--')
axs[0, 3].axhline(1,c='k',ls='--')
axs[0, 0].axvline(14.513,c='k',ls='--')
axs[0, 1].axvline(10,c='k',ls='--')
axs[0, 3].axvline(10,c='k',ls='--')
axs[1, 0].axhline(1.79993,c='k',ls='--')
axs[1, 0].axhline(0,c='k',ls='--')
axs[1, 1].axhline(0,c='k',ls='--')
axs[1, 3].axhline(0,c='k',ls='--')
axs[1, 0].axvline(14.513,c='k',ls='--')
axs[1, 1].axvline(10,c='k',ls='--')
axs[1, 3].axvline(10,c='k',ls='--')
axs[1, 2].axvline(30,c='k',ls='--')
axs[1, 2].axhline(0,c='k',ls='--')

axs[1,0].set_xlabel(r'$-\zeta$',fontsize=15)
axs[1,1].set_xlabel(r'$\zeta$',fontsize=15)
axs[1,2].set_xlabel(r'$-\zeta$',fontsize=15)
axs[1,3].set_xlabel(r'$\zeta$',fontsize=15)

cbar = fig.colorbar(sm, ax=axs, location='right', fraction=0.02, pad=0.02)
cbar.set_label(r'$y_B$', fontsize=13)
ticks = (yb - 0.1) / (0.6 - 0.1)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{v:.1f}' for v in yb])

for ax in axs.flat:
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=12)

plt.show()















































