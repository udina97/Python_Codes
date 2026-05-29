#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 02:46:03 2023

@author: u1450851
"""

import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_mom

from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 256
ny = 256
nz = 256

lx = 2*np.pi
ly = 2*np.pi
lz = 1

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.58
Tscale = 290
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%
sim = 'hom_eq_pfx'
path = '/uufs/chpc.utah.edu/common/home/calaf-group3/Giulia_research3/miscellaneous/' + sim
checkpnt = read_checkpoint_mom(path+'/output_checkpoint/',[2100000],nx,ny,nz)
# checkpnt_sfc = read_checkpoint_sfc(path+'/output_checkpoint/',[100000],nx,ny)


#%%

fig,axs = plt.subplots(1,1)

axs.plot(np.mean(checkpnt['u'][:,:,:-1],axis=(0,1))*uscale,z_uvp)

plt.show()

#%%Plot the surface scaling relations

phiM = checkpnt_sfc['phi_m']
phiH = checkpnt_sfc['phi_h']
psiM = checkpnt_sfc['psi_m']
psiH = checkpnt_sfc['psi_h']


from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,4,tight_layout=True,figsize=(10,5))

kde = gaussian_kde((phiM[:,:]).flatten())
x_pdf = np.linspace(min((phiM[:,:]).flatten()), max((phiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(phiM[:,:]).T,cmap='bwr',vmin=0,vmax=10)
axs[1,0].hist((phiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((phiH[:,:]).flatten())
x_pdf = np.linspace(min((phiH[:,:]).flatten()), max((phiH[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(phiH[:,:]).T,cmap='bwr',vmin=0,vmax=10)
axs[1,1].hist((phiH[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiM[:,:]).flatten())
x_pdf = np.linspace(min((psiM[:,:]).flatten()), max((psiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,(psiM[:,:]).T,cmap='bwr',vmin=-20,vmax=2)
axs[1,2].hist((psiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiH[:,:]).flatten())
x_pdf = np.linspace(min((psiH[:,:]).flatten()), max((psiH[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,3].pcolormesh(x,y,(psiH[:,:]).T,cmap='bwr',vmin=-20,vmax=5)
axs[1,3].hist((psiH[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,3].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,3].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$\phi_M$', fontsize=14)
axs[1,1].set_xlabel(r'$\phi_H$', fontsize=14)
axs[1,2].set_xlabel(r'$\psi_M$', fontsize=14)
axs[1,3].set_xlabel(r'$\psi_H$', fontsize=14)

axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Filter flux-gradietn scalings based on stability

# L = -(checkpnt_sfc['ustar']**3*checkpnt['SC'][:,:,0])/(0.4*(9.8*zi/uscale**2)*checkpnt_sfc['sfcFLUX'])
L = checkpnt_sfc['L']
zoverL = (dz/2)/L
zoverL_u = zoverL[(zoverL<0)]
zoverL_s = zoverL[(zoverL>0)]
phiM_u = phiM[(zoverL<0)]
phiH_u = phiH[(zoverL<0)]
phiM_s = phiM[(zoverL>0)]
phiH_s = phiH[(zoverL>0)]
psiM_u = psiM[(zoverL<0)]
psiH_u = psiH[(zoverL<0)]
psiM_s = psiM[(zoverL>0)]
psiH_s = psiH[(zoverL>0)]

#%%Plot pcolor and pdf of zeta

fig,axs = plt.subplots(2,1,tight_layout=True,figsize=(6,8))

kde = gaussian_kde((zoverL[:,:]).flatten())
x_pdf = np.linspace(min((zoverL[:,:]).flatten()), max((zoverL[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0].pcolormesh(x,y,(zoverL[:,:]).T,cmap='bwr',vmin=-10,vmax=10)
axs[1].hist((zoverL[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0].set_xlabel(r'$x/z_i$',fontsize=14)
axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[1].set_xlabel(r"$\zeta$",fontsize=14)

plt.show()

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.scatter(zoverL_s,zoverL_s,s=1,c='k')
axs.set_xscale('log')
axs.set_yscale('log')
axs.grid(True, which='both', linestyle='--', alpha=0.5)

plt.show()

#%%Mosso unstable

zeta = -np.logspace(-3, 3, 1000)
zeta_ax = -np.logspace(-3, 3, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]<-30,-30,zeta[i])
yb = np.linspace(0.1, 0.4, 4)

# Allocate arrays
phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))

# ---- Refined mesh integration function ----
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

    # zeta_fine = np.linspace(zeta0, zeta_in, npts)
    vertical_profile = np.zeros_like(zeta_fine)

    for k in range(0, npts):
        # phi_val = phi_func(zeta_fine[k])
        # vertical_profile[k] = (1.0 - phi_val) / zeta_fine[k]
        vertical_profile[k] = phi_func(zeta_fine[k])

    # Trapezoidal integration
    # psi_val = np.sum(0.5*(zeta_fine[1:] - zeta_fine[:-1]) *
    #                        (vertical_profile[1:] + vertical_profile[:-1]))
    psi_val = 0
    for i in range(1,npts):
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
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta)
    phih[:, j] = phiH_func(zeta)

    for i in range(len(zeta)):
        psim[i, j] = fit_phi_get_psi_refined(phiM_func, zeta[i], 0.1/zi, dz, npts=100)
        psih[i, j] = fit_phi_get_psi_refined(phiH_func, zeta[i], 0.01/zi, dz, npts=100)
        
# phiM_B = ((1 + 0.6*(zeta**2))/(1 - 7.5*(-zeta)))**(1/3)

# === Plot ===
fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)

# φ plots
for j in range(len(yb)):
    axs[0, 0].plot(-zeta_ax, phim[:, j], c=cmap(yb[j]))
    axs[0, 1].plot(-zeta_ax, phih[:, j], c=cmap(yb[j]))

axs[0, 0].scatter(abs(zoverL_u),phiM_u,s=1,c='k')
axs[1, 0].scatter(abs(zoverL_u),psiM_u,s=1,c='k')
# axs[0, 0].plot(zeta,phiM_B,c='k')
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[0, 1].set_xscale('log')
axs[0, 1].set_yscale('log')
axs[0, 0].invert_xaxis()
axs[0, 1].invert_xaxis()
axs[0, 0].set_ylabel(r"$\phi_m(\zeta)$")
axs[0, 1].set_ylabel(r"$\phi_h(\zeta)$")
axs[0, 0].axhline(1,c='k',ls='--')
axs[0, 1].axhline(1,c='k',ls='--')
axs[0, 0].axvline(14.513,c='k',ls='--')

# ψ plots
for j in range(len(yb)):
    axs[1, 0].plot(-zeta_ax, psim[:, j], c=cmap(yb[j]))
    axs[1, 1].plot(-zeta_ax, psih[:, j], c=cmap(yb[j]))

axs[0, 1].scatter(abs(zoverL_u),phiH_u,s=1,c='k')
axs[1, 1].scatter(abs(zoverL_u),psiH_u,s=1,c='k')
# axs[1, 0].set_xlim(0, 1)
# axs[1, 1].set_xlim(0, 1)
# axs[1, 0].set_ylim(-5, 5)
# axs[1, 1].set_ylim(-5, 5)
axs[1, 0].set_xscale('log')
axs[1, 1].set_xscale('log')
axs[1, 0].invert_xaxis()
axs[1, 1].invert_xaxis()
axs[1, 0].set_ylabel(r"$\psi_m(\zeta)$")
axs[1, 1].set_ylabel(r"$\psi_h(\zeta)$")
axs[1, 0].axhline(0,c='k',ls='--')
axs[1, 1].axhline(0,c='k',ls='--')
axs[1, 0].axhline(1.8,c='k',ls='--')
axs[1, 0].axvline(14.513,c='k',ls='--')

for ax in axs.flat:
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\zeta$')

plt.show()

#%%Stable flux gradient

zeta = np.logspace(-3, 5, 1000)
zeta_ax = np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]>10,10,zeta[i])
yb = np.linspace(0.1, 0.4, 5)

phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))
    
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

    # zeta_fine = np.linspace(zeta0, zeta_in, npts)
    vertical_profile = np.zeros_like(zeta_fine)

    for k in range(0, npts):
        # phi_val = phi_func(zeta_fine[k])
        # vertical_profile[k] = (1.0 - phi_val) / zeta_fine[k]
        vertical_profile[k] = phi_func(zeta_fine[k])

    # Trapezoidal integration
    # psi_val = np.sum(0.5*(zeta_fine[1:] - zeta_fine[:-1]) *
    #                        (vertical_profile[1:] + vertical_profile[:-1]))
    psi_val = 0
    for i in range(1,npts):
        psi_val = psi_val + 0.5*(zeta_fine[i] - zeta_fine[i-1])*((1-vertical_profile[i])/zeta_fine[i] + (1-vertical_profile[i-1])/zeta_fine[i-1])
        
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
    phim[:, j] = phiM_func(zeta)
    phih[:, j] = phiH_func(zeta)

    for i in range(len(zeta)):
        psim[i, j] = fit_phi_get_psi_refined(phiM_func, zeta[i], 0.1/zi, dz, npts=100)
        psih[i, j] = fit_phi_get_psi_refined(phiH_func, zeta[i], 0.01/zi, dz, npts=100)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

# φM
for j in range(len(yb)):
    axs[0, 0].plot(zeta_ax, phim[:, j], c=cmap(yb[j]))
axs[0, 0].scatter(zoverL_s,phiM_s,s=1,c='k')
axs[0, 0].set_xscale("log")
axs[0, 0].set_yscale("log")
axs[0, 0].set_xlabel(r'$\zeta$')
axs[0, 0].set_ylabel(r"$\phi_m(\zeta)$")
axs[0, 0].grid(True, which="both", ls="--", lw=0.5)
axs[0, 0].axhline(1,c='k',ls='--')

# φH
for j in range(len(yb)):
    axs[0, 1].plot(zeta_ax, phih[:, j], c=cmap(yb[j]))
axs[0, 1].scatter(zoverL_s,phiH_s,s=1,c='k')
axs[0, 1].set_xscale("log")
axs[0, 1].set_yscale("log")
axs[0, 1].set_xlabel(r'$\zeta$')
axs[0, 1].set_ylabel(r"$\phi_h(\zeta)$")
axs[0, 1].grid(True, which="both", ls="--", lw=0.5)
axs[0, 1].axhline(1,c='k',ls='--')

# ψM
for j in range(len(yb)):
    axs[1, 0].plot(zeta_ax, psim[:, j], c=cmap(yb[j]))
axs[1, 0].scatter(zoverL_s,psiM_s,s=1,c='k')
axs[1, 0].set_xlabel(r'$\zeta$')
axs[1, 0].set_ylabel(r"$\psi_m(\zeta)$")
axs[1, 0].set_xscale("log")
# axs[1, 0].set_xlim(0, 5)
# axs[1, 0].set_ylim(-20, 5)075790
axs[1, 0].grid(True, ls="--", lw=0.5)
axs[1, 0].axhline(0,c='k',ls='--')

# ψH
for j in range(len(yb)):
    axs[1, 1].plot(zeta_ax, psih[:, j], c=cmap(yb[j]))
axs[1, 1].scatter(zoverL_s,psiH_s,s=1,c='k')
axs[1, 1].set_xlabel(r'$\zeta$')
axs[1, 1].set_ylabel(r"$\psi_h(\zeta)$")
axs[1, 1].set_xscale("log")
# axs[1, 1].set_xlim(0, 5)
# axs[1, 1].set_ylim(-20, 5)
axs[1, 1].grid(True, ls="--", lw=0.5)
axs[1, 1].axhline(0,c='k',ls='--')

plt.show()


#%%Plot histogram and pdf of u,v,w values at different heights

from scipy.stats import gaussian_kde

u = checkpnt['u']
v = checkpnt['v']
w = checkpnt['w']

zlevel = 2

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((u[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((u[:,:,zlevel]).flatten()), max((u[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(u[:,:,zlevel]).T,cmap='bwr')
axs[1,0].hist((u[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((v[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((v[:,:,zlevel]).flatten()), max((v[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(v[:,:,zlevel]).T,cmap='bwr')
axs[1,1].hist((v[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((w[:,:,zlevel]).flatten())
x_pdf = np.linspace(min((w[:,:,zlevel]).flatten()), max((w[:,:,zlevel]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,(w[:,:,zlevel]).T,cmap='bwr')
axs[1,2].hist((w[:,:,zlevel]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(f'u at {zlevel*dz*zi + dz*zi/2 :.02f}m', fontsize=14)
axs[1,1].set_xlabel(f'v at {zlevel*dz*zi + dz*zi/2 :.02f}m', fontsize=14)
axs[1,2].set_xlabel(f'w at {zlevel*dz*zi :.02f}m', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(2):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Plot pdf for ustar and heatflux at the surface

ustar3D = checkpnt_sfc['ustar']
    
heatflux3D = checkpnt_sfc['sfcFLUX']

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((ustar3D[:,:]).flatten())
x_pdf = np.linspace(min((ustar3D[:,:]).flatten()), max((ustar3D[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,ustar3D[:,:].T,cmap='bwr',vmin=0,vmax=1)
axs[1,0].hist((ustar3D[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((heatflux3D[:,:]).flatten())
x_pdf = np.linspace(min((heatflux3D[:,:]).flatten()), max((heatflux3D[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,heatflux3D[:,:].T,cmap='bwr',vmin=-0.001,vmax=0.001)
axs[1,1].hist((heatflux3D[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$u_*$ at sfc', fontsize=14)
axs[1,1].set_xlabel(r'$\overline{wT}$ at sfc', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,6)
axs[1,1].set_xlim(-0.001,0.005)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Plot of ustar and heat flux at the surface as a function of stability

fig,axs = plt.subplots(2,2,tight_layout=True,sharey='row',sharex='col')

axs[0,0].scatter(-zoverL_u,ustar3D[(zoverL<0)],s=1,c='k')
axs[1,0].scatter(-zoverL_u,heatflux3D[(zoverL<0)],s=1,c='k')
axs[0,1].scatter(zoverL_s,ustar3D[(zoverL>0)],s=1,c='k')
axs[1,1].scatter(zoverL_s,heatflux3D[(zoverL>0)],s=1,c='k')

ax = axs.flatten()
for i in range(len(ax)):
    ax[i].set_xscale('log')
    
axs[0,0].invert_xaxis()
axs[0,0].set_xlim(1e5,1e-3)
axs[0,1].set_xlim(1e-3,1e5)
axs[1,0].set_xlabel(r"$-\zeta$",fontsize=14)
axs[1,1].set_xlabel(r"$\zeta$",fontsize=14)
axs[0,0].set_ylabel(r"$u_*$",fontsize=14)
axs[1,0].set_ylabel(r"$\overline{w'T'}$",fontsize=14)

fig.suptitle(sim,fontsize=14)

plt.show()


#%%Plot surface temperature

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,checkpnt_sfc['sfcVAL'].T,cmap='hot_r')
cbar = plt.colorbar(p)
axs.set_xlabel(r'$x/z_i$', fontsize=14)
axs.set_ylabel(r'$y/z_i$', fontsize=14)
axs.set_title('Sfc T', fontsize=14)

plt.show()

#%%Compute sgs tke

tke_sgs = checkpnt['txx_new'] + checkpnt['tyy_new'] + checkpnt['tzz_new']

#%% pcolor test plots

var = 'SC'
tmp = copy.deepcopy(checkpnt[var][:,:,0:nz])

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 40 #int(nz/2)
zslice = 10

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')

# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
axs.set_title(var)
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()


#%%Compute Reynolds stresses using the on-the-fly averages

uu = checkpnt['uu_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = checkpnt['vv_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = checkpnt['ww_new'][:,:,:nz] - checkpnt['w_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]#+ checkpnt['tzz_new'][:,:,:nz]
uv = checkpnt['uv_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = checkpnt['uw_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]- checkpnt['txz_new'][:,:,:nz]
vw = checkpnt['vw_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]- checkpnt['tyz_new'][:,:,:nz]

tke = uu + vv + ww

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

#%%
ustar = (uw**2 + vw**2)**(1/4)
log_yb = np.log10(yB)
a_yb_u = 0.784*log_yb**0 - 2.582*log_yb
uu_u = (np.sqrt(uu[:,:,0])/ustar[:,:,0])[(zoverL<0)]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.scatter(abs(zoverL_u),uu_u,s=1,c='k')
# for i in range(len(yb)):
#     axs.plot(zeta,sc25_u[:,i],c=cmap(yb[i]))
axs.set_xscale('log')
axs.set_xlabel(r"$\zeta$",fontsize=14)
axs.set_ylabel(r"$\phi_M$",fontsize=14)
axs.set_ylim(0,10)
axs.set_xlim(10e-3,10e3)
axs.invert_xaxis()

plt.show()

#%%
for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nz):
            if tke[i,j,k]<0:
                print(i,j,k)
                
#%%

fig,axs = plt.subplots(1,1)

axs.plot(np.mean(checkpnt['w_new'][:,:,:nz],axis=(0,1)),np.arange(0,nz))
# axs.plot(np.mean(checkpnt['tzz_new'][:,:,:-1],axis=(0,1)),np.arange(0,nz))

#%%Plot the on-the-fly REynolds Stresses

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 40 #int(nz/2)
zslice = 10

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

plt1 = axs.pcolormesh(x_ax,z_ax,ww[:,yslice,:].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,z_ax,uu[:,:,zslice].T,cmap= 'coolwarm')

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%% PLot Anisotropy

# X, Y = np.meshgrid(x_ax,z_ax)
yslice = 30 #int(nz/2)
zslice = 40

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

plt1 = axs.pcolormesh(x,z_uvp,yB[:,yslice,:].T,cmap= ColorAnisotropy())
# plt1 = axs.pcolormesh(x,z_uvp,yB[:,:,zslice].T,cmap= ColorAnisotropy())

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()














































