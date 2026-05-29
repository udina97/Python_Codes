#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 12:30:54 2025

@author: u1450851
"""

#%%

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
aniso = 'patch128_aniso_1ms_v2'
classic = 'patch128_classic_1ms_v2'
# aniso = 'homo_unstable_aniso_9ms'
# classic = 'homo_unstable_classic_9ms'

step = 100000

aniso3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/output_checkpoint/',[step],nx,ny,nz)
anisoSFC = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+aniso+'/output_checkpoint/',[step],nx,ny)
classic3D = read_checkpoint_aniso('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/output_checkpoint/',[step],nx,ny,nz)
classicSFC = read_checkpoint_sfc_L('/scratch/general/nfs1/u1450851/LES_Sims/'+classic+'/output_checkpoint/',[step],nx,ny)

#%%Compute ustar and heatflux PDFs

ustar_a = anisoSFC['ustar'].flatten()
wT_a = anisoSFC['sfcFLUX'].flatten()

ustar_c = classicSFC['ustar'].flatten()
wT_c = classicSFC['sfcFLUX'].flatten()

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

kde = gaussian_kde(ustar_a)
x_pdf = np.linspace(min(ustar_a), max(ustar_a), 1000)
pdf_a = kde(x_pdf)

kde = gaussian_kde(ustar_c)
x_pdf = np.linspace(min(ustar_c), max(ustar_c), 1000)
pdf_c = kde(x_pdf)

axs[0].plot(x_pdf, pdf_a, 'r-', label=r"$\psi(\zeta,y_B)$")
axs[0].plot(x_pdf, pdf_c, 'g-', label=r"$\psi(\zeta)$")

kde = gaussian_kde(wT_a)
x_pdf = np.linspace(min(wT_a), max(wT_a), 1000)
pdf_a = kde(x_pdf)

kde = gaussian_kde(wT_c)
x_pdf = np.linspace(min(wT_c), max(wT_c), 1000)
pdf_c = kde(x_pdf)

axs[1].plot(x_pdf, pdf_a, 'r-', label=r"$\psi(\zeta,y_B)$")
axs[1].plot(x_pdf, pdf_c, 'g-', label=r"$\psi(\zeta)$")

axs[0].set_xlabel(r'$u_*$ at sfc', fontsize=14)
axs[1].set_xlabel(r'$\overline{wT}$ at sfc', fontsize=14)
axs[0].set_ylabel(r'$PDF$',fontsize=14)
axs[1].set_ylabel(r'$PDF$',fontsize=14)
axs[1].set_xticks([0,0.001,0.002,0.003,0.004])
axs[0].set_xlim(0)
for i in range(len(axs)):
    axs[i].set_ylim(0)
    axs[i].legend()

plt.show()

#%%PDF of the velocity and temoerature gradient at the surface

dudz = (anisoSFC['ustar']*uscale).flatten()/(0.4*0.5*dz*zi)*anisoSFC['phi_m'].flatten()
dTdz = (anisoSFC['sfcFLUX']*uscale*Tscale).flatten()/(0.4*0.5*dz*zi*(anisoSFC['ustar']*uscale).flatten())*anisoSFC['phi_h'].flatten()

dudz_c = (classicSFC['ustar']*uscale).flatten()/(0.4*0.5*dz*zi)*classicSFC['phi_m'].flatten()
dTdz_c = (classicSFC['sfcFLUX']*uscale*Tscale).flatten()/(0.4*0.5*dz*zi*(classicSFC['ustar']*uscale).flatten())*classicSFC['phi_h'].flatten()

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

kde = gaussian_kde(dudz)
x_pdf = np.linspace(min(dudz), max(dudz), 1000)
pdf_a = kde(x_pdf)

kde = gaussian_kde(dudz_c)
x_pdf = np.linspace(min(dudz_c), max(dudz_c), 1000)
pdf_c = kde(x_pdf)

axs[0].plot(x_pdf, pdf_a, 'r-', label=r"$\psi(\zeta,y_B)$")
axs[0].plot(x_pdf, pdf_c, 'g-', label=r"$\psi(\zeta)$")

kde = gaussian_kde(dTdz)
x_pdf = np.linspace(min(dTdz), max(dTdz), 1000)
pdf_a = kde(x_pdf)

kde = gaussian_kde(dTdz_c)
x_pdf = np.linspace(min(dTdz_c), max(dTdz_c), 1000)
pdf_c = kde(x_pdf)

axs[1].plot(x_pdf, pdf_a, 'r-', label=r"$\psi(\zeta,y_B)$")
axs[1].plot(x_pdf, pdf_c, 'g-', label=r"$\psi(\zeta)$")

axs[0].set_xlabel(r'$\frac{\partial u}{\partial z}$ at sfc', fontsize=14)
axs[1].set_xlabel(r'$\frac{\partial \theta}{\partial z}$ at sfc', fontsize=14)
axs[0].set_ylabel(r'$PDF$',fontsize=14)
axs[1].set_ylabel(r'$PDF$',fontsize=14)
# axs[1].set_xticks([0,0.001,0.002,0.003,0.004])
# axs[0].set_xlim(0)
for i in range(len(axs)):
    axs[i].set_ylim(0)
    axs[i].legend()

plt.show()

#%%Domain averaged profiles of u,v,w,T

fig,axs = plt.subplots(1,4,tight_layout=True,sharey=True)

axs[0].plot(np.mean(aniso3D['u'][:,:,:-1],axis=(0,1)),z_uvp,c='k',label='Aniso')
axs[1].plot(np.mean(aniso3D['v'][:,:,:-1],axis=(0,1)),z_uvp,c='k',label='Aniso')
axs[2].plot(np.mean(aniso3D['w'][:,:,:-1],axis=(0,1)),z_w,c='k',label='Aniso')
axs[3].plot(np.mean(aniso3D['SC'][:,:,:-1],axis=(0,1)),z_uvp,c='k',label='Aniso')

axs[0].plot(np.mean(classic3D['u'][:,:,:-1],axis=(0,1)),z_uvp,c='r',label='Ref')
axs[1].plot(np.mean(classic3D['v'][:,:,:-1],axis=(0,1)),z_uvp,c='r',label='Ref')
axs[2].plot(np.mean(classic3D['w'][:,:,:-1],axis=(0,1)),z_w,c='r',label='Ref')
axs[3].plot(np.mean(classic3D['SC'][:,:,:-1],axis=(0,1)),z_uvp,c='r',label='Ref')

axs[0].set_ylim(0,z_uvp[-1])
axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
axs[0].set_xlabel(r"$u/u_s$",fontsize=14)
axs[1].set_xlabel(r"$v/u_s$",fontsize=14)
axs[2].set_xlabel(r"$z/u_s$",fontsize=14)
axs[3].set_xlabel(r"$\theta/\theta_s$",fontsize=14)

axs[3].legend()

plt.show()

#%%Reference scaling functions

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
        if abs(zeta[i])>30:
            phi_ref[i] = (0.33 + 0.057*30**0.78)/(0.33 + 30**0.78)
        else:
            phi_ref[i] = (0.33 + 0.057*abs(zeta[i])**0.78)/(0.33 + abs(zeta[i])**0.78)
            
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
            zeta[i] = -30
    psi_ref = ((1-0.057)/0.78)*np.log((0.33 + abs(zeta)**(0.78))/0.33)
            
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

#%%Psi integration

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

#%%

import matplotlib.ticker as mticker

L = anisoSFC['L'].flatten()
zoverL = (dz/2)/L
zoverL_u = zoverL[(zoverL<0)]
zoverL_s = zoverL[(zoverL>0)]

phiM_u = (anisoSFC['phi_m'].flatten())[(zoverL<0)]
phiH_u = (anisoSFC['phi_h'].flatten())[(zoverL<0)]
phiM_s = (anisoSFC['phi_m'].flatten())[(zoverL>0)]
phiH_s = (anisoSFC['phi_h'].flatten())[(zoverL>0)]
psiM_u = (anisoSFC['psi_m'].flatten())[(zoverL<0)]
psiH_u = (anisoSFC['psi_h'].flatten())[(zoverL<0)]
psiM_s = (anisoSFC['psi_m'].flatten())[(zoverL>0)]
psiH_s = (anisoSFC['psi_h'].flatten())[(zoverL>0)]

zeta = -np.logspace(-3, 4, 1000)
zeta_ax = -np.logspace(-3, 4, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]<-30,-30,zeta[i])
yb = np.linspace(0.1, 0.4, 4)

# Allocate arrays
phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))

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

    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta)
    phih[:, j] = phiH_func(zeta)

    psim[:, j] = psiM_int(phiM_func, zeta, 0.1/zi, dz, npts=100)
    psih[:, j] = psiH_int(phiH_func, zeta, 0.01/zi, dz, npts=100)

# === Plot ===
fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True,sharex=True)

# φ plots
for j in range(len(yb)):
    axs[0, 0].plot(-zeta_ax, phim[:, j], c=cmap(yb[j]))
    axs[0, 1].plot(-zeta_ax, phih[:, j], c=cmap(yb[j]))

# axs[0, 0].plot(-zeta_ax,phiM_u_ref(zeta_ax),c='g',linewidth=2)
# axs[0, 1].plot(-zeta_ax,phiH_u_ref(zeta_ax),c='g',linewidth=2)

axs[0, 0].scatter(abs(zoverL_u),phiM_u,s=1,c='k',alpha=0.2)
axs[1, 0].scatter(abs(zoverL_u),psiM_u,s=1,c='k',alpha=0.2)
axs[0, 0].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['phi_m'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')
axs[0, 1].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['phi_h'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[0, 1].set_xscale('log')
axs[0, 1].set_yscale('log')
axs[0, 0].set_ylabel(r"$\phi_m(\zeta,y_B)$",fontsize=20)
axs[0, 1].set_ylabel(r"$\phi_h(\zeta,y_B)$",fontsize=20)
axs[0, 0].axhline(1,c='k',ls='--')
axs[0, 1].axhline(1,c='k',ls='--')
axs[0, 0].axvline(14.513,c='k',ls='--')

# ψ plots
for j in range(len(yb)):
    axs[1, 0].plot(-zeta_ax, psim[:, j], c=cmap(yb[j]))
    axs[1, 1].plot(-zeta_ax, psih[:, j], c=cmap(yb[j]))

axs[0, 1].scatter(abs(zoverL_u),phiH_u,s=1,c='k',alpha=0.2)
axs[1, 1].scatter(abs(zoverL_u),psiH_u,s=1,c='k',alpha=0.2)
axs[1, 0].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['psi_m'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')
axs[1, 1].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['psi_h'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')

# axs[1, 0].plot(-zeta_ax,psiM_u_ref(zeta_ax),c='g',linewidth=2)
# axs[1, 1].plot(-zeta_ax,psiH_u_ref(zeta_ax),c='g',linewidth=2)
axs[1, 0].set_xscale('log')
axs[1, 1].set_xscale('log')
axs[1, 0].invert_xaxis()
axs[1, 0].set_ylabel(r"$\psi_m(\zeta,y_B)$",fontsize=20)
axs[1, 1].set_ylabel(r"$\psi_h(\zeta,y_B)$",fontsize=20)
axs[1, 0].axhline(0,c='k',ls='--')
axs[1, 1].axhline(0,c='k',ls='--')
axs[1, 0].axhline(1.8,c='k',ls='--')
axs[1, 0].axvline(14.513,c='k',ls='--')

for ax in axs.flat:
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
axs[1, 0].set_xlabel(r'$-\zeta$',fontsize=20)
axs[1, 1].set_xlabel(r'$-\zeta$',fontsize=20)
axs[1, 0].tick_params(axis='both',which='major',labelsize=14)
axs[1, 1].tick_params(axis='both',which='major',labelsize=14)
axs[0, 0].tick_params(axis='y',which='major',labelsize=14)
axs[0, 0].set_yticks([1])
axs[0, 0].yaxis.set_major_formatter(mticker.LogFormatterMathtext())
axs[0, 0].yaxis.set_minor_formatter(mticker.NullFormatter())
axs[0, 1].tick_params(axis='y',which='major',labelsize=14)

plt.show()

#%%Stable 

zeta = np.logspace(-3, 5, 1000)
zeta_ax = np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]>10,10,zeta[i])
yb = np.linspace(0.1, 0.4, 4)

phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))

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

    psim[:, j] = psiM_int(phiM_func, zeta, 0.1/zi, dz, npts=100)
    psih[:, j] = psiH_int(phiH_func, zeta, 0.01/zi, dz, npts=100)

fig, axs = plt.subplots(2, 2,  figsize=(10, 8), tight_layout=True, sharex=True)

# φM
for j in range(len(yb)):
    axs[0, 0].plot(zeta_ax, phim[:, j], c=cmap(yb[j]))
axs[0, 0].scatter(zoverL_s,phiM_s,s=1,c='k',alpha=0.5)
axs[0, 0].plot(zeta_ax, phiM_s_ref(zeta_ax),c='g',linewidth=2)
axs[0, 0].set_xscale("log")
axs[0, 0].set_yscale("log")
axs[0, 0].set_ylabel(r"$\phi_m(\zeta,y_B)$",fontsize=20)
axs[0, 0].grid(True, which="both", ls="--", lw=0.5)
axs[0, 0].axhline(1,c='k',ls='--')

# φH
for j in range(len(yb)):
    axs[0, 1].plot(zeta_ax, phih[:, j], c=cmap(yb[j]))
axs[0, 1].scatter(zoverL_s,phiH_s,s=1,c='k',alpha=0.5)
axs[0, 1].plot(zeta_ax, phiH_s_ref(zeta_ax),c='g',linewidth=2)
axs[0, 1].set_xscale("log")
axs[0, 1].set_yscale("log")
axs[0, 1].set_ylabel(r"$\phi_h(\zeta,y_B)$",fontsize=20)
axs[0, 1].grid(True, which="both", ls="--", lw=0.5)
axs[0, 1].axhline(1,c='k',ls='--')

# ψM
for j in range(len(yb)):
    axs[1, 0].plot(zeta_ax, psim[:, j], c=cmap(yb[j]))
axs[1, 0].scatter(zoverL_s,psiM_s,s=1,c='k',alpha=0.5)
axs[1, 0].plot(zeta_ax, psiM_s_ref(zeta_ax),c='g',linewidth=2)
axs[1, 0].set_xlabel(r'$\zeta$')
axs[1, 0].set_ylabel(r"$\psi_m(\zeta,y_B)$",fontsize=20)
axs[1, 0].set_xscale("log")
axs[1, 0].grid(True, ls="--", lw=0.5)
axs[1, 0].axhline(0,c='k',ls='--')

# ψH
for j in range(len(yb)):
    axs[1, 1].plot(zeta_ax, psih[:, j], c=cmap(yb[j]))
axs[1, 1].scatter(zoverL_s,psiH_s,s=1,c='k',alpha=0.5)
axs[1, 1].plot(zeta_ax, psiH_s_ref(zeta_ax),c='g',linewidth=2)
axs[1, 1].set_xlabel(r'$\zeta$')
axs[1, 1].set_ylabel(r"$\psi_h(\zeta,y_B)$",fontsize=20)
axs[1, 1].set_xscale("log")
axs[1, 1].grid(True, ls="--", lw=0.5)
axs[1, 1].axhline(0,c='k',ls='--')

axs[1, 0].set_xlabel(r'$\zeta$',fontsize=20)
axs[1, 1].set_xlabel(r'$\zeta$',fontsize=20)
axs[1, 0].tick_params(axis='both',which='major',labelsize=14)
axs[1, 1].tick_params(axis='both',which='major',labelsize=14)
axs[0, 0].tick_params(axis='y',which='major',labelsize=14)
axs[0, 1].tick_params(axis='y',which='major',labelsize=14)

plt.show()

#%%

zeta_u = -np.logspace(-3, 5, 1000)
zeta_s = np.logspace(-3, 5, 1000)
yb = np.linspace(0.1, 0.4, 4)

fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True,sharex='col')

phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
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

    def phiH_func(z):
        phi = np.zeros_like(z)
        for i in range(len(z)):
            if z[i]<-30:
                phi[i] = d * ((3 - 2.5 * (-30)) / (1 - 10*(-30) + 50*(-30)**2))**(1/3)
            else:
                phi[i] = d * ((3 - 2.5 * z[i]) / (1 - 10*z[i] + 50*z[i]**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi
    
    phim[:, j] = phiM_func(zeta_u)
    phih[:, j] = phiH_func(zeta_u)


for j in range(len(yb)):
    axs[0, 0].plot(abs(zeta_u), phim[:, j], c=cmap(yb[j]))
    axs[1, 0].plot(abs(zeta_u), phih[:, j], c=cmap(yb[j]))

axs[0, 0].scatter(abs(zoverL_u),phiM_u,s=1,c='k',alpha=0.2)
axs[1, 0].scatter(abs(zoverL_u),phiH_u,s=1,c='k',alpha=0.2)
axs[0, 0].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['phi_m'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')
axs[1, 0].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['phi_h'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[1, 0].set_xscale('log')
axs[1, 0].set_yscale('log')
axs[0, 0].set_ylabel(r"$\phi_m(\zeta,y_B)$",fontsize=20)
axs[1, 0].set_ylabel(r"$\phi_h(\zeta,y_B)$",fontsize=20)
axs[0, 0].axhline(1,c='k',ls='--')
axs[1, 0].axhline(1,c='k',ls='--')
axs[0, 0].axvline(14.513,c='k',ls='--')
axs[0, 0].invert_xaxis()

phim = np.zeros((len(zeta), len(yb)))
phih = np.zeros((len(zeta), len(yb)))
for j in range(len(yb)):
    a = 0.76 + 1.5 * yb[j]
    b = 6.3 - 4.3 * yb[j]
    c = np.where(yb[j] > 0.6, 0.34, 1.9 - 2.6 * yb[j])
    d = 6.7 - 10.0 * yb[j]

    # Define phi functions for this yb[j]
    def phiM_func(z):
        phi = np.zeros_like(z)
        z_abs = np.abs(z)
        for i in range(len(z)):
            if z[i]>10:
                phi[i] = (a + b * 10)
            else:
                phi[i] = (a + b * z[i])
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    def phiH_func(z):
        phi = np.zeros_like(z)
        for i in range(len(z)):
            if z[i]>10:
                phi[i] = (c + d * 10)
            else:
                phi[i] = (c + d * z[i])
        phi = np.maximum(phi, 1.0)  # enforce phi >= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta_s)
    phih[:, j] = phiH_func(zeta_s)

for j in range(len(yb)):
    axs[0, 1].plot(abs(zeta_s), phim[:, j], c=cmap(yb[j]))
    axs[1, 1].plot(abs(zeta_s), phih[:, j], c=cmap(yb[j]))
    
axs[0, 1].scatter(abs(zoverL_s),phiM_s,s=1,c='k',alpha=0.2)
axs[1, 1].scatter(abs(zoverL_s),phiH_s,s=1,c='k',alpha=0.2)
axs[0, 1].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() > 0)]),classicSFC['phi_m'].flatten()[(classicSFC['L'].flatten() > 0)],s=2,c='g')
axs[1, 1].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() > 0)]),classicSFC['phi_h'].flatten()[(classicSFC['L'].flatten() > 0)],s=2,c='g')
axs[0, 1].set_xscale('log')
axs[0, 1].set_yscale('log')
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')
axs[0, 1].axhline(1,c='k',ls='--')
axs[1, 1].axhline(1,c='k',ls='--')
axs[0, 0].axvline(14.513,c='k',ls='--')

axs[1, 0].set_xlabel(r'$-\zeta$',fontsize=20)
axs[1, 1].set_xlabel(r'$\zeta$',fontsize=20)
axs[1, 0].tick_params(axis='both',which='major',labelsize=14)
axs[1, 1].tick_params(axis='both',which='major',labelsize=14)
axs[0, 0].tick_params(axis='y',which='major',labelsize=14)
axs[0, 1].tick_params(axis='y',which='major',labelsize=14)
axs[0, 0].set_yticks([1])
axs[0, 0].yaxis.set_major_formatter(mticker.LogFormatterMathtext())
axs[0, 0].yaxis.set_minor_formatter(mticker.NullFormatter())
axs[1, 0].set_xticks([0.01,1,100,10000])
axs[1, 0].xaxis.set_major_formatter(mticker.LogFormatterMathtext())
axs[1, 0].xaxis.set_minor_formatter(mticker.NullFormatter())
axs[1, 1].set_xticks([0.01,1,100,10000])
axs[1, 1].xaxis.set_major_formatter(mticker.LogFormatterMathtext())
axs[1, 1].xaxis.set_minor_formatter(mticker.NullFormatter())

axs_flat = axs.flatten()
for i in range(len(axs.flat)):
    axs_flat[i].grid(True, ls="--", lw=0.5)

plt.show()

#%%

import matplotlib.ticker as mticker

fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True,sharex='col')

zeta = -np.logspace(-3, 5, 1000)
zeta_ax = -np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]<-30,-30,zeta[i])
yb = np.linspace(0.1, 0.4, 4)
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))

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

    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    psim[:, j] = psiM_int(phiM_func, zeta, 0.1/zi, dz, npts=100)
    psih[:, j] = psiH_int(phiH_func, zeta, 0.01/zi, dz, npts=100)
    
for j in range(len(yb)):
    axs[0, 0].plot(-zeta_ax, psim[:, j], c=cmap(yb[j]))
    axs[1, 0].plot(-zeta_ax, psih[:, j], c=cmap(yb[j]))

axs[0, 0].scatter(abs(zoverL_u),psiM_u,s=1,c='k',alpha=0.2)
axs[1, 0].scatter(abs(zoverL_u),psiH_u,s=1,c='k',alpha=0.2)
axs[0, 0].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['psi_m'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')
axs[1, 0].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() < 0)]),classicSFC['psi_h'].flatten()[(classicSFC['L'].flatten() < 0)],s=2,c='g')
axs[0, 0].set_xscale('log')
axs[1, 0].set_xscale('log')
axs[0, 0].invert_xaxis()
axs[0, 0].set_ylabel(r"$\psi_m(\zeta,y_B)$",fontsize=20)
axs[1, 0].set_ylabel(r"$\psi_h(\zeta,y_B)$",fontsize=20)
axs[0, 0].axhline(0,c='k',ls='--')
axs[1, 0].axhline(0,c='k',ls='--')
axs[0, 0].axhline(1.8,c='k',ls='--')
axs[0, 0].axvline(14.513,c='k',ls='--')

zeta = np.logspace(-3, 5, 1000)
zeta_ax = np.logspace(-3, 5, 1000)
for i in range(len(zeta)):
    zeta[i] = np.where(zeta[i]>10,10,zeta[i])
yb = np.linspace(0.1, 0.4, 4)
psim = np.zeros((len(zeta), len(yb)))
psih = np.zeros((len(zeta), len(yb)))

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

    psim[:, j] = psiM_int(phiM_func, zeta, 0.1/zi, dz, npts=100)
    psih[:, j] = psiH_int(phiH_func, zeta, 0.01/zi, dz, npts=100)

for j in range(len(yb)):
    axs[0, 1].plot(zeta_ax, psim[:, j], c=cmap(yb[j]))
    axs[1, 1].plot(zeta_ax, psih[:, j], c=cmap(yb[j]))

axs[0, 1].scatter(abs(zoverL_s),psiM_s,s=1,c='k',alpha=0.2)
axs[1, 1].scatter(abs(zoverL_s),psiH_s,s=1,c='k',alpha=0.2)
axs[0, 1].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() > 0)]),classicSFC['psi_m'].flatten()[(classicSFC['L'].flatten() > 0)],s=2,c='g')
axs[1, 1].scatter(abs(((0.5*dz)/classicSFC['L'].flatten())[(classicSFC['L'].flatten() > 0)]),classicSFC['psi_h'].flatten()[(classicSFC['L'].flatten() > 0)],s=2,c='g')
axs[0, 1].set_xscale('log')
axs[1, 1].set_xscale('log')

for ax in axs.flat:
    ax.grid(True, linestyle='--', alpha=0.5)
    
axs[1, 0].set_xlabel(r'$-\zeta$',fontsize=20)
axs[1, 1].set_xlabel(r'$\zeta$',fontsize=20)
axs[1, 0].tick_params(axis='both',which='major',labelsize=14)
axs[1, 1].tick_params(axis='both',which='major',labelsize=14)
axs[0, 0].tick_params(axis='y',which='major',labelsize=14)
axs[0, 1].tick_params(axis='y',which='major',labelsize=14)
axs[1, 0].set_xticks([0.01,1,100,10000])
axs[1, 0].xaxis.set_major_formatter(mticker.LogFormatterMathtext())
axs[1, 0].xaxis.set_minor_formatter(mticker.NullFormatter())
axs[1, 1].set_xticks([0.01,1,100,10000])
axs[1, 1].xaxis.set_major_formatter(mticker.LogFormatterMathtext())
axs[1, 1].xaxis.set_minor_formatter(mticker.NullFormatter())

plt.show()




















































