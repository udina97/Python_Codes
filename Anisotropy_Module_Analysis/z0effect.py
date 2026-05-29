#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:26:53 2026

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from scipy.stats import gaussian_kde

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso
from read_checkpoint_SC import read_checkpoint_sfc, read_checkpoint_sfc_L
from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering, Anisotropy
cmap = ColorAnisotropy()

#%%Simulation parameters

nx = 256
ny = 256
nz = 256

lx = 2*np.pi
ly = 2*np.pi
lz = 2

dx = lx/nx
dy = ly/ny
dz = lz/nz
dt = 0.1

zi = 1000
uscale = 0.4
Tscale = 290
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

Ug = 1
path = '/scratch/general/nfs1/u1450851/LES_Sims/'

#%%Load checkpoint data

check_step = 100000

sim = 'test_z0_256'

data3D = read_checkpoint_aniso(path+sim+'/output_checkpoint/',[check_step],nx,ny,nz)
data2D = read_checkpoint_sfc_L(path+sim+'/output_checkpoint/',[check_step],nx,ny)

#%%

from matplotlib import ticker as mticker

fig,ax = plt.subplots(figsize=(8,6),tight_layout=True)

diagnostics = np.genfromtxt(path+sim+'/running_diagnostics.txt')
ax.plot(diagnostics[:,0],diagnostics[:,4])

ax.set_title('Check $u_{*}$ for convergence')
ax.set_xlabel(r'Iteration Step')
ax.set_ylabel(r'$u_{*}$')
ax.set_xlim(0,diagnostics[-1,0])
# ax.set_ylim(0,0.0007)
# ax.legend()
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.show()

#%%Compute anisotropy

uu = data3D['uu_new'][:,:,:nz] - data3D['u_new'][:,:,:nz]*data3D['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = data3D['vv_new'][:,:,:nz] - data3D['v_new'][:,:,:nz]*data3D['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = data3D['ww_new'][:,:,:nz] - data3D['w_new'][:,:,:nz]*data3D['w_new'][:,:,:nz]#- checkpnt['tzz_new'][:,:,:nz]
uv = data3D['uv_new'][:,:,:nz] - data3D['u_new'][:,:,:nz]*data3D['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = data3D['uw_new'][:,:,:nz] - data3D['u_new'][:,:,:nz]*data3D['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = data3D['vw_new'][:,:,:nz] - data3D['v_new'][:,:,:nz]*data3D['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke = 0.5*(uu + vv + ww)

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

#%%Anisotropy tensor terms

b11 = uu/(2*tke) - 1/3
b22 = vv/(2*tke) - 1/3
b33 = ww/(2*tke) - 1/3
b12 = uv/(2*tke)
b13 = uw/(2*tke)
b23 = vw/(2*tke)

trace = b11+b22+b33

#%%PDF of yB  at the surface

fig,axs = plt.subplots(1,1,tight_layout=True)

x_pdf= np.linspace(min(yB[:,:,0].flatten()),max(yB[:,:,0].flatten()),1000)
# pdf = np.zeros((len(x_pdf)))
kde = gaussian_kde(yB[:,:,0].flatten())
pdf = kde(x_pdf)
axs.plot(x_pdf,(pdf),c='k',label='Classic')
# axs.fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.5)
axs.axvline(0.1,c='k',ls='--')
axs.set_xlabel(f"yB",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xlim(0,0.4)
axs.set_ylim(0,)
axs.set_title(f"{sim}",fontsize=12)
plt.show()

#%%Plot pdfs of the the reynolds stress components/anisotropy tensor components

# fig,axs = plt.subplots(1,1,tight_layout=True)

# x_pdf = np.linspace(min(uu[:,:,0].flatten()),max(uu[:,:,0].flatten()),1000)
# kde = gaussian_kde(uu[:,:,0].flatten())
# pdf = kde(x_pdf)
# axs.plot(x_pdf,pdf,label='uu')
# x_pdf = np.linspace(min(vv[:,:,0].flatten()),max(vv[:,:,0].flatten()),1000)
# kde = gaussian_kde(vv[:,:,0].flatten())
# pdf = kde(x_pdf)
# axs.plot(x_pdf,pdf,label='vv')
# x_pdf = np.linspace(min(ww[:,:,0].flatten()),max(ww[:,:,0].flatten()),1000)
# kde = gaussian_kde(ww[:,:,0].flatten())
# pdf = kde(x_pdf)
# axs.plot(x_pdf,pdf,label='ww')

# axs.set_xlabel(r"$\overline{u_{i}'u_{j}'}$", fontsize=14)
# axs.set_ylabel(r"$PDF$", fontsize=14)
# axs.legend()
# plt.show()

fig,axs = plt.subplots(1,1,tight_layout=True)

x_pdf = np.linspace(min(b11[:,:,0].flatten()),max(b11[:,:,0].flatten()),1000)
kde = gaussian_kde(b11[:,:,0].flatten())
pdf = kde(x_pdf)
axs.plot(x_pdf,pdf,label='b11')
x_pdf = np.linspace(min(b22[:,:,0].flatten()),max(b22[:,:,0].flatten()),1000)
kde = gaussian_kde(b22[:,:,0].flatten())
pdf = kde(x_pdf)
axs.plot(x_pdf,pdf,label='b22')
x_pdf = np.linspace(min(b33[:,:,0].flatten()),max(b33[:,:,0].flatten()),1000)
kde = gaussian_kde(b33[:,:,0].flatten())
pdf = kde(x_pdf)
axs.plot(x_pdf,pdf,label='b33')

x_pdf = np.linspace(min(lamba3[:,:,0].flatten()),max(lamba3[:,:,0].flatten()),1000)
kde = gaussian_kde(lamba3[:,:,0].flatten())
pdf = kde(x_pdf)
axs.plot(x_pdf,pdf,label='lambda')

axs.set_xlabel(r"$\overline{u_{i}'u_{j}'}/2*k - 1/3\delta_{i,j}$", fontsize=14)
axs.set_ylabel(r"$PDF$", fontsize=14)
axs.legend()
axs.set_title(f"{sim}",fontsize=12)
plt.show()

#%%Plot surface scaling functions

zeta = -np.logspace(-5, 3, 1000)
zeta_ax = -np.logspace(-5, 3, 1000)
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
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        phi = np.where(z<-0.41**-3,1,phi)
        return phi

    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta)
    phih[:, j] = phiH_func(zeta)

    for i in range(len(zeta)):
        psim[i, j] = fit_phi_get_psi_refined(phiM_func, zeta[i], 0.0/zi, dz, npts=100)
        yy = 0.41**-3
        xx = (yy/0.33)**(1/3)
        psi_zero = -np.log(0.33) + ((3**0.5)*0.41*(0.33**(1/3))*(np.pi/6))
        psim[i,j] = np.where(zeta[i]<-0.41**-3,np.log(0.33+yy) - (3*0.41*(yy**(1/3))) + (0.41*(0.33**(1/3)))/2*np.log((1+xx)**2/(1-xx+xx**2)) + 
                            (3**0.5*0.41*(0.33**(1/3)))*np.arctan((2*xx-1)/(3**0.5)) + psi_zero, psim[i, j])
        psih[i, j] = fit_phi_get_psi_refined(phiH_func, zeta[i], 0.0/zi, dz, npts=100)
        
# phiM_B = ((1 + 0.6*(zeta**2))/(1 - 7.5*(-zeta)))**(1/3)

# === Plot ===
fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)

# φ plots
for j in range(len(yb)):
    axs[0, 0].plot(-zeta_ax, phim[:, j], c=cmap(yb[j]))
    axs[0, 1].plot(-zeta_ax, phih[:, j], c=cmap(yb[j]))

axs[0, 0].scatter(abs(0.5*dz/data2D['L'][data2D['L']<0]).flatten(),data2D['phi_m'][data2D['L']<0].flatten(),s=1,c='k')
axs[1, 0].scatter(abs(0.5*dz/data2D['L'][data2D['L']<0]).flatten(),data2D['psi_m'][data2D['L']<0].flatten(),s=1,c='k')
# axs[1, 0].axhline(np.log(0.5*dz/(0.01/zi)),ls='--',c='k')
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

axs[0, 1].scatter(abs(0.5*dz/data2D['L'][data2D['L']<0]).flatten(),data2D['phi_h'][data2D['L']<0].flatten(),s=1,c='k')
axs[1, 1].scatter(abs(0.5*dz/data2D['L'][data2D['L']<0]).flatten(),data2D['psi_h'][data2D['L']<0].flatten(),s=1,c='k')
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
axs[1, 0].axhline(1.79993,c='k',ls='--')
axs[1, 0].axvline(14.513,c='k',ls='--')

for ax in axs.flat:
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$-\zeta$')

fig.suptitle(sim,fontsize=14)
plt.show()

#%%

yb = np.linspace(0.1,0.4,4)
zeta = -np.logspace(-7,3,1000)
dzeta = abs(zeta[1]-zeta[0])
tmp = np.zeros((len(yb),len(zeta)))
phim = np.zeros((len(zeta),len(yb)))
psi_int = np.zeros((len(zeta),len(yb)))
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
        phi = np.where(z>0.41**-3,1,phi)
        return phi

    def phiH_func(z):
        phi = d * ((3 - 2.5 * z) / (1 - 10*z + 50*z**2))**(1/3)
        phi = np.minimum(phi, 1.0)  # enforce phi <= 1.0
        return phi

    # Evaluate phi and psi for all zeta
    phim[:, j] = phiM_func(zeta)
    tmp[j,:] = (1-phim[:,j])/zeta

fig,axs = plt.subplots(1,1)
for i in range(len(yb)):
    axs.plot(abs(zeta),tmp[i,:],c=cmap(yb[i]))
    # axs.plot(zeta,phim[:,i],c=cmap(yb[i]))
    # axs.scatter([zeta[np.argmax(tmp[0,:])]],[phim[:,0][np.argmax(tmp[0,:])]],s=10)
axs.set_xscale('log')
axs.invert_xaxis()
plt.show()

#%%

def compute_PDF(data,n=1000):
    from scipy.stats import gaussian_kde
    x_pdf = np.linspace(min(data.flatten()),max(data.flatten()),n)
    kde = gaussian_kde(data.flatten())
    pdf = kde(x_pdf)
    return x_pdf,pdf

fig,axs = plt.subplots(1,1,tight_layout=True)


[x_pdf,pdf] = compute_PDF((data2D['sfcVAL'] - data3D['SC'][:,:,0])*0.4*data2D['ustar'],10000)
axs.plot(x_pdf,pdf)
print(f'Expected value: {np.trapz(pdf*x_pdf,x_pdf)}')

# axs.legend()
plt.show()


#%%Plot yB 

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))
axs[0].contourf(x,y,yB[:,:,0].T,levels=10,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[1].contourf(x,z_uvp,yB[:,ny//2,:].T,levels=10,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[0].set_xlabel(r"$x/z_i$",fontsize=14)
axs[0].set_ylabel(r"$y/z_i$",fontsize=14)
axs[1].set_xlabel(r"$x/z_i$",fontsize=14)
axs[1].set_ylabel(r"$z/z_i$",fontsize=14)
# axs[1].set_ylim(z_uvp[0],0.1)
plt.show()

#%%Pdf of zeta

zeta = 0.5*dz/data2D['L']

x_pdf = np.logspace(-3,5,1000)
kde = gaussian_kde(abs(zeta.flatten()))
pdf = kde(x_pdf)

fig,axs = plt.subplots(1,1,tight_layout=True)
# axs.plot(abs(x_pdf),pdf*x_pdf,c='k')
axs.hist(abs(zeta.flatten()),bins=np.logspace(-3,5,1000),density=True)
axs.set_xscale('log')
axs.invert_xaxis()
plt.show()

#%%Pdf of surface var

var = 'sfcFLUX'

fig,axs = plt.subplots(1,1,tight_layout=True)
x_pdf = np.linspace(min(data2D[var].flatten()),max(data2D[var].flatten()),1000)
kde = gaussian_kde(data2D[var].flatten())
pdf = kde(x_pdf)
axs.plot(x_pdf,pdf,c='k')
plt.show()














































































