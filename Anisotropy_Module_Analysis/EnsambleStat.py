#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:29:53 2026

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

nx = 128
ny = 128
nz = 128

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

Ug = 9
sfc = 'Homog'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sfc+'/'+str(nx)+'/'+str(Ug)+'ms/'

Niter = 105000

#%%Load RAV data and checkpoint data for the classic scaling

# cases = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
cases = ['A','B','C','D','E','F','G','H','I','J']
# cases = ['A','B']

RAV_time = '30'
check_step = 60000

mom3D_c = dict(); mom3D_a = dict()
mom2D_c = dict(); mom2D_a = dict()
sc3D_c = dict(); sc3D_a = dict()
sc2D_c = dict(); sc2D_a = dict()

for i in range(len(cases)):
    mom3D_c[cases[i]] = xr.open_dataarray(path+'Classic_'+cases[i]+'/data/Momentum3D/Data_Momentum_'+RAV_time+'min.nc')
    mom2D_c[cases[i]] = xr.open_dataarray(path+'Classic_'+cases[i]+'/data/Momentum2D/Data_Momentum_2D_'+RAV_time+'min.nc')
    sc3D_c[cases[i]] = xr.open_dataarray(path+'Classic_'+cases[i]+'/data/Scalar3D/Data_Scalar_'+RAV_time+'min.nc')
    sc2D_c[cases[i]] = xr.open_dataarray(path+'Classic_'+cases[i]+'/data/Scalar2D/Data_Scalar_2D_'+RAV_time+'min.nc')
    mom3D_a[cases[i]] = xr.open_dataarray(path+'Aniso_'+cases[i]+'/data/Momentum3D/Data_Momentum_'+RAV_time+'min.nc')
    mom2D_a[cases[i]] = xr.open_dataarray(path+'Aniso_'+cases[i]+'/data/Momentum2D/Data_Momentum_2D_'+RAV_time+'min.nc')
    sc3D_a[cases[i]] = xr.open_dataarray(path+'Aniso_'+cases[i]+'/data/Scalar3D/Data_Scalar_'+RAV_time+'min.nc')
    sc2D_a[cases[i]] = xr.open_dataarray(path+'Aniso_'+cases[i]+'/data/Scalar2D/Data_Scalar_2D_'+RAV_time+'min.nc')


# cp_c = dict(); cp_a = dict()
# cp_sfc_c = dict(); cp_sfc_a = dict()

# for i in range(len(cases)):
#     cp_c[cases[i]] = read_checkpoint_aniso(path+'Classic_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny,nz)
#     cp_sfc_c[cases[i]] = read_checkpoint_sfc_L(path+'Classic_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny)
#     cp_a[cases[i]] = read_checkpoint_aniso(path+'Aniso_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny,nz)
#     cp_sfc_a[cases[i]] = read_checkpoint_sfc_L(path+'Aniso_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny)

#%%Plot the evolution of ustar from the running averages

rd = ['iter','time','cfl','mke',r'$u_*$','L',r'$w\theta |_s$']
rd_var = 4

from matplotlib import ticker as mticker

fig, axs = plt.subplots(1, 1, figsize=(7, 6))

tmp_c = np.zeros((len(cases), Niter))
tmp_a = np.zeros((len(cases), Niter))
for i in range(len(cases)):
    diagnostics = np.genfromtxt(path+'Classic_'+cases[i]+'/running_diagnostics.txt')
    tmp_c[i,:] = diagnostics[:,rd_var]
    diagnostics = np.genfromtxt(path+'Aniso_'+cases[i]+'/running_diagnostics.txt')
    tmp_a[i,:] = diagnostics[:,rd_var]

axs.plot(diagnostics[:,0], np.median(tmp_c, axis=0), label='Classic', c='r')
axs.plot(diagnostics[:,0], np.median(tmp_a, axis=0), label='Aniso', c='k')
axs.fill_between(diagnostics[:,0], np.quantile(tmp_c, 0.25, axis=0), np.quantile(tmp_c, 0.75, axis=0), color='red', alpha=0.3)
axs.fill_between(diagnostics[:,0], np.quantile(tmp_a, 0.25, axis=0), np.quantile(tmp_a, 0.75, axis=0), color='black', alpha=0.3)
# ax.set_ylabel(rd[rd_var] + ' [m/s]', fontsize=15)
axs.set_title(f'Ug = {Ug} m/s', fontsize=14)
axs.tick_params(labelsize=12)
axs.legend(fontsize=14)
axs.grid()

diff = np.median(tmp_c, axis=0)[-1] / np.median(tmp_a, axis=0)[-1]
if diff < 1:
    print(f'Ug={Ug} - The median difference is {(1-diff)*100:.2f}%')
else:
    print(f'Ug={Ug} - The median difference is {(diff-1)*100:.2f}%')

axs.set_xlabel(r'Iteration Step', fontsize=15)
axs.xaxis.set_major_locator(plt.MaxNLocator(6))
# fig.text(0.04, 0.5, rd[rd_var] + r' $[W/m^2]$', va='center', ha='center', rotation='vertical', fontsize=15)
# fig.subplots_adjust(left=0.1)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Proposal/sfcWT_ts_128_256_1ms_9ms.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()

#%%

# ISTANTANEOUS DATA ANLAYSIS - CHECKPOINTS/SNAPSHOTS

#%%Plot PDF

# cases = ['A','B','C','D','E','F','G','J']

var = 'psi_h'

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(len(cases)):
    kde = gaussian_kde(cp_sfc_c[cases[i]][var].flatten())
    x_pdf = np.linspace(min(cp_sfc_c[cases[i]][var].flatten()),max(cp_sfc_c[cases[i]][var].flatten()),1000)
    pdf = kde(x_pdf)
    axs.plot(x_pdf,pdf,label=cases[i])
    kde = gaussian_kde(cp_sfc_a[cases[i]][var].flatten())
    x_pdf = np.linspace(min(cp_sfc_a[cases[i]][var].flatten()),max(cp_sfc_a[cases[i]][var].flatten()),1000)
    pdf = kde(x_pdf)
    axs.plot(x_pdf,pdf,label=[cases[i]])
axs.set_xlabel(f"{var}",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
# axs.legend()
plt.show()

fig,axs = plt.subplots(1,1,tight_layout=True)

x_pdf= np.linspace(min(cp_sfc_c[case][var].min() for case in cases),max(cp_sfc_c[case][var].max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(cp_sfc_c[cases[i]][var].flatten())
    pdf[:,i] = kde(x_pdf)
axs.plot(x_pdf,np.median(pdf,axis=(1)),c='k',label='Classic')
axs.fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.5)

x_pdf= np.linspace(min(cp_sfc_a[case][var].min() for case in cases),max(cp_sfc_a[case][var].max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(cp_sfc_a[cases[i]][var].flatten())
    pdf[:,i] = kde(x_pdf)
axs.plot(x_pdf,np.median(pdf,axis=(1)),c='r',label='Aniso')
axs.fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.5)
axs.legend()
axs.set_xlabel(f"{var}",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xlim(0,2)
axs.set_ylim(0,)
plt.show()

#%%Compute anisotropy with the checkpoint data

uu = cp_a[cases[0]]['uu_new'][:,:,:nz] - cp_a[cases[0]]['u_new'][:,:,:nz]*cp_a[cases[0]]['u_new'][:,:,:nz]#- checkpnt['txx_new'][:,:,:nz]
vv = cp_a[cases[0]]['vv_new'][:,:,:nz] - cp_a[cases[0]]['v_new'][:,:,:nz]*cp_a[cases[0]]['v_new'][:,:,:nz]#- checkpnt['tyy_new'][:,:,:nz]
ww = cp_a[cases[0]]['ww_new'][:,:,:nz] - cp_a[cases[0]]['w_new'][:,:,:nz]*cp_a[cases[0]]['w_new'][:,:,:nz]#- checkpnt['tzz_new'][:,:,:nz]
uv = cp_a[cases[0]]['uv_new'][:,:,:nz] - cp_a[cases[0]]['u_new'][:,:,:nz]*cp_a[cases[0]]['v_new'][:,:,:nz]#- checkpnt['txy_new'][:,:,:nz]
uw = cp_a[cases[0]]['uw_new'][:,:,:nz] - cp_a[cases[0]]['u_new'][:,:,:nz]*cp_a[cases[0]]['w_new'][:,:,:nz]#- checkpnt['txz_new'][:,:,:nz]
vw = cp_a[cases[0]]['vw_new'][:,:,:nz] - cp_a[cases[0]]['v_new'][:,:,:nz]*cp_a[cases[0]]['w_new'][:,:,:nz]#- checkpnt['tyz_new'][:,:,:nz]

tke = uu + vv + ww

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu[:,:,:],vv[:,:,:],ww[:,:,:],uv[:,:,:],uw[:,:,:],vw[:,:,:])

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

x_pdf= np.linspace(min(yB[:,:,0].flatten()),max(yB[:,:,0].flatten()),1000)
# pdf = np.zeros((len(x_pdf)))
kde = gaussian_kde(yB[:,:,0].flatten())
pdf = kde(x_pdf)
axs.plot(x_pdf,(pdf),c='k',label='Classic')
# axs.fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.5)

axs.set_xlabel(f"yB",fontsize=14)
axs.set_ylabel(r"PDF",fontsize=14)
axs.set_xlim(0,1)
axs.set_ylim(0,)
plt.show()

#%%

# zeta = np.zeros((100))
# z_sfc = (0.01/zi)/cp_sfc_a[cases[0]]['L'][2,2]
# z_end = 0.5*dz/cp_sfc_a[cases[0]]['L'][2,2]
# for i in range(1,101):
#     zeta[i-1] = z_sfc + (z_end-z_sfc)*(i-1)/(100-1)
    
# phi = np.zeros((100))
# a = 0.24 - 0.38*0.1
# b = 0.061
# c = 0.45 - 0.53*0.1
# n = -0.12 + 6.4*0.1
# for i in range(100):
#     phi[i] = (a + b*abs(zeta[i])**n)/(a + abs(zeta[i])**n) + c*abs(zeta[i])**(1/3)
    
# psi = 0
# for i in range(100-1):
#     psi = psi + 0.5*(zeta[i+1] - zeta[i])*((1 - phi[i])/zeta[i] + (1 - phi[i+1])/zeta[i+1])

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
        # if i == 1:
        #     psi_val = psi_val + 0.5*(zeta_fine[i])*((1-vertical_profile[i])/zeta_fine[i])
        # else:
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
        psim[i, j] = fit_phi_get_psi_refined(phiM_func, zeta[i], 0.01/zi, dz, npts=100)
        yy = 0.41**-3
        xx = (yy/0.33)**(1/3)
        psi_zero = -np.log(0.33) + ((3**0.5)*0.41*(0.33**(1/3))*(np.pi/6))
        psim[i,j] = np.where(zeta[i]<-0.41**-3,np.log(0.33+yy) - (3*0.41*(yy**(1/3))) + (0.41*(0.33**(1/3)))/2*np.log((1+xx)**2/(1-xx+xx**2)) + 
                            (3**0.5*0.41*(0.33**(1/3)))*np.arctan((2*xx-1)/(3**0.5)) + psi_zero, psim[i, j])
        psih[i, j] = fit_phi_get_psi_refined(phiH_func, zeta[i], 0.001/zi, dz, npts=100)
        
# phiM_B = ((1 + 0.6*(zeta**2))/(1 - 7.5*(-zeta)))**(1/3)

# === Plot ===
fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)

# φ plots
for j in range(len(yb)):
    axs[0, 0].plot(-zeta_ax, phim[:, j], c=cmap(yb[j]))
    axs[0, 1].plot(-zeta_ax, phih[:, j], c=cmap(yb[j]))

axs[0, 0].scatter(abs(0.5*dz/cp_sfc_a[cases[0]]['L']).flatten(),cp_sfc_a[cases[0]]['phi_m'].flatten(),s=1,c='k')
axs[1, 0].scatter(abs(0.5*dz/cp_sfc_a[cases[0]]['L']).flatten(),cp_sfc_a[cases[0]]['psi_m'].flatten(),s=1,c='k')
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

axs[0, 1].scatter(abs(0.5*dz/cp_sfc_a[cases[0]]['L']).flatten(),cp_sfc_a[cases[0]]['phi_h'].flatten(),s=1,c='k')
axs[1, 1].scatter(abs(0.5*dz/cp_sfc_a[cases[0]]['L']).flatten(),cp_sfc_a[cases[0]]['psi_h'].flatten(),s=1,c='k')
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
    ax.set_xlabel(r'$-\zeta$')

# fig.suptitle(sim,fontsize=14)
plt.show()

#%%Plot Box PLot

var = 'ustar'

Box = []

Box.append(np.concatenate([cp_sfc_c[case][var].flatten() for case in cases]))
Box.append(np.concatenate([cp_sfc_a[case][var].flatten() for case in cases]))
fig,axs = plt.subplots(1,1,tight_layout=True)

axs.boxplot(Box,tick_labels=['Classic','Aniso'], showfliers=False)
axs.set_ylabel(f"{var}", fontsize=14)
plt.show()

#%%Compute statistical parameters

var = 'ustar'

medians = []
means = []
stds = []
vars = []
cv = []

for i in range(len(cases)):
    medians.append(np.median(cp_sfc_c[cases[i]][var]))
    means.append(np.mean(cp_sfc_c[cases[i]][var]))
    stds.append(np.std(cp_sfc_c[cases[i]][var]))
    vars.append(np.var(cp_sfc_c[cases[i]][var]))
    cv.append(stds[i]/means[i]*100)

#%%

# RUNNING AVERAGE DATA ANALAYSIS

#%%Plot ustar,wstar,heatflux pdf from RAV data

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,3))

x_pdf= np.linspace(min(mom2D_a[cases[i]][:,:,0].data.min() for case in cases),max(mom2D_a[cases[i]][:,:,0].data.max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(mom2D_a[cases[i]][:,:,0].data.flatten())
    pdf[:,i] = kde(x_pdf)
axs[0].plot(x_pdf,np.median(pdf,axis=(1)),c='r')
axs[0].fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.3,color='red')

x_pdf= np.linspace(min(mom2D_c[cases[i]][:,:,0].data.min() for case in cases),max(mom2D_c[cases[i]][:,:,0].data.max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(mom2D_c[cases[i]][:,:,0].data.flatten())
    pdf[:,i] = kde(x_pdf)
axs[0].plot(x_pdf,np.median(pdf,axis=(1)),c='k')
axs[0].fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.3,color='black')

x_pdf= np.linspace(min(sc2D_a[cases[i]][:,:,0].data.min() for case in cases),max(sc2D_a[cases[i]][:,:,0].data.max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(sc2D_a[cases[i]][:,:,0].data.flatten())
    pdf[:,i] = kde(x_pdf)
axs[1].plot(x_pdf,np.median(pdf,axis=(1)),c='r')
axs[1].fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.3,color='red')

x_pdf= np.linspace(min(sc2D_c[cases[i]][:,:,0].data.min() for case in cases),max(sc2D_c[cases[i]][:,:,0].data.max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(sc2D_c[cases[i]][:,:,0].data.flatten())
    pdf[:,i] = kde(x_pdf)
axs[1].plot(x_pdf,np.median(pdf,axis=(1)),c='k')
axs[1].fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.3,color='black')

x_pdf= np.linspace(min(sc2D_a[cases[i]][:,:,-1].data.min() for case in cases),max(sc2D_a[cases[i]][:,:,-1].data.max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(sc2D_a[cases[i]][:,:,-1].data.flatten())
    pdf[:,i] = kde(x_pdf)
axs[2].plot(x_pdf,np.median(pdf,axis=(1)),c='r',label='Aniso')
axs[2].fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.3,color='red')

x_pdf= np.linspace(min(sc2D_c[cases[i]][:,:,-1].data.min() for case in cases),max(sc2D_c[cases[i]][:,:,-1].data.max() for case in cases),1000)
pdf = np.zeros((len(x_pdf),len(cases)))
for i in range(len(cases)):
    kde = gaussian_kde(sc2D_c[cases[i]][:,:,-1].data.flatten())
    pdf[:,i] = kde(x_pdf)
axs[2].plot(x_pdf,np.median(pdf,axis=(1)),c='k',label='Classic')
axs[2].fill_between(x_pdf,np.quantile(pdf,0.25,axis=(1)),np.quantile(pdf,0.75,axis=(1)),alpha=0.3,color='black')

axs[0].set_xlabel(r"$u_*$",fontsize=14)
axs[1].set_xlabel(r"$w_*$",fontsize=14)
axs[2].set_xlabel(r"$\overline{w'\theta'}_S$",fontsize=14)
axs[0].set_ylabel(r"PDF",fontsize=14)

# axs[0].set_xlim(0,0.5)
# axs[1].set_xlim(0,3)
# axs[2].set_xlim(0,0.001)
axs[2].legend()
fig.suptitle(sfc+f" - {nx} - {Ug}m/s", fontsize=12)

plt.show()

#%%Box plot of surface variables ustar, wT, L, z/L

ustar = []
wT = []
L = []
zeta = []

fig,axs = plt.subplots(1,4,tight_layout=True, figsize=(10,4))

for i in range(len(cases)):
    ustar.append(mom2D_c[cases[i]][:,:,0].data.flatten())
    wT.append(sc2D_c[cases[i]][:,:,-1].data.flatten())
    L.append(sc2D_c[cases[i]][:,:,1].data.flatten())
    zeta.append(dz/sc2D_c[cases[i]][:,:,1].data.flatten())

axs[0].boxplot(np.concatenate(ustar),showfliers=False, tick_labels='C', positions=[1])
axs[1].boxplot(np.concatenate(wT),showfliers=False, tick_labels='C', positions=[1])
axs[2].boxplot(np.concatenate(L),showfliers=False, tick_labels='C', positions=[1])
axs[3].boxplot(np.concatenate(zeta),showfliers=False, tick_labels='C', positions=[1])

for i in range(len(cases)):
    ustar.append(mom2D_a[cases[i]][:,:,0].data.flatten())
    wT.append(sc2D_a[cases[i]][:,:,-1].data.flatten())
    L.append(sc2D_a[cases[i]][:,:,1].data.flatten())
    zeta.append(dz/sc2D_a[cases[i]][:,:,1].data.flatten())

axs[0].boxplot(np.concatenate(ustar),showfliers=False, tick_labels='A', positions=[1.5])
axs[1].boxplot(np.concatenate(wT),showfliers=False, tick_labels='A', positions=[1.5])
axs[2].boxplot(np.concatenate(L),showfliers=False, tick_labels='A', positions=[1.5])
axs[3].boxplot(np.concatenate(zeta),showfliers=False, tick_labels='A', positions=[1.5])

axs[0].set_ylabel(r'$u_*$',fontsize=14)
axs[1].set_ylabel(r'$\overline{wT}$',fontsize=14)
axs[2].set_ylabel(r'$L$',fontsize=14)
axs[3].set_ylabel(r'$\zeta$',fontsize=14)

for i in range(len(axs)):
    axs[i].tick_params(axis='x',labelsize=12)
    axs[i].tick_params(axis='y',labelsize=12)
    axs[i].grid(color='grey',alpha=0.3)
    
fig.suptitle(sfc+f" - {nx} - {Ug}m/s", fontsize=12)

plt.show()

#%%Compute a signal-to noise ratio

var_ens_A = []
var_ens_C = []

for i in range(len(cases)):
    var_ens_C.append(mom2D_c[cases[i]][:,:,0].data.flatten()) #ustar
    # var_ens_C.append(sc2D_c[cases[i]][:,:,-1].data.flatten()) #heatflux
    # var_ens_C.append(sc2D_c[cases[i]][:,:,1].data.flatten()) #L
    # var_ens_C.append(dz/sc2D_c[cases[i]][:,:,1].data.flatten()) #zeta
    
    var_ens_A.append(mom2D_a[cases[i]][:,:,0].data.flatten()) #ustar
    # var_ens_A.append(sc2D_a[cases[i]][:,:,-1].data.flatten()) #heatflux
    # var_ens_A.append(sc2D_a[cases[i]][:,:,1].data.flatten()) #L
    # var_ens_A.append(dz/sc2D_a[cases[i]][:,:,1].data.flatten()) #zeta
    
mean_A = np.mean(np.concatenate(var_ens_A))
mean_C = np.mean(np.concatenate(var_ens_C))
std_A = np.std(np.concatenate(var_ens_A))
std_C = np.std(np.concatenate(var_ens_C))

R = abs(mean_A-mean_C)/np.sqrt(((len(cases)-1)*std_A**2 + (len(cases)-1)*std_C**2)/(len(cases)*2-2))

print(f"Signal-to-noise ratio: {R}")

#%%Plot pcolor of sfc variables

fig,axs = plt.subplots(2,10,tight_layout=True,figsize=(14,3))

for i in range(len(cases)):
    axs[0,i].pcolormesh(x,y,(mom2D_c[cases[i]][:,:,0].data).T,cmap='viridis',vmin=0,vmax=0.5)
    axs[1,i].pcolormesh(x,y,(mom2D_a[cases[i]][:,:,0].data).T,cmap='viridis',vmin=0,vmax=0.5)
    
    # axs[0,i].pcolormesh(x,y,(sc2D_c[cases[i]][:,:,0].data).T,cmap='viridis',vmin=0,vmax=4)
    # axs[1,i].pcolormesh(x,y,(sc2D_a[cases[i]][:,:,0].data).T,cmap='viridis',vmin=0,vmax=4)
    
    # axs[0,i].pcolormesh(x,y,(sc2D_c[cases[i]][:,:,-1].data).T,cmap='viridis',vmin=0,vmax=0.001)
    # axs[1,i].pcolormesh(x,y,(sc2D_a[cases[i]][:,:,-1].data).T,cmap='viridis',vmin=0,vmax=0.001)
    
plt.show()

#%%

#Plot mean profiles

#%%Compute Reynods stress terms, ustar at the surface and convective velocity scale

uu_a = dict(); uu_c = dict()
vv_a = dict(); vv_c = dict()
ww_a = dict(); ww_c = dict()
uv_a = dict(); uv_c = dict()
uw_a = dict(); uw_c = dict()
vw_a = dict(); vw_c = dict()

for i in range(len(cases)):
    uu_a[cases[i]] = mom3D_a[cases[i]][:,:,:,4].data - mom3D_a[cases[i]][:,:,:,0].data*mom3D_a[cases[i]][:,:,:,0].data #- mom3D_a[cases[i]][:,:,:,16].data
    vv_a[cases[i]] = mom3D_a[cases[i]][:,:,:,5].data - mom3D_a[cases[i]][:,:,:,1].data*mom3D_a[cases[i]][:,:,:,1].data #- mom3D_a[cases[i]][:,:,:,17].data
    ww_a[cases[i]] = wnode2uvpnode(mom3D_a[cases[i]][:,:,:,6].data - mom3D_a[cases[i]][:,:,:,2].data*mom3D_a[cases[i]][:,:,:,2].data) #- mom3D_a[cases[i]][:,:,:,18].data
    uv_a[cases[i]] = mom3D_a[cases[i]][:,:,:,13].data - mom3D_a[cases[i]][:,:,:,0].data*mom3D_a[cases[i]][:,:,:,1].data #- mom3D_a[cases[i]][:,:,:,19].data
    uw_a[cases[i]] = wnode2uvpnode(mom3D_a[cases[i]][:,:,:,14].data - uvpnode2wnode(mom3D_a[cases[i]][:,:,:,0].data)*mom3D_a[cases[i]][:,:,:,2].data) #- mom3D_a[cases[i]][:,:,:,20].data)
    vw_a[cases[i]] = wnode2uvpnode(mom3D_a[cases[i]][:,:,:,15].data - uvpnode2wnode(mom3D_a[cases[i]][:,:,:,1].data)*mom3D_a[cases[i]][:,:,:,2].data) #- mom3D_a[cases[i]][:,:,:,21].data)
    
    uu_c[cases[i]] = mom3D_c[cases[i]][:,:,:,4].data - mom3D_c[cases[i]][:,:,:,0].data*mom3D_c[cases[i]][:,:,:,0].data #- mom3D_c[cases[i]][:,:,:,16].data
    vv_c[cases[i]] = mom3D_c[cases[i]][:,:,:,5].data - mom3D_c[cases[i]][:,:,:,1].data*mom3D_c[cases[i]][:,:,:,1].data #- mom3D_c[cases[i]][:,:,:,17].data
    ww_c[cases[i]] = wnode2uvpnode(mom3D_c[cases[i]][:,:,:,6].data - mom3D_c[cases[i]][:,:,:,2].data*mom3D_c[cases[i]][:,:,:,2].data) #- mom3D_c[cases[i]][:,:,:,18].data
    uv_c[cases[i]] = mom3D_c[cases[i]][:,:,:,13].data - mom3D_c[cases[i]][:,:,:,0].data*mom3D_c[cases[i]][:,:,:,1].data #- mom3D_c[cases[i]][:,:,:,19].data
    uw_c[cases[i]] = wnode2uvpnode(mom3D_c[cases[i]][:,:,:,14].data - uvpnode2wnode(mom3D_c[cases[i]][:,:,:,0].data)*mom3D_c[cases[i]][:,:,:,2].data) #- mom3D_c[cases[i]][:,:,:,20].data)
    vw_c[cases[i]] = wnode2uvpnode(mom3D_c[cases[i]][:,:,:,15].data - uvpnode2wnode(mom3D_c[cases[i]][:,:,:,1].data)*mom3D_c[cases[i]][:,:,:,2].data) #- mom3D_c[cases[i]][:,:,:,21].data)
    
ustar_a = dict()
ustar_c = dict()

for i in range(len(cases)):
    ustar_a[cases[i]] = (uw_a[cases[i]][:,:,0]**2 + vw_a[cases[i]][:,:,0]**2)**(0.25)
    ustar_c[cases[i]] = (uw_c[cases[i]][:,:,0]**2 + vw_c[cases[i]][:,:,0]**2)**(0.25)

wT_a = dict(); wT_c = dict()

for i in range(len(cases)):
    wT_c[cases[i]] = wnode2uvpnode(sc3D_c[cases[i]][:,:,:,4].data - uvpnode2wnode(sc3D_c[cases[i]][:,:,:,0].data)*mom3D_c[cases[i]][:,:,:,2].data) #- sc3D_c[cases[i]][:,:,:,7].data)
    wT_a[cases[i]] = wnode2uvpnode(sc3D_a[cases[i]][:,:,:,4].data - uvpnode2wnode(sc3D_a[cases[i]][:,:,:,0].data)*mom3D_a[cases[i]][:,:,:,2].data) #- sc3D_a[cases[i]][:,:,:,7].data)

zi_a = dict()
zi_c = dict()
for i in range(len(cases)):
    zi_a[cases[i]] = np.median(np.argmin(wT_a[cases[i]],axis=2)*dz + 0.5*dz)
    zi_c[cases[i]] = np.median(np.argmin(wT_c[cases[i]],axis=2)*dz + 0.5*dz)
    
wstar_a = dict()
wstar_c = dict()
for i in range(len(cases)):
    wstar_a[cases[i]] = ((9.81*zi/(uscale**2))*zi_a[cases[i]]*np.median(wT_a[cases[i]][:,:,0])/np.median(sc3D_a[cases[i]][:,:,0,0].data))**(1/3)
    wstar_c[cases[i]] = ((9.81*zi/(uscale**2))*zi_c[cases[i]]*np.median(wT_c[cases[i]][:,:,0])/np.median(sc3D_c[cases[i]][:,:,0,0].data))**(1/3)

TT_a = dict(); TT_c = dict()
for i in range(len(cases)):
    TT_a[cases[i]] = sc3D_a[cases[i]][:,:,:,1].data - sc3D_a[cases[i]][:,:,:,0].data*sc3D_a[cases[i]][:,:,:,0].data
    TT_c[cases[i]] = sc3D_c[cases[i]][:,:,:,1].data - sc3D_c[cases[i]][:,:,:,0].data*sc3D_c[cases[i]][:,:,:,0].data

#%%Mean profile of the velocity components

fig,axs = plt.subplots(1,3,tight_layout=True,sharey=True)

vel = np.zeros((len(cases),nz))
for i in range(len(cases)):
    vel[i,:] = np.mean((mom3D_c[cases[i]][:,:,:,0].data)*uscale,axis=(0,1))
    # vel[i,:] = np.mean(mom3D_c[cases[i]][:,:,:,0].data/ustar_c[cases[i]][:,:,np.newaxis],axis=(0,1))
    # vel[i,:] = np.mean(mom3D_c[cases[i]][:,:,:,0].data/wstar_c[cases[i]],axis=(0,1))
axs[0].plot(np.median(vel,axis=(0)),z_uvp,c='k')
axs[0].fill_betweenx(z_uvp,np.quantile(vel,0.25,axis=(0)),np.quantile(vel,0.75,axis=(0)),alpha=0.5)
vel = np.zeros((len(cases),nz))
for i in range(len(cases)):
    vel[i,:] = np.mean((mom3D_a[cases[i]][:,:,:,0].data)*uscale,axis=(0,1))
    # vel[i,:] = np.mean(mom3D_a[cases[i]][:,:,:,0].data/ustar_a[cases[i]][:,:,np.newaxis],axis=(0,1))
    # vel[i,:] = np.mean(mom3D_a[cases[i]][:,:,:,0].data/wstar_a[cases[i]],axis=(0,1))
axs[0].plot(np.median(vel,axis=(0)),z_uvp,c='r')
axs[0].fill_betweenx(z_uvp,np.quantile(vel,0.25,axis=(0)),np.quantile(vel,0.75,axis=(0)),alpha=0.5)

vel = np.zeros((len(cases),nz))
for i in range(len(cases)):
    vel[i,:] = np.mean((mom3D_c[cases[i]][:,:,:,1].data)*uscale,axis=(0,1))
    # vel[i,:] = np.mean(mom3D_c[cases[i]][:,:,:,1].data/ustar_c[cases[i]][:,:,np.newaxis],axis=(0,1))
    # vel[i,:] = np.mean(mom3D_c[cases[i]][:,:,:,1].data/wstar_c[cases[i]],axis=(0,1))
axs[1].plot(np.median(vel,axis=(0)),z_uvp,c='k')
axs[1].fill_betweenx(z_uvp,np.quantile(vel,0.25,axis=(0)),np.quantile(vel,0.75,axis=(0)),alpha=0.5)
vel = np.zeros((len(cases),nz))
for i in range(len(cases)):
    vel[i,:] = np.mean((mom3D_a[cases[i]][:,:,:,1].data)*uscale,axis=(0,1))
    # vel[i,:] = np.mean(mom3D_a[cases[i]][:,:,:,1].data/ustar_a[cases[i]][:,:,np.newaxis],axis=(0,1))
    # vel[i,:] = np.mean(mom3D_a[cases[i]][:,:,:,1].data/wstar_a[cases[i]],axis=(0,1))
axs[1].plot(np.median(vel,axis=(0)),z_uvp,c='r')
axs[1].fill_betweenx(z_uvp,np.quantile(vel,0.25,axis=(0)),np.quantile(vel,0.75,axis=(0)),alpha=0.5)

vel = np.zeros((len(cases),nz))
for i in range(len(cases)):
    vel[i,:] = np.mean((mom3D_c[cases[i]][:,:,:,2].data)*uscale,axis=(0,1))
    # vel[i,:] = np.mean(mom3D_c[cases[i]][:,:,:,2].data/ustar_c[cases[i]][:,:,np.newaxis],axis=(0,1))
    # vel[i,:] = np.mean(mom3D_c[cases[i]][:,:,:,2].data/wstar_c[cases[i]],axis=(0,1))
axs[2].plot(np.median(vel,axis=(0)),z_uvp,c='k')
axs[2].fill_betweenx(z_uvp,np.quantile(vel,0.25,axis=(0)),np.quantile(vel,0.75,axis=(0)),alpha=0.5)
vel = np.zeros((len(cases),nz))
for i in range(len(cases)):
    vel[i,:] = np.mean((mom3D_a[cases[i]][:,:,:,2].data)*uscale,axis=(0,1))
    # vel[i,:] = np.mean(mom3D_a[cases[i]][:,:,:,2].data/ustar_a[cases[i]][:,:,np.newaxis],axis=(0,1))
    # vel[i,:] = np.mean(mom3D_a[cases[i]][:,:,:,2].data/wstar_a[cases[i]],axis=(0,1))
axs[2].plot(np.median(vel,axis=(0)),z_uvp,c='r')
axs[2].fill_betweenx(z_uvp,np.quantile(vel,0.25,axis=(0)),np.quantile(vel,0.75,axis=(0)),alpha=0.5)
    
axs[0].set_xlabel(r"$u/u_G$",fontsize=14)
axs[1].set_xlabel(r"$v/u_G$",fontsize=14)
axs[2].set_xlabel(r"$w/u_G$",fontsize=14)
axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
for i in range(len(axs)):
    # axs[i].legend()
    axs[i].set_ylim(0,z_uvp[-1])

fig.suptitle(sfc+f" - {nx} - {Ug}m/s", fontsize=12)
plt.show()


#%%Plot profiles of the Reynolds stress components

fig,axs = plt.subplots(1,6,tight_layout=True,figsize=(12,4))

axs[0].plot(np.median(np.array([np.mean(uu_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k')
axs[0].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(uu_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(uu_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[0].plot(np.median(np.array([np.mean(uu_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r')
axs[0].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(uu_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(uu_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs[1].plot(np.median(np.array([np.mean(vv_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k')
axs[1].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(vv_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(vv_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[1].plot(np.median(np.array([np.mean(vv_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r')
axs[1].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(vv_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(vv_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs[2].plot(np.median(np.array([np.mean(ww_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k')
axs[2].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(ww_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(ww_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[2].plot(np.median(np.array([np.mean(ww_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r')
axs[2].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(ww_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(ww_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs[3].plot(np.median(np.array([np.mean(uv_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k')
axs[3].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(uv_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(uv_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[3].plot(np.median(np.array([np.mean(uv_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r')
axs[3].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(uv_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(uv_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs[4].plot(np.median(np.array([np.mean(uw_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k')
axs[4].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(uw_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(uw_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[4].plot(np.median(np.array([np.mean(uw_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r')
axs[4].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(uw_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(uw_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs[5].plot(np.median(np.array([np.mean(vw_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k',label='Classic')
axs[5].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(vw_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(vw_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[5].plot(np.median(np.array([np.mean(vw_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r',label='Aniso')
axs[5].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(vw_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(vw_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)


# for i in range(len(cases)):
#     axs[0].plot(np.mean(uu[cases[i]],axis=(0,1)),z_uvp,label=cases[i])
#     axs[1].plot(np.mean(vv[cases[i]],axis=(0,1)),z_uvp,label=cases[i])
#     axs[2].plot(np.mean(ww[cases[i]],axis=(0,1)),z_uvp,label=cases[i])
#     axs[3].plot(np.mean(uv[cases[i]],axis=(0,1)),z_uvp,label=cases[i])
#     axs[4].plot(np.mean(uw[cases[i]],axis=(0,1)),z_uvp,label=cases[i])
#     axs[5].plot(np.mean(vw[cases[i]],axis=(0,1)),z_uvp,label=cases[i])
    
axs[0].set_xlabel(r"$\overline{u'u'}$",fontsize=14)
axs[1].set_xlabel(r"$\overline{v'v'}$",fontsize=14)
axs[2].set_xlabel(r"$\overline{w'w'}$",fontsize=14)
axs[3].set_xlabel(r"$\overline{u'v'}$",fontsize=14)
axs[4].set_xlabel(r"$\overline{u'w'}$",fontsize=14)
axs[5].set_xlabel(r"$\overline{v'w'}$",fontsize=14)
axs[0].set_ylabel(r"$z/z_i$",fontsize=14)
for i in range(len(axs)):
    # axs[i].legend()
    axs[i].set_ylim(0,z_uvp[-1])
    
axs[-1].legend()
fig.suptitle(sfc+f" - {nx} - {Ug}m/s", fontsize=12)
plt.show()

#%%Plot the mean profile of temperature, temperature variance and heatflux

fig,axs = plt.subplots(1,3,tight_layout=True,sharey=True)

axs[0].plot(np.median(np.array([np.mean(sc3D_c[case][:,:,:,0].data, axis=(0,1)) for case in cases]), axis=0)*Tscale,z_uvp,c='k')
axs[0].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(sc3D_c[case][:,:,:,0].data, axis=(0,1)) for case in cases]),0.25, axis=0)*Tscale,np.quantile(np.array([np.mean(sc3D_c[case][:,:,:,0].data, axis=(0,1)) for case in cases])*Tscale,0.75, axis=0),alpha=0.5)
axs[0].plot(np.median(np.array([np.mean(sc3D_a[case][:,:,:,0].data, axis=(0,1)) for case in cases]), axis=0)*Tscale,z_uvp,c='r')
axs[0].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(sc3D_a[case][:,:,:,0].data, axis=(0,1)) for case in cases]),0.25, axis=0)*Tscale,np.quantile(np.array([np.mean(sc3D_a[case][:,:,:,0].data, axis=(0,1)) for case in cases])*Tscale,0.75, axis=0),alpha=0.5)

axs[0].set_xlabel(r"$\theta$",fontsize=14)

axs[1].plot(np.median(np.array([np.mean(TT_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k',label='Classic')
axs[1].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(TT_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(TT_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[1].plot(np.median(np.array([np.mean(TT_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r',label='Aniso')
axs[1].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(TT_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(TT_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs[1].set_xlabel(r"$\overline{\theta'\theta'}$",fontsize=14)

axs[2].plot(np.median(np.array([np.mean(wT_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k',label='Classic')
axs[2].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(wT_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(wT_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs[2].plot(np.median(np.array([np.mean(wT_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r',label='Aniso')
axs[2].fill_betweenx(z_uvp,np.quantile(np.array([np.mean(wT_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(wT_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs[2].legend()
axs[0].set_ylabel(r"$z/z_i$",fontsize = 14)
axs[2].set_xlabel(r"$\overline{w'\theta'}$",fontsize=14)
axs[0].set_ylim(0,z_uvp[-1])
fig.suptitle(sfc+f" - {nx} - {Ug}m/s", fontsize=12)
plt.show()

#%%Profile of TKE

TKE_a = dict()
TKE_c = dict()

for i in range(len(cases)):
    TKE_a[cases[i]] = 0.5*(uu_a[cases[i]] + vv_a[cases[i]] + ww_a[cases[i]])
    TKE_c[cases[i]] = 0.5*(uu_c[cases[i]] + vv_c[cases[i]] + ww_c[cases[i]])

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.median(np.array([np.mean(TKE_c[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='k',label='Classic')
axs.fill_betweenx(z_uvp,np.quantile(np.array([np.mean(TKE_c[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(TKE_c[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)
axs.plot(np.median(np.array([np.mean(TKE_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp,c='r',label='Aniso')
axs.fill_betweenx(z_uvp,np.quantile(np.array([np.mean(TKE_a[case], axis=(0,1)) for case in cases]),0.25, axis=0),np.quantile(np.array([np.mean(TKE_a[case], axis=(0,1)) for case in cases]),0.75, axis=0),alpha=0.5)

axs.set_xlabel(r"$\overline{e}$",fontsize=14)
axs.legend()
axs.set_ylabel(r"$z/z_i$",fontsize = 14)
axs.set_ylim(0,z_uvp[-1])
fig.suptitle(sfc+f" - {nx} - {Ug}m/s", fontsize=12)
plt.show()

#%%Dispersive Fluxes

u_D_a = dict(); v_D_a = dict(); w_D_a = dict()
u_D_c = dict(); v_D_c = dict(); w_D_c = dict()

uu_D_a = dict(); vv_D_a = dict(); ww_D_a = dict()
uu_D_c = dict(); vv_D_c = dict(); ww_D_c = dict()

for i in range(len(cases)):
    u_D_a[cases[i]] = mom3D_a[cases[i]][:,:,:,0].data - np.mean(mom3D_a[cases[i]][:,:,:,0].data,axis=(0,1),keepdims=True)
    v_D_a[cases[i]] = mom3D_a[cases[i]][:,:,:,1].data - np.mean(mom3D_a[cases[i]][:,:,:,1].data,axis=(0,1),keepdims=True)
    w_D_a[cases[i]] = mom3D_a[cases[i]][:,:,:,2].data - np.mean(mom3D_a[cases[i]][:,:,:,2].data,axis=(0,1),keepdims=True)
    
    u_D_c[cases[i]] = mom3D_c[cases[i]][:,:,:,0].data - np.mean(mom3D_c[cases[i]][:,:,:,0].data,axis=(0,1),keepdims=True)
    v_D_c[cases[i]] = mom3D_c[cases[i]][:,:,:,1].data - np.mean(mom3D_c[cases[i]][:,:,:,1].data,axis=(0,1),keepdims=True)
    w_D_c[cases[i]] = mom3D_c[cases[i]][:,:,:,2].data - np.mean(mom3D_c[cases[i]][:,:,:,2].data,axis=(0,1),keepdims=True)
    
    uu_D_a[cases[i]] = uw_a[cases[i]] - np.mean(uw_a[cases[i]],axis=(0,1),keepdims=True)
    
fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.median(np.array([np.mean(uu_D_a[case], axis=(0,1)) for case in cases]), axis=0),z_uvp)

plt.show()

#%%Wind veer

def compute_wind_direction(u, v):
    """
    Compute meteorological wind direction from u and v components.

    Meteorological convention: 0° = wind from North, 90° = from East,
    increasing clockwise. This is the direction the wind is coming FROM.

    Parameters
    ----------
    u : np.ndarray
        Zonal (West-East) wind component. Shape: (nx, ny, nz) or any shape.
    v : np.ndarray
        Meridional (South-North) wind component. Same shape as u.

    Returns
    -------
    wdir : np.ndarray
        Wind direction in degrees [0, 360). Same shape as input.
    """
    # atan2 gives the direction the wind is going TO (math convention)
    # Adding 180° converts to the direction wind is coming FROM
    wdir = (np.degrees(np.arctan2(u, v)) + 180.0) % 360.0
    return wdir


def compute_wind_veering(u, v, z_axis=-1, reference="bottom"):
    """
    Compute wind veering across a 3D LES wind field.

    Veering is defined as the clockwise change in wind direction with height.
    Positive veering = clockwise rotation with height (typical in NH boundary layer).
    Negative veering = counter-clockwise (backing).

    Parameters
    ----------
    u : np.ndarray
        Zonal wind component. Shape: (nx, ny, nz).
    v : np.ndarray
        Meridional wind component. Shape: (nx, ny, nz).
    z_axis : int
        Axis corresponding to the vertical (z) dimension. Default: -1 (last axis).
    reference : str or int
        How to compute veering:
        - "bottom"  : veering relative to the lowest level (default).
        - "adjacent": veering between each level and the one below it (layer-by-layer).
        - int       : veering relative to a specific level index.

    Returns
    -------
    veering : np.ndarray
        Wind veering in degrees. Shape matches input; veering at the
        reference level is 0° (for "bottom"/int modes) or NaN at level 0
        (for "adjacent" mode).
    wdir : np.ndarray
        Wind direction at every grid point [degrees], same shape as u/v.
    """
    wdir = compute_wind_direction(u, v)

    nz = u.shape[z_axis]

    if reference == "bottom":
        # Veering relative to the lowest level
        ref_slice = [slice(None)] * wdir.ndim
        ref_slice[z_axis] = 0
        ref_dir = np.take(wdir, 0, axis=z_axis)
        ref_dir = np.expand_dims(ref_dir, axis=z_axis)
        veering = _angle_difference(wdir, ref_dir)

    elif reference == "adjacent":
        # Layer-by-layer veering (finite difference in z)
        wdir_upper = np.take(wdir, range(1, nz), axis=z_axis)
        wdir_lower = np.take(wdir, range(0, nz - 1), axis=z_axis)
        dv = _angle_difference(wdir_upper, wdir_lower)

        # Pad with NaN at the bottom level to preserve shape
        pad_shape = list(u.shape)
        pad_shape[z_axis] = 1
        pad = np.full(pad_shape, np.nan)
        veering = np.concatenate([pad, dv], axis=z_axis)

    elif isinstance(reference, int):
        # Veering relative to an arbitrary reference level
        ref_dir = np.take(wdir, reference, axis=z_axis)
        ref_dir = np.expand_dims(ref_dir, axis=z_axis)
        veering = _angle_difference(wdir, ref_dir)

    else:
        raise ValueError(
            f"reference must be 'bottom', 'adjacent', or an int. Got: {reference}"
        )

    return veering, wdir


def _angle_difference(angle_a, angle_b):
    """
    Compute the signed angular difference (a - b), wrapped to [-180, 180].

    Positive = clockwise rotation from b to a (veering).
    Negative = counter-clockwise (backing).
    """
    diff = (angle_a - angle_b + 180.0) % 360.0 - 180.0
    return diff


def horizontal_mean_veering_profile(u, v, x_axis=0, y_axis=1, z_axis=2):
    """
    Compute the horizontally averaged veering profile (standard LES diagnostic).

    Averages u and v over x and y first, then computes veering from the
    mean profile — consistent with how veering is typically reported from LES.

    Parameters
    ----------
    u, v : np.ndarray
        3D wind components, shape (nx, ny, nz).
    x_axis, y_axis, z_axis : int
        Axes for x, y, z dimensions.

    Returns
    -------
    veering_profile : np.ndarray
        1D array of veering [degrees] at each height level, relative to z=0.
    wdir_profile : np.ndarray
        1D mean wind direction profile [degrees].
    """
    u_mean = u.mean(axis=(x_axis, y_axis))
    v_mean = v.mean(axis=(x_axis, y_axis))

    wdir_profile = compute_wind_direction(u_mean, v_mean)
    veering_profile = _angle_difference(wdir_profile, wdir_profile[0])

    return veering_profile, wdir_profile


veer3D_a = dict(); wdir3D_a = dict()
veer3D_c = dict(); wdir3D_c = dict()
for i in range(len(cases)):
    veer3D_a[cases[i]], wdir3D_a[cases[i]] = compute_wind_veering(mom3D_a[cases[i]][:,:,:,0].data, mom3D_a[cases[i]][:,:,:,1].data, z_axis=-1, reference="bottom")
    veer3D_c[cases[i]], wdir3D_c[cases[i]] = compute_wind_veering(mom3D_c[cases[i]][:,:,:,0].data, mom3D_c[cases[i]][:,:,:,1].data, z_axis=-1, reference="bottom")


#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm


# ─────────────────────────────────────────────
#  Synthetic LES data  (replace with your own)
# ─────────────────────────────────────────────
# rng = np.random.default_rng(42)
# nx, ny, nz = 48, 48, 60
# z = np.linspace(0, 1000, nz)                   # heights in metres

# speed_prof  = 4 + 8 * np.log1p(z / 80)         # log-ish speed increase
# dir_prof    = np.radians(200 + 38 * (z / z[-1]))# veer from 200° → 238°

# u_1d = -speed_prof * np.sin(dir_prof)
# v_1d = -speed_prof * np.cos(dir_prof)

u3d = (mom3D_c[cases[0]][:,:,:,0].data)*uscale
v3d = (mom3D_c[cases[0]][:,:,:,1].data)*uscale

# ─────────────────────────────────────────────
#  Compute horizontal-mean profile
# ─────────────────────────────────────────────
u_mean = u3d.mean(axis=(0, 1))
v_mean = v3d.mean(axis=(0, 1))

speed_mean = np.sqrt(u_mean**2 + v_mean**2)
wdir_mean  = (np.degrees(np.arctan2(u_mean, v_mean)) + 180) % 360   # met convention
veering    = ((wdir_mean - wdir_mean[0]) + 180) % 360 - 180          # signed, rel. surface

# ─────────────────────────────────────────────
#  Choose display levels (evenly spaced in z)
# ─────────────────────────────────────────────
n_display = 28
idx = np.round(np.linspace(0, nz - 1, n_display)).astype(int)

# ─────────────────────────────────────────────
#  Colour map: speed → blue → cyan → orange
# ─────────────────────────────────────────────
cmap   = plt.get_cmap("coolwarm_r")
norm   = mcolors.Normalize(vmin=speed_mean.min(), vmax=speed_mean.max())
sm     = cm.ScalarMappable(cmap=cmap, norm=norm)

# ─────────────────────────────────────────────
#  Helper: draw one 3D wind barb
#    origin (x0, y0, z0), wind (u, v), at height z_val
# ─────────────────────────────────────────────
def draw_barb_3d(ax, x0, y0, z0, u, v, spd, color, shaft_scale=55, barb_scale=50):
    """
    Draws a meteorological wind barb in the XY plane at height z0.
    The barb shaft points INTO the wind (from-direction).
    Barbs hang off the shaft at 60° in the XY plane.
    """
    if spd < 0.5:
        ax.plot([x0], [y0], [z0], 'o', color=color, ms=3, zorder=5)
        return

    # Unit vector pointing INTO the wind (from-direction)
    ux, uy = u / spd, v / spd

    # Shaft tip
    xtip = x0 + ux * shaft_scale
    ytip = y0 + uy * shaft_scale
    ax.plot([x0, xtip], [y0, ytip], [z0, z0],
            color=color, lw=1.2, solid_capstyle='round', zorder=5)

    # Perpendicular unit vector (left of shaft direction)
    px, py = -uy, ux

    # Decode speed into barbs/flags
    s = round(spd / 2.5) * 2.5
    flags = int(s // 50);  s %= 50
    fulls = int(s // 10);  s %= 10
    halfs = int(s //  5)

    spacing  = shaft_scale * 0.14
    barb_len = barb_scale  * 0.55
    cursor   = shaft_scale          # start at shaft tip, walk toward origin

    # 50-unit flags (filled pennant)
    for _ in range(flags):
        base1x = x0 + ux * cursor
        base1y = y0 + uy * cursor
        base2x = x0 + ux * (cursor - spacing * 1.5)
        base2y = y0 + uy * (cursor - spacing * 1.5)
        tipx   = base1x + px * barb_len
        tipy   = base1y + py * barb_len
        tri_x  = [base1x, base2x, tipx, base1x]
        tri_y  = [base1y, base2y, tipy, base1y]
        tri_z  = [z0] * 4
        ax.plot(tri_x, tri_y, tri_z, color=color, lw=1.0, zorder=5)
        ax.add_collection3d(
            __import__('mpl_toolkits.mplot3d.art3d', fromlist=['Poly3DCollection'])
            .Poly3DCollection([list(zip(tri_x[:3], tri_y[:3], tri_z[:3]))],
                              color=color, alpha=0.75, zorder=5)
        )
        cursor -= spacing * 1.8

    # Full barbs
    for _ in range(fulls):
        bx = x0 + ux * cursor
        by = y0 + uy * cursor
        ax.plot([bx, bx + px * barb_len],
                [by, by + py * barb_len],
                [z0, z0], color=color, lw=1.2, zorder=5)
        cursor -= spacing

    # Half barbs
    for _ in range(halfs):
        bx = x0 + ux * cursor
        by = y0 + uy * cursor
        ax.plot([bx, bx + px * barb_len * 0.5],
                [by, by + py * barb_len * 0.5],
                [z0, z0], color=color, lw=1.2, zorder=5)
        cursor -= spacing


# ─────────────────────────────────────────────
#  Figure setup
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(10, 12), facecolor='#0b1520')
ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.set_facecolor('#0b1520')
fig.patch.set_facecolor('#0b1520')

# ─────────────────────────────────────────────
#  Draw each barb level
# ─────────────────────────────────────────────
for i in idx:
    color = sm.to_rgba(speed_mean[i])
    draw_barb_3d(ax, 0, 0, z_uvp[i], u_mean[i], v_mean[i], speed_mean[i],
                 color=color, shaft_scale=60, barb_scale=55)

# ─────────────────────────────────────────────
#  Wind hodograph trace (thin line connecting barb tips)
# ─────────────────────────────────────────────
shaft_scale = 60
tip_x = u_mean[idx] / speed_mean[idx] * shaft_scale
tip_y = v_mean[idx] / speed_mean[idx] * shaft_scale
tip_z = z_uvp[idx]

# Colour each segment by speed
segments = [[(tip_x[k], tip_y[k], tip_z[k]),
             (tip_x[k+1], tip_y[k+1], tip_z[k+1])]
            for k in range(len(idx)-1)]
seg_spd  = (speed_mean[idx[:-1]] + speed_mean[idx[1:]]) / 2
seg_cols = [sm.to_rgba(s) for s in seg_spd]
lc = Line3DCollection(segments, colors=seg_cols, linewidths=1.2,
                      alpha=0.55, linestyle='--', zorder=3)
ax.add_collection3d(lc)

# ─────────────────────────────────────────────
#  Vertical axis reference line
# ─────────────────────────────────────────────
ax.plot([0, 0], [0, 0], [0, z_uvp[-1]],
        color='#3a6a8a', lw=0.8, alpha=0.6, linestyle=':', zorder=2)

# ─────────────────────────────────────────────
#  Height tick marks & labels
# ─────────────────────────────────────────────
h_ticks = np.arange(0, z_uvp[-1], 200)
for h in h_ticks:
    ax.text(-85, 0, h, f'{int(h)} m', color='#6aabcc',
            fontsize=7.5, va='center', ha='right', fontfamily='monospace')
    ax.plot([-5, 5], [0, 0], [h, h], color='#3a6a8a', lw=0.7, alpha=0.5)

# ─────────────────────────────────────────────
#  Compass labels at base
# ─────────────────────────────────────────────
compass = {'N': (0, 90, '↑'), 'S': (0, -90, '↓'),
           'E': (90, 0, '→'), 'W': (-90, 0, '←')}
for label, (cx, cy, arrow) in compass.items():
    ax.text(cx, cy, -60, label, color='#4a8fa8',
            fontsize=9, ha='center', va='center', fontfamily='monospace')

# ─────────────────────────────────────────────
#  Styling
# ─────────────────────────────────────────────
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_zlim(-0.8, z_uvp[-1])

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#122030')
ax.yaxis.pane.set_edgecolor('#122030')
ax.zaxis.pane.set_edgecolor('#122030')

ax.grid(True, color='#122030', linewidth=0.5, alpha=0.5)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

ax.view_init(elev=18, azim=-60)

# ─────────────────────────────────────────────
#  Colour bar
# ─────────────────────────────────────────────
cbar_ax = fig.add_axes([0.82, 0.22, 0.02, 0.5])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Wind Speed  [m s⁻¹]', color='#6aabcc',
               fontsize=9, labelpad=10, fontfamily='monospace')
cbar.ax.yaxis.set_tick_params(color='#3a6a8a')
plt.setp(cbar.ax.yaxis.get_ticklabels(),
         color='#6aabcc', fontsize=8, fontfamily='monospace')
cbar.outline.set_edgecolor('#3a6a8a')

# ─────────────────────────────────────────────
#  Barb legend
# ─────────────────────────────────────────────
legend_ax = fig.add_axes([0.02, 0.04, 0.28, 0.12])
legend_ax.set_facecolor('#0d1e2e')
legend_ax.set_xlim(0, 10); legend_ax.set_ylim(0, 3)
legend_ax.axis('off')
legend_ax.text(0.2, 2.5, 'BARB LEGEND', color='#6aabcc',
               fontsize=7.5, fontfamily='monospace', va='top', fontweight='bold')
items = [('━━━━|', '10 m/s  full barb'),
         ('━━━━╴', ' 5 m/s  half barb'),
         ('━━━━▶', '50 m/s  flag')]
for k, (sym, label) in enumerate(items):
    legend_ax.text(0.2, 2.0 - k*0.65, sym,  color='#00cfee', fontsize=8, fontfamily='monospace')
    legend_ax.text(2.2, 2.0 - k*0.65, label, color='#8ab8cc', fontsize=7.5, fontfamily='monospace')

# ─────────────────────────────────────────────
#  Stats annotation
# ─────────────────────────────────────────────
stats_str = (
    f"Surface dir :  {wdir_mean[0]:.1f}°\n"
    f"Top dir     :  {wdir_mean[-1]:.1f}°\n"
    f"Total veer  :  {veering[-1]:+.1f}°\n"
    f"Max speed   :  {speed_mean.max():.1f} m/s\n"
    f"Avg — {nx}×{ny}×{nz} pts"
)
fig.text(0.04, 0.92, stats_str,
         color='#6aabcc', fontsize=8.2, fontfamily='monospace',
         va='top', linespacing=1.7,
         bbox=dict(facecolor='#0d1e2e', edgecolor='#1e4060',
                   boxstyle='round,pad=0.6', alpha=0.9))

# ─────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────
fig.text(0.5, 0.97, 'Wind Barbs Profile  ·  LES Horizontal Mean',
         color='#c8e8f8', fontsize=13, ha='center', va='top',
         fontfamily='monospace', fontweight='bold')
fig.text(0.5, 0.94, 'barb shaft → from-direction  |  color → wind speed  |  dashed → hodograph trace',
         color='#4a7a95', fontsize=8, ha='center', va='top', fontfamily='monospace')

# plt.savefig('/mnt/user-data/outputs/wind_barbs_profile_3d.png',
#             dpi=180, bbox_inches='tight',
#             facecolor=fig.get_facecolor())
# print("Saved.")

#%%Compute Spectogram

from matplotlib.colors import LogNorm

field = wnode2uvpnode(cp_c[cases[0]]['w'][:,:,:nz])
z = z_uvp
# def premultiplied_2d_spectra(field, dx, dy, dz=None, z=None):
    # """
    # Compute and plot pre-multiplied 2D spectra of a 3D field.
    
    # Parameters:
    #     field : np.ndarray, shape (nx, ny, nz)
    #     dx, dy : grid spacing in x and y
    #     dz : grid spacing in z (optional, used if z not provided)
    #     z  : 1D array of z coordinates (optional)
    # """
nx, ny, nz = field.shape

# --- Fluctuations: subtract planar (x,y) mean at each z ---
field_fluc = field - field.mean(axis=(0, 1), keepdims=True)

# --- Wavenumbers ---
kx = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi

# --- 2D FFT in x and y for each z ---
fft2 = np.fft.rfft2(field_fluc, axes=(0, 1))  # shape: (nx//2+1, ny, nz)

# Power spectral density (normalize)
psd = (np.abs(fft2) ** 2) / (nx * ny) ** 2  # shape: (nx//2+1, ny, nz)

# Account for one-sided spectrum (rfft): double non-zero freqs
psd[1:, :, :] *= 2
psd_2d = psd.reshape(nx * (ny//2 + 1), nz)     # shape: (nx//2+1 * ny, nz)

# --- Shell averaging over k_h = sqrt(kx^2 + ky^2) ---
KX, KY = np.meshgrid(kx, ky, indexing='ij')  # shape: (nx//2+1, ny)
K_h = np.sqrt(KX**2 + KY**2)

# Define k_h bins (log-spaced)
kh_flat = K_h.flatten()
kh_min = kh_flat[kh_flat > 0].min()
kh_max = kh_flat.max()
k_bins = np.geomspace(kh_min, kh_max, 50)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

print(psd.shape, psd_2d.shape, kh_flat.shape)

# Bin the spectra
E = np.zeros((len(k_centers), nz))
for ik, (k_lo, k_hi) in enumerate(zip(k_bins[:-1], k_bins[1:])):
    mask = ((K_h >= k_lo) & (K_h < k_hi)).flatten()  # shape: (nx//2+1, ny)
    if mask.any():
        E[ik, :] = psd_2d[mask,:].sum(axis=0)

# --- Pre-multiply by k_h ---
E_pm = k_centers[:, np.newaxis] * E  # shape: (n_bins, nz)

# --- Z axis ---
if z is None:
    z = np.arange(nz) * (dz if dz is not None else 1.0)

# --- Wavelength axis ---
lam = 2 * np.pi / k_centers

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
Z_grid, L_grid = np.meshgrid(z, lam)
cf = ax.contourf(Z_grid, L_grid, E_pm, levels=20, cmap='inferno')
plt.colorbar(cf, ax=ax, label=r'$k_h \cdot E(k_h, z)$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('z')
ax.set_ylabel(r'$\lambda_h = 2\pi / k_h$')
ax.set_title('Pre-multiplied 2D Spectra')
plt.tight_layout()
plt.show()

    # return k_centers, z, E_pm

# k_centers, z, E_pm = premultiplied_2d_spectra(mom3D_a[cases[0]][:,:,:,0].data, dx, dy, z=z_uvp)



#%%

# Spatial Spectra computed by subtracting spatial means at a specific height to get a fluctuating field

#%%

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# INPUT DATA
# ============================================================

u = mom3D_a[cases[0]][:,:,:,0].data

z_indices = [5, 15, 30, 50]

# ============================================================
# WAVENUMBERS
# ============================================================

# 1D streamwise wavenumbers
kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8,6))

for iz in z_indices:

    # --------------------------------------------------------
    # Horizontal plane at height iz
    #
    # Shape: (nx, ny)
    # --------------------------------------------------------
    u_plane = u[:, :, iz]

    # --------------------------------------------------------
    # Remove mean along x for each y
    #
    # u'(x,y) = u(x,y) - <u>_x(y)
    #
    # Mean taken over x-direction
    # --------------------------------------------------------
    u_fluc = u_plane - np.mean(u_plane, axis=0, keepdims=True)

    # --------------------------------------------------------
    # Compute spectra for each y
    # --------------------------------------------------------
    spectra_y = []

    for j in range(ny):

        # 1D signal along x
        signal = u_fluc[:, j]

        # ----------------------------------------------------
        # FFT along x
        # ----------------------------------------------------
        uhat = np.fft.rfft(signal)

        # ----------------------------------------------------
        # 1D energy spectrum
        # ----------------------------------------------------
        Euu = (np.abs(uhat)**2) / nx

        spectra_y.append(Euu)

    # --------------------------------------------------------
    # Average spectra across y
    # --------------------------------------------------------
    Euu_avg = np.mean(spectra_y, axis=0)

    # --------------------------------------------------------
    # Premultiplied spectrum
    # --------------------------------------------------------
    kxEuu = kx * Euu_avg

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plt.loglog(
        kx[1:],          # skip zero wavenumber
        kxEuu[1:],
        label=f'z-index = {iz}'
    )

# ============================================================
# FIGURE SETTINGS
# ============================================================

plt.xlabel(r'Wavenumber $k_x$')
plt.ylabel(r'Premultiplied Spectrum $k_x E_{uu}(k_x)$')

plt.grid(True, which='major')
plt.legend()

plt.tight_layout()
plt.show()
































































