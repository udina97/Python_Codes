#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:34:42 2026

@author: u1450851
"""

#%%Import libraries
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom, read_checkpoint_tke
from Anisotropy_Functions import ColorAnisotropy, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Simulation parameters

nx = 256
ny = 256
nz = 256

lx = 2000*np.pi
ly = 2000*np.pi
lz = 1000

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
# zbl = 1099
uscale = 1
# wstar = 2.05
# Tscale = 290
# qstar = 0.24
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sims = ['neut_9ms_256_Dear_Tdyn']#,'neut_9ms_64_Dear_Tdyn']#,'neut_9ms_64_Smag_Tfix','neut_9ms_64_Smag_T1hr','neut_9ms_64_Smag_T5hr']

step = 936000
time_ext = '30min'
bools = [False,False]

cases = ['D']#,'F']
# cases = sims

mom3D = dict(); tke3D = dict(); mom2D = dict(); aniso3D = dict()#; sc3D = dict(); sc2D = dict()

for i,case in enumerate(cases):
    aniso3D[case] = xr.open_dataarray(path+sims[i]+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
    mom3D[case] = xr.open_dataarray(path+sims[i]+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
    # sc3D[case] = xr.open_dataarray(path+sims[i]+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
    tke3D[case] = xr.open_dataarray(path+sims[i]+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
    # mom2D[case] = xr.open_dataarray(path+sims[i]+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
    # sc2D[case] = xr.open_dataarray(path+sims[i]+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

aniso_C = dict(); tke_C = dict(); mom_C = dict()

for i,case in enumerate(cases):
    aniso_C[case] = read_checkpoint_aniso(path+sims[i]+'/output_checkpoint/',[step],nx,ny,nz)
    tke_C[case] = read_checkpoint_tke(path+sims[i]+'/output_checkpoint/',[step],nx,ny,nz)
    # mom2D[case] = read_checkpoint_sfc_L(path+sims[i]+'/output_checkpoint/',[step],nx,ny)
    mom_C[case] = read_checkpoint_mom(path+sims[i]+'/output_checkpoint/',[step],nx,ny,nz,bools[i])

#%%Compute anisotropy using RAV data

xB = dict(); yB = dict(); l3 = dict()


for i in cases:
    uu = mom3D[i][:,:,:,4] - mom3D[i][:,:,:,0]**2 - mom3D[i][:,:,:,16] + (2/3)*tke3D[i][:,:,:,1]
    vv = mom3D[i][:,:,:,5] - mom3D[i][:,:,:,1]**2 - mom3D[i][:,:,:,17] + (2/3)*tke3D[i][:,:,:,1]
    ww = wnode2uvpnode(mom3D[i][:,:,:,6] - mom3D[i][:,:,:,2]**2) - mom3D[i][:,:,:,18] + (2/3)*tke3D[i][:,:,:,1]
    uv = mom3D[i][:,:,:,13] - mom3D[i][:,:,:,0]*mom3D[i][:,:,:,1] - mom3D[i][:,:,:,19]
    uw = wnode2uvpnode(mom3D[i][:,:,:,14] - uvpnode2wnode(mom3D[i][:,:,:,0])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,20])
    vw = wnode2uvpnode(mom3D[i][:,:,:,15] - uvpnode2wnode(mom3D[i][:,:,:,1])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,21])
    
    [xB[i],yB[i],l3[i]] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%%Plot anisotropy slice

fig,axs = plt.subplots(2,2,sharey=True,sharex=True,figsize=(11,4))

for i,case in enumerate(cases):
    # p = axs[0,i].pcolormesh(x,z_uvp,np.mean(yB[case],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    # p = axs[1,i].pcolormesh(x,z_uvp,np.mean(aniso3D[case][:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[0,i].pcolormesh(x,z_uvp,yB[case][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[1,i].pcolormesh(x,z_uvp,aniso3D[case][:,ny//2,:,1].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    axs[0,i].set_title(cases[i])


cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
fig.suptitle('yB')

plt.show()

#%%Plot the velocity field checkpoint

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(14,4),sharex=True)

for i,case in enumerate(cases):
    axs[i,0].pcolormesh(x,z_uvp,mom_C[case]['u'][:,ny//2,:nz].T,cmap='viridis')
    axs[i,1].pcolormesh(x,z_uvp,mom_C[case]['v'][:,ny//2,:nz].T,cmap='viridis')
    axs[i,2].pcolormesh(x,z_w,mom_C[case]['w'][:,ny//2,:nz].T,cmap='viridis')

axs[0,0].set_title(r'u',fontsize=12)
axs[0,1].set_title(r'v',fontsize=12)
axs[0,2].set_title(r'w',fontsize=12)

axs[0,0].set_ylabel(r'z [m]',fontsize=12)
axs[1,0].set_ylabel(r'z [m]',fontsize=12)

for i in range(len(axs)):
    axs[1,i].set_xlabel(r'x [m]',fontsize=12)

plt.show()

#%%
height = np.arange(nz)*dz

fig,axs = plt.subplots(1,2,figsize=(10, 6),sharey=True)
for i,case in enumerate(cases):

    D = aniso3D[case][:, :, :, 1]  # expected shape: (nx, ny, nz)
    
    # One common value grid for all heights
    valid = D[np.isfinite(D)]
    x_pdf = np.linspace(valid.min(), valid.max(), 1000)
    
    nz = D.shape[2]
    pdf_by_height = np.zeros((nz, len(x_pdf)))
    
    for k in range(nz):
        values = D[:, :, k].ravel()
        values = values[np.isfinite(values)]
    
        if len(values) > 1 and np.std(values) > 0:
            y_pdf = gaussian_kde(values)(x_pdf)
            pdf_by_height[k, :] = y_pdf

    p = axs[i].pcolormesh(x_pdf, height, pdf_by_height, shading="auto", cmap="magma")
    cbar = plt.colorbar(p,label="Probability density")
    axs[i].set_xlabel("yB")
axs[0].set_ylabel("Height / vertical level")
fig.suptitle("PDF of yB across horizontal planes")
plt.show()

#%%Compute and plot shear stress

sgs = dict()

for i,case in enumerate(cases):
    # sgs[case] = - mom3D[case][:,:,:,16]- mom3D[case][:,:,:,17]- mom3D[case][:,:,:,18]
    sgs[case] = - aniso_C[case]['txx_new']- aniso_C[case]['tyy_new']- aniso_C[case]['tzz_new']


#%%Plot PDF of yB at the surface

fig,axs = plt.subplots(1,1,tight_layout=True)

y_pdf = gaussian_kde(yB['D'][:,:,0].flatten())
x_pdf = np.linspace(yB['D'][:,:,0].flatten().min(),yB['D'][:,:,0].flatten().max(),1000)
axs.plot(x_pdf,y_pdf(x_pdf),c='k',ls='-')

y_pdf = gaussian_kde(aniso3D['D'][:,:,0,1].flatten())
x_pdf = np.linspace(aniso3D['D'][:,:,0,1].flatten().min(),aniso3D['D'][:,:,0,1].flatten().max(),1000)
axs.plot(x_pdf,y_pdf(x_pdf),c='k',ls='--')

plt.show()


#%%Plot the dynamic memory time scale

from matplotlib.colors import LogNorm
T = ((3/aniso_C['D']['S_uvp'][:,:,:nz])*(zi/uscale))/60

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

p = axs[0].pcolormesh(x,z_uvp,np.mean(T[:,:,:],axis=(1)).T,cmap='viridis',norm=LogNorm(vmin=T[T > 0].min(), vmax=T.max()))
cs = axs[0].contour(x,z_uvp,np.mean(T, axis=1).T,levels=[1, 10, 30, 60],colors='k')
axs[0].clabel(cs, inline=True, fontsize=8, fmt='%g')
cbar = plt.colorbar(p)
axs[0].set_xlabel(r'x [m]',fontsize=12)
axs[0].set_ylabel(r'z [m]',fontsize=12)
axs[0].set_title(r'T', fontsize=12)

z_lvl = 120
axs[0].axhline(z_lvl*dz + dz/2,c='r',ls='--')
y_pdf = gaussian_kde(T[:,:,z_lvl].flatten())
x_pdf = np.linspace(T[:,:,z_lvl].flatten().min(),T[:,:,z_lvl].flatten().max(),1000)
axs[1].plot(x_pdf,y_pdf(x_pdf),c='k',ls='-')
axs[1].set_xlabel(r'T [min]',fontsize=12)
axs[1].set_title(f'uvp level: {z_lvl}', fontsize=12)
axs[1].set_ylim(0)

plt.show()

#%%Plot velocity profiles

fig,axs = plt.subplots(1,1,tight_layout=True)
ls = ['-','--']
for i,case in enumerate(cases):
    axs.plot(np.mean(mom3D[case][:,:,:,0],axis=(0,1))*uscale,z_uvp,c='k',ls=ls[i])
    axs.plot(np.mean(mom3D[case][:,:,:,1],axis=(0,1))*uscale,z_uvp,c='r',ls=ls[i])
    axs.plot(np.mean(mom3D[case][:,:,:,2],axis=(0,1))*uscale,z_w,c='g',ls=ls[i])

axs.set_ylim(0,z_uvp[-1])
axs.set_ylabel(r'z [m]', fontsize=12)
axs.set_xlabel(r'$u_i$ [m/s]', fontsize=12)
plt.show()

#%%Compare Reynolds stresses

uu = dict(); vv = dict(); ww = dict(); uv = dict(); uw = dict(); vw = dict()
uu_otf = dict(); vv_otf = dict(); ww_otf = dict(); uv_otf = dict(); uw_otf = dict(); vw_otf = dict()
tke_rav = dict(); tke_otf = dict()

for i in cases:
    uu[i] = mom3D[i][:,:,:,4] - mom3D[i][:,:,:,0]**2 - mom3D[i][:,:,:,16] + (2/3)*tke3D[i][:,:,:,1]
    vv[i] = mom3D[i][:,:,:,5] - mom3D[i][:,:,:,1]**2 - mom3D[i][:,:,:,17] + (2/3)*tke3D[i][:,:,:,1]
    ww[i] = wnode2uvpnode(mom3D[i][:,:,:,6] - mom3D[i][:,:,:,2]**2) - mom3D[i][:,:,:,18] + (2/3)*tke3D[i][:,:,:,1]
    uv[i] = mom3D[i][:,:,:,13] - mom3D[i][:,:,:,0]*mom3D[i][:,:,:,1] - mom3D[i][:,:,:,19]
    uw[i] = wnode2uvpnode(mom3D[i][:,:,:,14] - uvpnode2wnode(mom3D[i][:,:,:,0])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,20])
    vw[i] = wnode2uvpnode(mom3D[i][:,:,:,15] - uvpnode2wnode(mom3D[i][:,:,:,1])*mom3D[i][:,:,:,2] - mom3D[i][:,:,:,21])
    
    uu_otf[i] = aniso_C[i]['uu_new'][:,:,:nz] - aniso_C[i]['u_new'][:,:,:nz]**2 - aniso_C[i]['txx_new'][:,:,:nz] + (2/3)*aniso_C[i]['tke_sgs_new'][:,:,:nz]
    vv_otf[i] = aniso_C[i]['vv_new'][:,:,:nz] - aniso_C[i]['v_new'][:,:,:nz]**2 - aniso_C[i]['tyy_new'][:,:,:nz] + (2/3)*aniso_C[i]['tke_sgs_new'][:,:,:nz]
    ww_otf[i] = aniso_C[i]['ww_new'][:,:,:nz] - aniso_C[i]['w_new'][:,:,:nz]**2 - aniso_C[i]['tzz_new'][:,:,:nz] + (2/3)*aniso_C[i]['tke_sgs_new'][:,:,:nz]
    uv_otf[i] = aniso_C[i]['uv_new'][:,:,:nz] - aniso_C[i]['u_new'][:,:,:nz]*aniso_C[i]['v_new'][:,:,:nz] - aniso_C[i]['txy_new'][:,:,:nz]
    uw_otf[i] = aniso_C[i]['uw_new'][:,:,:nz] - aniso_C[i]['u_new'][:,:,:nz]*aniso_C[i]['w_new'][:,:,:nz] - aniso_C[i]['txz_new'][:,:,:nz]
    vw_otf[i] = aniso_C[i]['vw_new'][:,:,:nz] - aniso_C[i]['v_new'][:,:,:nz]*aniso_C[i]['w_new'][:,:,:nz] - aniso_C[i]['tyz_new'][:,:,:nz]
    
    tke_rav[i] = uu[i] + vv[i] + ww[i]
    tke_otf[i] = uu_otf[i] + vv_otf[i] + ww_otf[i]


#%%Plot Reynolds stress

fig,axs = plt.subplots(2,2,sharey=True,sharex=True,figsize=(11,4))

for i,case in enumerate(cases):
    # p = axs[0,i].pcolormesh(x,z_uvp,np.mean(yB[case],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    # p = axs[1,i].pcolormesh(x,z_uvp,np.mean(aniso3D[case][:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
    p = axs[0,i].pcolormesh(x,z_uvp,tke_rav[case][:,ny//2,:].T,cmap='viridis',vmin=0,vmax=2)
    p = axs[1,i].pcolormesh(x,z_uvp,tke_otf[case][:,ny//2,:].T,cmap='viridis',vmin=0,vmax=2)
    axs[0,i].set_title(cases[i])


cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
fig.suptitle('yB')

plt.show()

#%%Compute yB approx

yBa_rav = dict(); yBa_otf = dict()

for i,case in enumerate(cases):
    yBa_rav[case] = (ww[case]/tke_rav[case])#*(3*np.sqrt(3)/4)
    yBa_otf[case] = (ww_otf[case]/tke_otf[case])#*(3*np.sqrt(3)/4)

fig,axs = plt.subplots(2,2,figsize=(10,5))

for i,case in enumerate(cases):
    axs[0,i].pcolormesh(x,z_uvp,yBa_rav[case][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=1)
    p = axs[1,i].pcolormesh(x,z_uvp,yBa_otf[case][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=1)

cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
fig.colorbar(p, cax=cax, orientation='vertical')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# SPECTRAL ANALYSIS

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Spatial Spectra for a 2D plane at a given height, can average over multiple snapshots to smooth the curves. The spectra is expected to show the -5/3
# close to the surface, but as one moves further up the spectra starts to show the -1 regime for the large scale motions too

def RadialPowerSpectra(var_snap, zref, Nx, Ny, dx, dy):
    # This function computes the radial power spectra of a 2D field.
    # It takes no arguments and returns the radial wavenumber and the corresponding power spectrum.

    var_prime = var_snap - np.mean(var_snap, axis=(0, 1))  # Example for u component; adjust as needed for v, w, T

    # Compute the 2D FFT of the variable at the reference height (z = 10dz):
    signal = np.squeeze(var_prime[:,:,zref])
    fft_signal = np.fft.fft2(signal,s=(Nx,Ny),norm='forward') #Here, the nyquist frequency is at the center of the array.

    # Shift the zero-frequency component to the center of the spectrum:
    shifted_fft = np.fft.fftshift(fft_signal)
    PSD = np.abs(shifted_fft*np.conj(shifted_fft))

    # Compute the wavelengths corresponding to the FFT frequencies in both x and y directions:    
    kx = np.fft.fftfreq(signal.shape[0], d=dx) #This is the Wavenumber vector in the x-direction. We only need to keep the positive frequencies (first half), since the negative frequencies are redundant for real-valued signals.
    ky = np.fft.fftfreq(signal.shape[1], d=dy) #This is the Wavenumber vector in the y-direction. We only need to keep the positive frequencies, since the negative frequencies are redundant for real-valued signals.

    # Keep the PSD and frequency/wavelength grid the same size and ordering
    shifted_kx = np.fft.fftshift(kx)
    shifted_ky = np.fft.fftshift(ky)
    kx2, ky2 = np.meshgrid(shifted_kx, shifted_ky, indexing='ij')
    k_grid = np.sqrt(kx2**2 + ky2**2)

    #Remove the zero-frequency component from the PSD and wavelength grid to avoid division by zero in subsequent calculations:
    trim_PSD = PSD[1:,1:]
    trim_k_grid = k_grid[1:,1:]

    #Define the radial coordinate:
    center_y, center_x = np.array(trim_PSD.shape) // 2
    x, y = np.indices(trim_PSD.shape)
    r = (np.sqrt((x - center_x)**2 + (y - center_y)**2))

    #Sort the radial distances and corresponding correlation values
    r_flat = r.ravel()
    spectrum_flat = trim_PSD.ravel()
    k_flat = trim_k_grid.ravel()

    #Use a histogram-like method to bin the data
    num_bins = int(Nx/2) -1
    r_bins = np.linspace(0, int(Nx/2) -1, num_bins)
    r_avg = np.zeros(num_bins - 1)
    spectrum_radial = np.zeros(num_bins - 1)
    k_radial = np.zeros(num_bins - 1)

    for i in range(num_bins - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.any(mask):
                r_avg[i] = (np.mean(r_flat[mask]))
                spectrum_radial[i] = np.mean(spectrum_flat[mask])
                k_radial[i] = np.mean(k_flat[mask])
            

    PSpectra = 2*spectrum_radial
    
    return PSpectra, k_radial

def load_output3_snapshots(path, nx, ny, nz, num_snapshots, start_iteration):
    """Load u, v, w snapshots created by output3 in IO_main.f90."""
    from pathlib import Path
    import re
    import numpy as np
    pattern = re.compile(r"^snapshots_(\d+)\.dat$")
    path = Path(path)

    files = sorted(
        (
            (int(match.group(1)), file)
            for file in path.iterdir()
            if (match := pattern.match(file.name))
            and int(match.group(1)) >= start_iteration
        ),
        key=lambda item: item[0],
    )

    if len(files) < num_snapshots:
        raise FileNotFoundError(
            f"Found only {len(files)} snapshots at or after iteration "
            f"{start_iteration}; requested {num_snapshots}."
        )

    shape = (nx, ny, nz)
    npoints = nx * ny * nz
    u = np.empty((*shape, num_snapshots))
    v = np.empty((*shape, num_snapshots))
    w = np.empty((*shape, num_snapshots))

    for t, (_, filename) in enumerate(files[:num_snapshots]):
        data = np.fromfile(filename, dtype=np.float64)

        if data.size != 3 * npoints:
            raise ValueError(f"Unexpected size for snapshot file: {filename}")

        # output3 writes u, then v, then w in Fortran array order.
        u[..., t] = data[:npoints].reshape(shape, order="F")
        v[..., t] = data[npoints : 2 * npoints].reshape(shape, order="F")
        w[..., t] = data[2 * npoints :].reshape(shape, order="F")

    return u, v, w

def radial_power_spectrum(var_snap, zref, dx, dy, nbins=None):
    signal = var_snap[:, :, zref]
    signal = signal - signal.mean()

    nx, ny = signal.shape
    F = np.fft.fft2(signal, norm="forward")
    power2d = np.abs(F)**2

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kx2, ky2 = np.meshgrid(kx, ky, indexing="ij")
    kr = np.hypot(kx2, ky2)

    # Exclude only the actual DC component.
    valid = kr > 0
    k_values = kr[valid]
    power_values = power2d[valid]

    if nbins is None:
        nbins = min(nx, ny) // 2

    edges = np.linspace(0, k_values.max(), nbins + 1)
    power_shell, _ = np.histogram(k_values, bins=edges, weights=power_values)
    counts, _ = np.histogram(k_values, bins=edges)

    k_radial = 0.5 * (edges[:-1] + edges[1:])
    power_mean = np.divide(
        power_shell, counts,
        out=np.full(nbins, np.nan),
        where=counts > 0,
    )

    return power_mean, k_radial  # or return power_mean, k_radial

#%%Load Snapshot data

sim = 'neut_9ms_64_Dear_Tdyn'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'

nx = 64
ny = 64
nz = 64

lx = 2000*np.pi
ly = 2000*np.pi
lz = 1000

dx = lx/nx
dy = ly/ny
dz = lz/nz

x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

nsnap = 10
snap_start = 937000

u,v,w = load_output3_snapshots(path+sim+'/output_snap/', nx, ny, nz, nsnap, snap_start)

#%%Load RAV data to compute ustar

time_ext = '30min'
mom = xr.open_dataarray(path+sim+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data

ustar = np.mean((((mom[:,:,:,14] - uvpnode2wnode(mom[:,:,:,0])*mom[:,:,:,2] - mom[:,:,:,20])**2 + \
    (mom[:,:,:,15] - uvpnode2wnode(mom[:,:,:,1])*mom[:,:,:,2] - mom[:,:,:,21])**2)**(1/4))[:,:,0])

#%%Plot pcolor ofsnapshot fields

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(14,4))

axs[0].pcolormesh(x,z_uvp,u[:,ny//2,:,0].T,cmap='viridis')
axs[1].pcolormesh(x,z_uvp,v[:,ny//2,:,0].T,cmap='viridis')
axs[2].pcolormesh(x,z_w,w[:,ny//2,:,0].T,cmap='viridis')

axs[0].set_title(r'u',fontsize=12)
axs[1].set_title(r'v',fontsize=12)
axs[2].set_title(r'w',fontsize=12)

axs[0].set_ylabel(r'z [m]',fontsize=12)

for i in range(len(axs)):
    axs[i].set_xlabel(r'x [m]',fontsize=12)

plt.show()

#%%Compute spatial spectra at a given height

Spectra = 0
levels = [1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61]#,63]
cmap = plt.colormaps['viridis']
c = cmap(np.linspace(0,1,len(levels)))
fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(len(levels)):
    for j in range(0,nsnap):
        spectra, k_rad = RadialPowerSpectra(u[:,:,:,j], levels[i], nx, ny, dx, dy)
        # spectra, k_rad = radial_power_spectrum(u[:,:,:,j], levels[i], dx, dy)
        # print(k_rad)
        Spectra = spectra if j == 0 else np.vstack((Spectra, spectra))
    # axs.loglog(k_rad,np.mean(Spectra,axis=0),linewidth=2)
    axs.loglog(k_rad*(levels[i]*dz + dz/2),spectra/(ustar**2*(levels[i]*dz + dz/2)),linewidth=2,c=c[i],label=str(levels[i]*dz + dz/2))

axs.set_xlabel(r'$kz$')
axs.set_ylabel(r'E/$u_*^2$z')

coef = 1e-7 #This is a coefficient that can be adjusted to fit the $k^{-5/3}$ line to the data. You can change this value to see how it affects the fit.
axs.loglog(k_rad[12:]*(levels[i]*dz + dz/2),coef*k_rad[12:]**(-5/3)/(ustar**2*(levels[i]*dz + dz/2)),label=r'$k^{-5/3}$',linestyle='--',color='black')
coef = 1.5e-5
axs.loglog(k_rad[:11]*(levels[i]*dz + dz/2),coef*k_rad[:11]**(-1)/(ustar**2*(levels[i]*dz + dz/2)),label=r'$k^{-1}$',linestyle=':',color='black')
axs.legend()

plt.show()

#%%Graphical representation of the corresponding 1D Power Spectra:

lambda_radial = 1/k_rad

fig, axs=plt.subplots(1,1,figsize=(15,5))

# for t in range(0,t_stamps):
# axs.loglog(k_rad,spectra)

axs.loglog(k_rad,np.mean(Spectra,axis=0),color='black',linewidth=2)
axs.set_xlabel(r'$k$')
axs.set_ylabel(r'1D Energy Spectrum')

coef = 1e-8 #This is a coefficient that can be adjusted to fit the $k^{-5/3}$ line to the data. You can change this value to see how it affects the fit.
axs.loglog(k_rad,coef*k_rad**(-5/3),label=r'$k^{-5/3}$',linestyle='--',color='black')

plt.show()


#%%Reading the tower timeseries

def read_tower(path, nz):
    """Read a twr_ts_*.out file; return an array shaped (nz, nt)."""
    from pathlib import Path
    nz = int(nz)
    if nz <= 0:
        raise ValueError("nz must be positive")

    record = np.dtype([
        ("step", np.int32),
        ("profile", np.float64, (nz,)),
    ])

    path = Path(path)
    if path.stat().st_size % record.itemsize:
        raise ValueError("File size is not a whole number of tower records; check nz")

    data = np.fromfile(path, dtype=record)
    return data["profile"].T.copy()

import numpy as np

def cosp(u, w, hz):
    """
    Compute a cospectrum and its bin-averaged version.

    Returns
    -------
    f_bin, Co_uw_bin, f, Co_uw
    """
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)

    if u.ndim != 1 or w.ndim != 1 or len(u) != len(w):
        raise ValueError("u and w must be 1D arrays of equal length")
    if len(u) < 4 or hz <= 0:
        raise ValueError("At least 4 samples and a positive sampling frequency are required")

    n = len(u)
    df = hz / n

    # Equivalent to the real/imaginary products in the MATLAB code.
    U = np.fft.rfft(u) / n
    W = np.fft.rfft(w) / n
    Co_uw = 2 * np.real(U * np.conj(W)) / df

    # Preserves the frequency values in the supplied MATLAB code.
    f = np.arange(1, len(Co_uw) + 1) * df

    f_bin = list(f[:3])
    Co_uw_bin = list(Co_uw[:3])

    start = 3                  # Python index corresponding to MATLAB bin 4
    end = 2 * start            # Exclusive Python slice endpoint
    while end <= len(f):
        f_bin.append(np.mean(f[start:end]))
        Co_uw_bin.append(np.mean(Co_uw[start:end]))
        start = end
        end = 2 * (start + 1)

    return np.array(f_bin), np.array(Co_uw_bin), f, Co_uw

#%%Load timeseries

npoints = 4
nsteps = 36000
nz = 64

idx = [16,48]
idy = [16,48]

path = '/scratch/general/nfs1/u1450851/LES_Sims/neut_9ms_64_Dear_Tdyn/output_twr/'

u_ts = np.zeros((npoints,nz,nsteps))
v_ts = np.zeros((npoints,nz,nsteps))
w_ts = np.zeros((npoints,nz,nsteps))

pt = 0
stop = False

for i in range(len(idx)):
    for j in range(len(idy)):
        u_ts[pt,:,:] = read_tower(path + 'twr_ts_0'+str(idx[i])+'_0'+str(idy[j])+'_u.out', nz)*uscale #in m/s
        v_ts[pt,:,:] = read_tower(path + 'twr_ts_0'+str(idx[i])+'_0'+str(idy[j])+'_v.out', nz)*uscale #in m/s
        w_ts[pt,:,:] = read_tower(path + 'twr_ts_0'+str(idx[i])+'_0'+str(idy[j])+'_w.out', nz)*uscale #in m/s
        pt += 1
        if pt == npoints:
            stop = True
            break
    if stop:
        break

dt = 0.1 #in s
freq = 1/dt
time = np.arange(0,u_ts.shape[2])*dt # in s


#%%Plot time series at different heights

fig, axs = plt.subplots(1, 1, tight_layout=True,figsize=(10,3))

for i in range(npoints):
    label_u = 'u' if i == 0 else None
    label_v = 'v' if i == 0 else None
    label_w = 'w' if i == 0 else None

    axs.plot(time, u_ts[i, 0, :], c='k', ls='-', label=label_u)
    axs.plot(time, v_ts[i, 0, :], c='r', ls='-', label=label_v)
    axs.plot(time, w_ts[i, 1, :], c='g', ls='-', label=label_w)

    axs.plot(time, u_ts[i, nz//2, :], c='k', ls='--')
    axs.plot(time, v_ts[i, nz//2, :], c='r', ls='--')
    axs.plot(time, w_ts[i, nz//2, :], c='g', ls='--')

axs.set_xlabel(r't [s]', fontsize=12)
axs.set_ylabel(r'$u_i$ [m/s]', fontsize=12)
axs.legend()
axs.set_xlim(time[0], time[-1])

plt.show()

#%%Compute Spectra of timeseries

Spectra = 0
Spectra_bin = 0
levels = [60,61,62,63]
fact = [0.1,5,100]
c = ['k','r','g','b']

fig,axs = plt.subplots(1,1,tight_layout=True)
for l in range(len(levels)):
    for n in range(npoints):
        # u_fluc = u_ts[n,levels[l],:len(time)//2]-np.mean(u_ts[n,levels[l],:len(time)//2])
        u_fluc = w_ts[n,levels[l],:len(time)//2]-np.mean(w_ts[n,levels[l],:len(time)//2])
        f_bin,spec_bin,f,spectra = cosp(u_fluc, u_fluc, freq)
        # print(f_bin)
        Spectra = spectra if n == 0 else np.vstack((Spectra, spectra))
        Spectra_bin = spec_bin if n==0 else np.vstack((Spectra_bin,spec_bin))

    axs.loglog(f[1:], np.mean(Spectra,axis=(0))[1:], alpha=0.35,c=c[l],label=str(levels[l]*dz + dz/2))
    # axs.loglog(f_bin[1:], np.mean(Spectra_bin,axis=(0))[1:],
    #               "o-",c=c[l],label=str(levels[l]*dz/2 + dz/2))
    
# coef = 1e-0 #This is a coefficient that can be adjusted to fit the $k^{-5/3}$ line to the data. You can change this value to see how it affects the fit.
# axs.loglog(f[10:],coef*f[10:]**(-5/3),label=r'$k^{-5/3}$',linestyle='--',color='black')
# coef = 1e2 #This is a coefficient that can be adjusted to fit the $k^{-5/3}$ line to the data. You can change this value to see how it affects the fit.
# axs.loglog(f[:100],coef*f[:100]**(-1),label=r'$k^{-1}$',linestyle=':',color='black')
axs.set_xlabel("Frequency (Hz)")
axs.set_ylabel("Cospectral density")
axs.legend()
plt.show()























































