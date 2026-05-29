#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:29:11 2026

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
path = '/scratch/general/nfs1/u1450851/LES_Sims/'

#%%Load RAV data and checkpoint data for the classic scaling

# cases = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
cases128 = ['A','B','C','D','E','F','G','H','I','J']
cases256 = ['A','B']

sfc_type = 'Homog'
RAV_time = '30'
check_step = 60000
Niter = 105000

#%%Plot running diagnostic time series 3 (mke), 4 (ustar), 5 (L), 6 (wT)

rd = ['iter','time','cfl','mke',r'$u_*$','L',r'$w\theta |_s$']
rd_var = 6

from matplotlib import ticker as mticker

fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

for ax, Ug in zip(axs, [9, 1]):
    tmp_c = np.zeros((len(cases128), Niter))
    tmp_a = np.zeros((len(cases128), Niter))
    for i in range(len(cases128)):
        diagnostics = np.genfromtxt(path+sfc_type+'/'+str(nx)+'/'+str(Ug)+'ms/'+'Classic_'+cases128[i]+'/running_diagnostics.txt')
        tmp_c[i,:] = diagnostics[:,rd_var]*uscale*Tscale*1000
        diagnostics = np.genfromtxt(path+sfc_type+'/'+str(nx)+'/'+str(Ug)+'ms/'+'Aniso_'+cases128[i]+'/running_diagnostics.txt')
        tmp_a[i,:] = diagnostics[:,rd_var]*uscale*Tscale*1000

    ax.plot(diagnostics[:,0], np.median(tmp_c, axis=0), label='Classic 128', c='r')
    ax.plot(diagnostics[:,0], np.median(tmp_a, axis=0), label='Aniso 128', c='k')
    ax.fill_between(diagnostics[:,0], np.quantile(tmp_c, 0.25, axis=0), np.quantile(tmp_c, 0.75, axis=0), color='red', alpha=0.3)
    ax.fill_between(diagnostics[:,0], np.quantile(tmp_a, 0.25, axis=0), np.quantile(tmp_a, 0.75, axis=0), color='black', alpha=0.3)
    # ax.set_ylabel(rd[rd_var] + ' [m/s]', fontsize=15)
    ax.set_title(f'Ug = {Ug} m/s', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=14)
    ax.grid()

    diff = np.median(tmp_c, axis=0)[-1] / np.median(tmp_a, axis=0)[-1]
    if diff < 1:
        print(f'Ug={Ug} - The median difference is {(1-diff)*100:.2f}%')
    else:
        print(f'Ug={Ug} - The median difference is {(diff-1)*100:.2f}%')
        
for ax, Ug in zip(axs, [9, 1]):
    tmp_c = np.zeros((len(cases256), Niter))
    tmp_a = np.zeros((len(cases256), Niter))
    for i in range(len(cases256)):
        diagnostics = np.genfromtxt(path+sfc_type+'/'+str(256)+'/'+str(Ug)+'ms/'+'Classic_'+cases256[i]+'/running_diagnostics.txt')
        tmp_c[i,:] = diagnostics[:,rd_var]*uscale*Tscale*1000
        diagnostics = np.genfromtxt(path+sfc_type+'/'+str(256)+'/'+str(Ug)+'ms/'+'Aniso_'+cases256[i]+'/running_diagnostics.txt')
        tmp_a[i,:] = diagnostics[:,rd_var]*uscale*Tscale*1000

    ax.plot(diagnostics[:,0], np.median(tmp_c, axis=0), label='Classic 256', c='r', ls='--')
    ax.plot(diagnostics[:,0], np.median(tmp_a, axis=0), label='Aniso 256', c='k', ls='--')
    ax.fill_between(diagnostics[:,0], np.quantile(tmp_c, 0.25, axis=0), np.quantile(tmp_c, 0.75, axis=0), color='red', alpha=0.3)
    ax.fill_between(diagnostics[:,0], np.quantile(tmp_a, 0.25, axis=0), np.quantile(tmp_a, 0.75, axis=0), color='black', alpha=0.3)
    # ax.set_ylabel(rd[rd_var] + ' [m/s]', fontsize=15)
    # ax.set_title(f'Ug = {Ug} m/s', fontsize=14)
    # ax.tick_params(labelsize=12)
    ax.legend(fontsize=14)
    # ax.grid()

    diff = np.median(tmp_c, axis=0)[-1] / np.median(tmp_a, axis=0)[-1]
    if diff < 1:
        print(f'Ug={Ug} - The median difference is {(1-diff)*100:.2f}%')
    else:
        print(f'Ug={Ug} - The median difference is {(diff-1)*100:.2f}%')

axs[1].set_xlabel(r'Iteration Step', fontsize=15)
axs[0].xaxis.set_major_locator(plt.MaxNLocator(6))
fig.text(0.04, 0.5, rd[rd_var] + r' $[W/m^2]$', va='center', ha='center', rotation='vertical', fontsize=15)
# fig.subplots_adjust(left=0.1)

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Proposal/sfcWT_ts_128_256_1ms_9ms.png', dpi = 300,  facecolor='None', edgecolor='None')

plt.show()


#%%



















































