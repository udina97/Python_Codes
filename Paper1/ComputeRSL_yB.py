#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:53:39 2025

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")
from functions import build_phi, build_intf

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from Anisotropy_Functions import ColorAnisotropy
cmap = ColorAnisotropy()

#%%

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from functions2 import compute_d_twr,find_coordinates,average_over_selected_coords

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_h[:, :, -1]
  return phi_c

#%%Paths to data

pathFig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/'
directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'

flat = 'simflat_256x256x384_out5hr_v2'
atto = 'simATTO_256x256x384_full_out5hr'
sin = 'simbicheng_hill_256x256x384_out5hr'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/' 

path_flat = pathOUT + flat + '/'
path_sin = pathOUT + sin + '/'
path_atto = pathOUT + atto + '/'

#%%Loading Topo data

zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py

tpath_flat = directory + flat + '/output/'
tpath_sin = directory + sin + '/output/'
tpath_atto = directory + atto + '/output/'

# Read parameter file for ta1_field
with open(tpath_flat + 'ta1_field/parameters.txt', 'r') as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

Nx = int(param[0])
Ny = int(param[1])
Nz = int(param[2])
Lx = param[3]
Ly = param[4]
Lz = param[5]
dx = param[6]
dy = param[7]
dz = param[8]
mpiProc = int(param[11])
nzTot = Nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

cases = ['Flat','Sinusoidal','ATTO']
# cases_2 = ['flat','sin','atto']
ustar = dict()

# for i in range(len(cases)):

#     path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
#     data = xr.open_dataarray(path_to_data + cases[i] + '/dataTKE.nc')
#     terms_bdg = xr.open_dataarray(path_to_data + cases[i] + '/TKE_terms.nc')
    
#     from functions import build_phi, build_intf
    
#     phi = build_phi(path_to_data + cases[i] + '/phi_functions/', Nx, Ny, Nz, mpiProc)
#     intf, iintf = build_intf(phi, dz)
    
#     z_profile = np.arange(Nz) * dz * zi
#     zeds = np.ones((Nx, Ny, 1)) * z_profile
#     dist = copy.deepcopy(zeds)
#     del zeds,z_profile
#     dist -= intf[:, :, np.newaxis] * zi
#     mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
#     mask4D = np.expand_dims(mask, axis=-1)
#     data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
#     terms_bdg.data[:, :, :, :] = np.where(mask4D, np.nan, terms_bdg.data[:, :, :, :])
#     del mask,mask4D
    
#     ustar_tmp = np.zeros((Nx,Ny),order='F')
    
#     # Pre-extract needed variable indices
#     idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
#     idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24
#     idx_P, idx_D = 14, 11
    
#     z_start = np.argmax(dist > 0, axis=2) - 5
#     z_idx_ustar = z_start + 16
    
#     ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    
#     Ruw = (data.data[ii, jj, z_idx_ustar, idx_uw]
#             - data.data[ii, jj, z_idx_ustar, idx_u] * data.data[ii, jj, z_idx_ustar, idx_w]
#             - data.data[ii, jj, z_idx_ustar, idx_txz])
    
#     Rvw = (data.data[ii, jj, z_idx_ustar, idx_vw]
#             - data.data[ii, jj, z_idx_ustar, idx_v] * data.data[ii, jj, z_idx_ustar, idx_w]
#             - data.data[ii, jj, z_idx_ustar, idx_tyz])
    
#     ustar_tmp = (Ruw**2 + Rvw**2)**0.25
#     ustar[cases[i]] = ustar_tmp

# Assuming you have a function build_phi to load phi data from a file
phi_flat = build_phi(tpath_flat + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_flat, iintf_flat = build_intf(phi_flat, dz)
phi_sin = build_phi(tpath_sin + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_sin, iintf_sin = build_intf(phi_sin, dz)
phi_atto = build_phi(tpath_atto + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_atto, iintf_atto = build_intf(phi_atto, dz)

intf = {
        'Flat' : intf_flat,
        'Sinusoidal' : intf_sin,
        'ATTO' : intf_atto
        }

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
Nz_SLayer = 300 #int(Nz/2)

zeds = np.zeros((Nx,Ny,Nz))

for i in range(0,Nx):
    for j in range(0,Ny):
        zeds[i,j,:] = np.arange(0,Nz)*(dz*zi)
    
dist_flat = copy.deepcopy(zeds); dist_sin = copy.deepcopy(zeds); dist_atto = copy.deepcopy(zeds)

for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz):
            dist_flat[i,j,k] = dist_flat[i,j,k] - intf_flat[i,j]*zi
            dist_sin[i,j,k] = dist_sin[i,j,k] - intf_sin[i,j]*zi
            dist_atto[i,j,k] = dist_atto[i,j,k] - intf_atto[i,j]*zi

dist = {
        'Flat' : dist_flat,
        'Sinusoidal' : dist_sin,
        'ATTO' : dist_atto
        }    

#%%Load anisotropy data
            
Aniso_data_flat = xr.open_dataarray(path_flat + 'Anisotropy_clustering_'+flat+'_v2.nc')
Aniso_data_sin = xr.open_dataarray(path_sin + 'Anisotropy_clustering_'+sin+'_v2.nc')
Aniso_data_atto = xr.open_dataarray(path_atto + 'Anisotropy_clustering_'+atto+'_v2.nc')

aniso_flat = dict(); aniso_sin = dict(); aniso_atto = dict()
aniso_var = ['xB','yB','type']

for i in range(len(aniso_var)):
    aniso_flat[aniso_var[i]] = np.reshape(np.copy(Aniso_data_flat.data[:,i]),(Nx,Ny,Nz)) 
    aniso_sin[aniso_var[i]] = np.reshape(np.copy(Aniso_data_sin.data[:,i]),(Nx,Ny,Nz)) 
    aniso_atto[aniso_var[i]] = np.reshape(np.copy(Aniso_data_atto.data[:,i]),(Nx,Ny,Nz)) 

for i in range(len(aniso_var)):
    aniso_flat[aniso_var[i]][(dist_flat[:,:,:]<0)] = float('nan')
    aniso_sin[aniso_var[i]][(dist_sin[:,:,:]<0)] = float('nan')
    aniso_atto[aniso_var[i]][(dist_atto[:,:,:]<0)] = float('nan')

aniso = {
    'Flat' : aniso_flat,
    'Sinusoidal' : aniso_sin,
    'ATTO' : aniso_atto
    }

#%%Select grid points where yBis roughly 0.38 and save the heights from the surface
from scipy.stats import gaussian_kde
case = 'Sinusoidal'

# yb = aniso[case]['yB'][(aniso[case]['yB'] > 0.37) & (aniso[case]['yB'] < 0.39) & (dist[case]>40) & (dist[case]<40*15)]
rsl = dist[case][(aniso[case]['yB'] > 0.37) & (aniso[case]['yB'] < 0.39) & (dist[case]>40) & (dist[case]<40*15)]/39

rsl_samp = np.random.choice(rsl,size=200000,replace=False)

kde = gaussian_kde(rsl_samp)
x_pdf = np.linspace(min(rsl_samp),max(rsl_samp),1000)
pdf = kde(x_pdf)
    
fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.hist2d(yb,rsl,bins=[100,100])
# axs[0].hist(uu_samp,bins=100,density=True,alpha=0.4)
axs.plot(x_pdf,pdf,'r-')

plt.show()


#%%select tower coordinates

Ntwr = 100

# coord_f = find_coordinates(intf,Ntwr,'flat')
coord_p = find_coordinates(intf[case],Ntwr,'max')
coord_v = find_coordinates(intf[case],Ntwr,'min')

yb_p = np.zeros((Ntwr,Nz_SLayer),'d',order='F')
yb_v = np.zeros((Ntwr,Nz_SLayer),'d',order='F')

for k, (ix, iy) in enumerate(coord_p):
    z_start = np.argmax(dist[case][ix, iy, :] > 0)
        
    loc = coord_p[k]
    
    yb_p[k,:] = aniso[case]['yB'][ix,iy,z_start:z_start+Nz_SLayer]

for k, (ix, iy) in enumerate(coord_v):
    z_start = np.argmax(dist[case][ix, iy, :] > 0)
        
    loc = coord_v[k]
    
    yb_v[k,:] = aniso[case]['yB'][ix,iy,z_start:z_start+Nz_SLayer]

#%%Plot the profiles

fig,axs = plt.subplots(1,1,tight_layout=True)

# for i in range(0,Ntwr):
#     axs.plot(yb_v[i,:],np.arange(0,Nz_SLayer)*dz)
    
axs.plot(np.mean(yb_v,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi)+dz/2/(39/zi))
axs.axvline(0.38,c='k',ls='--')
plt.show()



























































