#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:05:47 2024

@author: u1450851
"""

#%%

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
cases_2 = ['flat','sin','atto']
ustar = dict()

for i in range(len(cases)):

    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + cases[i] + '/dataTKE.nc')
    terms_bdg = xr.open_dataarray(path_to_data + cases[i] + '/TKE_terms.nc')
    
    from functions import build_phi, build_intf
    
    phi = build_phi(path_to_data + cases[i] + '/phi_functions/', Nx, Ny, Nz, mpiProc)
    intf, iintf = build_intf(phi, dz)
    
    z_profile = np.arange(Nz) * dz * zi
    zeds = np.ones((Nx, Ny, 1)) * z_profile
    dist = copy.deepcopy(zeds)
    del zeds,z_profile
    dist -= intf[:, :, np.newaxis] * zi
    mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
    mask4D = np.expand_dims(mask, axis=-1)
    data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
    terms_bdg.data[:, :, :, :] = np.where(mask4D, np.nan, terms_bdg.data[:, :, :, :])
    del mask,mask4D
    
    ustar_tmp = np.zeros((Nx,Ny),order='F')
    
    # Pre-extract needed variable indices
    idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
    idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24
    idx_P, idx_D = 14, 11
    
    z_start = np.argmax(dist > 0, axis=2) - 5
    z_idx_ustar = z_start + 16
    
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    
    Ruw = (data.data[ii, jj, z_idx_ustar, idx_uw]
            - data.data[ii, jj, z_idx_ustar, idx_u] * data.data[ii, jj, z_idx_ustar, idx_w]
            - data.data[ii, jj, z_idx_ustar, idx_txz])
    
    Rvw = (data.data[ii, jj, z_idx_ustar, idx_vw]
            - data.data[ii, jj, z_idx_ustar, idx_v] * data.data[ii, jj, z_idx_ustar, idx_w]
            - data.data[ii, jj, z_idx_ustar, idx_tyz])
    
    ustar_tmp = (Ruw**2 + Rvw**2)**0.25
    ustar[cases_2[i]] = ustar_tmp

# Assuming you have a function build_phi to load phi data from a file
phi_flat = build_phi(tpath_flat + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_flat, iintf_flat = build_intf(phi_flat, dz)
phi_sin = build_phi(tpath_sin + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_sin, iintf_sin = build_intf(phi_sin, dz)
phi_atto = build_phi(tpath_atto + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_atto, iintf_atto = build_intf(phi_atto, dz)

intf = {
        'flat' : intf_flat,
        'sin' : intf_sin,
        'atto' : intf_atto
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
        'flat' : dist_flat,
        'sin' : dist_sin,
        'atto' : dist_atto
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
    'flat' : aniso_flat,
    'sin' : aniso_sin,
    'atto' : aniso_atto
    }

#%%Load TKE data

TKE_flat = np.load(path_flat + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_sin = np.load(path_sin + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_atto = np.load(path_atto + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()

TKE = {
       'flat' : TKE_flat,
       'sin' : TKE_sin,
       'atto' : TKE_atto
       }

#%%Plot the composite plot of TKE res, Yb 

from matplotlib.transforms import ScaledTranslation
import matplotlib.gridspec as gridspec
cases = ['flat','sin','atto']
yslice = [100,150,63]
levels=[-100,-1,1,100]
levels_2 = [0.36,0.4]
# levels_3 = [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11]
levels_3 = [-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1]
levels_yb = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
colors=['blue','white','red']
labels = ["a1)", "a2)", "a3)", "b1)", "b2)",'b3)','c1)','c2)','c3)','d1)','d2)','d3)','e1)','e2)','e3)']  # Subplot labels

fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(6, 3, height_ratios=[1,1,1,0.1,1,1], figure=fig)

axs = [[None]*3 for _ in range(6)]

# Row 1 Flat
for i in range(3):
    axs[0][i] = fig.add_subplot(gs[0, i], sharey=axs[0][0] if i > 0 else None)

# Row 2 Sinusoidal
for i in range(3):
    axs[1][i] = fig.add_subplot(gs[1, i], sharey=axs[1][0] if i > 0 else None)

# Row 3 ATTO
for i in range(3):
    axs[2][i] = fig.add_subplot(gs[2, i], sharey=axs[2][0] if i > 0 else None)

# Row 4 g800
for i in range(3):
    axs[4][i] = fig.add_subplot(gs[4, i], sharey=axs[4][0] if i > 0 else None)
    
# Row 5 i800
for i in range(3):
    axs[5][i] = fig.add_subplot(gs[5, i], sharey=axs[5][0] if i > 0 else None)

for i in range(0,3):
    tmpRES = copy.deepcopy((TKE[cases[i]]['prod'] - TKE[cases[i]]['totdis'])[:,:,5:]*((39/zi)/ustar[cases[i]][:,:,np.newaxis]**3))
    tmpRES[(dist[cases[i]][:,:,5:]<0)] = float('nan')
    # p1 = axs[0,i].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmpRES[:,yslice[i],:]).T,cmap='bwr',alpha=1,vmin=-10,vmax=10)
    p1 = axs[i][0].contourf(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmpRES[:,yslice[i],:]).T,cmap='bwr',levels=levels_3,extend='both')
    p1.cmap.set_under('blue')
    p1.cmap.set_over('red')
    axs[i][0].plot(np.arange(0,Nx)*dx,(intf[cases[i]][:,yslice[i]]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    axs[i][0].plot(np.arange(0,Nx)*dx,(intf[cases[i]][:,yslice[i]]-z_shift)/(39/zi),ls='-',c='k')
    
    axs[i][1].plot(np.arange(0,Nx)*dx,(intf[cases[i]][:,yslice[i]]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    axs[i][1].plot(np.arange(0,Nx)*dx,(intf[cases[i]][:,yslice[i]]-z_shift)/(39/zi),ls='-',c='k')
    sc1 = axs[i][1].contour(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(aniso[cases[i]]['yB'][:,yslice[i],5:]).T,levels=levels_2,colors=['black'])
    # axs[i][1].contour(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(aniso[cases[i]]['yB'][:,yslice[i],5:]).T,levels=[0.36,0.4],colors=['black'])
    sc2 = axs[i][1].contourf(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),np.transpose(aniso[cases[i]]['yB'][:,yslice[i],5:]), levels=levels_yb, cmap = cmap, vmin = 0, vmax = np.sqrt(3)/2)

    tmpRES = copy.deepcopy((TKE[cases[i]]['prod'] - TKE[cases[i]]['totdis'])[:,:,5:])
    tmpDIS = copy.deepcopy((TKE[cases[i]]['totdis'])[:,:,5:])
    tmpRES[(abs(tmpRES)<1)] = 0
    tmpNorm = (tmpRES)/abs(tmpDIS)
    p2 = axs[i][2].contourf(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmpNorm[:,yslice[i],:]*100).T,colors=colors,alpha=0.5,levels=levels,extend='both')#vmin=-10,vmax=10)
    p2.cmap.set_under('blue')
    p2.cmap.set_over('red')
    sc = axs[i][2].contour(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(aniso[cases[i]]['yB'][:,yslice[i],5:]).T,levels=levels_2,colors=['black'])
    axs[i][2].plot(np.arange(0,Nx)*dx,(intf[cases[i]][:,yslice[i]]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    axs[i][2].plot(np.arange(0,Nx)*dx,(intf[cases[i]][:,yslice[i]]-z_shift)/(39/zi),ls='-',c='k')

cases = ['Gap_8_9mps','Patch_8_9mps']
Nx_G = 256
Nz_G = 256
Lx_G = 2*np.pi
Lz_G = 1
dx_G = Lx_G/Nx_G
dz_G = Lz_G/Nz_G
x_G = np.arange(0,Nx_G)*dx_G
z_w_G = np.arange(0,Nz_G)*dz_G
z_uvp_G = np.arange(0,Nz_G)*dz_G + dz_G/2
zi_G = 1000
Hcanopy_G = 39/zi_G

for i in range(len(cases)):
    path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path+cases[i]+'/Data_Momentum_4TKE.nc')
    terms_ptb = xr.open_dataarray(path+cases[i]+'/terms_ptb.nc')
    terms_bdg = xr.open_dataarray(path+cases[i] + '/TKE_terms.nc')
    anisotropy = xr.open_dataarray(path+cases[i] + '/anisotropy.nc')
    Res = (terms_bdg.data[:,:,:,-1]+terms_bdg.data[:,:,:,11])#*(Hcanopy)/(ustar[:,:,np.newaxis]**3)
    # ResNorm[(abs(ResNorm) < 50)] = 0

    T_13 = wnode2uvpnode(((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23])))
    T_23 = wnode2uvpnode(((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24])))

    cov_turb = -np.sqrt((T_13)**2 + (T_23)**2)
    ustar_G = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height

    del T_13,T_23,cov_turb

    ResNorm = Res*(Hcanopy_G)/(ustar_G[:,:,np.newaxis]**3)
    
    p1 = axs[i+4][0].contourf(x_G,z_uvp_G/Hcanopy_G,ResNorm[:,190,:].T,cmap='bwr',levels=levels_3,extend='both')
    p1.cmap.set_under('blue')
    p1.cmap.set_over('red')
    axs[i+4][0].axhline(0,ls='-',color='k')
    axs[i+4][0].axhline(1,ls='--',color='k')
    
    axs[i+4][1].axhline(0,ls='-',color='k')
    axs[i+4][1].axhline(1,ls='--',color='k')
    sc1 = axs[i+4][1].contour(x_G,z_uvp_G/Hcanopy_G,anisotropy[:,190,:,1].T,levels=levels_2,colors=['black'])
    sc2 = axs[i+4][1].contourf(x_G,z_uvp_G/Hcanopy_G,anisotropy[:,190,:,1].T,cmap = cmap, levels=levels_yb , vmin = 0, vmax = np.sqrt(3)/2)
    
    tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
    tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])
    # tmpRES = np.where(np.abs(ResNorm_v2) < 0.14, 0, tmpRES)
    tmpRES = np.where(np.abs(Res) < 20, 0, tmpRES)

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
    
    p2 = axs[i+4][2].contourf(x_G,z_uvp_G/Hcanopy_G,tmpNorm[:,190,:].T,colors=colors,alpha=0.5,levels=levels,extend='both')#vmin=-10,vmax=10)
    p2.cmap.set_under('blue')
    p2.cmap.set_over('red')
    sc = axs[i+4][2].contour(x_G,z_uvp_G/Hcanopy_G,anisotropy[:,190,:,1].T,levels=levels_2,colors=['black'])
    axs[i+4][2].axhline(0,ls='-',color='k')
    axs[i+4][2].axhline(1,ls='--',color='k')
    
cbar_ax = fig.add_axes([0.083, 0.07, 0.28, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p1, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$R\cdot\frac{h_C}{u_{*}^{3}}$ ", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size
# cbar1.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1])
# cbar1.set_ticklabels(['-1', '-0.8', '-0.6', '-0.4', '-0.2', '0.2', '0.4', '0.6', '0.8', '1'])
cbar1.set_ticks([-0.9, -0.6, -0.3, 0.3, 0.6, 0.9])
cbar1.set_ticklabels(['-0.9', '-0.6', '-0.3', '0.3', '0.6', '0.9'])

cbar_ax = fig.add_axes([0.39, 0.07, 0.28, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(sc2, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$yB$", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size

cbar_ax = fig.add_axes([0.70, 0.07, 0.28, 0.01])  # Position and size of the colorbar
cbar1 = fig.colorbar(p2, cax=cbar_ax, orientation="horizontal")
cbar1.set_label(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')  # Add label to colorbar if needed
cbar1.ax.tick_params(labelsize=9)  # Set colorbar tick label size

for row in range(3):
    for col in range(1, 3):  # columns 1 and 2
        axs[row][col].tick_params(labelleft=False)  # hide tick labels
        axs[row][col].set_ylabel("")  
for row in range(4,6):
    for col in range(1, 3):  # columns 1 and 2
        axs[row][col].tick_params(labelleft=False)  # hide tick labels
        axs[row][col].set_ylabel("")  
    
for i in range(3):
    axs[0][i].tick_params(labelbottom=False)  # hide tick labels
    axs[1][i].tick_params(labelbottom=False)  # hide tick labels
    axs[4][i].tick_params(labelbottom=False)  # hide tick labels
    axs[0][i].set_ylim(0,20)
    axs[1][i].set_ylim(0,20)
    axs[2][i].set_ylim(0,20)
    axs[4][i].set_ylim(0,20)
    axs[5][i].set_ylim(0,20)

for i in range(3):
    axs[i][0].set_ylabel(r"$z/h_C$", fontsize=14)
for i in range(4,6):
    axs[i][0].set_ylabel(r"$z/h_C$", fontsize=14)

for j in range(3):
    axs[2][j].set_xlabel(r"$x/z_i$", fontsize=14)
    axs[5][j].set_xlabel(r"$x/z_i$", fontsize=14)

plt.subplots_adjust(left=0.08,bottom=0.15,right=0.98,top=0.96,wspace=0.08,hspace=0.4)

l = 0
for m in range(3):
    for n in range(3):
        axs[m][n].text(0.0, 1.0, labels[l],transform=axs[m][n].transAxes + ScaledTranslation(-5/72, +5/72, fig.dpi_scale_trans), fontsize=12, \
                       va="bottom", ha="left", fontfamily="serif")
        l += 1
for m in range(4,6):
    for n in range(3):
        axs[m][n].text(0.0, 1.0, labels[l],transform=axs[m][n].transAxes + ScaledTranslation(-5/72, +5/72, fig.dpi_scale_trans), fontsize=12, \
                       va="bottom", ha="left", fontfamily="serif")
        l += 1
        
# plt.savefig(pathFig + 'ResYB_Combo_All.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()










































# %%
