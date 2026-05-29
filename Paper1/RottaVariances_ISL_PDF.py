#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:50:37 2025

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
    ustar[cases[i]] = ustar_tmp

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

#%%Load TKE data

TKE_flat = np.load(path_flat + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_sin = np.load(path_sin + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_atto = np.load(path_atto + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()

TKE = {
       'Flat' : TKE_flat,
       'Sinusoidal' : TKE_sin,
       'ATTO' : TKE_atto
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


#%%Compute and plot the residual
from matplotlib.transforms import ScaledTranslation

colors = ['#0072B2','#E69F00','#CC79A7']
layout = [['a)', 'b)'],
          ['c)', 'd)']]

fig, axs_dict = plt.subplot_mosaic(layout, layout='constrained',figsize=(6,6),sharey=True)

for label, ax in axs_dict.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-5/72, +7/72, fig.dpi_scale_trans)),
        fontsize=15, va='bottom', fontfamily='serif')

from scipy.stats import gaussian_kde
# fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(12,3))
# axs = np.array([axs_dict[label] for label in layout[0]])
axs = np.array([[axs_dict[label] for label in row] for row in layout])

# for j in range(axs.shape[1]):        # loop over columns
#     for i in range(1, axs.shape[0]): # loop over rows in that column
#         axs[i, j].sharex(axs[0, j])  # share with the top axis in that column
        
# # Hide inner labels
# for ax in axs.ravel():
#     ax.label_outer()

for i in range(len(cases)):
    case = cases[i]
    # case = 'atto'
    levels = [-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1]
    
    # tmpRES = copy.deepcopy((TKE[case]['prod'] - TKE[case]['totdis'])[:,:,5:]*((39/zi)/ustar[case][:,:,np.newaxis]**3))
    # tmpRES[(dist[case][:,:,5:]<0)] = float('nan')
    
    # tmpRES = copy.deepcopy((TKE[case]['prod'] - TKE[case]['totdis'])[:,:,5:])
    # tmpDIS = copy.deepcopy((TKE[case]['totdis'])[:,:,5:])
    # tmpRES[(abs(tmpRES)<1)] = 0
    # tmpNorm = (tmpRES)/abs(tmpDIS)*100
    
    # fig,axs = plt.subplots(1,1,tight_layout=True)
    
    # p1 = axs.contourf(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmpNorm[:,150,:]).T,cmap='bwr',levels=levels,extend='both')
    # p1.cmap.set_under('blue')
    # p1.cmap.set_over('red')
    # axs.plot(np.arange(0,Nx)*dx,(intf[case][:,150]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    # axs.plot(np.arange(0,Nx)*dx,(intf[case][:,150]-z_shift)/(39/zi),ls='-',c='k')
    
    # cbar = plt.colorbar(p1)
    
    # plt.show()
    
    #%Compute velocity variance
    
    # case = 'ATTO'
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + case + '/dataTKE.nc')
    
    uu = data.data[:,:,:,4] - data.data[:,:,:,0]*data.data[:,:,:,0] #+ data.data[:,:,:,19]
    vv = data.data[:,:,:,5] - data.data[:,:,:,1]*data.data[:,:,:,1] #+ data.data[:,:,:,20]
    ww = data.data[:,:,:,6] - data.data[:,:,:,2]*data.data[:,:,:,2] #+ data.data[:,:,:,21]
    uw = abs(data.data[:,:,:,8] - data.data[:,:,:,0]*data.data[:,:,:,2]) #+ data.data[:,:,:,23])
    
    tke = 0.5*(uu + vv + ww)
    
    uu_e = uu/tke
    vv_e = vv/tke
    ww_e = ww/tke
    uw_e = uw/tke
    
    #%Plot the variances
    
    # fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(12,4),sharey=True)
    
    # p1 = axs[0].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/canopyH,uu_e[:,150,:].T,cmap='bwr', vmin = 0, vmax = 2)
    # p2 = axs[1].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/canopyH,vv_e[:,150,:].T,cmap='bwr',vmin = 0, vmax = 2)
    # p3 = axs[2].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/canopyH,ww_e[:,150,:].T,cmap='bwr', vmin = 0, vmax = 2)
    
    # axs[0].set_title(r"$\frac{\overline{u'u'}}{\overline{e}}$",fontsize=15)
    # axs[1].set_title(r"$\frac{\overline{v'v'}}{\overline{e}}$",fontsize=15)
    # axs[2].set_title(r"$\frac{\overline{w'w'}}{\overline{e}}$",fontsize=15)
    
    # axs[0].set_ylabel(r"$z/h_c$",fontsize=14)
    
    # for i in range(len(axs)):
    #     axs[i].set_xlabel(r"$x/z_i$",fontsize=14)
    
    # cbar = plt.colorbar(p1)
    # cbar = plt.colorbar(p2)
    # cbar = plt.colorbar(p3)
    
    # plt.show()
    
    #%Select a subset of the variances data seta where P-D = 0
    
    # uu_filt = uu_e[(abs(tmpNorm)<10) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    # vv_filt = vv_e[(abs(tmpNorm)<10) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    # ww_filt = ww_e[(abs(tmpNorm)<10) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    # uw_filt = uw_e[(abs(tmpNorm)<10) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    
    # case = 'atto'
    
    uu_filt = uu_e[(aniso[case]['yB'][:,:,5:]<0.39) & (aniso[case]['yB'][:,:,5:]>0.37) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    vv_filt = vv_e[(aniso[case]['yB'][:,:,5:]<0.39) & (aniso[case]['yB'][:,:,5:]>0.37) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    ww_filt = ww_e[(aniso[case]['yB'][:,:,5:]<0.39) & (aniso[case]['yB'][:,:,5:]>0.37) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    uw_filt = uw_e[(aniso[case]['yB'][:,:,5:]<0.39) & (aniso[case]['yB'][:,:,5:]>0.37) & (dist[case][:,:,5:]>40) & (dist[case][:,:,5:]<40*15)]
    
    nsamples = 200000
    
    uu_samp = np.random.choice(uu_filt,size=nsamples,replace=False)
    vv_samp = np.random.choice(vv_filt,size=nsamples,replace=False)
    ww_samp = np.random.choice(ww_filt,size=nsamples,replace=False)
    uw_samp = np.random.choice(uw_filt,size=nsamples,replace=False)
    
    #%Plot PDF of the variance values in the region where P-D=0
    
    # uu_r = 2/3*(1/0.9 + 1)
    # vv_r = 2/3*(-1/(2*0.9) + 1)
    # ww_r = 2/3*(-1/(2*0.9) + 1)
    
    # from scipy.stats import gaussian_kde
    # fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(12,4))
    
    kde = gaussian_kde(uu_samp)
    x_pdf = np.linspace(min(uu_samp),max(uu_samp),1000)
    pdf = kde(x_pdf)
    # axs[0].hist(uu_samp,bins=100,density=True,alpha=0.4)
    axs[0,0].plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(uu_samp)
    median_val = np.median(uu_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    # axs[0].plot([mean_val, mean_val], [0, y_mean], c=colors[i], ls='-')
    axs[0,0].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    # axs[0].axvline(np.mean(uu_samp),c=colors[i],ls='-')
    # axs[0].axvline(np.median(uu_samp),c=colors[i],ls='--')
    # axs[0].axvline(uu_r,c='k',ls='-.')
    
    kde = gaussian_kde(vv_samp)
    x_pdf = np.linspace(min(vv_samp),max(vv_samp),1000)
    pdf = kde(x_pdf)
    # axs[1].hist(vv_samp,bins=100,density=True,alpha=0.4)
    axs[0,1].plot(x_pdf,pdf,c=colors[i])
    mean_val = np.mean(vv_samp)
    median_val = np.median(vv_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    # axs[1].plot([mean_val, mean_val], [0, y_mean], c=colors[i], ls='-')
    axs[0,1].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    # axs[1].axvline(np.mean(vv_samp),c=colors[i],ls='-')
    # axs[1].axvline(np.median(vv_samp),c=colors[i],ls='--')
    # axs[1].axvline(vv_r,c='k',ls='-.')
    
    kde = gaussian_kde(ww_samp)
    x_pdf = np.linspace(min(ww_samp),max(ww_samp),1000)
    pdf = kde(x_pdf)
    # axs[2].hist(ww_samp,bins=100,density=True,alpha=0.4)
    axs[1,0].plot(x_pdf,pdf,colors[i])
    mean_val = np.mean(ww_samp)
    median_val = np.median(ww_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    # axs[2].plot([mean_val, mean_val], [0, y_mean], c=colors[i], ls='-')
    axs[1,0].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    # axs[2].axvline(np.mean(ww_samp),c=colors[i],ls='-')
    # axs[2].axvline(np.median(ww_samp),c=colors[i],ls='--')
    # axs[2].axvline(ww_r,c='k',ls='-.')
    
    kde = gaussian_kde(uw_samp)
    x_pdf = np.linspace(min(uw_samp),max(uw_samp),1000)
    pdf = kde(x_pdf)
    # axs[2].hist(ww_samp,bins=100,density=True,alpha=0.4)
    axs[1,1].plot(x_pdf,pdf,colors[i])
    mean_val = np.mean(uw_samp)
    median_val = np.median(uw_samp)
    y_mean = np.interp(mean_val, x_pdf, pdf)
    y_median = np.interp(median_val, x_pdf, pdf)
    # axs[2].plot([mean_val, mean_val], [0, y_mean], c=colors[i], ls='-')
    axs[1,1].plot([median_val, median_val], [0, y_median], c=colors[i], ls='-')
    # axs[2].axvline(np.mean(ww_samp),c=colors[i],ls='-')
    # axs[2].axvline(np.median(ww_samp),c=colors[i],ls='--')
    # axs[2].axvline(ww_r,c='k',ls='-.')

cr = [0.9,1.8]
cr_ls = [':','-.']

for i in range(len(cr)):
    uu_r = 2/3*(1/cr[i] + 1)
    vv_r = 2/3*(-1/(2*cr[i]) + 1)
    ww_r = 2/3*(-1/(2*cr[i]) + 1)
    
    axs[0,0].axvline(uu_r,c='k',ls=cr_ls[i])
    axs[0,1].axvline(vv_r,c='k',ls=cr_ls[i])
    axs[1,0].axvline(ww_r,c='k',ls=cr_ls[i])
    
axs[0,0].set_xlabel(r"$\frac{\overline{u'u'}}{\overline{e}}$",fontsize=16)
axs[0,1].set_xlabel(r"$\frac{\overline{v'v'}}{\overline{e}}$",fontsize=16)
axs[1,0].set_xlabel(r"$\frac{\overline{w'w'}}{\overline{e}}$",fontsize=16)
axs[1,1].set_xlabel(r"$\frac{|\overline{u'w'}|}{\overline{e}}$",fontsize=16)

axs[0,0].set_ylabel(r"PDF",fontsize=15)
axs[1,0].set_ylabel(r"PDF",fontsize=15)

axs[0,0].set_xlim(0.5,1.5)
axs[0,1].set_xlim(0.25,1)
axs[1,0].set_xlim(0.25,0.75)
axs[1,1].set_xlim(0,0.5)

    
for i in range(len(axs[0])):
    axs[0,i].set_ylim(0)
    axs[1,i].set_ylim(0)
    axs[0,i].tick_params(axis='both', which='major', labelsize=12)
    axs[1,i].tick_params(axis='both', which='major', labelsize=12)
    
# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/' + 'Variances_RottaModel_uw.png',dpi=300,edgecolor='white',facecolor='white')

plt.show()

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,z_uvp[:-5],uw_e[:,100,:].T,cmap='bwr',vmin=0,vmax=1.5)

cbar = plt.colorbar(p)

plt.show()

#%%A different way to visualize it, scatter plot

uu_flat = uu[(abs(tmpRES)<0.2)]
vv_flat = vv[(abs(tmpRES)<0.2)]
ww_flat = ww[(abs(tmpRES)<0.2)]
tke_flat = tke[(abs(tmpRES)<0.2)]

indx = np.random.choice(len(tke_flat),size=500000, replace=False)

x_r = np.linspace(0,3,100)
y_r_uu = uu_r*x_r
y_r_vv = vv_r*x_r
y_r_ww = ww_r*x_r

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(12,4))

axs[0].plot(x_r,y_r_uu,'k')
axs[0].scatter(tke_flat[indx],uu_flat[indx],s=0.1,alpha=0.1)
axs[0].hist2d(tke_flat[indx],uu_flat[indx],bins=50,cmap='hot_r',density=False)

axs[1].plot(x_r,y_r_vv,'k')
axs[1].scatter(tke_flat[indx],vv_flat[indx],s=0.1,alpha=0.1)
axs[1].hist2d(tke_flat[indx],vv_flat[indx],bins=50,cmap='hot_r',density=False)

axs[2].plot(x_r,y_r_ww,'k')
axs[2].scatter(tke_flat[indx],ww_flat[indx],s=0.1,alpha=0.1)
axs[2].hist2d(tke_flat[indx],ww_flat[indx],bins=50,cmap='hot_r',density=False)

for i in range(len(axs)):
    axs[i].set_xlim(0,3)
    axs[i].set_ylim(0,3)
    axs[i].set_xlabel(r"$\overline{e}$",fontsize=14)
    
axs[0].set_ylabel(r"$\overline{u'u'}$",fontsize=14)
axs[1].set_ylabel(r"$\overline{v'v'}$",fontsize=14)
axs[2].set_ylabel(r"$\overline{w'w'}$",fontsize=14)

plt.show()

#%%Compute anisotropy using a diagonalized reynolds stress tensor, only the variances

b11 = uu/(2*tke) - 1/3
b22 = vv/(2*tke) - 1/3
b33 = ww/(2*tke) - 1/3

b11_m = 1/(3*0.9)
b22_m = -1/(6*0.9)
b33_m = -1/(0.9*6)

b11_filt = np.random.choice(b11[(abs(tmpRES)<0.2)],size=100000,replace=False)
b22_filt = np.random.choice(b22[(abs(tmpRES)<0.2)],size=100000,replace=False)
b33_filt = np.random.choice(b33[(abs(tmpRES)<0.2)],size=100000,replace=False)

from scipy.stats import gaussian_kde
fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(12,4))

kde = gaussian_kde(b11_filt)
x_pdf = np.linspace(min(b11_filt),max(b11_filt),1000)
pdf = kde(x_pdf)
axs[0].hist(b11_filt,bins=100,density=True,alpha=0.4)
axs[0].plot(x_pdf,pdf,'r-')
axs[0].axvline(np.mean(b11_filt),c='r',ls='-')
axs[0].axvline(np.median(b11_filt),c='r',ls='--')
axs[0].axvline(b11_m,c='k',ls='-.')

kde = gaussian_kde(b22_filt)
x_pdf = np.linspace(min(b22_filt),max(b22_filt),1000)
pdf = kde(x_pdf)
axs[1].hist(b22_filt,bins=100,density=True,alpha=0.4)
axs[1].plot(x_pdf,pdf,'r-')
axs[1].axvline(np.mean(b22_filt),c='r',ls='-')
axs[1].axvline(np.median(b22_filt),c='r',ls='--')
axs[1].axvline(b22_m,c='k',ls='-.')

kde = gaussian_kde(b33_filt)
x_pdf = np.linspace(min(b33_filt),max(b33_filt),1000)
pdf = kde(x_pdf)
axs[2].hist(b33_filt,bins=100,density=True,alpha=0.4)
axs[2].plot(x_pdf,pdf,'r-')
axs[2].axvline(np.mean(b33_filt),c='r',ls='-')
axs[2].axvline(np.median(b33_filt),c='r',ls='--')
axs[2].axvline(b33_m,c='k',ls='-.')

# axs[0].set_xlabel(r"$\frac{\overline{u'u'}}{\overline{e}}$",fontsize=15)
# axs[1].set_xlabel(r"$\frac{\overline{v'v'}}{\overline{e}}$",fontsize=15)
# axs[2].set_xlabel(r"$\frac{\overline{w'w'}}{\overline{e}}$",fontsize=15)

plt.show()











































