#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:40:18 2025

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
height = math.ceil(canopyH/dz)

# Build normalized axes
x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

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
        'sin_min' : dist_sin,
        'atto_min' : dist_atto,
        'sin_max' : dist_sin,
        'atto_max' : dist_atto
        }

#%%Import data

import pickle

# var = ['u','v','w','uu','vv','ww','uv','uw','vw']

var = ['u','v','w','uu','uv','uw','vv','vw','ww','txx','txy','txz','tyy','tyz',\
       'tzz','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz']

with open(path_flat+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_flat = dict()
for i in range(len(var)):
    data_flat[var[i]] = np.mean(data[var[i]],axis=3)

with open(path_sin+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_sin = dict()
for i in range(len(var)):
    data_sin[var[i]] = np.mean(data[var[i]],axis=3)

with open(path_atto+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_atto = dict()
for i in range(len(var)):
    data_atto[var[i]] = np.mean(data[var[i]],axis=3)

velocity = {
    'flat':data_flat,
    'sin_min':data_sin,
    'sin_max':data_sin,
    'atto_min':data_atto,
    'atto_max':data_atto
    }

del data, data_flat, data_sin, data_atto

#%%Select coordinates
import random
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex

N_twrs = 100
Nz_SLayer = 300
LAD = [0.15655190, 0.20633190, 0.24492203, 0.28024144, 0.33267326, 0.33145316, 0.32065714, 0.28729650, 0.25240169, 0.17358901, 0.11740349, 0.064294815, \
    0.041340224, 0.023756023, 0.013134912, 0.013107183]

coord_atto_max = find_coordinates(intf['atto'],N_twrs,'max') #min,max
d_dim_atto_max = compute_d_twr(velocity['atto_max'],coord_atto_max,dist['atto_max'],height,dz,zi,u_scale,LAD,'ATTO')
coord_sin_max = find_coordinates(intf['sin'],N_twrs,'max') #min,max
d_dim_sin_max = compute_d_twr(velocity['sin_max'],coord_sin_max,dist['sin_max'],height,dz,zi,u_scale,LAD,'ATTO')
coord_atto_min = find_coordinates(intf['atto'],N_twrs,'min') #min,max
d_dim_atto_min = compute_d_twr(velocity['atto_min'],coord_atto_max,dist['atto_min'],height,dz,zi,u_scale,LAD,'ATTO')
coord_sin_min = find_coordinates(intf['sin'],N_twrs,'min') #min,max
d_dim_sin_min = compute_d_twr(velocity['sin_min'],coord_atto_max,dist['sin_min'],height,dz,zi,u_scale,LAD,'ATTO')
coord_flat = []
for i in range(0,N_twrs):
    coord_flat.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))
d_dim_flat = compute_d_twr(velocity['flat'],coord_flat,dist['flat'],height,dz,zi,u_scale,LAD,'ATTO')

coord = {
    'flat' : coord_flat,
    'sin_min' : coord_sin_min,
    'atto_min' : coord_atto_min,
    'sin_max' : coord_sin_max,
    'atto_max' : coord_atto_max
    }

d_dim = {
    'flat' : d_dim_flat,
    'sin_min' : d_dim_sin_min,
    'atto_min' : d_dim_atto_min,
    'sin_max' : d_dim_sin_max,
    'atto_max' : d_dim_atto_max
    }

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
  phi_c[:, :, -1] = phi_c[:, :, -2]
  return phi_c

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
    'sin_min' : aniso_sin,
    'atto_min' : aniso_atto,
    'sin_max' : aniso_sin,
    'atto_max' : aniso_atto
    }

del Aniso_data_flat, Aniso_data_sin, Aniso_data_atto, aniso_flat, aniso_sin, aniso_atto

#%%Load TKE data

TKE_flat = np.load(path_flat + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_sin = np.load(path_sin + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()
TKE_atto = np.load(path_atto + 'TKE_terms_v2.npy',allow_pickle='TRUE').item()

TKE = {
       'flat' : TKE_flat,
       'sin_min' : TKE_sin,
       'atto_min' : TKE_atto,
       'sin_max' : TKE_sin,
       'atto_max' : TKE_atto
       }

cases = ['flat','sin_min','sin_max','atto_min','atto_max']

TKE['flat']['totdis'] = - TKE['flat']['totdis']
TKE['sin_min']['totdis'] =  TKE['sin_min']['totdis']
TKE['sin_max']['totdis'] = - TKE['sin_max']['totdis']
TKE['atto_min']['totdis'] =  TKE['atto_min']['totdis']
TKE['atto_max']['totdis'] = - TKE['atto_max']['totdis']

TKE['flat']['res'] = TKE['flat']['prod']+TKE['flat']['totdis']
TKE['sin_min']['res'] = TKE['sin_min']['prod']+TKE['sin_min']['totdis']
TKE['sin_max']['res'] = TKE['sin_max']['prod']+TKE['sin_max']['totdis']
TKE['atto_min']['res'] = TKE['atto_min']['prod']+TKE['atto_min']['totdis']
TKE['atto_max']['res'] = TKE['atto_max']['prod']+TKE['atto_max']['totdis']

del TKE_flat, TKE_sin, TKE_atto

#%%
import matplotlib.gridspec as gridspec

lengths = [r'$L_s$',r'$L_\epsilon$',r'$L_P$']
cases = ['flat','sin_min','sin_max','atto_min','atto_max']
labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]  # Subplot labels
yB_cross = {
    'flat' : [3,5],
    'sin_min' : [4,5],
    'atto_min' : [10,12],
    'sin_max' : [2],
    'atto_max' : [3,8]
    }
# markers = ['o','v','s','*','X','+']
# colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
positions = [[0.33, 0.57, 0.2, 0.38], [0.61, 0.57, 0.2, 0.38], 
             [0.33, 0.08, 0.2, 0.38], [0.61, 0.08, 0.2, 0.38]]

fig, axs = plt.subplots(2, 3, figsize=(12, 12))#, gridspec_kw={'width_ratios': [1, 1, 1]})
gs = gridspec.GridSpec(2, 3, width_ratios=[1.6, 1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.3)
axs = axs.flatten()
legend_lines, legend_labels = None, None

for j in range(len(cases)):
    
    # tmpLs_U = np.zeros((N_twrs,Nz_SLayer),'d',order='F'); tmpLs_V = np.zeros((N_twrs,Nz_SLayer),'d',order='F'); tmpLs_W = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
    tmpUmag = np.sqrt(velocity[cases[j]]['u']**2 + velocity[cases[j]]['v']**2)*u_scale
    tmpDUDZ = ((velocity[cases[j]]['u']*velocity[cases[j]]['dudz'] + velocity[cases[j]]['v']*velocity[cases[j]]['dvdz'])/\
                np.sqrt(velocity[cases[j]]['u']**2 + velocity[cases[j]]['v']**2))*(u_scale/zi)
    # tmpDUDZ = wnode2uvpnode(get_dphidz(tmpUmag, dz))
    tmpLs_mag = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
    
    tmpLe = np.zeros((N_twrs,Nz_SLayer),'d',order='F');ustar = np.zeros((N_twrs,Nz_SLayer),'d',order='F');
    tmpDis = abs(TKE[cases[j]]['dissip'])*((u_scale**3)/zi)
    
    tmpLp = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
    tmpProd = abs(TKE[cases[j]]['prod'])*((u_scale**3)/zi)
    
    tmpYb = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
    
    Ruw = (velocity[cases[j]]['uw'] - velocity[cases[j]]['u']*velocity[cases[j]]['w'] - velocity[cases[j]]['txz'])*(u_scale**2)
    Rvw = (velocity[cases[j]]['vw'] - velocity[cases[j]]['v']*velocity[cases[j]]['w'] - velocity[cases[j]]['tyz'])*(u_scale**2)
    
    lines = []
    
    for k in range(len(coord[cases[j]])):
        loc = coord[cases[j]][k]
        
        # tmpLs_U[k,:] = velocity[cases[j]]['u'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
        #                                       int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]/\
        #     velocity[cases[j]]['dudz'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
        #                                           int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        # tmpLs_V[k,:] = velocity[cases[j]]['v'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
        #                                       int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]/\
        #     velocity[cases[j]]['dvdz'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
        #                                           int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        # tmpLs_W[k,:] = velocity[cases[j]]['w'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
        #                                       int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]/\
        #     velocity[cases[j]]['dwdz'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
        #                                           int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        tmpLs_mag[k,:] = (tmpUmag[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16]/\
                          tmpDUDZ[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])/\
                                                    (0.41*((np.arange(0,Nz_SLayer)*dz*zi + dz*zi/2)-d_dim[cases[j]][k]))
                
        ustar[k,:] = ((Ruw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                              int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])**2 + \
                        (Rvw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                             int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])**2)**(1/4)
        tmpLe[k,:] = ((ustar[k,:]**(3))/\
                        tmpDis[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                  int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])/\
                                                    (0.41*((np.arange(0,Nz_SLayer)*dz*zi + dz*zi/2)-d_dim[cases[j]][k]))
                            
        tmpLp[k,:] = (ustar[k,:]**(3))/\
                        tmpProd[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                  int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]/\
                                                    (0.41*((np.arange(0,Nz_SLayer)*dz*zi + dz*zi/2)-d_dim[cases[j]][k]))
                            
        tmpYb[k,:] = aniso[cases[j]]['yB'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                  int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
                
    line, = axs[j].plot(np.nanmean(tmpLs_mag,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),linewidth=3,c='b')
    lines.append(line)
    line, = axs[j].plot(np.nanmean(tmpLe,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),linewidth=3,c='g')
    lines.append(line)
    line, = axs[j].plot(np.nanmean(tmpLp,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),linewidth=3,c='r')
    lines.append(line)
    
    # axs[j].plot(np.nanmean(tmpYb,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),linewidth=1,c='k')
    
    axs[j].axvline(0,c='k')
    axs[j].axvline(1,c='k',ls='--')
    for val in range(len(yB_cross[cases[j]])):
        axs[j].axhline(yB_cross[cases[j]][val],c='k',ls='-.')
    # axs[j].axvline(0.36,c='k',ls='--')
    # axs[j].axvline(0.38,c='k',ls='--')
    
    # axs.plot(np.nanmean(tmpLs_U,axis=(0))*zi/39,np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='Ls_U',linewidth=3,c='k')
    # axs.plot(np.nanmean(tmpLs_V,axis=(0))*zi/39,np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='Ls_V',linewidth=3,c='g')
    # axs.plot(np.nanmean(tmpLs_W,axis=(0))*zi/39,np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='Ls_W',linewidth=3,c='r')
    
    if j==0:
        axs[j].set_position([0.06, 0.32, 0.2, 0.38])
    else:
        axs[j].set_position(positions[j-1])
    
    axs[j].set_xlabel('Length Scale / ' + r'$k(z-d_0)$',fontsize=15)
    axs[j].text(0.12, 0.98, labels[j], transform=axs[j].transAxes, fontsize=12, va="top", ha="right")
    axs[j].set_ylabel(r'$z/h_C$',fontsize=15)
    axs[j].set_ylim(1,12)
    axs[j].set_xlim(0,8)
    axs[j].grid()
    axs[j].tick_params(axis='both', which='major', labelsize=12)
    axs[j].axhline(1,ls='--',c='k')
    # axs[j].legend(loc='upper right',fontsize=12)
    axs[j].set_title(f'{cases[j]}', fontsize=15)
    # plt.savefig(pathFig+f'{cases[j]}_TKEprofTWR_ustar.png',dpi=300,facecolor='None', edgecolor='None')

axs[-1].legend(lines, lengths, loc="center", fontsize=15)
axs[-1].set_position([0.8,0.42,0.2,0.2])
axs[-1].axis("off")
axs[2].set_yticklabels([]), axs[4].set_yticklabels([])
axs[2].set_ylabel(''), axs[4].set_ylabel('')
axs[1].set_xlabel(''), axs[2].set_xlabel('')
# plt.savefig(pathFig+f'LengthScales_prof_tij.png',dpi=300,facecolor='None', edgecolor='None')
# fig.delaxes(axs[-1])
plt.show()


#%%

fig,axs = plt.subplots(1,1)
axs.plot(np.mean(tmpDis,axis=0),np.arange(0,Nz_SLayer)*dz)
plt.show()




































