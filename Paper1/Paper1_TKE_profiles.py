#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 01:14:04 2024

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

var = ['u','v','w','uw','vw']

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

    
#%%Select coordinates
import random
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex

N_twrs = 100
Nz_SLayer = 300

coord_atto_max = find_coordinates(intf['atto'],N_twrs,'max') #min,max
coord_sin_max = find_coordinates(intf['sin'],N_twrs,'max') #min,max
coord_atto_min = find_coordinates(intf['atto'],N_twrs,'min') #min,max
coord_sin_min = find_coordinates(intf['sin'],N_twrs,'min') #min,max
coord_flat = []
for i in range(0,N_twrs):
    coord_flat.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))

coord = {
    'flat' : coord_flat,
    'sin_min' : coord_sin_min,
    'atto_min' : coord_atto_min,
    'sin_max' : coord_sin_max,
    'atto_max' : coord_atto_max
    }

#%%
import matplotlib.gridspec as gridspec

TKE_terms = ['res','adv','ttrans','ptrans','totdis','prod']
cases = ['flat','sin_min','sin_max','atto_min','atto_max']
labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]  # Subplot labels
markers = ['o','v','s','*','X','+']
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
positions = [[0.33, 0.57, 0.2, 0.38], [0.61, 0.57, 0.2, 0.38], 
             [0.33, 0.08, 0.2, 0.38], [0.61, 0.08, 0.2, 0.38]]

fig, axs = plt.subplots(2, 3, figsize=(12, 12))#, gridspec_kw={'width_ratios': [1, 1, 1]})
gs = gridspec.GridSpec(2, 3, width_ratios=[1.6, 1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.3)
axs = axs.flatten()
legend_lines, legend_labels = None, None

for j in range(len(cases)):
    
    # fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,6))
    
    # axs.axvline(0,c='k')
    lines = []
    for i in range(len(TKE_terms)):
        tmpTKE = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
        Ruw = (velocity[cases[j]]['uw'] - velocity[cases[j]]['u']*velocity[cases[j]]['w'])*(u_scale**2)
        Rvw = (velocity[cases[j]]['vw'] - velocity[cases[j]]['v']*velocity[cases[j]]['w'])*(u_scale**2)
    
        for k in range(len(coord[cases[j]])):
            loc = coord[cases[j]][k]
            ustar = ((Ruw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2 + \
                            (Rvw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2)**(1/4)
            tmpTKE[k,:] = TKE[cases[j]][TKE_terms[i]][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                              int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]*(u_scale**3/zi)*(39/ustar**3)
    
        line, = axs[j].plot(np.nanmean(tmpTKE,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label=TKE_terms[i],linewidth=1.5,\
                    marker=markers[i], markevery=7, c=colors[i])
        lines.append(line)
    
    if j==0:
        axs[j].set_position([0.06, 0.32, 0.2, 0.38])
    else:
        axs[j].set_position(positions[j-1])
    
    axs[j].set_xlabel(r'$\left \langle \frac{\partial e}{\partial t} \right \rangle \frac{h_C}{u_{*}^{3}}$',fontsize=15)
    axs[j].text(0.12, 0.98, labels[j], transform=axs[j].transAxes, fontsize=12, va="top", ha="right")
    axs[j].set_ylabel(r'$z/h_C$',fontsize=15)
    axs[j].set_ylim(0,10)
    axs[j].set_xlim(-10,10)
    axs[j].grid()
    axs[j].tick_params(axis='both', which='major', labelsize=12)
    axs[j].axhline(1,ls='--',c='k')
    # axs[j].legend(loc='upper right',fontsize=12)
    axs[j].set_title(f'{cases[j]}', fontsize=15)
    # plt.savefig(pathFig+f'{cases[j]}_TKEprofTWR_ustar.png',dpi=300,facecolor='None', edgecolor='None')

axs[-1].legend(lines, TKE_terms, loc="center", fontsize=15)
axs[-1].set_position([0.8,0.42,0.2,0.2])
axs[-1].axis("off")
axs[2].set_yticklabels([]), axs[4].set_yticklabels([])
axs[2].set_ylabel(''), axs[4].set_ylabel('')
axs[1].set_xlabel(''), axs[2].set_xlabel('')
# plt.savefig(pathFig+f'TKEprofTWR_ustar_tij.png',dpi=300,facecolor='None', edgecolor='None')
# fig.delaxes(axs[-1])
plt.show()

#%% Profiles of TKE Budget

# z_ax = np.arange(0,nz)*dz + dz/2

# plt.figure(figsize=(4,8))
# plt.plot(np.nanmean(terms_bdg['res'],axis=(0,1)),z_ax,c='pink',marker='o',markevery=8,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0,1)),z_ax,c='red',marker='s',markevery=8,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0,1)),z_ax,c='purple',marker='^',markevery=8,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0,1)),z_ax,c='brown',marker='D',markevery=8,label='P_trans')
# # plt.plot(np.nanmean(-terms_bdg['dissip'],axis=(0,1)),z_ax,c='green',marker='v',markevery=8,label='Diss')
# # plt.plot(np.nanmean(-terms_bdg['canopy'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=8,label='Can')
# plt.plot(np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='black',marker='o',markevery=8,label='TotDis')
# plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod')
# # plt.plot(np.nanmean(terms_bdg['prod']-terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod-Diss')
# # plt.plot(np.nanmean(terms_bdg['adv']+terms_bdg['ttrans']+terms_bdg['ptrans'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='A+TT+PT')

# # plt.plot(np.nanmean(terms_bdg['res'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='pink',marker='o',markevery=5,label='res')
# # plt.plot(np.nanmean(terms_bdg['adv'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# # plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# # plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# # plt.plot(np.nanmean(-terms_bdg['dissip'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='green',marker='v',markevery=5,label='Diss')
# # plt.plot(np.nanmean(-terms_bdg['canopy'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=5,label='Can')
# # plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
# plt.hlines(canopyH,-80,80,linestyle='--',colors='k')
# plt.hlines(2*canopyH,-80,80,linestyle='--',colors='k')
# plt.vlines(0,0,5,linestyle='--',colors='grey')
# # plt.vlines(1,0,5,linestyle='--',colors='grey')
# # plt.vlines(-1,0,5,linestyle='--',colors='grey')
# plt.ylim(z_ax[0],z_ax[-1])
# # plt.xlim(-50,50)
# # plt.title(r'Real')
# plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# # plt.xlabel(r'$\left \langle de/dt \right \rangle$')
# plt.ylabel(r'$z/z_i$')
# plt.legend(loc='upper right')
# plt.show()






































