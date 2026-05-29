#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:05:44 2024

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import os
import sys
import copy

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

#%%
#inputs
# sim = 'flat_256x256x384_out5hr'
# sim = 'flat_256x256x384_out5hr_v2'
sim = 'flat_256x256x384_001_1hr'
# sim = 'flat_256x256x384_005_1hr'
# sim = 'flat_256x256x384_001_2min'
# sim = 'ATTO1_256x256x384_v2_out5hr'
# sim = 'ATTO2_256x256x384_out5hr'
# sim = 'ATTO12_256x256x384_out5hr'
# sim = 'ATTO_256x256x384_full_out5hr'
# sim = 'bicheng_hill_256x256x384_out5hr'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/sim'+sim+'/'

if os.path.exists(pathOUT):
    print(f"The forlder '{pathOUT}' aready exists.")
else:
    os.makedirs(pathOUT)
    print(f"Created folder '{pathOUT}'.")

pcnt3 = 36000 #18000; %Important! used in get_var 
avg_time = pcnt3*10 #total timesteps averaged  Important! used in get_var
startavg = 1
endavg = 1 # max is avg_time/pcnt3
incskip = 0 #should be 0 unless simulation incomplete (# of avgs incompleted)

completed_sim = True
infinite_geom = True
slice_profiles = False
shifting_z = False
tke_budget_flag = True
derivatives_flag = True
PCON_flag = False
PCON_TEMP_flag = False
v_fracflag = False
SAVEAS_flag = False
clear_var_flag = False

T_STC = 305 #320; %298.15; %[K], temperature scale
dt = 0.01 #05; %0.000005;
zi = 1000.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = '/scratch/general/nfs1/u1450851/LES_Sims/'
# simPath = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'

# figPath = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/ZevLES/ATTO1_JETtest/'
# simPath = '/uufs/chpc.utah.edu/common/home/calaf-group2/Zev_research2/ATTO_sims/';

# if SAVEAS_flag:
#     output_directory = os.path.join(simPath, f'sim{sim}', f'output/sim{sim}DOEplots')
#     os.makedirs(output_directory, exist_ok=True)
#     plotsPath = output_directory

# Change the current working directory
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/MATLAB_Codes/MATLAB_postproc_Zev')

# Append a directory to the Python path
sys.path.append('./functions/')

#%% Kinetic Energy for convergence

parentpath = simPath + f'sim{sim}/'
file_name = 'ke.txt'

# Load and import data
ke = np.genfromtxt(parentpath + file_name)
ts = np.linspace(0, len(ke) * wbase, len(ke))

#%% Simulation parameters
# Input and output paths
ipath = simPath + f'sim{sim}/output/ta1_field/'
opath = simPath + f'sim{sim}/output/'

# Read parameter file for ta1_field
with open(ipath + 'parameters.txt', 'r') as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

nx = int(param[0])
ny = int(param[1])
nz = int(param[2])
lx = param[3]
ly = param[4]
lz = param[5]
dx = param[6]
dy = param[7]
dz = param[8]
nt = int(param[9])
actLaunch = int(param[10])
mpiProc = int(param[11])
Re = param[12]
Ri = param[13]
Pr = param[14]
alpha = param[15]
SGS = param[16]
S_FLAG = param[17]
ibm = param[18]
rot = param[19]
nzTot = nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, nx) * dx
y = np.arange(0, ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

if shifting_z:
    # Modify axes to move 0 to IBM surface
    z_uvp = z_uvp[5:]
    z_uvp = z_uvp - z_uvp[0]
    z_w = z_w[5:]
    z_w = z_w - z_w[0]

# Build interface and solidity of IBM
if not completed_sim:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))
    nt = len(ke) // pcnt3
elif ibm == 1:
    # Assuming you have a function build_phi to load phi data from a file
    phi = build_phi(opath + 'phi_functions/', nx, ny, nzTot, mpiProc)
    intf, iintf = build_intf(phi, dz)
else:
    intf = np.zeros((nx, ny))
    iintf = np.round(np.zeros((nx, ny)))

if ibm:
    phi_uv = np.zeros_like(phi)
    for k in range(nz - 1):
        phi_uv[:, :, k] = (phi[:, :, k] + phi[:, :, k + 1]) / 2.0

if PCON_flag and v_fracflag:
    v_frac = get_var(opath + 'cut_cell/', 'v_frac', nx, ny, nzTot, 1, iintf, mpiProc)
    a_cut = get_var(opath + 'cut_cell/', 'a_cut', nx, ny, nzTot, 1, iintf, mpiProc)

#%% Loading the topography data:

zeds = np.zeros((nx,ny,nz))

for i in range(0,nx):
    for j in range(0,ny):
        zeds[i,j,:] = np.arange(0,nz)*(dz*zi)
    
dist = copy.deepcopy(zeds)

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nz):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi
            
#%%

nt_tot = nt * actLaunch - incskip
print('nt_tot is', nt_tot)

avgT = int(avg_time / pcnt3)
print('avgT is', avgT)

print('startavg is', startavg)
print('endavg is', endavg)

# timescale = zi / ustar
# timesteps = pcnt3 * (list(range(nt_tot - nt + 1, nt_tot + 1)))
# time = [t * timescale * dt / 60 for t in timesteps]

#%% Loading the Variance and TKE terms

terms_UU = np.load(pathOUT + 'UU_terms.npy',allow_pickle='TRUE').item()
terms_VV = np.load(pathOUT + 'VV_terms.npy',allow_pickle='TRUE').item()
terms_WW = np.load(pathOUT + 'WW_terms.npy',allow_pickle='TRUE').item()
terms_TKE = np.load(pathOUT + 'TKE_terms.npy',allow_pickle='TRUE').item()

var_bdg = ['dissip', 'canopy', 'adv','res','ptrans','ttrans','prod']

for i in range(len(var_bdg)):
    terms_UU[var_bdg[i]][(dist < 0)] = float("nan")
    terms_VV[var_bdg[i]][(dist < 0)] = float("nan")
    terms_WW[var_bdg[i]][(dist < 0)] = float("nan")
    terms_TKE[var_bdg[i]][(dist < 0)] = float("nan")

#%%Checking the value of R_u/S, R_v/S and R_w/s

R_u_S = -2/3 + terms_VV['prod']/(3*terms_UU['prod'])

fig, axs = plt.subplots(1,1,tight_layout=True)
axs.plot(np.nanmean(R_u_S,axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),c='k')
# axs.plot(np.nanmean(terms_UU['pturb_s']/terms_UU['prod'],axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),c='r')
# axs.plot(np.nanmean(-2/3 + 1/(3*terms_UU['prod'])*(terms_WW['ttrans']+terms_WW['pturb_t']+terms_VV['prod']+terms_VV['ttrans']+terms_VV['pturb_t']-2*terms_UU['ttrans']-2*terms_UU['pturb_t']),axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),c='g')
axs.axhline(1,-1000,1000,c='k',ls='--')
axs.axvline(-2/3,-1000,1000,c='k',ls='--')
axs.set_xlabel('$R_u/S_u$')
axs.set_ylabel('$z/H_c$')
# axs.set_xlim(-5,1)
axs.set_ylim(0,20)
plt.show()

#%% Pcolor plots of TKE vs the sum of the variance terms

i=4

terms = ['dissip', 'adv','res','ptrans','ttrans','prod','canopy']

TKE_term = terms_TKE[terms[i]][:,:,5:]
VAR_term = 0.5*(terms_UU[terms[i]] + terms_VV[terms[i]] + terms_WW[terms[i]])[:,:,5:]

yslice = 150

fig,axs = plt.subplots(2,1,figsize=(10,8),tight_layout=True)
pc1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-5)*dz/(39/zi),VAR_term[:,yslice,:].T,cmap='coolwarm',vmin=-50,vmax=50)
pc2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-5)*dz/(39/zi),TKE_term[:,yslice,:].T,cmap='coolwarm',vmin=-50,vmax=50)
axs[0].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs[1].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs[0].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
axs[1].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
axs[1].set_xlabel(r'$x/z_i$')
axs[0].set_ylabel(r'$z/h_c$')
axs[1].set_ylabel(r'$z/h_c$')
axs[0].set_title(terms[i] + ' from sum of VAR')
axs[1].set_title(terms[i] + ' from TKE')
cbar1 = plt.colorbar(pc1)
cbar2 = plt.colorbar(pc2)
plt.show()

#%% Compare budget terms between the variance equations

yslice = 63

terms = ['dissip', 'adv','res','ptrans','ttrans','prod','canopy','pturb_s','pturb_t']

for i in range(len(terms)):

    fig,axs = plt.subplots(3,1,figsize=(9,8),tight_layout=True)
    pc1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-5)*dz/(39/zi),terms_UU[terms[i]][:,yslice,5:].T,cmap='bwr',vmin=-40,vmax=40)
    pc2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-5)*dz/(39/zi),terms_VV[terms[i]][:,yslice,5:].T,cmap='bwr',vmin=-40,vmax=40)
    pc3 = axs[2].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-5)*dz/(39/zi),terms_WW[terms[i]][:,yslice,5:].T,cmap='bwr',vmin=-40,vmax=40)
    axs[0].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    axs[1].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    axs[2].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
    axs[0].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
    axs[1].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
    axs[2].plot(np.arange(0,nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
    axs[2].set_xlabel(r'$x/z_i$')
    axs[0].set_ylabel(r'$z/h_c$')
    axs[1].set_ylabel(r'$z/h_c$')
    axs[2].set_ylabel(r'$z/h_c$')
    axs[0].set_title(terms[i] + ' from UU')
    axs[1].set_title(terms[i] + ' from VV')
    axs[2].set_title(terms[i] + ' from WW')
    cbar1 = plt.colorbar(pc1)
    cbar2 = plt.colorbar(pc2)
    cbar3 = plt.colorbar(pc3)
    plt.show()

#%% Comparing Profiles for the terms of the variance budget

i=2

terms = ['dissip', 'adv','res','ptrans','ttrans','prod','canopy','pturb_s','pturb_t']

# for i in range(len(terms)):

fig,axs = plt.subplots(1,1,figsize=(4,8),tight_layout=True)
axs.plot(np.nanmean(terms_UU[terms[i]],axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),color='red',marker='o',markevery=10,label='UU')
axs.plot(np.nanmean(terms_VV[terms[i]],axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),color='green',marker='^',markevery=10,label='VV')
axs.plot(np.nanmean(terms_WW[terms[i]],axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),color='black',marker='s',markevery=10,label='WW')
axs.axvline(0,0,20,ls='--',c='k')
axs.axhline(1,-60,60,ls='--',c='k')
# axs.axhline(3,-60,60,ls='--',c='k')
axs.set_xlim(-2,2)
axs.set_xlabel('Variance Budget Term')
axs.set_ylabel(r'$z/h_c$')
axs.set_title(terms[i])
axs.legend()
axs.grid()
axs.set_ylim(0,20)
plt.show()
    
# plt.savefig(pathOUT +'Figures/Res_zoom_flat_01.png',dpi=300,facecolor='white', edgecolor='white')
    
# %% Profiles for the sum of turbulent transport and pressure transport

fig,axs = plt.subplots(1,1,figsize=(4,8),tight_layout=True)
axs.plot(np.nanmean(terms_UU['pturb_t'] + terms_UU['ttrans'],axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),color='red',marker='o',markevery=10,label='UU')
axs.plot(np.nanmean(terms_VV['pturb_t'] + terms_VV['ttrans'],axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),color='green',marker='^',markevery=10,label='VV')
axs.plot(np.nanmean(terms_WW['pturb_t'] + terms_WW['ttrans'],axis=(0,1))[5:],np.arange(0,nz-5)*dz/(39/zi),color='black',marker='s',markevery=10,label='WW')
axs.axvline(0,0,20,ls='--',c='k')
axs.axhline(1,-60,60,ls='--',c='k')
axs.axhline(3,-60,60,ls='--',c='k')
axs.set_xlabel('Variance Budget Term')
axs.set_ylabel(r'$z/h_c$')
# axs.set_title('Pressure + Turbulent Transport')
axs.legend()
axs.grid()
axs.set_ylim(0,20)

# plt.savefig(pathOUT +'Figures/Turb_Pres_trans_flat_var_budget.png',dpi=300,facecolor='white', edgecolor='white')


#%% Compare vertical profiles of variance sums and TKE

i=1

terms = ['dissip', 'adv','res','ptrans','ttrans','prod','canopy']

TKE_term = terms_TKE[terms[i]][:,:,5:]
VAR_term = 0.5*(terms_UU[terms[i]] + terms_VV[terms[i]] + terms_WW[terms[i]])[:,:,5:]

fig,axs = plt.subplots(1,1,figsize=(4,8),tight_layout=True)
axs.plot(np.nanmean(VAR_term,axis=(0,1)),np.arange(0,nz-5)*dz/(39/zi),color='red',marker='o',markevery=8,label='Sum of VAR')
axs.plot(np.nanmean(TKE_term,axis=(0,1)),np.arange(0,nz-5)*dz/(39/zi),color='green',label='TKE')
axs.axvline(0,0,20,ls='--',c='k')
axs.axhline(1,-60,60,ls='--',c='k')
axs.set_xlabel('Variance Budget Term')
axs.set_ylabel(r'$z/h_c$')
axs.set_title(terms[i])
axs.legend()
axs.grid()
axs.set_ylim(0,20)

#%%
# FOR TOPOGRAPHIES
#%%

#%% Profiles for towers in the real ATTO case

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex

N_twrs = 100
Nz_SLayer = 300

coord = find_coordinates(intf,N_twrs,'max') #min,max

# topo_twr = plot_topo_twr(zi,nx,ny,dx,dy,intf,coord,textsize)

i=5

terms = ['dissip', 'adv','res','ptrans','ttrans','prod','canopy']

TKE_twr = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
VAR_twr = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

for j in range(len(coord)):
    loc = coord[j]
    TKE_twr[j,:] = terms_TKE[terms[i]][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    VAR_twr[j,:] = 0.5*(terms_UU[terms[i]] + terms_VV[terms[i]] + terms_WW[terms[i]])[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        

fig,axs = plt.subplots(1,1,figsize=(4,8),tight_layout=True)
axs.plot(np.nanmean(VAR_twr,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='red',marker='o',markevery=8,label='Sum of VAR')
axs.plot(np.nanmean(TKE_twr,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='green',label='TKE')
axs.axvline(0,0,20,ls='--',c='k')
axs.axhline(1,-60,60,ls='--',c='k')
axs.set_xlabel('Variance Budget Term')
axs.set_ylabel(r'$z/h_c$')
axs.set_title(terms[i])
axs.legend()
axs.grid()
axs.set_ylim(0,20)

#%%Compare profiles of variance budget terms for UU,VV,WW

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex
import copy

N_twrs = 100
Nz_SLayer = 300

coord = find_coordinates(intf,N_twrs,'max') #min,max

#%%
# topo_twr = plot_topo_twr(zi,nx,ny,dx,dy,intf,coord,textsize)

# i=8

terms = ['dissip', 'adv','res','ptrans','ttrans','prod','canopy','pturb_s','pturb_t']

for i in range(len(terms)):

    tmp_UU = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
    tmp_VV = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
    tmp_WW = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
    
    for j in range(len(coord)):
        loc = coord[j]
        tmp_UU[j,:] = copy.deepcopy(terms_UU[terms[i]])[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        tmp_VV[j,:] = copy.deepcopy(terms_VV[terms[i]])[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        tmp_WW[j,:] = copy.deepcopy(terms_WW[terms[i]])[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    
    fig,axs = plt.subplots(1,1,figsize=(4,8),tight_layout=True)
    axs.plot(np.nanmean(tmp_UU,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='red',marker='o',markevery=10,label='UU')
    axs.plot(np.nanmean(tmp_VV,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='green',marker='^',markevery=10,label='VV')
    axs.plot(np.nanmean(tmp_WW,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='black',marker='s',markevery=10,label='WW')
    axs.axvline(0,0,20,ls='--',c='k')
    axs.axhline(1,-60,100,ls='--',c='k')
    # axs.axhline(5,-60,60,ls='--',c='k')
    axs.set_xlim(-40,75)
    # axs.axhline(3,-60,60,ls='--',c='k')
    axs.set_xlabel('Variance Budget Term')
    axs.set_ylabel(r'$z/h_c$')
    axs.set_title(terms[i])
    axs.legend()
    axs.grid()
    axs.set_ylim(0,20)
    
    # plt.savefig(pathOUT +'Figures/' + terms[i] + '_peaks_var_budget.png',dpi=300,facecolor='white', edgecolor='white')

#%%

tmp_UU = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmp_VV = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmp_WW = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

for j in range(len(coord)):
    loc = coord[j]
    tmp_UU[j,:] = copy.deepcopy(terms_UU['pturb_t'] + terms_UU['ttrans'])[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmp_VV[j,:] = copy.deepcopy(terms_VV['pturb_t'] + terms_VV['ttrans'])[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmp_WW[j,:] = copy.deepcopy(terms_WW['pturb_t'] + terms_WW['ttrans'])[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]

fig,axs = plt.subplots(1,1,figsize=(4,8),tight_layout=True)
axs.plot(np.nanmean(tmp_UU,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='red',marker='o',markevery=10,label='UU')
axs.plot(np.nanmean(tmp_VV,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='green',marker='^',markevery=10,label='VV')
axs.plot(np.nanmean(tmp_WW,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi),color='black',marker='s',markevery=10,label='WW')
axs.axvline(0,0,20,ls='--',c='k')
axs.axhline(1,-60,60,ls='--',c='k')
axs.axhline(5,-60,60,ls='--',c='k')
# axs.axhline(3,-60,60,ls='--',c='k')
axs.set_xlabel('Variance Budget Term')
axs.set_ylabel(r'$z/h_c$')
# axs.set_title(terms[i])
axs.legend()
axs.grid()
axs.set_ylim(0,20)

# plt.savefig(pathOUT +'Figures/Turb_Pres_trans_valley_var_budget.png',dpi=300,facecolor='white', edgecolor='white')

 #%%Profile plots

z_ax = (np.arange(0,nz-5)*dz+dz/2)/(39/zi)

plt.figure(figsize=(4,8))
plt.plot(np.nanmean(terms_UU['res'][:,:,5:],axis=(0,1)),z_ax,c='blue',marker='o',markevery=8,label='Res')
plt.plot(np.nanmean(terms_UU['adv'][:,:,5:],axis=(0,1)),z_ax,c='pink',marker='o',markevery=8,label='Adv')
plt.plot(np.nanmean(terms_UU['prod'][:,:,5:],axis=(0,1)),z_ax,c='red',marker='s',markevery=8,label='Prod')
plt.plot(np.nanmean(terms_UU['ttrans'][:,:,5:],axis=(0,1)),z_ax,c='purple',marker='^',markevery=8,label='T_trans')
plt.plot(np.nanmean(terms_UU['pturb_s'][:,:,5:],axis=(0,1)),z_ax,c='black',marker='D',markevery=8,label='Return')
plt.plot(np.nanmean(terms_UU['pturb_t'][:,:,5:],axis=(0,1)),z_ax,c='brown',marker='D',markevery=8,label='P_trans')
plt.plot(np.nanmean(-terms_UU['dissip'][:,:,5:],axis=(0,1)),z_ax,c='green',marker='v',markevery=8,label='Diss')
plt.plot(np.nanmean(-terms_UU['canopy'][:,:,5:],axis=(0,1)),z_ax,c='orange',marker='o',markevery=8,label='Can')
# plt.plot(np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='black',marker='o',markevery=8,label='TotDis')
# plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod')
# plt.plot(np.nanmean(terms_bdg['prod']-terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='Prod-Diss')
# plt.plot(np.nanmean(terms_bdg['adv']+terms_bdg['ttrans']+terms_bdg['ptrans'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='A+TT+PT')
plt.hlines(1,-80,80,linestyle='--',colors='k')
plt.hlines(2,-80,80,linestyle='--',colors='k')
plt.vlines(0,0,50,linestyle='--',colors='grey')
plt.ylim(z_ax[0],z_ax[-1])
plt.xlim(-80,80)
# plt.title(r'Real')
plt.xlabel(r'$\left \langle duu/dt \right \rangle$')
# plt.xlabel(r'$\left \langle de/dt \right \rangle$')
plt.ylabel(r'$z/h_c$')
plt.legend(loc='upper right')
plt.show()




































