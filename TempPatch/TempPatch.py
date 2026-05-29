#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 02:19:14 2025

@author: u1450851
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
import os
import sys
import copy
import mpl_scatter_density

#%%Simulation Parameters

nx = 128
ny = 128
nz = 128

zi = 1000
uscale = 0.4
Tscale = 290
dt = 0.1
ug = 1

lx = 2000*np.pi/zi
ly = 2000*np.pi/zi
lz = 2000/zi
dx = lx/nx
dy = ly/ny
dz = lz/nz

sim_a = '800patch_9ms_aniso_grn'
path_data_a = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_a+'/data/'
sim_na = '800patch_9ms_noaniso_fix_grn'
path_data_na = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_na+'/data/'

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/TempPatch/'

#%%Load the data
StartTime = 162000
AvgIter = 18000

#Momentum fields 3D

var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC','avgL3D']
dataM_a = xr.open_dataarray(path_data_a+'Momentum3D/Data_Momentum_'+str(StartTime)+'.nc')
dataM_na = xr.open_dataarray(path_data_na+'Momentum3D/Data_Momentum_'+str(StartTime)+'.nc')
data_mom_a = dict()
data_mom_na = dict()
for i in range(len(var)):
    data_mom_a[var[i]] = dataM_a.data[:,:,:,i]
    data_mom_na[var[i]] = dataM_na.data[:,:,:,i]
del dataM_na,dataM_a
keys_list = list(data_mom_na.keys())
if keys_list[-1]=='avgL3D':
    data_mom_a['avgL3D'] = data_mom_a['avgL3D']*AvgIter
    data_mom_na['avgL3D'] = data_mom_na['avgL3D']*AvgIter

# Surface momentum

var2D = ['avgUstar']
dataM2D_a = xr.open_dataarray(path_data_a+'Momentum2D/Data_Momentum_2D_'+str(StartTime)+'.nc')
dataM2D_na = xr.open_dataarray(path_data_na+'Momentum2D/Data_Momentum_2D_'+str(StartTime)+'.nc')
data_mom_2D_a = dict()
data_mom_2D_na = dict()
for i in range(len(var2D)):
    data_mom_2D_a[var2D[i]] = dataM2D_a.data[:,:,i]
    data_mom_2D_na[var2D[i]] = dataM2D_na.data[:,:,i]
del dataM2D_na,dataM2D_a

# Scalar fields 3D
    
varS = ['avgT','avgT2','avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus','avg_ds']
dataS_a = xr.open_dataarray(path_data_a+'Scalar3D/Data_Scalar_'+str(StartTime)+'.nc')
dataS_na = xr.open_dataarray(path_data_na+'Scalar3D/Data_Scalar_'+str(StartTime)+'.nc')
data_sc_a = dict()
data_sc_na = dict()
for i in range(len(varS)):
    data_sc_a[varS[i]] = dataS_a.data[:,:,:,i]
    data_sc_na[varS[i]] = dataS_na.data[:,:,:,i]
del dataS_na,dataS_a

# Surface Scalar

varS2D = ['avgWstar','avgL','avgPHIm','avgPSIm','avgPHIh','avgPSIh','avgSFCval','avgSFCflux']
dataS2D_a = xr.open_dataarray(path_data_a+'Scalar2D/Data_Scalar_2D_'+str(StartTime)+'.nc')
dataS2D_na = xr.open_dataarray(path_data_na+'Scalar2D/Data_Scalar_2D_'+str(StartTime)+'.nc')
data_sc_2D_a = dict()
data_sc_2D_na = dict()
for i in range(len(varS2D)):
    data_sc_2D_a[varS2D[i]] = dataS2D_a.data[:,:,i]
    data_sc_2D_na[varS2D[i]] = dataS2D_na.data[:,:,i]
del dataS2D_na,dataS2D_a

# Anisotropy

varA = ['avgXB','avgYB']#,'avgPHIM','avgPHIH','avgPSIM','avgPSIH','avgL3D','avgustar3D','avgSCF3D']
dataA_a = xr.open_dataarray(path_data_a+'Anisotropy/Data_Anisotropy_'+str(StartTime)+'.nc')
dataA_na = xr.open_dataarray(path_data_na+'Anisotropy/Data_Anisotropy_'+str(StartTime)+'.nc')
data_aniso_a = dict()
data_aniso_na = dict()
for i in range(len(varA)):
    data_aniso_a[varA[i]] = dataA_a.data[:,:,:,i]
    data_aniso_na[varA[i]] = dataA_na.data[:,:,:,i]
del dataA_na,dataA_a

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

#%% Check values of Anisotropy

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nz):
            if data_aniso_a['avgYB'][i,j,k]<0:
                print(i,j,k)
                
#%%Flow overview plots, mean profile of temperature, wind speed, shear stress

u_w = uvpnode2wnode(data_mom_a['avgU'])
v_w = uvpnode2wnode(data_mom_a['avgV'])

Re_a = {
      'uu' : data_mom_a['avgU2'] - data_mom_a['avgU']*data_mom_a['avgU'],
      'vv' : data_mom_a['avgV2'] - data_mom_a['avgV']*data_mom_a['avgV'],
      'ww' : data_mom_a['avgW2'] - data_mom_a['avgW']*data_mom_a['avgW'],
      'uv' : data_mom_a['avgUV'] - data_mom_a['avgU']*data_mom_a['avgV'],
      'uw' : data_mom_a['avgUW'] - u_w*data_mom_a['avgW'],
      'vw' : data_mom_a['avgVW'] - v_w*data_mom_a['avgW']
      }

u_w = uvpnode2wnode(data_mom_na['avgU'])
v_w = uvpnode2wnode(data_mom_na['avgV'])

Re_na = {
      'uu' : data_mom_na['avgU2'] - data_mom_na['avgU']*data_mom_na['avgU'],
      'vv' : data_mom_na['avgV2'] - data_mom_na['avgV']*data_mom_na['avgV'],
      'ww' : data_mom_na['avgW2'] - data_mom_na['avgW']*data_mom_na['avgW'],
      'uv' : data_mom_na['avgUV'] - data_mom_na['avgU']*data_mom_na['avgV'],
      'uw' : data_mom_na['avgUW'] - u_w*data_mom_na['avgW'],
      'vw' : data_mom_na['avgVW'] - v_w*data_mom_na['avgW']
      }

wT_a = (data_sc_a['avgWT'] - data_mom_a['avgW']*uvpnode2wnode(data_sc_a['avgT']) - data_sc_a['avgWT_sgs'])
wT_na = (data_sc_na['avgWT'] - data_mom_na['avgW']*uvpnode2wnode(data_sc_na['avgT']) - data_sc_na['avgWT_sgs'])

xlabels = ['U [m/s]','V [m/s]','W [m/s]','T [K]',r"$\overline{u'w'}$ $[m^2s^{-2}]$",r"$\overline{v'w'}$ $[m^2s^{-2}]$",r"$\overline{w^'\theta^'}$"]

fig,axs = plt.subplots(2,7,figsize=(16,7),tight_layout=True)#,gridspec_kw={'hspace':0.6})

axs[0,0].plot(np.mean(data_mom_a['avgU'],axis=(0,1))*uscale,np.arange(0,nz)*dz+dz/2,c='k',label='U')
axs[0,0].fill_betweenx(np.arange(0,nz)*dz+dz/2,np.mean(data_mom_a['avgU'],axis=(0,1))*uscale - np.std(data_mom_a['avgU'],axis=(0,1))*uscale,\
                        np.mean(data_mom_a['avgU'],axis=(0,1))*uscale + np.std(data_mom_a['avgU'],axis=(0,1))*uscale,color='gray',alpha=0.4)
axs[0,1].plot(np.mean(data_mom_a['avgV'],axis=(0,1))*uscale,np.arange(0,nz)*dz+dz/2,c='k',label='V')
axs[0,1].fill_betweenx(np.arange(0,nz)*dz+dz/2,np.mean(data_mom_a['avgV'],axis=(0,1))*uscale - np.std(data_mom_a['avgV'],axis=(0,1))*uscale,\
                        np.mean(data_mom_a['avgV'],axis=(0,1))*uscale + np.std(data_mom_a['avgV'],axis=(0,1))*uscale,color='gray',alpha=0.4)
axs[0,2].plot(np.mean(data_mom_a['avgW'],axis=(0,1))*uscale,np.arange(0,nz)*dz,c='k',label='W')
axs[0,2].fill_betweenx(np.arange(0,nz)*dz,np.mean(data_mom_a['avgW'],axis=(0,1))*uscale - np.std(data_mom_a['avgW'],axis=(0,1))*uscale,\
                        np.mean(data_mom_a['avgW'],axis=(0,1))*uscale + np.std(data_mom_a['avgW'],axis=(0,1))*uscale,color='gray',alpha=0.4)
axs[0,3].plot(np.mean(data_sc_a['avgT'],axis=(0,1))*Tscale,np.arange(0,nz)*dz+dz/2,c='k',label='T')
axs[0,3].fill_betweenx(np.arange(0,nz)*dz+dz/2,np.mean(data_sc_a['avgT'],axis=(0,1))*Tscale - np.std(data_sc_a['avgT'],axis=(0,1))*Tscale,\
                        np.mean(data_sc_a['avgT'],axis=(0,1))*Tscale + np.std(data_sc_a['avgT'],axis=(0,1))*Tscale,color='gray',alpha=0.4)
axs[0,4].plot(np.mean(Re_a['uw'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='k',label='Ruw')
axs[0,4].fill_betweenx(np.arange(0,nz)*dz, np.mean(Re_a['uw'],axis=(0,1))*(uscale**2) - np.std(Re_a['uw'],axis=(0,1))*(uscale**2),\
                        np.mean(Re_a['uw'],axis=(0,1))*(uscale**2) + np.std(Re_a['uw'],axis=(0,1))*(uscale**2),color='gray',alpha=0.4)
axs[0,4].plot(np.mean(-data_mom_a['avgtxz'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='r',label='txz')
axs[0,4].fill_betweenx(np.arange(0,nz)*dz,np.mean(-data_mom_a['avgtxz'],axis=(0,1))*(uscale**2) - np.std(-data_mom_a['avgtxz'],axis=(0,1))*(uscale**2),\
                        np.mean(-data_mom_a['avgtxz'],axis=(0,1))*(uscale**2) + np.std(-data_mom_a['avgtxz'],axis=(0,1))*(uscale**2),color='red',alpha=0.4)
axs[0,5].plot(np.mean(Re_a['vw'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='k',label='Rvw')
axs[0,5].fill_betweenx(np.arange(0,nz)*dz, np.mean(Re_a['vw'],axis=(0,1))*(uscale**2) - np.std(Re_a['vw'],axis=(0,1))*(uscale**2),\
                        np.mean(Re_a['vw'],axis=(0,1))*(uscale**2) + np.std(Re_a['vw'],axis=(0,1))*(uscale**2),color='gray',alpha=0.4)
axs[0,5].plot(np.mean(-data_mom_a['avgtyz'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='r',label='tyz')
axs[0,5].fill_betweenx(np.arange(0,nz)*dz,np.mean(-data_mom_a['avgtyz'],axis=(0,1))*(uscale**2) - np.std(-data_mom_a['avgtyz'],axis=(0,1))*(uscale**2),\
                        np.mean(-data_mom_a['avgtyz'],axis=(0,1))*(uscale**2) + np.std(-data_mom_a['avgtyz'],axis=(0,1))*(uscale**2),color='red',alpha=0.4)
axs[0,6].plot(np.mean(wT_a,axis=(0,1))*uscale*Tscale,np.arange(0,nz)*dz,c='k',label='wT')
axs[0,6].fill_betweenx(np.arange(0,nz)*dz,np.mean(wT_a,axis=(0,1))*(uscale*Tscale) - np.std(wT_a,axis=(0,1))*(uscale*Tscale),\
                        np.mean(wT_a,axis=(0,1))*(uscale*Tscale) + np.std(wT_a,axis=(0,1))*(uscale*Tscale),color='gray',alpha=0.4)
    
axs[1,0].plot(np.mean(data_mom_na['avgU'],axis=(0,1))*uscale,np.arange(0,nz)*dz+dz/2,c='k',label='U')
axs[1,0].fill_betweenx(np.arange(0,nz)*dz+dz/2,np.mean(data_mom_na['avgU'],axis=(0,1))*uscale - np.std(data_mom_na['avgU'],axis=(0,1))*uscale,\
                       np.mean(data_mom_na['avgU'],axis=(0,1))*uscale + np.std(data_mom_na['avgU'],axis=(0,1))*uscale,color='gray',alpha=0.4)
axs[1,1].plot(np.mean(data_mom_na['avgV'],axis=(0,1))*uscale,np.arange(0,nz)*dz+dz/2,c='k',label='V')
axs[1,1].fill_betweenx(np.arange(0,nz)*dz+dz/2,np.mean(data_mom_na['avgV'],axis=(0,1))*uscale - np.std(data_mom_na['avgV'],axis=(0,1))*uscale,\
                       np.mean(data_mom_na['avgV'],axis=(0,1))*uscale + np.std(data_mom_na['avgV'],axis=(0,1))*uscale,color='gray',alpha=0.4)
axs[1,2].plot(np.mean(data_mom_na['avgW'],axis=(0,1))*uscale,np.arange(0,nz)*dz,c='k',label='W')
axs[1,2].fill_betweenx(np.arange(0,nz)*dz,np.mean(data_mom_na['avgW'],axis=(0,1))*uscale - np.std(data_mom_na['avgW'],axis=(0,1))*uscale,\
                       np.mean(data_mom_na['avgW'],axis=(0,1))*uscale + np.std(data_mom_na['avgW'],axis=(0,1))*uscale,color='gray',alpha=0.4)
axs[1,3].plot(np.mean(data_sc_na['avgT'],axis=(0,1))*Tscale,np.arange(0,nz)*dz+dz/2,c='k',label='T')
axs[1,3].fill_betweenx(np.arange(0,nz)*dz+dz/2,np.mean(data_sc_na['avgT'],axis=(0,1))*Tscale - np.std(data_sc_na['avgT'],axis=(0,1))*Tscale,\
                       np.mean(data_sc_na['avgT'],axis=(0,1))*Tscale + np.std(data_sc_na['avgT'],axis=(0,1))*Tscale,color='gray',alpha=0.4)
axs[1,4].plot(np.mean(Re_na['uw'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='k',label='Ruw')
axs[1,4].fill_betweenx(np.arange(0,nz)*dz, np.mean(Re_na['uw'],axis=(0,1))*(uscale**2) - np.std(Re_na['uw'],axis=(0,1))*(uscale**2),\
                       np.mean(Re_na['uw'],axis=(0,1))*(uscale**2) + np.std(Re_na['uw'],axis=(0,1))*(uscale**2),color='gray',alpha=0.4)
axs[1,4].plot(np.mean(-data_mom_na['avgtxz'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='r',label='txz')
axs[1,4].fill_betweenx(np.arange(0,nz)*dz,np.mean(-data_mom_na['avgtxz'],axis=(0,1))*(uscale**2) - np.std(-data_mom_na['avgtxz'],axis=(0,1))*(uscale**2),\
                       np.mean(-data_mom_na['avgtxz'],axis=(0,1))*(uscale**2) + np.std(-data_mom_na['avgtxz'],axis=(0,1))*(uscale**2),color='red',alpha=0.4)
axs[1,5].plot(np.mean(Re_na['vw'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='k',label='Rvw')
axs[1,5].fill_betweenx(np.arange(0,nz)*dz, np.mean(Re_na['vw'],axis=(0,1))*(uscale**2) - np.std(Re_na['vw'],axis=(0,1))*(uscale**2),\
                       np.mean(Re_na['vw'],axis=(0,1))*(uscale**2) + np.std(Re_na['vw'],axis=(0,1))*(uscale**2),color='gray',alpha=0.4)
axs[1,5].plot(np.mean(-data_mom_na['avgtyz'],axis=(0,1))*(uscale**2),np.arange(0,nz)*dz,c='r',label='tyz')
axs[1,5].fill_betweenx(np.arange(0,nz)*dz,np.mean(-data_mom_na['avgtyz'],axis=(0,1))*(uscale**2) - np.std(-data_mom_na['avgtyz'],axis=(0,1))*(uscale**2),\
                       np.mean(-data_mom_na['avgtyz'],axis=(0,1))*(uscale**2) + np.std(-data_mom_na['avgtyz'],axis=(0,1))*(uscale**2),color='red',alpha=0.4)
axs[1,6].plot(np.mean(wT_na,axis=(0,1))*uscale*Tscale,np.arange(0,nz)*dz,c='k',label='wT')
axs[1,6].fill_betweenx(np.arange(0,nz)*dz,np.mean(wT_na,axis=(0,1))*(uscale*Tscale) - np.std(wT_na,axis=(0,1))*(uscale*Tscale),\
                        np.mean(wT_na,axis=(0,1))*(uscale*Tscale) + np.std(wT_na,axis=(0,1))*(uscale*Tscale),color='gray',alpha=0.4)

for i in range(len(axs)):
    for j in range(len(axs[0,:])):
        axs[i,j].legend()
        # axs[i].axhline(0.8,c='k',ls='--')
        axs[1,j].set_xlabel(xlabels[j],fontsize=15)
        axs[i,j].set_ylim(0,10*dz)
        axs[i,j].grid()
    axs[i,0].set_ylabel('z/zi',fontsize=15)
    
for i in range(len(axs)):
    axs[i,0].set_xlim(0,10)
    axs[i,1].set_xlim(-0.5,2.5)
    axs[i,3].set_xlim(285,295)
    axs[i,4].set_xlim(-0.35,0.1)
    axs[i,5].set_xlim(-0.15,0.1)
    
fig.text(0.5,0.95,'Anisotropy',ha='center',va='center',fontsize=14)
fig.text(0.5,0.48,'No Anisotropy',ha='center',va='center',fontsize=14)

plt.tight_layout(rect=[0,0,1,0.95])
plt.subplots_adjust(hspace=0.6)
    
plt.show()

#%% Graphical Representation of U,W,TKE:

yslice = 128
    
tke_a = (Re_a['uu'] + Re_a['vv'] + wnode2uvpnode(Re_a['ww']))/2 
tke_na = (Re_na['uu'] + Re_na['vv'] + wnode2uvpnode(Re_na['ww']))/2  

#Mean Velocity colorplots
fig, axs=plt.subplots(3,1,figsize=(8,6), constrained_layout=True)
# plt1 = axs[0].pcolormesh(x_ax,z_ax,np.nanmean(tmp_u,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
# plt2 = axs[1].pcolormesh(x_ax,z_ax,np.nanmean(tmp_w,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
# plt3 = axs[2].pcolormesh(x_ax,z_ax,np.nanmean(tmp_tke,axis=(1)).T,cmap='YlGnBu',shading='gouraud')
plt1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,data_mom_a['avgU'][:,yslice,:].T,cmap='YlGnBu',shading='gouraud',vmin=0,vmax=5)
plt2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_mom_a['avgW'][:,yslice,:].T,cmap='YlGnBu',shading='gouraud',vmin=-1,vmax=1)
plt3 = axs[2].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,tke_a[:,yslice,:].T,cmap='YlGnBu',shading='gouraud',vmin=0,vmax=10)
# plt1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,data_mom_na['avgU'][:,yslice,:].T,cmap='YlGnBu',shading='gouraud',vmin=0,vmax=5)
# plt2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_mom_na['avgW'][:,yslice,:].T,cmap='YlGnBu',shading='gouraud',vmin=-1,vmax=1)
# plt3 = axs[2].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,tke_na[:,yslice,:].T,cmap='YlGnBu',shading='gouraud',vmin=0,vmax=10)

fig.colorbar(plt1,ax=axs[0])
fig.colorbar(plt2,ax=axs[1])
fig.colorbar(plt3,ax=axs[2])


# axs[0].set_ylim(0,1);axs[1].set_ylim(0,1);axs[2].set_ylim(0,1)
# axs[0].set_xlim(x_ax[0],x_ax[-1]);axs[1].set_xlim(x_ax[0],x_ax[-1]);axs[2].set_xlim(x_ax[0],x_ax[-1])
axs[0].set_ylabel(r'$z/z_i$'); #axs[0].set_xlabel(r'$x/z_i$')
axs[0].set_title(r'$\overline{u}(x,y_{nz/2},z)/u_*$')

axs[1].set_ylabel(r'$z/z_i$');#axs[1].set_xlabel(r'$x/z_i$') 
axs[1].set_title(r'$\overline{w}(x,y_{nz/2},z)/u_*$')

axs[2].set_xlabel(r'$x/z_i$'); axs[2].set_ylabel(r'$z/z_i$')
axs[2].set_title(r'$\overline{e}(x,y_{nz/2},z)/u_*^2$')


plt.show()

#%% Surface temperature

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

p1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_a['avgSFCval']*Tscale).T,cmap='hot_r',alpha=0.5)
p2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_na['avgSFCval']*Tscale).T,cmap='hot_r',alpha=0.5)
c1 = axs[0].contour(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_mom_2D_a['avgUstar']*uscale).T,levels=[0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8],colors='black')
c2 = axs[1].contour(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_mom_2D_na['avgUstar']*uscale).T,levels=[0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8],colors='black')
axs[0].clabel(c1,inline=True,fontsize=8,fmt="%.2f")
axs[1].clabel(c2,inline=True,fontsize=8,fmt="%.2f")
axs[0].set_ylabel(r'$y/z_i$ [m]', fontsize=15)
fig.suptitle('Surface temperature', fontsize=15)
for i in range(len(axs)):
    axs[i].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[0].text(0.1, 0.9,f"{np.mean(data_sc_2D_a['avgSFCval'] * Tscale):.2f}", transform=axs[0].transAxes, fontsize=15)
axs[1].text(0.1, 0.9,f"{np.mean(data_sc_2D_na['avgSFCval'] * Tscale):.2f}", transform=axs[1].transAxes, fontsize=15)
axs[0].set_title('Anisotropy',fontsize=15)
axs[1].set_title('No Anisotropy',fontsize=15)
cbar1 = plt.colorbar(p1)
cbar1.set_label(label='T [K]',fontsize=15)
cbar2 = plt.colorbar(p2)
cbar2.set_label(label='T [K]',fontsize=15)
plt.show()

#%%Surface heat flux

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

p1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_a['avgSFCflux']*Tscale*uscale*1.2*1005).T,cmap='hot_r',vmin=-250,vmax=350)
p2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_na['avgSFCflux']*Tscale*uscale*1.2*1005).T,cmap='hot_r',vmin=-250,vmax=350)
axs[0].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[1].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[0].set_ylabel(r'$y/z_i$ [m]', fontsize=15)
fig.suptitle('Surface heat flux', fontsize=15)
axs[0].text(0.1, 0.9,f"{np.median(data_sc_2D_a['avgSFCflux'] * Tscale*uscale*1.2*1005):.2f}", transform=axs[0].transAxes, fontsize=15)
axs[1].text(0.1, 0.9,f"{np.median(data_sc_2D_na['avgSFCflux'] * Tscale*uscale*1.2*1005):.2f}", transform=axs[1].transAxes, fontsize=15)
axs[0].set_title('Anisotropy',fontsize=15)
axs[1].set_title('No Anisotropy',fontsize=15)
cbar2 = plt.colorbar(p1)
cbar2 = plt.colorbar(p2)
cbar2.set_label(label=r"$\overline{w'\theta'}$ [$K-ms^{-1}$]",fontsize=15)
plt.show()

#%%Surface friction velocity

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

p1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_mom_2D_a['avgUstar']*uscale).T,cmap='jet',vmin=0.0,vmax=0.6)
p2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_mom_2D_na['avgUstar']*uscale).T,cmap='jet',vmin=0.0,vmax=0.6)
# c1 = axs[0].contour(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_a['avgSFCval']*Tscale).T,levels=[280,282,285,288,290,295,300],colors='black')
# c2 = axs[1].contour(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_na['avgSFCval']*Tscale).T,levels=[280,282,285,288,290,295,300],colors='black')
# axs[0].clabel(c1,inline=True,fontsize=8,fmt="%.2f")
# axs[1].clabel(c2,inline=True,fontsize=8,fmt="%.2f")
axs[0].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[1].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[0].set_ylabel(r'$y/z_i$ [m]', fontsize=15)
fig.suptitle('Surface friction velocity', fontsize=15)
axs[0].text(0.1, 0.9,f"{np.mean(data_mom_2D_a['avgUstar'] * uscale):.2f}", transform=axs[0].transAxes, fontsize=15)
axs[1].text(0.1, 0.9,f"{np.mean(data_mom_2D_na['avgUstar'] * uscale):.2f}", transform=axs[1].transAxes, fontsize=15)
axs[0].set_title('Anisotropy',fontsize=15)
axs[1].set_title('No Anisotropy',fontsize=15)
cbar2 = plt.colorbar(p1)
cbar2 = plt.colorbar(p2)
cbar2.set_label(label=r"$u_*$ [$ms^{-1}$]",fontsize=15)
plt.show()

#%%Plot surface Phi and Psi functions

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

p1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_a['avgPSIm']).T,cmap='jet')#,vmin=-0.5,vmax=0.5)
p2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_na['avgPSIm']).T,cmap='jet')#,vmin=-0.5,vmax=0.5)
axs[0].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[1].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[0].set_ylabel(r'$y/z_i$ [m]', fontsize=15)
fig.suptitle('Surface scaling', fontsize=15)
# axs[0].text(0.1, 0.9,f"{np.mean(data_mom_2D_a['avgUstar'] * uscale):.2f}", transform=axs[0].transAxes, fontsize=15)
# axs[1].text(0.1, 0.9,f"{np.mean(data_mom_2D_na['avgUstar'] * uscale):.2f}", transform=axs[1].transAxes, fontsize=15)
axs[0].set_title('Anisotropy',fontsize=15)
axs[1].set_title('No Anisotropy',fontsize=15)
cbar2 = plt.colorbar(p1)
cbar2 = plt.colorbar(p2)
# cbar2.set_label(label=r"$u_*$ [$ms^{-1}$]",fontsize=15)
plt.show()

#%%Surface obukhov length

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

p1 = axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_a['avgL']*zi).T,cmap='jet',vmin=-100,vmax=100)
p2 = axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,(data_sc_2D_na['avgL']*zi).T,cmap='jet',vmin=-100,vmax=100)
axs[0].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[1].set_xlabel(r'$x/z_i$ [m]', fontsize=15)
axs[0].set_ylabel(r'$y/z_i$ [m]', fontsize=15)
fig.suptitle('Surface Obukhov length', fontsize=15)
axs[0].text(0.1, 0.9,f"{np.median(data_sc_2D_a['avgL'] * zi):.2f}", transform=axs[0].transAxes, fontsize=15)
axs[1].text(0.1, 0.9,f"{np.median(data_sc_2D_na['avgL'] * zi):.2f}", transform=axs[1].transAxes, fontsize=15)
axs[0].set_title('Anisotropy',fontsize=15)
axs[1].set_title('No Anisotropy',fontsize=15)
cbar1 = plt.colorbar(p1)
# cbar1.set_label(label=r"$L$ [$m$]",fontsize=15)
cbar1 = plt.colorbar(p2)
cbar1.set_label(label=r"$L$ [$m$]",fontsize=15)
plt.show()

#%%Compute ustar and temperature flux

ustar_a = ((Re_a['uw'] - data_mom_a['avgtxz'])**2 + (Re_a['vw'] - data_mom_a['avgtyz'])**2)**(1/4) 
ustar_na = ((Re_na['uw'] - data_mom_na['avgtxz'])**2 + (Re_na['vw'] - data_mom_na['avgtyz'])**2)**(1/4) 

wT_a = (data_sc_a['avgWT'] - data_mom_a['avgW']*uvpnode2wnode(data_sc_a['avgT']) - data_sc_a['avgWT_sgs'])
wT_na = (data_sc_na['avgWT'] - data_mom_na['avgW']*uvpnode2wnode(data_sc_na['avgT']) - data_sc_na['avgWT_sgs'])

#%%Some vertical slices of things
os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering, Anisotropy

zslice = 10
yslice = 128
xslice = 128

var_a = data_mom_a['avgU']
var_na = data_mom_na['avgU']

fig,axs = plt.subplots(2,3,tight_layout=True,sharex=True,figsize=(11,8))

p1 = axs[0,0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,var_a[:,yslice,:].T,cmap=ColorAnisotropy())#,vmin=0,vmax=2)
p2 = axs[1,0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,var_na[:,yslice,:].T,cmap=ColorAnisotropy())#,vmin=0,vmax=2)

cbar1 = plt.colorbar(p1)
cbar1 = plt.colorbar(p2)

p1 = axs[0,1].pcolormesh(np.arange(0,ny)*dy,np.arange(0,nz)*dz,var_a[xslice,:,:].T,cmap=ColorAnisotropy())#,vmin=0,vmax=2)
p2 = axs[1,1].pcolormesh(np.arange(0,ny)*dy,np.arange(0,nz)*dz,var_na[xslice,:,:].T,cmap=ColorAnisotropy())#,vmin=0,vmax=2)

cbar1 = plt.colorbar(p1)
cbar1 = plt.colorbar(p2)

p1 = axs[0,2].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,var_a[:,:,zslice].T,cmap=ColorAnisotropy())#,vmin=0,vmax=2)
p2 = axs[1,2].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,var_na[:,:,zslice].T,cmap=ColorAnisotropy())#,vmin=0,vmax=2)

cbar1 = plt.colorbar(p1)
cbar1 = plt.colorbar(p2)

for i in range(len(axs[0,:])):
    axs[1,i].set_xlabel('x(y)/zi',fontsize=15)
for i in range(len(axs)):
    # axs[i].set_ylim(0,1.5)
    axs[i,0].set_ylabel('z(y)/zi', fontsize=15)
    
axs[0,1].set_title('Anisotropy',fontsize=15)
axs[1,1].set_title('No Anisotropy',fontsize=15)
# cbar1 = plt.colorbar(p1)
# cbar1.set_label()
# cbar2 = plt.colorbar(p2)
# cbar2.set_label(label=r"$L$ [$m$]",fontsize=15)

plt.show()

#%%Compute the flux Richardson number

T_w_a = uvpnode2wnode(data_sc_a['avgT'])
T_w_a[:,:,0] = data_sc_2D_a['avgSFCval']
T_w_na = uvpnode2wnode(data_sc_na['avgT'])
T_w_na[:,:,0] = data_sc_2D_na['avgSFCval']

Ri_f_a = (9.81*zi/uscale**2)/T_w_a*wT_a/((Re_a['uw']-data_mom_a['avgtxz'])*data_mom_a['avgdudz'] + (Re_a['vw']-data_mom_a['avgtyz'])*data_mom_a['avgdvdz'])
Ri_f_na = (9.81*zi/uscale**2)/T_w_na*wT_na/((Re_na['uw']-data_mom_na['avgtxz'])*data_mom_na['avgdudz'] + (Re_na['vw']-data_mom_na['avgtyz'])*data_mom_na['avgdvdz'])

z3D = np.zeros((nx,ny,50),order='F')

for i in range(0,nx):
    for j in range(0,ny):
        z3D[i,j,:] = np.arange(0,50)*dz*zi

z3D[:,:,0] = 0.1

L_a = (-ustar_a[:,:,0:50]**3*T_w_a[:,:,0:50]/(0.41*(9.81*zi/uscale**2)*wT_a[:,:,0:50]))*zi
L_na = (-ustar_na[:,:,0:50]**3*T_w_na[:,:,0:50]/(0.41*(9.81*zi/uscale**2)*wT_na[:,:,0:50]))*zi

#%%Box plot

z0 = 0.1

fig,axs = plt.subplots(1,7,tight_layout=True,figsize=(14,4))

# axs[0].boxplot([data_mom_2D_a['avgUstar'].flatten(),data_mom_2D_na['avgUstar'].flatten()],showfliers=False)
axs[0].boxplot([ustar_a[:,:,0:20].flatten(),ustar_na[:,:,0:20].flatten()],showfliers=False)
axs[1].boxplot([data_sc_2D_a['avgPHIm'].flatten(),data_sc_2D_na['avgPHIm'].flatten()],showfliers=False)
axs[2].boxplot([data_sc_2D_a['avgPHIh'].flatten(),data_sc_2D_na['avgPHIh'].flatten()],showfliers=False)
axs[3].boxplot([data_sc_2D_a['avgPSIm'].flatten(),data_sc_2D_na['avgPSIm'].flatten()],showfliers=False)
axs[4].boxplot([data_sc_2D_a['avgPSIh'].flatten(),data_sc_2D_na['avgPSIh'].flatten()],showfliers=False)
# axs[5].boxplot([z0/(data_sc_2D_a['avgL'].flatten()*zi),z0/(zi*data_sc_2D_na['avgL'].flatten())],showfliers=False)
axs[5].boxplot([(z3D/L_a).flatten(),(z3D/L_na).flatten()],showfliers=False)
axs[6].boxplot([Ri_f_a[:,:,0].flatten(),Ri_f_na[:,:,0].flatten()],showfliers=False)

axs[0].set_title(r"$u_*$",fontsize=15)
axs[1].set_title(r"$\phi_M$",fontsize=15)
axs[2].set_title(r"$\phi_H$",fontsize=15)
axs[3].set_title(r"$\psi_M$",fontsize=15)
axs[4].set_title(r"$\psi_H$",fontsize=15)
axs[5].set_title(r"$z/L$",fontsize=15)
axs[6].set_title(r"$Ri_f$",fontsize=15)

axs[0].set_xticklabels(['A','NA'],fontsize=15)
axs[1].set_xticklabels(['A','NA'],fontsize=15)
axs[2].set_xticklabels(['A','NA'],fontsize=15)
axs[3].set_xticklabels(['A','NA'],fontsize=15)
axs[4].set_xticklabels(['A','NA'],fontsize=15)
axs[5].set_xticklabels(['A','NA'],fontsize=15)
axs[6].set_xticklabels(['A','NA'],fontsize=15)

plt.show()

#%%Compute a(yB)

a_yB = np.zeros((nx,ny,nz-1),order='F')
val = [0.784,-2.582]
coef_u = 0

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(1,nz):
            for idx in range(len(val)):
                tmp_a = val[idx]*(np.log10(data_aniso_a['avgYB'][i,j,k]))**idx
                a_yB[i,j,k-1] = a_yB[i,j,k-1] + tmp_a
            
#%%

for i in range(0,nx):
    for j in range(0,ny):
        for k in range(0,nz):
            if (Re_a['uu'][i,j,k]-data_mom_a['avgtxx'][i,j,k]<0):
                print(i,j,k)
                
#%%Computing the scaling relations from Marc and Iva for unstable and stable stratification

zeta_vec_u = np.array([-1e2,-0.5e2,-1e1,-0.5e1,-1e0,-0.5e0,-1e-1,-5e-2,-1e-2,-5e-3,-1e-3,-5e-4,-1e-4])
zeta_vec_s = abs(zeta_vec_u[::-1])
y_b_vec = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
phi_u_fit_s = np.zeros((len(zeta_vec_s),len(y_b_vec)))
phi_u_fit_u = np.zeros((len(zeta_vec_u),len(y_b_vec)))
coef_a_u = [0.784,-2.582]
coef_a_s = [2.332,-2.047,2.672]
coef_c_s = [0.255,-1.76,5.6,-6.8,2.65]

for i in range(len(y_b_vec)):
    a_u=0
    a_s=0
    c_s=0
    for j in range(len(coef_a_u)):
        tmp_a_u = coef_a_u[j]*(np.log10(y_b_vec[i]))**(j)
        a_u = tmp_a_u + a_u
    for j in range(len(coef_a_s)):
        tmp_a_s = coef_a_s[j]*(y_b_vec[i])**(j)
        a_s = tmp_a_s + a_s
    for j in range(len(coef_c_s)):
        tmp_c_s = coef_c_s[j]*(y_b_vec[i])**(j)
        c_s = tmp_c_s + c_s
    phi_u_fit_u[:,i] = ((1 - 3*zeta_vec_u)**(1/3))
    phi_u_fit_s[:,i] = a_s*((1 + 3*zeta_vec_s)**(c_s))
    print(c_s)
    
    
#%%Plot sigma vs z/L

phi_a = uvpnode2wnode(np.sqrt(abs(Re_a['uu']-data_mom_a['avgtxx'])))/(ustar_a)
phi_na = uvpnode2wnode(np.sqrt(abs(Re_na['uu']-data_mom_na['avgtxx'])))/(ustar_na)

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
nbins = 200

fig = plt.figure(tight_layout=True)
axs1=fig.add_subplot(2,2,1,projection='scatter_density')
axs2=fig.add_subplot(2,2,2,projection='scatter_density')
axs3=fig.add_subplot(2,2,3,projection='scatter_density')
axs4=fig.add_subplot(2,2,4,projection='scatter_density')
    
axs1.scatter_density(abs((z3D/L_a)[:,:,1:50].flatten()[((z3D/L_a)[:,:,1:50].flatten()<0)]),(phi_a[:,:,1:50]/a_yB[:,:,0:49]).flatten()[((z3D/L_a)[:,:,1:50].flatten()<0)],cmap='hot_r')
axs2.scatter_density(abs((z3D/L_a)[:,:,1:50].flatten()[((z3D/L_a)[:,:,1:50].flatten()>0)]),(phi_a[:,:,1:50]/a_yB[:,:,0:49]).flatten()[((z3D/L_a)[:,:,1:50].flatten()>0)],cmap='hot_r')
axs3.scatter_density(abs((z3D/L_na).flatten()[((z3D/L_na).flatten()<0)]),phi_na[:,:,:50].flatten()[((z3D/L_na).flatten()<0)],cmap='hot_r')
axs4.scatter_density(abs((z3D/L_na).flatten()[((z3D/L_na).flatten()>0)]),phi_na[:,:,:50].flatten()[((z3D/L_na).flatten()>0)],cmap='hot_r')

for i in range(len(y_b_vec)):
    axs1.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))

axs1.set_ylim(0,10)
axs2.set_ylim(0,10)
axs3.set_ylim(0,10)
axs4.set_ylim(0,10)

axs1.set_xscale('log')
axs2.set_xscale('log')
axs3.set_xscale('log')
axs4.set_xscale('log')

# for i in range(len(y_b_vec)):
#     axs1.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
#     axs2.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))
#     axs3.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
#     axs4.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))

axs1.set_xlim(1e-4,120)
axs2.set_xlim(1e-4,120)
axs3.set_xlim(1e-4,120)
axs4.set_xlim(1e-4,120)

axs1.invert_xaxis()
axs3.invert_xaxis()

axs3.set_xlabel(r'$-\zeta$',fontsize=15)
axs1.set_ylabel(r'$\phi_u$',fontsize=15)
axs4.set_xlabel(r'$\zeta$',fontsize=15)
axs3.set_ylabel(r'$\phi_u$',fontsize=15)

axs1.set_title('Unstable',fontsize=15)
axs2.set_title('Stable',fontsize=15)

axs1.text(0.7, 0.9,f"w/ Aniso", transform=axs1.transAxes, fontsize=12)
axs2.text(0.7, 0.9,f"w/ Aniso", transform=axs2.transAxes, fontsize=12)
axs3.text(0.7, 0.9,f"Classic", transform=axs3.transAxes, fontsize=12)
axs4.text(0.7, 0.9,f"Classic", transform=axs4.transAxes, fontsize=12)

plt.show()

#%%Compute anisotropy using RAV data

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering, Anisotropy

[xB_a,yB_a,l3_a] = Anisotropy(nx,ny,nz-1,uvpnode2wnode(Re_a['uu']-data_mom_a['avgtxx'])[:,:,1:],uvpnode2wnode(Re_a['vv']-data_mom_a['avgtyy'])[:,:,1:],\
                              Re_a['ww'][:,:,1:]+uvpnode2wnode(data_mom_a['avgtzz'])[:,:,1:],uvpnode2wnode(Re_a['uv']-data_mom_a['avgtxy'])[:,:,1:],\
                                  Re_a['uw'][:,:,1:]-data_mom_a['avgtxz'][:,:,1:],Re_a['vw'][:,:,1:]-data_mom_a['avgtyz'][:,:,1:])

[xB_na,yB_na,l3_na] = Anisotropy(nx,ny,nz-1,uvpnode2wnode(Re_na['uu']-data_mom_na['avgtxx'])[:,:,1:],uvpnode2wnode(Re_na['vv']-data_mom_na['avgtyy'])[:,:,1:],\
                              Re_na['ww'][:,:,1:]+uvpnode2wnode(data_mom_na['avgtzz'])[:,:,1:],uvpnode2wnode(Re_na['uv']-data_mom_na['avgtxy'])[:,:,1:],\
                                  Re_na['uw'][:,:,1:]-data_mom_na['avgtxz'][:,:,1:],Re_na['vw'][:,:,1:]-data_mom_na['avgtyz'][:,:,1:])


#%%Plot Anisotropy onthe-fly and RAV

slc = 128

fig,axs = plt.subplots(2,2,tight_layout=True,sharex=True,sharey=True,figsize=(12,7))
p1 = axs[0,0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_aniso_a['avgYB'][:,slc,:].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
axs[0,0].contour(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_aniso_a['avgYB'][:,slc,:].T,levels=[0.38],colors='black')
p2 = axs[0,1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-1)*dz,yB_a[:,slc,:].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
axs[0,1].contour(np.arange(0,nx)*dx,np.arange(0,nz-1)*dz,yB_a[:,slc,:].T,levels=[0.38],colors='black')
p3 = axs[1,0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_aniso_na['avgYB'][:,slc,:].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
axs[1,0].contour(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_aniso_na['avgYB'][:,slc,:].T,levels=[0.38],colors='black')
p4 = axs[1,1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz-1)*dz,yB_na[:,slc,:].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
axs[1,1].contour(np.arange(0,nx)*dx,np.arange(0,nz-1)*dz,yB_na[:,slc,:].T,levels=[0.38],colors='black')

axs[0,0].set_ylabel(r"$z/z_i$",fontsize=15)
axs[1,0].set_ylabel(r"$z/z_i$",fontsize=15)

axs[1,0].set_xlabel(r"$x/z_i$",fontsize=15)
axs[1,1].set_xlabel(r"$x/z_i$",fontsize=15)

axs[0,0].set_title(r"A - OTF",fontsize=15)
axs[0,1].set_title(r"A - RAV",fontsize=15)
axs[1,0].set_title(r"NA - OTF",fontsize=15)
axs[1,1].set_title(r"NA - RAV",fontsize=15)

cbar = plt.colorbar(p1)
cbar = plt.colorbar(p2)
cbar = plt.colorbar(p3)
cbar = plt.colorbar(p4)

for i in range(len(axs)):
    for j in range(len(axs[0,:])):
        axs[i,j].set_ylim(0,0.5)

plt.show()






























