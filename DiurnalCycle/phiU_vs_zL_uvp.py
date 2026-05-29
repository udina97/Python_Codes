#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 01:58:23 2025

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

#%%#Simulation parameters

path_fig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/DiurnalCycle/'
sim_a = 'diurnal_c_aniso_L3D'
path_a = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_a+'/data/Momentum3D/'
sim_na = 'diurnal_c_noaniso_L3D_fix'
path_na = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim_na+'/data/Momentum3D/'
StartTime = 1296000
AvgIter = 24000
StartFile = 27
EndFile = 40

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.075 #05; %0.000005;
zi = 3000.0 #in m
uscale = 0.4 #set equal to whatever is in parameters.py
ug = 9.5
Tscale = 288 #in K
textsize = 20
wbase = 1000

nx = 128
ny = 128
nz = 384
lx = 1000*np.pi/zi
ly = 1000*np.pi/zi
lz = 3000/zi
dx = lx/nx
dy = ly/ny
dz = lz/nz

plot_profile = 0
plot_color = 0

#%% Simulation path

# simPath = path

#%%Bicheng functions

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

#%% Select a random x and y coordinate

import random

x_rnd = random.randint(0,nx)
y_rnd = random.randint(0,ny)

#%%Import L from surface RAV files

L_sfc_a = dict()
L_sfc_na = dict()

for i in range(StartFile,EndFile):
    L_sfc_a[str(StartTime + i*AvgIter)] = xr.open_dataarray(path_a+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,1]
    L_sfc_na[str(StartTime + i*AvgIter)] = xr.open_dataarray(path_na+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,1]
    
    print(f'Done with File: {i}')
    
#%%Plot Pcolor of L from the RAV LES

idx = 35

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,4))

p1=axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,zi*L_sfc_a[str(StartTime + idx*AvgIter)].T,cmap='jet')
p2=axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,zi*L_sfc_na[str(StartTime + idx*AvgIter)].T,cmap='jet')

for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=15)
    
axs[0].set_ylabel(r"$y/z_i$",fontsize=15)
axs[0].set_title(r"Aniso", fontsize=15)
axs[1].set_title(r"No Aniso",fontsize=15)

axs[0].text(0.1,0.8,f"Median = {np.median(L_sfc_a[str(StartTime + idx*AvgIter)])*zi:.2f}",fontsize=15,color='black',bbox=dict(facecolor='white',alpha=0.6))
axs[1].text(0.1,0.8,f"Median = {np.median(L_sfc_na[str(StartTime + idx*AvgIter)])*zi:.2f}",fontsize=15,color='black',bbox=dict(facecolor='white',alpha=0.6))
            
cbar=plt.colorbar(p1)
cbar=plt.colorbar(p2)

plt.show()

#%%Box plot of L from surface RAV

L_flat_a = np.array([])
L_flat_na = np.array([])

for i in range(StartFile,EndFile):
    L_flat_a = np.concatenate((L_flat_a,((L_sfc_a[str(StartTime + i*AvgIter)]*zi)).flatten()))
    L_flat_na = np.concatenate((L_flat_na,((L_sfc_na[str(StartTime + i*AvgIter)]*zi)).flatten()))

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,4))

axs.boxplot([L_flat_a[(L_flat_a<0)],L_flat_na[(L_flat_na<0)]],showfliers=False)

axs.set_title(r"$z/L$",fontsize=15)

axs.set_xticklabels(['A','NA'],fontsize=15)

plt.show()

#%%Compute L using RAV files at first uvp node

L_rav_a = dict()
L_rav_na = dict()
kvonk = 0.41
g=9.81*zi/(uscale**2)

for i in range(StartFile,EndFile):
    T = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,0]
    tsz = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,7]
    txz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,20]
    tyz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,21]
    
    ustar_sfc_a = ((- txz)**2 + (- tyz)**2)**(1/4)
    wT_a = (- tsz)
    
    L_rav_a[str(StartTime + AvgIter*i)] = -(ustar_sfc_a**3)*T/(kvonk*g*wT_a)
    
    #-----------------------------------------------------------------------------------------------------------------
    T = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,0]
    tsz = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,7]
    txz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,20]
    tyz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0,21]
    
    ustar_sfc_na = ((- txz)**2 + (- tyz)**2)**(1/4)
    wT_na = (- tsz)
    
    L_rav_na[str(StartTime + AvgIter*i)] = -(ustar_sfc_na**3)*T/(kvonk*g*wT_na)
    
    print(f'Done with File: {i}')


#%%Plot Pcolor of L from the RAV LES

idx = 30

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,4))

p1=axs[0].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,zi*L_rav_a[str(StartTime + idx*AvgIter)].T,cmap='jet')
p2=axs[1].pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,zi*L_rav_na[str(StartTime + idx*AvgIter)].T,cmap='jet')

for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=15)
    
axs[0].set_ylabel(r"$y/z_i$",fontsize=15)
axs[0].set_title(r"Aniso", fontsize=15)
axs[1].set_title(r"No Aniso",fontsize=15)

axs[0].text(0.1,0.8,f"Median = {np.median(L_rav_a[str(StartTime + idx*AvgIter)])*zi:.2f}",fontsize=15,color='black',bbox=dict(facecolor='white',alpha=0.6))
axs[1].text(0.1,0.8,f"Median = {np.median(L_rav_na[str(StartTime + idx*AvgIter)])*zi:.2f}",fontsize=15,color='black',bbox=dict(facecolor='white',alpha=0.6))
            
cbar=plt.colorbar(p1)
cbar=plt.colorbar(p2)

plt.show()

#%%Box plot of L from surface RAV

L_flat_a = np.array([])
L_flat_na = np.array([])

for i in range(StartFile,EndFile):
    L_flat_a = np.concatenate((L_flat_a,((L_rav_a[str(StartTime + i*AvgIter)]*zi)).flatten()))
    L_flat_na = np.concatenate((L_flat_na,((L_rav_na[str(StartTime + i*AvgIter)]*zi)).flatten()))

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,4))

axs.boxplot([L_flat_a[(L_flat_a>0)],L_flat_na[(L_flat_na>0)]],showfliers=False)

axs.set_title(r"$z/L$",fontsize=15)

axs.set_xticklabels(['A','NA'],fontsize=15)

plt.show()

#%%

# COMPUTING SIGMA_U AND L AS 3D FIELDS IN THE SURFACE LAYER ALL ON UVP NODES

#%%

#%%Compute sigmaU at first uvp node

sigma_a = dict()
sigma_na = dict()

for i in range(StartFile,EndFile):
    u = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    # u_w = uvpnode2wnode(u)
    uu = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    # uu_w = uvpnode2wnode(uu)
    txx = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,16]
    # txx = uvpnode2wnode(txx)
    sigma_a[str(StartTime + AvgIter*i)] = np.sqrt((uu - u*u - txx))
    
    u = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    # u_w = uvpnode2wnode(u)
    uu = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    # uu_w = uvpnode2wnode(uu)
    txx = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,16]
    # txx_w = uvpnode2wnode(txx)
    sigma_na[str(StartTime + AvgIter*i)] = np.sqrt((uu - u*u - txx))
    
    print(f'Done with File: {i}')
    
#%%Compute ustar at sfc
    
ustar_a = dict()
ustar_na = dict()

for i in range(StartFile,EndFile):
    u = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    v = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    
    uw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,14]
    uw_uvp = wnode2uvpnode(uw)
    vw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,15]
    vw_uvp = wnode2uvpnode(vw)
    txz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,20]
    txz_uvp = wnode2uvpnode(txz)
    tyz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,21]
    tyz_uvp = wnode2uvpnode(tyz)
    
    ustar_a[str(StartTime + AvgIter*i)] = ((uw_uvp - u*w_uvp - txz_uvp)**2 + (vw_uvp - v*w_uvp - tyz_uvp)**2)**(1/4)
    
    u = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    v = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    w = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    
    uw = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,14]
    uw_uvp = wnode2uvpnode(uw)
    vw = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,15]
    vw_uvp = wnode2uvpnode(vw)
    txz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,20]
    txz_uvp = wnode2uvpnode(txz)
    tyz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,21]
    tyz_uvp = wnode2uvpnode(tyz)
    
    ustar_na[str(StartTime + AvgIter*i)] = ((uw_uvp - u*w_uvp - txz_uvp)**2 + (vw_uvp - v*w_uvp - tyz_uvp)**2)**(1/4)
    print(f'Done with File: {i}')

#%%Import temperature

T3D_a = dict()
T3D_na = dict()

for i in range(StartFile,EndFile):
    T3D_a[str(StartTime + i*AvgIter)] = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + i*AvgIter)+'.nc').data[:,:,0:50,0]
    T3D_na[str(StartTime + i*AvgIter)] = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + i*AvgIter)+'.nc').data[:,:,0:50,0]
    print(f'Done with File: {i}')
    
#%%Compute the heat flux

wT_a = dict()
wT_na =dict()

for i in range(StartFile,EndFile):
    
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    wT = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    wT_uvp = wnode2uvpnode(wT)
    tsz = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,7]
    tsz_uvp = wnode2uvpnode(tsz)
    
    wT_a[str(StartTime + i*AvgIter)] = (wT_uvp - T3D_a[str(StartTime + i*AvgIter)]*w_uvp - tsz_uvp)
    
    w = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    w_uvp = wnode2uvpnode(w)
    wT = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    wT_uvp = wnode2uvpnode(wT)
    tsz = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,7]
    tsz_uvp = wnode2uvpnode(tsz)
    
    wT_na[str(StartTime + i*AvgIter)] = (wT_uvp - T3D_na[str(StartTime + i*AvgIter)]*w_uvp - tsz_uvp)
    
    print(f'Done with File: {i}')
    
#%%Compute L

L3D_a = dict()
L3D_na = dict()

for i in range(StartFile,EndFile):
    L3D_a[str(StartTime+i*AvgIter)] = -T3D_a[str(StartTime+i*AvgIter)]*ustar_a[str(StartTime+i*AvgIter)]**3/(g*kvonk*wT_a[str(StartTime+i*AvgIter)])
    L3D_na[str(StartTime+i*AvgIter)] = -T3D_na[str(StartTime+i*AvgIter)]*ustar_na[str(StartTime+i*AvgIter)]**3/(g*kvonk*wT_na[str(StartTime+i*AvgIter)])

#%%Compute phiU at the surface using aigmaU from the first uvp node and ustar from the surface

phiU_a = dict()
phiU_na = dict()

for i in range(StartFile,EndFile):
    phiU_a[str(StartTime+i*AvgIter)] = sigma_a[str(StartTime+i*AvgIter)]/ustar_a[str(StartTime+i*AvgIter)]
    phiU_na[str(StartTime+i*AvgIter)] = sigma_na[str(StartTime+i*AvgIter)]/ustar_na[str(StartTime+i*AvgIter)]

#%%Compute a 3D field for the vertical coordinate

z3D = np.zeros((nx,ny,50),order='F')

for i in range(0,nx):
    for j in range(0,ny):
        z3D[i,j,:] = np.arange(0,50)*dz + dz/2

#%%Create flat vectors

zeta_a = np.array([])
zeta_na = np.array([])
phiU_flat_a = np.array([])
phiU_flat_na = np.array([])

for i in range(StartFile,EndFile):
    zeta_a = np.concatenate((zeta_a,((z3D[:,:,0]*zi)/(zi*L_rav_a[str(StartTime+i*AvgIter)][:,:])).flatten()))
    zeta_na = np.concatenate((zeta_na,((z3D[:,:,0]*zi)/(zi*L_rav_na[str(StartTime+i*AvgIter)][:,:])).flatten()))
    phiU_flat_a = np.concatenate((phiU_flat_a,phiU_a[str(StartTime+i*AvgIter)][:,:,0].flatten()))
    phiU_flat_na = np.concatenate((phiU_flat_na,phiU_na[str(StartTime+i*AvgIter)][:,:,0].flatten()))

#%%Density plot using Gaussian KDE

import seaborn as sns
from matplotlib.colors import Normalize

x_zeta = abs(zeta_na[(zeta_na<0)])
y_phi = phiU_flat_na[(zeta_na<0)]

# x_zeta = abs(zeta_tot_na[(zeta_tot_na<0) & (phi_na>0)])
# y_phi = phi_na[(zeta_tot_na<0) & (phi_na>0)]

sample = 100000
index = np.random.choice(len(x_zeta),size=sample,replace=False)

x_sample = x_zeta[index]
y_sample = y_phi[index]
plt.figure(tight_layout=True)

plt.scatter(x_sample,y_sample,c='k',s=1)

#---------------------------------------------------------
# from scipy.stats import gaussian_kde

# xy = np.vstack([x_sample,y_sample])
# kde = gaussian_kde(xy)

# xgrid = np.linspace(x_sample.min(),x_sample.max(),100)
# ygrid = np.linspace(y_sample.min(),y_sample.max(),100)
# X,Y = np.meshgrid(xgrid,ygrid)
# Z = kde(np.vstack([X.ravel(),Y.ravel()])).reshape(X.shape)

# plt.pcolormesh(X,Y,Z,shading='auto',cmap='hot_r')
#----------------------------------------------------------

log_bin_x = np.logspace(np.log10(x_sample.min()), np.log10(x_sample.max()), 200)
bin_y = np.linspace((y_sample.min()),(y_sample.max()), 200)

h,xedge,yedge,img=plt.hist2d(x_sample,y_sample,bins=[log_bin_x,bin_y],cmap='viridis',density=False,norm=Normalize(vmin=0,vmax=1))
h[(h<np.nanmax(h)/100)] = np.nan
img.set_array(h.T.ravel())
img.set_array(img.get_array()/np.nanmax(h))

# sns.kdeplot(x=x_sample,y=y_sample,fill=True,cmap='hot_r',bw_adjust=0.5)

plt.ylim(0,10)

plt.xscale('log')

plt.plot(np.arange(1e-4,120,0.01),2.55*(1-3*(-np.arange(1e-4,120,0.01)))**(1/3),c='r')

# for i in range(len(y_b_vec)):
#     plt.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    # axs2.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))
    # axs3.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    # axs4.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))

plt.xlim(1e-4,120)

plt.gca().invert_xaxis()

plt.xlabel(r'$-\zeta$',fontsize=15)
plt.ylabel(r'$\phi_u$',fontsize=15)
# plt.title('No Anisotropy',fontsize=15)

plt.colorbar(img)

plt.show()

#%%

coef = np.polyfit(phiU_flat_a,phiU_flat_na,1)
poly = np.poly1d(coef)


fig = plt.figure()

# plt.scatter(phiU_flat_a,phiU_flat_na,s=1)
plt.plot(phiU_flat_a,phiU_flat_na,'yo',phiU_flat_a,poly(phiU_flat_a),'--k')

plt

plt.show()























