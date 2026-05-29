#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:16:23 2026

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

Ug = 1
path = '/scratch/general/nfs1/u1450851/LES_Sims/RandTurbStats/128/'

#%%Load RAV data and checkpoint data for the classic scaling

cases = ['A','B','C','D','E']#,'F','G','H','I','J']
# cases = ['A','B']
# cases = ['128']

RAV_time = '30'
check_step = 132000

mom3D_c = dict(); mom3D_a = dict()
mom2D_c = dict(); mom2D_a = dict()
sc3D_c = dict(); sc3D_a = dict()
sc2D_c = dict(); sc2D_a = dict()

for i in range(len(cases)):
    mom3D_c[cases[i]] = xr.open_dataarray(path+'Classic_1ms_'+cases[i]+'/data/Momentum3D/Data_Momentum_'+RAV_time+'min.nc')
    mom2D_c[cases[i]] = xr.open_dataarray(path+'Classic_1ms_'+cases[i]+'/data/Momentum2D/Data_Momentum_2D_'+RAV_time+'min.nc')
    sc3D_c[cases[i]] = xr.open_dataarray(path+'Classic_1ms_'+cases[i]+'/data/Scalar3D/Data_Scalar_'+RAV_time+'min.nc')
    sc2D_c[cases[i]] = xr.open_dataarray(path+'Classic_1ms_'+cases[i]+'/data/Scalar2D/Data_Scalar_2D_'+RAV_time+'min.nc')
    mom3D_a[cases[i]] = xr.open_dataarray(path+'Aniso_1ms_'+cases[i]+'/data/Momentum3D/Data_Momentum_'+RAV_time+'min.nc')
    mom2D_a[cases[i]] = xr.open_dataarray(path+'Aniso_1ms_'+cases[i]+'/data/Momentum2D/Data_Momentum_2D_'+RAV_time+'min.nc')
    sc3D_a[cases[i]] = xr.open_dataarray(path+'Aniso_1ms_'+cases[i]+'/data/Scalar3D/Data_Scalar_'+RAV_time+'min.nc')
    sc2D_a[cases[i]] = xr.open_dataarray(path+'Aniso_1ms_'+cases[i]+'/data/Scalar2D/Data_Scalar_2D_'+RAV_time+'min.nc')


cp_c = dict(); cp_a = dict()
cp_sfc_c = dict(); cp_sfc_a = dict()

for i in range(len(cases)):
    cp_c[cases[i]] = read_checkpoint_aniso(path+'Classic_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny,nz)
    cp_sfc_c[cases[i]] = read_checkpoint_sfc_L(path+'Classic_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny)
    cp_a[cases[i]] = read_checkpoint_aniso(path+'Aniso_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny,nz)
    cp_sfc_a[cases[i]] = read_checkpoint_sfc_L(path+'Aniso_1ms_'+cases[i]+'/output_checkpoint/',[check_step],nx,ny)

#%%Compute fluctuations of the checkpoint data by subtracting the time averaged

uprime_a = dict(); vprime_a = dict(); wprime_a = dict(); Tprime_a = dict()
uprime_c = dict(); vprime_c = dict(); wprime_c = dict(); Tprime_c = dict()
for i in range(len(cases)):
    uprime_a[cases[i]] = cp_a[cases[i]]['u'][:,:,:nz] - mom3D_a[cases[i]][:,:,:,0].data
    vprime_a[cases[i]] = cp_a[cases[i]]['v'][:,:,:nz] - mom3D_a[cases[i]][:,:,:,1].data
    wprime_a[cases[i]] = wnode2uvpnode(cp_a[cases[i]]['w'][:,:,:nz] - mom3D_a[cases[i]][:,:,:,2].data)
    Tprime_a[cases[i]] = cp_a[cases[i]]['SC'][:,:,:nz] - sc3D_a[cases[i]][:,:,:,0].data
    uprime_c[cases[i]] = cp_c[cases[i]]['u'][:,:,:nz] - mom3D_c[cases[i]][:,:,:,0].data
    vprime_c[cases[i]] = cp_c[cases[i]]['v'][:,:,:nz] - mom3D_c[cases[i]][:,:,:,1].data
    wprime_c[cases[i]] = wnode2uvpnode(cp_c[cases[i]]['w'][:,:,:nz] - mom3D_c[cases[i]][:,:,:,2].data)
    Tprime_c[cases[i]] = cp_c[cases[i]]['SC'][:,:,:nz] - sc3D_c[cases[i]][:,:,:,0].data

uu_a = dict(); uu_c = dict()
vv_a = dict(); vv_c = dict()
ww_a = dict(); ww_c = dict()
uv_a = dict(); uv_c = dict()
uw_a = dict(); uw_c = dict()
vw_a = dict(); vw_c = dict()
for i in range(len(cases)):
    uu_a[cases[i]] = uprime_a[cases[i]]*uprime_a[cases[i]]
    vv_a[cases[i]] = vprime_a[cases[i]]*vprime_a[cases[i]]
    ww_a[cases[i]] = wprime_a[cases[i]]*wprime_a[cases[i]]
    uv_a[cases[i]] = uprime_a[cases[i]]*vprime_a[cases[i]]
    uw_a[cases[i]] = uprime_a[cases[i]]*wprime_a[cases[i]]
    vw_a[cases[i]] = vprime_a[cases[i]]*wprime_a[cases[i]]
    
    uu_c[cases[i]] = uprime_c[cases[i]]*uprime_c[cases[i]]
    vv_c[cases[i]] = vprime_c[cases[i]]*vprime_c[cases[i]]
    ww_c[cases[i]] = wprime_c[cases[i]]*wprime_c[cases[i]]
    uv_c[cases[i]] = uprime_c[cases[i]]*vprime_c[cases[i]]
    uw_c[cases[i]] = uprime_c[cases[i]]*wprime_c[cases[i]]
    vw_c[cases[i]] = vprime_c[cases[i]]*wprime_c[cases[i]]

###FRICTION VELOCITY
ustar_a = dict(); ustar_c = dict()
for i in range(len(cases)):
    ustar_a[cases[i]] = (uw_a[cases[i]]**2 + vw_a[cases[i]]**2)**(0.25)
    ustar_c[cases[i]] = (uw_c[cases[i]]**2 + vw_c[cases[i]]**2)**(0.25)

###HEATFLUX
wT_a = dict(); wT_c = dict()
for i in range(len(cases)):
    wT_a[cases[i]] = wprime_a[cases[i]]*Tprime_a[cases[i]]
    wT_c[cases[i]] = wprime_c[cases[i]]*Tprime_c[cases[i]]

###STABILITY PARAMETER
L_a = dict(); L_c = dict()
for i in range(len(cases)):
    L_a[cases[i]] = -(((ustar_a[cases[i]])**3)*cp_a[cases[i]]['SC'][:,:,:nz])/(0.4*(9.81*zi/uscale**2)*(wT_a[cases[i]]))
    L_c[cases[i]] = -(((ustar_c[cases[i]])**3)*cp_c[cases[i]]['SC'][:,:,:nz])/(0.4*(9.81*zi/uscale**2)*(wT_c[cases[i]]))
    
#%%Compute REynolds stresses, ustar, wstar, heatflux

###REYNOLDS STRESSES
uu_a = dict(); uu_c = dict()
vv_a = dict(); vv_c = dict()
ww_a = dict(); ww_c = dict()
uv_a = dict(); uv_c = dict()
uw_a = dict(); uw_c = dict()
vw_a = dict(); vw_c = dict()

for i in range(len(cases)):
    uu_a[cases[i]] = mom3D_a[cases[i]][:,:,:,4].data - mom3D_a[cases[i]][:,:,:,0].data*mom3D_a[cases[i]][:,:,:,0].data - mom3D_a[cases[i]][:,:,:,16].data
    vv_a[cases[i]] = mom3D_a[cases[i]][:,:,:,5].data - mom3D_a[cases[i]][:,:,:,1].data*mom3D_a[cases[i]][:,:,:,1].data - mom3D_a[cases[i]][:,:,:,17].data
    ww_a[cases[i]] = wnode2uvpnode(mom3D_a[cases[i]][:,:,:,6].data - mom3D_a[cases[i]][:,:,:,2].data*mom3D_a[cases[i]][:,:,:,2].data) - mom3D_a[cases[i]][:,:,:,18].data
    uv_a[cases[i]] = mom3D_a[cases[i]][:,:,:,13].data - mom3D_a[cases[i]][:,:,:,0].data*mom3D_a[cases[i]][:,:,:,1].data - mom3D_a[cases[i]][:,:,:,19].data
    uw_a[cases[i]] = wnode2uvpnode(mom3D_a[cases[i]][:,:,:,14].data - uvpnode2wnode(mom3D_a[cases[i]][:,:,:,0].data)*mom3D_a[cases[i]][:,:,:,2].data - mom3D_a[cases[i]][:,:,:,20].data)
    vw_a[cases[i]] = wnode2uvpnode(mom3D_a[cases[i]][:,:,:,15].data - uvpnode2wnode(mom3D_a[cases[i]][:,:,:,1].data)*mom3D_a[cases[i]][:,:,:,2].data - mom3D_a[cases[i]][:,:,:,21].data)
    
    uu_c[cases[i]] = mom3D_c[cases[i]][:,:,:,4].data - mom3D_c[cases[i]][:,:,:,0].data*mom3D_c[cases[i]][:,:,:,0].data - mom3D_c[cases[i]][:,:,:,16].data
    vv_c[cases[i]] = mom3D_c[cases[i]][:,:,:,5].data - mom3D_c[cases[i]][:,:,:,1].data*mom3D_c[cases[i]][:,:,:,1].data - mom3D_c[cases[i]][:,:,:,17].data
    ww_c[cases[i]] = wnode2uvpnode(mom3D_c[cases[i]][:,:,:,6].data - mom3D_c[cases[i]][:,:,:,2].data*mom3D_c[cases[i]][:,:,:,2].data) - mom3D_c[cases[i]][:,:,:,18].data
    uv_c[cases[i]] = mom3D_c[cases[i]][:,:,:,13].data - mom3D_c[cases[i]][:,:,:,0].data*mom3D_c[cases[i]][:,:,:,1].data - mom3D_c[cases[i]][:,:,:,19].data
    uw_c[cases[i]] = wnode2uvpnode(mom3D_c[cases[i]][:,:,:,14].data - uvpnode2wnode(mom3D_c[cases[i]][:,:,:,0].data)*mom3D_c[cases[i]][:,:,:,2].data - mom3D_c[cases[i]][:,:,:,20].data)
    vw_c[cases[i]] = wnode2uvpnode(mom3D_c[cases[i]][:,:,:,15].data - uvpnode2wnode(mom3D_c[cases[i]][:,:,:,1].data)*mom3D_c[cases[i]][:,:,:,2].data - mom3D_c[cases[i]][:,:,:,21].data)

###FRICTION VELOCITY
ustar_a = dict()
ustar_c = dict()

for i in range(len(cases)):
    ustar_a[cases[i]] = (uw_a[cases[i]]**2 + vw_a[cases[i]]**2)**(0.25)
    ustar_c[cases[i]] = (uw_c[cases[i]]**2 + vw_c[cases[i]]**2)**(0.25)

###HEATFLUX
wT_a = dict(); wT_c = dict()

for i in range(len(cases)):
    wT_c[cases[i]] = wnode2uvpnode(sc3D_c[cases[i]][:,:,:,4].data - uvpnode2wnode(sc3D_c[cases[i]][:,:,:,0].data)*mom3D_c[cases[i]][:,:,:,2].data - sc3D_c[cases[i]][:,:,:,7].data)
    wT_a[cases[i]] = wnode2uvpnode(sc3D_a[cases[i]][:,:,:,4].data - uvpnode2wnode(sc3D_a[cases[i]][:,:,:,0].data)*mom3D_a[cases[i]][:,:,:,2].data - sc3D_a[cases[i]][:,:,:,7].data)

###BOUNDARY LAYER HEIGHT
zi_a = dict()
zi_c = dict()
for i in range(len(cases)):
    zi_a[cases[i]] = np.median(np.argmin(wT_a[cases[i]],axis=2)*dz + 0.5*dz)
    zi_c[cases[i]] = np.median(np.argmin(wT_c[cases[i]],axis=2)*dz + 0.5*dz)

###CONVECTIVE VELOCITY SCALE
wstar_a = dict()
wstar_c = dict()
for i in range(len(cases)):
    wstar_a[cases[i]] = ((9.81*zi/(uscale**2))*zi_a[cases[i]]*np.median(wT_a[cases[i]][:,:,0])/np.median(sc3D_a[cases[i]][:,:,0,0].data))**(1/3)
    wstar_c[cases[i]] = ((9.81*zi/(uscale**2))*zi_c[cases[i]]*np.median(wT_c[cases[i]][:,:,0])/np.median(sc3D_c[cases[i]][:,:,0,0].data))**(1/3)

###STABILITY PARAMETER
L_a = dict()
L_c = dict()
for i in range(len(cases)):
    L_a[cases[i]] = -(((ustar_a[cases[i]])**3)*sc3D_a[cases[i]][:,:,:,0])/(0.4*(9.81*zi/uscale**2)*(wT_a[cases[i]]))
    L_c[cases[i]] = -(((ustar_c[cases[i]])**3)*sc3D_c[cases[i]][:,:,:,0])/(0.4*(9.81*zi/uscale**2)*(wT_c[cases[i]]))

#%%Compute anisotropy

yB_a = dict(); xB_a = dict(); lam3_a = dict()

for i in range(len(cases)):
    [xB_a[cases[i]],yB_a[cases[i]],lam3_a[cases[i]]] = Anisotropy(nx,ny,nz,uu_a[cases[i]],vv_a[cases[i]],ww_a[cases[i]],uv_a[cases[i]],\
                                                                 uw_a[cases[i]],vw_a[cases[i]])

#%%Plot the variance vs zeta

k = 0

# phi_u = np.sqrt(np.where(uu_c[cases[k]] >= 0, uu_c[cases[k]], np.nan)) / ((ustar_c[cases[k]]))
phi_u = np.sqrt(np.where(uu_a[cases[k]] >= 0, uu_a[cases[k]], np.nan)) / (ustar_a[cases[k]][:,:,0][:,:,np.newaxis])
# phi_u = np.sqrt(np.where(uu_a[cases[k]] >= 0, uu_a[cases[k]], np.nan)) / ((mom2D_a[cases[k]][:,:,0].data)[:,:,np.newaxis])
z3D = np.ones((nx, ny, nz), order='F') * z_uvp
# zeta = z3D / (L_c[cases[k]].data)
zeta = z3D / (L_a[cases[k]])[:,:,0][:,:,np.newaxis]
# zeta = z3D / (sc2D_a[cases[k]][:,:,1].data)[:,:,np.newaxis]

lvl = 1

# Flatten and remove NaNs
# zeta_flat = np.asarray(np.abs(zeta[:,:,:][(yB_a[cases[k]]>0.4) & (yB_a[cases[k]]<0.5)])).ravel()
# phi_flat = np.asarray(phi_u[:,:,:][(yB_a[cases[k]]>0.4) & (yB_a[cases[k]]<0.5)]).ravel()
zeta_flat = np.asarray(np.abs(zeta[:,:,lvl])).ravel()
phi_flat = np.asarray(phi_u[:,:,lvl]).ravel()
mask = np.isfinite(zeta_flat) & np.isfinite(phi_flat)

fig, axs = plt.subplots(1, 1, tight_layout=True)

yb = np.arange(0.1,0.9,0.1)
zeta_vec = np.logspace(-3,3,1000)
a = [0.784,-2.582]
phi_u_ref = np.zeros((len(yb),len(zeta_vec)))
# coef = 0
for i in range(len(yb)):
    coef = 0
    for j in range(len(a)):
        coef = coef + a[j]*(np.log10(yb[i]))**j
        
    phi_u_ref[i,:] = coef*(1-3*(-zeta_vec))**(1/3)
    axs.plot(zeta_vec,phi_u_ref[i,:],c=cmap(yb[i]))

axs.hexbin(zeta_flat[mask], phi_flat[mask], bins=10, alpha=0.5, cmap='hot_r',mincnt=10,extent=(-3,4,0,10),xscale='log')
# axs.set_xscale('log')
axs.invert_xaxis()
axs.set_xlabel(r"$-\zeta$",fontsize=14)
axs.set_ylabel(r"$\sigma/u_*$",fontsize=14)
axs.set_ylim(0,10)
plt.show()
















































































