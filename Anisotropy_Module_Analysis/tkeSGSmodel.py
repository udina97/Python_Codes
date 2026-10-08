#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:32:04 2026

@author: u1450851
"""
#%%
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom, read_checkpoint_tke
from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Simulation parameters

nx = 128
ny = 128
nz = 128

lx = 5.120 #2*np.pi
ly = 5.120 #2*np.pi
lz = 2.048 #1

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1024
uscale = 1
Tscale = 300
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load surface checkpoint/istantaneous fields 
# sim = 'yb_test_v2'

sim = 'simC_Patton2011'

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

step = 51000
avgtime = 15   

# checkpnt = read_checkpoint_mom(path+sim+'/output_checkpoint/',[step],nx,ny,nz)
checkpnt_sc = read_checkpoint(path+sim+'/output_checkpoint/',[step],nx,ny,nz, False)
checkpnt_aniso = read_checkpoint_aniso(path+sim+'/output_checkpoint/',[step],nx,ny,nz)
checkpnt_tke = read_checkpoint_tke(path+sim+'/output_checkpoint/',[step],nx,ny,nz)
checkpnt_sfc = read_checkpoint_sfc_L(path+sim+'/output_checkpoint/',[step],nx,ny)
# RAV_aniso = xr.open_dataarray(path+sim+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
# RAV_mom = xr.open_dataarray(path+sim+'/data/Momentum3D/Data_Momentum_'+str(avgtime)+'min.nc').data

# NumVariables = 27
# NumIt = 1
# NumFiles = 1
# StartTime = 1300
# anisotropy_flag = False
# data = xr.DataArray(np.ones(shape = (nx,ny,nz,NumVariables),order='F'),\
#                         dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
#                         'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                         'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                         'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                         'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']}) 
    
# vardata = xr.DataArray(np.ones(shape = (nx,ny,nz,NumVariables),order='F'),\
#                     dims=('x', 'y','z','variable'), coords = {'variable':['avgU','avgV',\
#                     'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                     'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                     'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                     'avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']})

# data = load_momentum(nx,ny,nz,NumVariables,NumIt,NumFiles,StartTime,path+sim+'/output_RAV/',vardata,anisotropy_flag)

#%%Plot anisotropy invariant

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

axs[0].pcolormesh(x,z_uvp,np.median(checkpnt_aniso['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[0].pcolormesh(x,z_uvp,checkpnt_aniso['yB'][:,ny//2,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[0].pcolormesh(x,z_w,checkpnt_aniso['yB'][nx//2,:,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[0].pcolormesh(x,y,checkpnt_aniso['yB'][:,:,15].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x,z_uvp,np.median(RAV_aniso[:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

plt.show()

#%%Velocity magnitude at the top

U_mag = np.mean(np.sqrt(checkpnt_sc['u']**2 + checkpnt_sc['v']**2)[:,:,-1])*uscale
# U_mag = np.mean(np.sqrt(RAV_mom[:,:,:,0]**2 + RAV_mom[:,:,:,1]**2)[:,:,-1])*uscale
print('Velocity magnitude at the top: ',U_mag)

#%%Plot subgrid scale tke

# k_sgs = checkpnt_aniso["tke_sgs"][:,:,:nz]
k_sgs = checkpnt_tke['k_sgs_uvp'][:,:,:nz]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x,z_w,np.median(k_sgs,axis=(1)).T,cmap=cmap,vmin=0,vmax=1)
# axs.pcolormesh(x,z_w,k_sgs[nx//2,:,:].T,cmap=cmap)
# axs.pcolormesh(x,y,k_sgs[:,:,5].T,cmap=cmap)

plt.show()

#%%PLot velocity components

fig,axs = plt.subplots(2,3,tight_layout=True,figsize = (10,6))

axs[0,0].pcolormesh(x,z_uvp,checkpnt_aniso['u_new'][:,ny//2,:nz].T,cmap='viridis')
axs[0,1].pcolormesh(x,z_uvp,checkpnt_aniso['v_new'][:,ny//2,:nz].T,cmap='viridis')
axs[0,2].pcolormesh(x,z_uvp,checkpnt_aniso['w_new'][:,ny//2,:nz].T,cmap='viridis')

axs[1,0].pcolormesh(x,z_uvp,checkpnt_sc['u'][:,ny//2,:nz].T,cmap='viridis')
axs[1,1].pcolormesh(x,z_uvp,checkpnt_sc['v'][:,ny//2,:nz].T,cmap='viridis')
axs[1,2].pcolormesh(x,z_w,checkpnt_sc['w'][:,ny//2,:nz].T,cmap='viridis')

# axs[0].pcolormesh(y,z_uvp,checkpnt_aniso['u_new'][nx//2,:,:nz].T,cmap='viridis')
# axs[1].pcolormesh(y,z_uvp,checkpnt_aniso['v_new'][nx//2,:,:nz].T,cmap='viridis')
# axs[2].pcolormesh(y,z_w,checkpnt_aniso['w_new'][nx//2,:,:nz].T,cmap='viridis')

# axs[0].pcolormesh(x,y,checkpnt_aniso['u_new'][:,:,15].T,cmap='viridis')
# axs[1].pcolormesh(x,y,checkpnt_aniso['v_new'][:,:,15].T,cmap='viridis')
# axs[2].pcolormesh(x,y,checkpnt_aniso['w_new'][:,:,15].T,cmap='viridis')

axs[0,0].set_title('u')
axs[0,1].set_title('v')
axs[0,2].set_title('w')

plt.show()

#%%Plot pcolor of temperature

fig,axs = plt.subplots(1,1,tight_layout=True)
# axs.pcolormesh(x,z_uvp,np.median(checkpnt_sc['SC'],axis=(1))[:,:nz].T,cmap='viridis')
axs.plot(np.mean(checkpnt_sc['SC'],axis=(0,1))[:-1],z_uvp,c='blue')
plt.show()

#%%Compute vertical gradient of temperature and for each x,y select the vertical point where the gradient is max

def centered_difference_z(field, dz, axis=2):
    """Compute d(field)/dz with centered differences on interior levels."""
    field = np.asarray(field)
    gradient = np.empty_like(field, dtype=np.float64)

    interior = [slice(None)] * field.ndim
    lower = [slice(None)] * field.ndim
    upper = [slice(None)] * field.ndim
    interior[axis] = slice(1, -1)
    lower[axis] = slice(0, -2)
    upper[axis] = slice(2, None)
    gradient[tuple(interior)] = (field[tuple(upper)] - field[tuple(lower)])/(2*dz)

    first = [slice(None)] * field.ndim
    second = [slice(None)] * field.ndim
    first[axis] = 0
    second[axis] = 1
    gradient[tuple(first)] = (field[tuple(second)] - field[tuple(first)])/dz

    last = [slice(None)] * field.ndim
    next_to_last = [slice(None)] * field.ndim
    last[axis] = -1
    next_to_last[axis] = -2
    gradient[tuple(last)] = (field[tuple(last)] - field[tuple(next_to_last)])/dz

    return gradient


temperature = checkpnt_sc['SC'][:,:,:nz]
dTdz = centered_difference_z(temperature, dz)

max_dTdz_level = np.argmax(dTdz, axis=2)
max_dTdz_height = (max_dTdz_level + 0.5)*dz*zi

#%%Compute REynolds stresses and TKE, resolved and SGS

Rxx_res = checkpnt_aniso['uu_new'][:,:,:nz] - checkpnt_aniso['u_new'][:,:,:nz]*checkpnt_aniso['u_new'][:,:,:nz]
Rxx_sgs = - checkpnt_aniso['txx_new'][:,:,:nz] + (2/3)*checkpnt_aniso['tke_sgs_new'][:,:,:nz]
Rxx_tot = Rxx_res + Rxx_sgs
# Rxx_rav_res = RAV_mom[:,:,:,4] - RAV_mom[:,:,:,0]*RAV_mom[:,:,:,0]
# Rxx_rav_sgs = - RAV_mom[:,:,:,16] + (2/3)*wnode2uvpnode(RAV_mom[:,:,:,-1])
# Rxx_rav_tot = Rxx_rav_res + Rxx_rav_sgs

Ryy_res = checkpnt_aniso['vv_new'][:,:,:nz] - checkpnt_aniso['v_new'][:,:,:nz]*checkpnt_aniso['v_new'][:,:,:nz]
Ryy_sgs = - checkpnt_aniso['tyy_new'][:,:,:nz] + (2/3)*checkpnt_aniso['tke_sgs_new'][:,:,:nz]
Ryy_tot = Ryy_res + Ryy_sgs
# Ryy_rav_res = RAV_mom[:,:,:,5] - RAV_mom[:,:,:,1]*RAV_mom[:,:,:,1]
# Ryy_rav_sgs = - RAV_mom[:,:,:,17] + (2/3)*wnode2uvpnode(RAV_mom[:,:,:,-1])
# Ryy_rav_tot = Ryy_rav_res + Ryy_rav_sgs

Rzz_res = checkpnt_aniso['ww_new'][:,:,:nz] - checkpnt_aniso['w_new'][:,:,:nz]*checkpnt_aniso['w_new'][:,:,:nz]
Rzz_sgs = - checkpnt_aniso['tzz_new'][:,:,:nz] + (2/3)*checkpnt_aniso['tke_sgs_new'][:,:,:nz]
Rzz_tot = Rzz_res + Rzz_sgs
# Rzz_rav_res = wnode2uvpnode(RAV_mom[:,:,:,6] - RAV_mom[:,:,:,2]*RAV_mom[:,:,:,2])
# Rzz_rav_sgs = - RAV_mom[:,:,:,18] + (2/3)*wnode2uvpnode(RAV_mom[:,:,:,-1])
# Rzz_rav_tot = Rzz_rav_res + Rzz_rav_sgs

Rxy_res = checkpnt_aniso['uv_new'][:,:,:nz] - checkpnt_aniso['u_new'][:,:,:nz]*checkpnt_aniso['v_new'][:,:,:nz]
Rxy_sgs = - checkpnt_aniso['txy_new'][:,:,:nz]
Rxy_tot = Rxy_res + Rxy_sgs
# Rxy_rav_res = RAV_mom[:,:,:,13] - RAV_mom[:,:,:,1]*RAV_mom[:,:,:,0]
# Rxy_rav_sgs = - RAV_mom[:,:,:,19]
# Rxy_rav_tot = Rxy_rav_res + Rxy_rav_sgs

Rxz_res = checkpnt_aniso['uw_new'][:,:,:nz] - checkpnt_aniso['u_new'][:,:,:nz]*checkpnt_aniso['w_new'][:,:,:nz]
Rxz_sgs = - checkpnt_aniso['txz_new'][:,:,:nz]
Rxz_tot = Rxz_res + Rxz_sgs
# Rxz_rav_res = wnode2uvpnode(RAV_mom[:,:,:,14] - RAV_mom[:,:,:,2]*uvpnode2wnode(RAV_mom[:,:,:,0]))
# Rxz_rav_sgs = - wnode2uvpnode(RAV_mom[:,:,:,20])
# Rxz_rav_tot = Rxz_rav_res + Rxz_rav_sgs

Ryz_res = checkpnt_aniso['vw_new'][:,:,:nz] - checkpnt_aniso['v_new'][:,:,:nz]*checkpnt_aniso['w_new'][:,:,:nz]
Ryz_sgs = - checkpnt_aniso['tyz_new'][:,:,:nz]
Ryz_tot = Ryz_res + Ryz_sgs
# Ryz_rav_res = wnode2uvpnode(RAV_mom[:,:,:,15] - RAV_mom[:,:,:,2]*uvpnode2wnode(RAV_mom[:,:,:,1]))
# Ryz_rav_sgs = - wnode2uvpnode(RAV_mom[:,:,:,21])
# Ryz_rav_tot = Ryz_rav_res + Ryz_rav_sgs

TKE_res = 0.5*(Rxx_res + Ryy_res + Rzz_res)#*(uscale**2)
TKE_sgs = 0.5*(Rxx_sgs + Ryy_sgs + Rzz_sgs)#*(uscale**2)
TKE_tot = TKE_res + TKE_sgs
# TKE_rav_res = 0.5*(Rxx_rav_res + Ryy_rav_res + Rzz_rav_res)#*(uscale**2)
# TKE_rav_sgs = 0.5*(Rxx_rav_sgs + Ryy_rav_sgs + Rzz_rav_sgs)#*(uscale**2)
# TKE_rav_tot = TKE_rav_res + TKE_rav_sgs

TKE_res_rat = (TKE_res)/(TKE_tot)
TKE_sgs_rat = (TKE_sgs)/(TKE_tot)
# TKE_rav_res_rat = (TKE_rav_res)/(TKE_rav_tot)*100
# TKE_rav_sgs_rat = (TKE_rav_sgs)/(TKE_rav_tot)*100

#%%Plot pcolor of Reynolds terms

fig,axs = plt.subplots(2,3,tight_layout=True)

axs[0,0].pcolormesh(x,z_uvp,Rxx_res[:,ny//2,:].T,cmap='bwr',vmin=0,vmax=1)
axs[0,1].pcolormesh(x,z_uvp,Ryy_res[:,ny//2,:].T,cmap='bwr',vmin=0,vmax=1)
axs[0,2].pcolormesh(x,z_uvp,Rzz_res[:,ny//2,:].T,cmap='bwr',vmin=0,vmax=1)

# axs[1,0].pcolormesh(x,z_uvp,Rxx_rav_sgs[:,ny//2,:].T,cmap='bwr',vmin=0,vmax=2)
# axs[1,1].pcolormesh(x,z_uvp,Ryy_rav_sgs[:,ny//2,:].T,cmap='bwr',vmin=0,vmax=2)
# axs[1,2].pcolormesh(x,z_uvp,Rzz_rav_sgs[:,ny//2,:].T,cmap='bwr',vmin=0,vmax=2)

axs[0,0].set_title('Rxx')
axs[0,1].set_title('Ryy')
axs[0,2].set_title('Rzz')

plt.show()


#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x,z_uvp,checkpnt_aniso['tke_sgs_new'][:,ny//2,:nz].T,cmap='viridis',vmin=0,vmax=1)

plt.show()

#%%Compute anisotropy

xB_res,yB_res,lamba3_res = Anisotropy(nx,ny,nz,Rxx_res,Ryy_res,Rzz_res,Rxy_res,Rxz_res,Ryz_res)
xB_sgs,yB_sgs,lamba3_sgs = Anisotropy(nx,ny,nz,Rxx_sgs,Ryy_sgs,Rzz_sgs,Rxy_sgs,Rxz_sgs,Ryz_sgs)

#%%Plot yB computed using the resolved, sgs and total terms

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,3))

axs[0].pcolormesh(x,z_uvp,np.median(yB_res,axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[1].pcolormesh(x,z_uvp,np.median(yB_sgs,axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[2].pcolormesh(x,z_uvp,np.median(checkpnt_aniso['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

# axs[0].pcolormesh(x,z_uvp,(yB_res)[:,20,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x,z_uvp,(yB_sgs)[:,20,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[2].pcolormesh(x,z_uvp,(checkpnt_aniso['yB'])[:,20,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs[0].set_xlabel('x')
axs[0].set_ylabel('z')
axs[1].set_xlabel('x')
axs[2].set_xlabel('x')

axs[0].set_title(r'$y_B^{res} \quad b_{ij} = \frac{R_{ij}^{Res}}{2e^{Res}} - \frac{1}{3}\delta_{ij}$')
axs[1].set_title(r'$y_B^{sgs} \quad b_{ij} = \frac{R_{ij}^{SGS}}{2e^{SGS}} - \frac{1}{3}\delta_{ij}$')
axs[2].set_title(r'$y_B^{tot} \quad b_{ij} = \frac{R_{ij}^{TOT}}{2e^{TOT}} - \frac{1}{3}\delta_{ij}$')

plt.show()

#%%Plot the fractional contribution of TKE resolved and sgs to the total value

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

p1 = axs[0].pcolormesh(x,z_uvp,np.median(TKE_res_rat,axis=(1)).T,cmap='bwr',vmin=0,vmax=1)
p2 = axs[1].pcolormesh(x,z_uvp,np.median(TKE_sgs_rat,axis=(1)).T,cmap='bwr',vmin=0,vmax=1)

# p1 = axs[0].pcolormesh(x,z_uvp,(TKE_res_rat)[:,20,:].T,cmap='bwr',vmin=0,vmax=1)
# p2 = axs[1].pcolormesh(x,z_uvp,(TKE_sgs_rat)[:,20,:].T,cmap='bwr',vmin=0,vmax=1)

axs[0].set_title(r'$f^{res} = \frac{e^{res}}{e^{res} + e^{sgs}}$')
axs[1].set_title(r'$f^{sgs} = 1 - f^{res}$')

cbar1 = fig.colorbar(p1, ax=axs[0])
cbar2 = fig.colorbar(p2, ax=axs[1])

axs[0].set_xlabel('x')
axs[0].set_ylabel('z')
axs[1].set_xlabel('x')

plt.show()

#%%Plot profiles of the fractional contribution

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.median(TKE_res_rat,axis=(0,1)),z_uvp,label='TKE_res')
axs.axhline(dz*np.where(np.median(TKE_res_rat,axis=(0,1))>0.5)[0][0],color='k',linestyle='--')
axs.axhline(dz*np.where(np.median(TKE_res_rat,axis=(0,1))>0.6)[0][0],color='k',linestyle='--')
axs.axhline(dz*np.where(np.median(TKE_res_rat,axis=(0,1))>0.7)[0][0],color='k',linestyle='--')
axs.axhline(dz*np.where(np.median(TKE_res_rat,axis=(0,1))>0.8)[0][0],color='k',linestyle='--')

axs.set_ylabel('z')
axs.set_xlabel(r'$f^{res}$')

axs.set_xlim(0,1)
axs.set_ylim(0,1)
axs.grid()

plt.show()

#%%Compute yB by summing the resolved and sgs terms multiplied by the corresponding TKE ratio

fig,axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

axs[0].pcolormesh(x,z_uvp,np.median(yB_res*TKE_res_rat + yB_sgs*TKE_sgs_rat,axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs[1].pcolormesh(x,z_uvp,np.median(checkpnt_aniso['yB'],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

# axs[0].pcolormesh(x,z_uvp,(yB_res*TKE_res_rat + yB_sgs*TKE_sgs_rat)[:,20,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
# axs[1].pcolormesh(x,z_uvp,(checkpnt_aniso['yB'])[:,20,:].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs[0].set_title(r'$f^{res}y_B^{res} + f^{sgs}y_B^{sgs}$')
axs[1].set_title(r'$y_B^{tot}$')

axs[0].set_xlabel('x')
axs[0].set_ylabel('z')
axs[1].set_xlabel('x')

plt.show()

#%%Plot the TKE ratio PDFs

lvl = 40

TKE_res_lvl = TKE_res_rat[:,:,lvl].flatten()
TKE_sgs_lvl = TKE_sgs_rat[:,:,lvl].flatten()

TKE_res_lvl = TKE_res_lvl[np.isfinite(TKE_res_lvl)]
TKE_sgs_lvl = TKE_sgs_lvl[np.isfinite(TKE_sgs_lvl)]

x_pdf = np.linspace(min(TKE_res_lvl.min(),TKE_sgs_lvl.min()),\
                    max(TKE_res_lvl.max(),TKE_sgs_lvl.max()),1000)

kde_res = gaussian_kde(TKE_res_lvl)
pdf_res = kde_res(x_pdf)

kde_sgs = gaussian_kde(TKE_sgs_lvl)
pdf_sgs = kde_sgs(x_pdf)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(x_pdf,pdf_res,label='TKE_res')
axs.plot(x_pdf,pdf_sgs,label='TKE_sgs')

axs.set_xlabel('TKE')
axs.set_ylabel('PDF')
axs.set_title('PDFs at z level '+str(lvl))
axs.legend()

plt.show()

#%%Plot PDFs of TKE resolved and sgs at a selected vertical level

lvl = 0

TKE_res_lvl = TKE_res[:,:,lvl].flatten()
TKE_sgs_lvl = TKE_sgs[:,:,lvl].flatten()

TKE_res_lvl = TKE_res_lvl[np.isfinite(TKE_res_lvl)]
TKE_sgs_lvl = TKE_sgs_lvl[np.isfinite(TKE_sgs_lvl)]

x_pdf = np.linspace(min(TKE_res_lvl.min(),TKE_sgs_lvl.min()),\
                    max(TKE_res_lvl.max(),TKE_sgs_lvl.max()),1000)

kde_res = gaussian_kde(TKE_res_lvl)
pdf_res = kde_res(x_pdf)

kde_sgs = gaussian_kde(TKE_sgs_lvl)
pdf_sgs = kde_sgs(x_pdf)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(x_pdf,pdf_res,label='TKE_res')
axs.plot(x_pdf,pdf_sgs,label='TKE_sgs')

axs.set_xlabel('TKE')
axs.set_ylabel('PDF')
axs.set_title('PDFs at z level '+str(lvl))
axs.legend()

plt.show()

#%%Plot pcolor of TKE_tot

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x,z_uvp,(TKE_tot[:,ny//2,:nz]*(uscale**2)).T,cmap='viridis',vmin=0,vmax=2)

plt.show()

#%%Anisotropy tensor diagonal terms

Bxx = Rxx_tot/(2*TKE_tot) - 1/3
Byy = Ryy_tot/(2*TKE_tot) - 1/3
Bzz = Rzz_tot/(2*TKE_tot) - 1/3
B_trace  = Bxx + Byy + Bzz

#%%Compute the memory time scale

from matplotlib.colors import LogNorm

T = 3 / (checkpnt_aniso['S_uvp'][:, :, :nz] * uscale / zi) / 3600

T_plot = np.ma.masked_less_equal(T[:, ny // 2, :].T, 0)

fig, ax = plt.subplots(1, 1, tight_layout=True)

p = ax.pcolormesh(
    x,
    z_uvp,
    T_plot,
    cmap='viridis',
    norm=LogNorm(),
    shading='auto'
)

cbar = fig.colorbar(p, ax=ax)
cbar.set_label(r'$T$ [h]')

# Contour levels expressed in hours
levels = [1 / 60, 10/60, 1]

cs = ax.contour(
    x,
    z_uvp,
    T_plot,
    levels=levels,
    colors='white',
    linewidths=1.2
)

# Custom contour labels
labels = {
    1 / 60: '1 min',
    10 / 60: '10 min',
    1: '1 h'
}

ax.clabel(cs, fmt=labels, inline=True, fontsize=9)

plt.show()

#%%Plot PDF of the memory time scale a given vertical level

lvl = 40

T_lvl = T[:,:,lvl].flatten()*60

T_lvl = T_lvl[np.isfinite(T_lvl)]

x_pdf = np.linspace(min(T_lvl), max(T_lvl),1000)

kde = gaussian_kde(T_lvl)
pdf = kde(x_pdf)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(x_pdf,pdf,label='T')

axs.set_xlabel('T')
axs.set_ylabel('PDF')
axs.set_title('PDFs at z level '+str(lvl))
axs.legend()

plt.show()

#%%Resolution study on the height at which the resolved TKE dominates the SGS TKE from a median profile perspective
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#%%Import the data

zi = 1000
nz = [64, 128, 256]
step = 110000
path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 'test_tkesgs_tii_'
checkpnt_aniso = dict()
for n in range(0,len(nz)):
    checkpnt_aniso[str(nz[n])] = read_checkpoint_aniso(path+sim+str(nz[n])+'/output_checkpoint/',[step],nz[n],nz[n],nz[n])

#%%Compute REynolds stresses and TKE, resolved and SGS

Rxx_res = dict(); Rxx_sgs = dict(); Rxx_tot = dict()
Ryy_res = dict(); Ryy_sgs = dict(); Ryy_tot = dict()
Rzz_res = dict(); Rzz_sgs = dict(); Rzz_tot = dict()
Rxy_res = dict(); Rxy_sgs = dict(); Rxy_tot = dict()
Rxz_res = dict(); Rxz_sgs = dict(); Rxz_tot = dict()
Ryz_res = dict(); Ryz_sgs = dict(); Ryz_tot = dict()
TKE_res = dict(); TKE_sgs = dict(); TKE_tot = dict(); TKE_res_rat = dict(); TKE_sgs_rat = dict()

for i in range(0,len(nz)):
    
    Rxx_res[str(nz[i])] = checkpnt_aniso[str(nz[i])]['uu_new'][:,:,:nz[i]] - checkpnt_aniso[str(nz[i])]['u_new'][:,:,:nz[i]]*checkpnt_aniso[str(nz[i])]['u_new'][:,:,:nz[i]]
    Rxx_sgs[str(nz[i])] = - checkpnt_aniso[str(nz[i])]['txx_new'][:,:,:nz[i]] + (2/3)*checkpnt_aniso[str(nz[i])]['tke_sgs_new'][:,:,:nz[i]]
    Rxx_tot[str(nz[i])] = Rxx_res[str(nz[i])] + Rxx_sgs[str(nz[i])]

    Ryy_res[str(nz[i])] = checkpnt_aniso[str(nz[i])]['vv_new'][:,:,:nz[i]] - checkpnt_aniso[str(nz[i])]['v_new'][:,:,:nz[i]]*checkpnt_aniso[str(nz[i])]['v_new'][:,:,:nz[i]]
    Ryy_sgs[str(nz[i])] = - checkpnt_aniso[str(nz[i])]['tyy_new'][:,:,:nz[i]] + (2/3)*checkpnt_aniso[str(nz[i])]['tke_sgs_new'][:,:,:nz[i]]
    Ryy_tot[str(nz[i])] = Ryy_res[str(nz[i])] + Ryy_sgs[str(nz[i])]

    Rzz_res[str(nz[i])] = checkpnt_aniso[str(nz[i])]['ww_new'][:,:,:nz[i]] - checkpnt_aniso[str(nz[i])]['w_new'][:,:,:nz[i]]*checkpnt_aniso[str(nz[i])]['w_new'][:,:,:nz[i]]
    Rzz_sgs[str(nz[i])] = - checkpnt_aniso[str(nz[i])]['tzz_new'][:,:,:nz[i]] + (2/3)*checkpnt_aniso[str(nz[i])]['tke_sgs_new'][:,:,:nz[i]]
    Rzz_tot[str(nz[i])] = Rzz_res[str(nz[i])] + Rzz_sgs[str(nz[i])]

    Rxy_res[str(nz[i])] = checkpnt_aniso[str(nz[i])]['uv_new'][:,:,:nz[i]] - checkpnt_aniso[str(nz[i])]['u_new'][:,:,:nz[i]]*checkpnt_aniso[str(nz[i])]['v_new'][:,:,:nz[i]]
    Rxy_sgs[str(nz[i])] = - checkpnt_aniso[str(nz[i])]['txy_new'][:,:,:nz[i]]
    Rxy_tot[str(nz[i])] = Rxy_res[str(nz[i])] + Rxy_sgs[str(nz[i])]

    Rxz_res[str(nz[i])] = checkpnt_aniso[str(nz[i])]['uw_new'][:,:,:nz[i]] - checkpnt_aniso[str(nz[i])]['u_new'][:,:,:nz[i]]*checkpnt_aniso[str(nz[i])]['w_new'][:,:,:nz[i]]
    Rxz_sgs[str(nz[i])] = - checkpnt_aniso[str(nz[i])]['txz_new'][:,:,:nz[i]]
    Rxz_tot[str(nz[i])] = Rxz_res[str(nz[i])] + Rxz_sgs[str(nz[i])]

    Ryz_res[str(nz[i])] = checkpnt_aniso[str(nz[i])]['vw_new'][:,:,:nz[i]] - checkpnt_aniso[str(nz[i])]['v_new'][:,:,:nz[i]]*checkpnt_aniso[str(nz[i])]['w_new'][:,:,:nz[i]]
    Ryz_sgs[str(nz[i])] = - checkpnt_aniso[str(nz[i])]['tyz_new'][:,:,:nz[i]]
    Ryz_tot[str(nz[i])] = Ryz_res[str(nz[i])] + Ryz_sgs[str(nz[i])]

    TKE_res[str(nz[i])] = 0.5*(Rxx_res[str(nz[i])] + Ryy_res[str(nz[i])] + Rzz_res[str(nz[i])])#*(uscale**2)
    TKE_sgs[str(nz[i])] = 0.5*(Rxx_sgs[str(nz[i])] + Ryy_sgs[str(nz[i])] + Rzz_sgs[str(nz[i])])#*(uscale**2)
    TKE_tot[str(nz[i])] = TKE_res[str(nz[i])] + TKE_sgs[str(nz[i])]

    TKE_res_rat[str(nz[i])] = (TKE_res[str(nz[i])])/(TKE_tot[str(nz[i])])
    TKE_sgs_rat[str(nz[i])] = (TKE_sgs[str(nz[i])])/(TKE_tot[str(nz[i])])

#%%Plot profiles of the fractional contribution

lz = 1
dz = []
for i in range(0,len(nz)):
    dz.append(lz/nz[i])

fig,axs = plt.subplots(1,1,tight_layout=True)

colors = ['r','g','blue']

for i in range(0,len(nz)):
    
    axs.plot(np.mean(TKE_res_rat[str(nz[i])],axis=(0,1)),np.arange(0,nz[i])*float(dz[i]) + float(dz[i])/2,label=str(nz[i]),color=colors[i])
    # axs.axhline(dz*np.where(np.median(TKE_res_rat[str(nz[i])],axis=(0,1))>0.5)[0][0],color=colors[i],linestyle='--')
    # axs.axhline(dz*np.where(np.median(TKE_res_rat[str(nz[i])],axis=(0,1))>0.6)[0][0],color=colors[i],linestyle='--')
    # axs.axhline(dz*np.where(np.median(TKE_res_rat[str(nz[i])],axis=(0,1))>0.7)[0][0],color=colors[i],linestyle='--')
    # axs.axhline(float(dz[i])*np.where(np.median(TKE_res_rat[str(nz[i])],axis=(0,1))>0.8)[0][0],color=colors[i],linestyle='--')

axs.set_ylabel('z')
axs.set_xlabel(r'$f^{res}$')
axs.legend()
axs.set_xlim(0,1)
axs.set_ylim(0,1)
axs.grid()
axs.set_title('Low Shear')

plt.show()










































