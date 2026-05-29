# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 02:40:19 2025

@author: udina
"""

#Libraries and Functions
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

#%%Load the data

file_path = '/scratch/general/nfs1/u1450851/LES_Sims/'

# patch = '800'
# speed = 1
# case = patch+'patch_'+str(speed)+'ms_'+'noaniso_fix_grn'

case = 'diurnal_c_noaniso_L3D_fix'
step = 35
it = 1608000

os.chdir(file_path+case+'/data/') #Here one should change the text in between apostrphes for the corresponding Path to the data files. 

data = xr.open_dataarray('Momentum3D/Data_Momentum_'+str(it)+'.nc')  #This uploads the data from the NetCDF files in xarray form.
dataS = xr.open_dataarray('Scalar3D/Data_Scalar_'+str(it)+'.nc')
dataA = xr.open_dataarray('Anisotropy/Data_Anisotropy_'+str(it)+'.nc')

#%%Simulation parameters

NumVariables = 28
NumVariablesSC = 10

Nx = 128
Ny = 128
Nz = 384
Lx = 1*np.pi
Ly = 1*np.pi
Lz = 3
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz
z_i = 1

uscale = 0.40
Tscale = 290
zi = 3000
Nz_Slayer = 12

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

#%%Compute the phiU = sigmaU/ustar

u = data.data[:,:,0:Nz_Slayer,0]
uu = data.data[:,:,0:Nz_Slayer,4]
txx = data.data[:,:,0:Nz_Slayer,16]
sigmaU = np.sqrt((uu - u*u - txx))

v = data.data[:,:,0:Nz_Slayer,1]
vv = data.data[:,:,0:Nz_Slayer,5]
tyy = data.data[:,:,0:Nz_Slayer,17]
sigmaV = np.sqrt((vv - v*v - tyy))

w = data.data[:,:,0:Nz_Slayer,2]
w_uvp = wnode2uvpnode(w)
ww = data.data[:,:,0:Nz_Slayer,6]
ww_uvp = wnode2uvpnode(ww)
tzz = data.data[:,:,0:Nz_Slayer,18]
sigmaW = np.sqrt((ww_uvp - w_uvp*w_uvp))# - tzz))

uw = data.data[:,:,0:Nz_Slayer,14]
uw_uvp = wnode2uvpnode(uw)
vw = data.data[:,:,0:Nz_Slayer,15]
vw_uvp = wnode2uvpnode(vw)
txz = data.data[:,:,0:Nz_Slayer,20]
txz_uvp = wnode2uvpnode(txz)
tyz = data.data[:,:,0:Nz_Slayer,21]
tyz_uvp = wnode2uvpnode(tyz)

ustar = ((uw_uvp - u*w_uvp - txz_uvp)**2 + (vw_uvp - v*w_uvp - tyz_uvp)**2)**(1/4)

phiU = sigmaU/ustar
phiV = sigmaV/ustar
phiW = sigmaW/ustar

#%%Compute the obukhov length using RAV data
kvonk = 0.41
g=9.81*zi/(uscale**2)

T = dataS.data[:,:,0:Nz_Slayer,0]
    
wT = dataS.data[:,:,0:Nz_Slayer,4]
wT_uvp = wnode2uvpnode(wT)
tsz = dataS.data[:,:,0:Nz_Slayer,7]
tsz_uvp = wnode2uvpnode(tsz)

wT_flux = (wT_uvp - T*w_uvp- tsz_uvp)

L = -T*ustar**3/(g*kvonk*wT_flux)

#%%Compute a 3D field for the vertical coordinate

z3D = np.zeros((Nx,Ny,Nz_Slayer),order='F')

for i in range(0,Nx):
    for j in range(0,Ny):
        z3D[i,j,:] = np.arange(0,Nz_Slayer)*dz + dz/2

#%%Create flat vector

zeta = np.array([])
phiU_flat = np.array([])
phiV_flat = np.array([])
phiW_flat = np.array([])

zeta = np.concatenate((zeta,((z3D[:,:,0:Nz_Slayer]*zi)/(zi*L[:,:,0:Nz_Slayer])).flatten()))
phiU_flat = np.concatenate((phiU_flat,phiU[:,:,0:Nz_Slayer].flatten()))
phiV_flat = np.concatenate((phiV_flat,phiV[:,:,0:Nz_Slayer].flatten()))
phiW_flat = np.concatenate((phiW_flat,phiW[:,:,0:Nz_Slayer].flatten()))

#%%
# plt.figure()
# plt.hist(sigmaU[:,:,12].flatten(),bins=50)
# plt.show()

#%%Density plot using Gaussian KDE

# import seaborn as sns
from matplotlib.colors import Normalize

x_zeta = abs(zeta[(zeta>0)])
y_phiU = phiU_flat[(zeta>0)]
y_phiV = phiV_flat[(zeta>0)]
y_phiW = phiW_flat[(zeta>0)]

# x_zeta = abs(zeta_tot_na[(zeta_tot_na<0) & (phi_na>0)])
# y_phi = phi_na[(zeta_tot_na<0) & (phi_na>0)]

sample = 100000
index = np.random.choice(len(x_zeta),size=sample,replace=False)

x_sample = x_zeta[index]
y_sampleU = y_phiU[index]
y_sampleV = y_phiV[index]
y_sampleW = y_phiW[index]

fig,axs = plt.subplots(1,3,tight_layout=True,sharey=True,figsize=(15,5))

axs[0].scatter(x_sample,y_sampleU,c='k',s=1)
axs[1].scatter(x_sample,y_sampleV,c='k',s=1)
axs[2].scatter(x_sample,y_sampleW,c='k',s=1)

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
bin_yU = np.linspace((y_sampleU.min()),(y_sampleU.max()), 200)
bin_yV = np.linspace((y_sampleV.min()),(y_sampleV.max()), 200)
bin_yW = np.linspace((y_sampleW.min()),(y_sampleW.max()), 200)

h,xedge,yedge,img=axs[0].hist2d(x_sample,y_sampleU,bins=[log_bin_x,bin_yU],cmap='viridis',density=False,norm=Normalize(vmin=0,vmax=1))
h[(h<np.nanmax(h)/100)] = np.nan
img.set_array(h.T.ravel())
img.set_array(img.get_array()/np.nanmax(h))
h,xedge,yedge,img=axs[1].hist2d(x_sample,y_sampleV,bins=[log_bin_x,bin_yV],cmap='viridis',density=False,norm=Normalize(vmin=0,vmax=1))
h[(h<np.nanmax(h)/100)] = np.nan
img.set_array(h.T.ravel())
img.set_array(img.get_array()/np.nanmax(h))
h,xedge,yedge,img=axs[2].hist2d(x_sample,y_sampleW,bins=[log_bin_x,bin_yW],cmap='viridis',density=False,norm=Normalize(vmin=0,vmax=1))
h[(h<np.nanmax(h)/100)] = np.nan
img.set_array(h.T.ravel())
img.set_array(img.get_array()/np.nanmax(h))

# sns.kdeplot(x=x_sample,y=y_sample,fill=True,cmap='hot_r',bw_adjust=0.5)

axs[0].set_ylim(1e-1,1e2)

for i in range(len(axs)):
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlim(1e-4,120)
    axs[i].set_xlabel(r'$-\zeta$',fontsize=15)

# axs[0].plot(np.arange(1e-4,120,0.01),2.55*(1-3*(-np.arange(1e-4,120,0.01)))**(1/3),c='r')
# axs[1].plot(np.arange(1e-4,120,0.01),2.05*(1-3*(-np.arange(1e-4,120,0.01)))**(1/3),c='r')
# axs[2].plot(np.arange(1e-4,120,0.01),1.35*(1-3*(-np.arange(1e-4,120,0.01)))**(1/3),c='r')
axs[0].plot(np.arange(1e-4,120,0.01),2.06*np.arange(1e-4,120,0.01)**0,c='r')
axs[1].plot(np.arange(1e-4,120,0.01),2.06*np.arange(1e-4,120,0.01)**0,c='r')
axs[2].plot(np.arange(1e-4,120,0.01),1.6*np.arange(1e-4,120,0.01)**0,c='r')
# for i in range(len(y_b_vec)):
#     plt.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    # axs2.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))
    # axs3.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    # axs4.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))

# plt.gca().invert_xaxis()
for i in range(len(axs)):
    axs[i].invert_xaxis()

axs[0].set_ylabel(r'$\phi_u$',fontsize=15)
axs[1].set_ylabel(r'$\phi_v$',fontsize=15)
axs[2].set_ylabel(r'$\phi_w$',fontsize=15)

fig.suptitle(f"Patch size: {patch}m and forcing: {speed}m/s",fontsize=15)

# plt.colorbar(img)

plt.show()





























