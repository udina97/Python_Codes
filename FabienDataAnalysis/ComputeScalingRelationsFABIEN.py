#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:58:23 2025

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

#%%Simulation parameters

nx = 256
ny = 256
nz = 256
lx = 2*np.pi
ly = 2*np.pi
lz = 2
dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1000
uscale = 0.4
Tscale = 290

x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Some functions

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

#%%Load the data

path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/FabienData/'

case = 'P800_Ug1_Data'

dataM = xr.open_dataarray(path_to_data + case + '_Momentum.nc')
dataS = xr.open_dataarray(path_to_data + case + '_Scalar.nc')


#%%Compute the velocity gradient

kvonk = 0.4
g_hat = 9.81*zi/(uscale**2)

ustar3D = (((dataM[:,:,:,14] - uvpnode2wnode(dataM[:,:,:,0])*dataM[:,:,:,2] - dataM[:,:,:,20])**2 + \
          (dataM[:,:,:,15] - uvpnode2wnode(dataM[:,:,:,1])*dataM[:,:,:,2] - dataM[:,:,:,21])**2)**0.25).values
    
heatflux3D = (dataS[:,:,:,4] - uvpnode2wnode(dataS[:,:,:,0])*dataM[:,:,:,2] - dataS[:,:,:,7]).values

L3D = -(wnode2uvpnode(ustar3D)**3)*dataS[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D))

zoverL3D = np.ones((nx,ny,nz),order='F')*z_uvp/L3D

u = (dataM[:,:,:,0].values)
v = (dataM[:,:,:,1].values)
dudz = wnode2uvpnode(dataM[:,:,:,22].values)
dvdz = wnode2uvpnode(dataM[:,:,:,23].values)
meanDUDZ = (u*dudz + v*dvdz)/np.sqrt(u**2 + v**2)

phiM3D = (0.4*(np.ones((nx,ny,nz),order="F")*z_uvp)/wnode2uvpnode(ustar3D))*meanDUDZ

phiM_stable = phiM3D[:,:,0][(zoverL3D[:,:,0]>0)]
phiM_unstable = phiM3D[:,:,0][(zoverL3D[:,:,0]<0)]
zeta_stable = zoverL3D[:,:,0][(zoverL3D[:,:,0]>0)]
zeta_unstable = zoverL3D[:,:,0][(zoverL3D[:,:,0]<0)]

#%%Flux gradient relations 

zeta = np.linspace(10e-4,10e1,10000)
ho96 = (1-19*(-zeta))**(-0.25)
gr00 = (1-10*(-zeta))**(-1/3)
ky90 = ((1+0.6*(-zeta)**2)/(1-7.5*(-zeta)))**(1/3)

#%%Marc and iva's scaling relations

yb = np.linspace(0.1,0.8,8)
sc25_s = np.zeros((len(zeta),len(yb)))
sc25_u = np.zeros((len(zeta),len(yb)))

for i in range(len(yb)):
    if yb[i]<0.6:
        a = 0.24-0.38*yb[i]
    else:
        a = 0.012
        
    b = 0.061
    c = 0.45 - 0.53*yb[i]
    n = -0.12 + 6.4*yb[i]
    sc25_u[:,i] = (a + b*zeta**n)/(a+zeta**n) + c*zeta**(1/3)
    
    a = 0.76 + 1.5*yb[i]
    b = 6.3 - 4.3*yb[i]
    sc25_s[:,i] = a + b*zeta

#%%Plot the non dimensional velocity gradient

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy
cmap = ColorAnisotropy()

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

axs[0].scatter(abs(zeta_unstable),phiM_unstable,s=0.5,alpha=0.1)
axs[1].scatter(zeta_stable,phiM_stable,s=0.5,alpha=0.1)

# axs[0].plot(zeta,ho96,c='k')
# axs[0].plot(zeta,gr00,c='r')
# axs[0].plot(zeta,ky90,c='b')

for i in range(len(yb)):
    axs[0].plot(zeta,sc25_u[:,i],c=cmap(yb[i]))
    axs[1].plot(zeta,sc25_s[:,i],c=cmap(yb[i]))

for i in range(len(axs)):
    axs[i].set_xlim(10e-4,10e1)
    axs[i].set_ylim(10e-3,10e1)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')

axs[0].set_xlabel(r'$-\zeta$',fontsize=14)
axs[1].set_xlabel(r'$\zeta$',fontsize=14)
axs[0].set_ylabel(r'$\phi_M$',fontsize=14)

axs[0].invert_xaxis()

# fig.suptitle(sim,fontsize=14)

plt.show()

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
    phi_u_fit_u[:,i] = a_u*((1 - 3*zeta_vec_u)**(1/3))
    phi_u_fit_s[:,i] = a_s*((1 + 3*zeta_vec_s)**(c_s))
    print(c_s)
    

#%%Compute sigmaU

sigmaU = np.sqrt(dataM[:,:,:,4] - dataM[:,:,:,0]*dataM[:,:,:,0] - dataM[:,:,:,16]).values/wnode2uvpnode(ustar3D)

sigmaU_u = sigmaU[:,:,0][(zoverL3D[:,:,0]<0)]
sigmaU_s = sigmaU[:,:,0][(zoverL3D[:,:,0]>0)]
zeta_u = zoverL3D[:,:,0][(zoverL3D[:,:,0]<0)]
zeta_s = zoverL3D[:,:,0][(zoverL3D[:,:,0]>0)]


#%%

from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
import seaborn as sns
os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy
cmap = ColorAnisotropy()

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,5))

# log_bin_x = np.logspace(np.log10(x_sample.min()), np.log10(x_sample.max()), 200)
# bin_y = np.linspace((y_sample.min()),(y_sample.max()), 200)

# h,xedge,yedge,img=axs[0].hist2d(x_sample,y_sample,bins=[log_bin_x,bin_y],cmap='viridis',density=False,norm=Normalize(vmin=0,vmax=1))
# h[h==0] = np.nan
# img.set_array(h.T.ravel())
# img.set_array(img.get_array()/np.nanmax(h))


for i in range(len(y_b_vec)):
    axs[0].semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    axs[1].semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))

axs[0].scatter(abs(zeta_u),sigmaU_u,s=0.1,alpha=0.1)
axs[1].scatter(abs(zeta_s),sigmaU_s,s=0.1,alpha=0.1)

# axs[0].hist2d(abs(zeta_u),sigmaU_u,bins=[np.logspace(np.log10(abs(zeta_u).min()), np.log10(abs(zeta_u).max()), 200),\
#                                           np.linspace((sigmaU_u.min()),(sigmaU_u.max()), 20000)],cmap='hot_r')
# axs[1].hist2d(abs(zeta_s),sigmaU_s,bins=[np.logspace(np.log10(abs(zeta_s).min()), np.log10(abs(zeta_s).max()), 200),\
#                                           np.linspace((sigmaU_s.min()),(sigmaU_s.max()), 20000)],cmap='hot_r')


axs[0].set_ylim(0,10)
axs[0].set_xscale('log')
axs[0].set_xlim(1e-4,120)
axs[0].invert_xaxis()
axs[0].set_xlabel(r"$-\zeta$",fontsize=12)
axs[0].set_ylabel(r"$\phi_U$",fontsize=12)
axs[0].set_title(r"Unstable",fontsize=12)

axs[1].set_xscale('log')
axs[1].set_xlim(1e-4,120)
axs[1].set_xlabel(r"$\zeta$",fontsize=12)
axs[1].set_title(r"Stable",fontsize=12)

# fig.suptitle(sim,fontsize=15)
    
plt.show()




















































