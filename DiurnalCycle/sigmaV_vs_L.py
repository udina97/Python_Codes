#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 08:18:10 2025

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

#%%Import variables

Nfiles = 52

sigma_a = dict()
sigma_na = dict()
ustar_a = dict()
ustar_na = dict()

for i in range(0,Nfiles):
    u = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    u_w = uvpnode2wnode(u)
    
    v = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    v_w = uvpnode2wnode(v)
    vv = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,5]
    vv_w = uvpnode2wnode(vv)
    tyy = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,17]
    tyy_w = uvpnode2wnode(tyy)
    sigma_a[str(StartTime + AvgIter*i)] = (vv_w - v_w*v_w - tyy_w)
    
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    
    uw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,14]
    vw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,15]
    txz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,20]
    tyz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,21]
    
    ustar_a[str(StartTime + AvgIter*i)] = ((uw - u_w*w - txz)**2 + (vw - v_w*w - tyz)**2)**(1/4)
    
    #--------------------------------------------------------------------------------------------------------------------
    
    u = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    u_w = uvpnode2wnode(u)
    
    v = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    v_w = uvpnode2wnode(v)
    vv = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,5]
    vv_w = uvpnode2wnode(vv)
    tyy = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,17]
    tyy_w = uvpnode2wnode(tyy)
    sigma_na[str(StartTime + AvgIter*i)] = (vv_w - v_w*v_w - tyy_w)
    
    w = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    
    uw = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,14]
    vw = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,15]
    txz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,20]
    tyz = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,21]
    
    ustar_na[str(StartTime + AvgIter*i)] = ((uw - u_w*w - txz)**2 + (vw - v_w*w - tyz)**2)**(1/4)
    
    print(f'Done with File: {i}')
    
#%%Varibales to compute L using RAV

L_rav_a = dict()
L_rav_na = dict()
kvonk = 0.41
g=9.81*zi/(uscale**2)

for i in range(0,Nfiles):
    T = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    T_w = uvpnode2wnode(T)
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    wT = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    tsz = xr.open_dataarray(path_a+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,7]
    L = xr.open_dataarray(path_a+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,1]
    
    wT_a = (wT - T_w*w - tsz)
    
    L_rav_a[str(StartTime + AvgIter*i)] = -(ustar_a[str(StartTime + AvgIter*i)]**3)*T_w/(kvonk*g*wT_a)
    L_rav_a[str(StartTime + AvgIter*i)][:,:,0] = L
    
    #-----------------------------------------------------------------------------------------------------------------
    
    T = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    T_w = uvpnode2wnode(T)
    w = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    wT = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    tsz = xr.open_dataarray(path_na+'../Scalar3D/Data_Scalar_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,7]
    L = xr.open_dataarray(path_na+'../Scalar2D/Data_Scalar_2D_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,1]
    
    wT_a = (wT - T_w*w - tsz)
    
    L_rav_na[str(StartTime + AvgIter*i)] = -(ustar_na[str(StartTime + AvgIter*i)]**3)*T_w/(kvonk*g*wT_a)
    L_rav_na[str(StartTime + AvgIter*i)][:,:,0] = L
    
    print(f'Done with File: {i}')
    
#%% Import L

L_a = dict()
L_na = dict()

for i in range(0,Nfiles):
    L_a[str(StartTime + AvgIter*i)] = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,-1]*AvgIter
    L_na[str(StartTime + AvgIter*i)] = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,-1]*AvgIter
    print(f'Done with File: {i}')

#%%Create 3D field for z

z3D = np.zeros((nx,ny,50),order='F')

for i in range(0,nx):
    for j in range(0,ny):
        z3D[i,j,:] = np.arange(0,50)*dz*zi

z3D[:,:,0] = 0.03

#%%Plot 2D histogram

idx = 20
zeta_a = (z3D/(L_rav_a[str(StartTime + AvgIter*idx)]*zi)).flatten()
zeta_na = (z3D/(L_rav_na[str(StartTime + AvgIter*idx)]*zi)).flatten()

# fig,axs = plt.subplots(1,1,tight_layout=True,projection='scatter_density')

# axs.scatter_density(abs((z3D/(L_a[str(StartTime + AvgIter*idx)]*zi)).flatten()[(zeta_a<0)]),\
#            (np.sqrt(sigma_a[str(StartTime + AvgIter*idx)])/ustar_a[str(StartTime + AvgIter*idx)]).flatten()[(zeta_a<0)],cmap='hot_r')

fig = plt.figure()
axs=fig.add_subplot(1,1,1,projection='scatter_density')
axs.scatter_density(abs(zeta_na[(zeta_na<0)]),\
            (np.sqrt(sigma_na[str(StartTime + AvgIter*idx)])/ustar_na[str(StartTime + AvgIter*idx)]).flatten()[(zeta_na<0)],cmap='hot_r')
axs.set_ylim(0,10)
axs.set_xscale('log')

plt.show()

#%%

zeta_vec_u = np.array([-1e2,-1e1,-1e0,-1e-1,-1e-2,-1e-3,-1e-4])
zeta_vec_s = abs(zeta_vec_u[::-1])
y_b_vec = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
phi_v_fit_s = np.zeros((len(zeta_vec_s),len(y_b_vec)))
phi_v_fit_u = np.zeros((len(zeta_vec_u),len(y_b_vec)))
coef_a_u = [0.725,-2.702]
coef_a_s = [2.385,-2.781,3.771]
coef_c_s = [0.654,-6.282,21.975,-31.634,16.251]

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
    phi_v_fit_u[:,i] = a_u*((1 - 3*zeta_vec_u)**(1/3))
    phi_v_fit_s[:,i] = a_s*((1 + 3*zeta_vec_s)**(c_s))
    print(c_s)

#%%

zeta_tot_a = np.array([])
zeta_tot_na = np.array([])
phi_a = np.array([])
phi_na = np.array([])
for i in range(0,Nfiles):
    zeta_a = (z3D/(L_rav_a[str(StartTime + AvgIter*i)]*(zi))).flatten()
    zeta_na = (z3D/(L_rav_na[str(StartTime + AvgIter*i)]*(zi))).flatten()
    zeta_tot_a = np.concatenate((zeta_tot_a,zeta_a))
    zeta_tot_na = np.concatenate((zeta_tot_na,zeta_na))
    phi_a = np.concatenate((phi_a,(np.sqrt(sigma_a[str(StartTime + AvgIter*i)])/ustar_a[str(StartTime + AvgIter*i)]).flatten()))
    phi_na = np.concatenate((phi_na,(np.sqrt(sigma_na[str(StartTime + AvgIter*i)])/ustar_na[str(StartTime + AvgIter*i)]).flatten()))

#%%Plot sigma vs z/L for stable and unstable case and anisotropy and no anisotropy

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
nbins = 200

fig = plt.figure(tight_layout=True)
axs1=fig.add_subplot(2,2,1,projection='scatter_density')
axs2=fig.add_subplot(2,2,2,projection='scatter_density')
axs3=fig.add_subplot(2,2,3,projection='scatter_density')
axs4=fig.add_subplot(2,2,4,projection='scatter_density')
    
axs1.scatter_density(abs(zeta_tot_a[(zeta_tot_a<0)]),phi_a[(zeta_tot_a<0)],cmap='hot_r')
axs2.scatter_density(abs(zeta_tot_a[(zeta_tot_a>0)]),phi_a[(zeta_tot_a>0)],cmap='hot_r')
axs3.scatter_density(abs(zeta_tot_na[(zeta_tot_na<0)]),phi_na[(zeta_tot_na<0)],cmap='hot_r')
axs4.scatter_density(abs(zeta_tot_na[(zeta_tot_na>0)]),phi_na[(zeta_tot_na>0)],cmap='hot_r')

axs1.set_ylim(0,10)
axs2.set_ylim(0,10)
axs3.set_ylim(0,10)
axs4.set_ylim(0,10)

axs1.set_xscale('log')
axs2.set_xscale('log')
axs3.set_xscale('log')
axs4.set_xscale('log')

for i in range(len(y_b_vec)):
    axs1.semilogx(-zeta_vec_u,phi_v_fit_u[:,i],c=cmap(y_b_vec[i]))
    axs2.semilogx(zeta_vec_s,phi_v_fit_s[:,i],c=cmap(y_b_vec[i]))
    axs3.semilogx(-zeta_vec_u,phi_v_fit_u[:,i],c=cmap(y_b_vec[i]))
    axs4.semilogx(zeta_vec_s,phi_v_fit_s[:,i],c=cmap(y_b_vec[i]))

axs1.set_xlim(1e-4,120)
axs2.set_xlim(1e-4,120)
axs3.set_xlim(1e-4,120)
axs4.set_xlim(1e-4,120)

axs1.invert_xaxis()
axs3.invert_xaxis()

axs3.set_xlabel(r'$-\zeta$',fontsize=15)
axs1.set_ylabel(r'$\phi_v$',fontsize=15)
axs4.set_xlabel(r'$\zeta$',fontsize=15)
axs3.set_ylabel(r'$\phi_v$',fontsize=15)

axs1.set_title('Unstable',fontsize=15)
axs2.set_title('Stable',fontsize=15)

axs1.text(0.7, 0.9,f"w/ Aniso", transform=axs1.transAxes, fontsize=12)
axs2.text(0.7, 0.9,f"w/ Aniso", transform=axs2.transAxes, fontsize=12)
axs3.text(0.7, 0.9,f"Classic", transform=axs3.transAxes, fontsize=12)
axs4.text(0.7, 0.9,f"Classic", transform=axs4.transAxes, fontsize=12)

# plt.savefig(path_fig+'sigmaV_vs_zL.png',dpi=300)
plt.show()

#%%Plot the scaling as a function of zeta as a 2D PDF
os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
nbins = 200
idx = 40
# zeta_a = (z3D/(L_rav_a[str(StartTime + AvgIter*idx)]*(zi))).flatten()
# zeta_na = (z3D/(L_rav_na[str(StartTime + AvgIter*idx)]*(zi))).flatten()

# pdf, xedges, yedges = np.histogram2d(abs((z3D/(L_a[str(StartTime + AvgIter*idx)]*zi)).flatten()[(zeta_a<0)]),\
#                                      (np.sqrt(sigma_a[str(StartTime + AvgIter*idx)])/ustar_a[str(StartTime + AvgIter*idx)]).flatten()[(zeta_a<0)],bins=nbins,density=True)

# xcenters = 0.5 * (xedges[1:] + xedges[:-1])
# ycenters = 0.5 * (yedges[1:] + yedges[:-1])
# X, Y = np.meshgrid(xcenters, ycenters)

# fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
fig = plt.figure()
axs=fig.add_subplot(1,1,1,projection='scatter_density')
    
axs.scatter_density(abs(zeta_tot_a[(zeta_tot_a<0)]),phi_a[(zeta_tot_a<0)],cmap='hot_r')
axs.set_ylim(0,10)
axs.set_xscale('log')
for i in range(len(y_b_vec)):
    axs.semilogx(-zeta_vec,phi_v_fit[:,i],c=cmap(y_b_vec[i]))
# axs.contourf(X,Y,pdf.T,levels=100,cmap='hot_r')
# axs.scatter(abs((z3D/(L_a[str(StartTime + AvgIter*idx)]*zi)).flatten()[(zeta_a<0)]),\
#             (np.sqrt(sigma_a[str(StartTime + AvgIter*idx)])/ustar_a[str(StartTime + AvgIter*idx)]).flatten()[(zeta_a<0)])
# axs.contour(X,Y,pdf.T,levels=10,c='k')
# axs.plot([-1e-6], [0], alpha=0)
# axs.set_xscale('symlog',linthresh=0.1)
# axs.set_xlim(-1e2,-1e-6)
axs.set_xlim(1e-4,1e8)
axs.invert_xaxis()
# axs.set_ylim(0,10)
# axs.autoscale(False)
# axs.set_xlabel(r'$\zeta$',fontsize=15)
# axs.set_ylabel(r'$\phi$',fontsize=15)
# axs.set_xticks([-1e2, -1e1, -1e0, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5, -1e-6])
# axs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

plt.show()











































