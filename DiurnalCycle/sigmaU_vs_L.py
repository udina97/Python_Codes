#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 07:53:33 2025

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
    uu = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    uu_w = uvpnode2wnode(uu)
    txx = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,16]
    txx_w = uvpnode2wnode(txx)
    sigma_a[str(StartTime + AvgIter*i)] = (uu_w - u_w*u_w - txx_w)
    
    v = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    v_w = uvpnode2wnode(v)
    
    w = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,2]
    
    uw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,14]
    vw = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,15]
    txz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,20]
    tyz = xr.open_dataarray(path_a+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,21]
    
    ustar_a[str(StartTime + AvgIter*i)] = ((uw - u_w*w - txz)**2 + (vw - v_w*w - tyz)**2)**(1/4)
    
    #--------------------------------------------------------------------------------------------------------------------
    
    u = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,0]
    u_w = uvpnode2wnode(u)
    uu = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,4]
    uu_w = uvpnode2wnode(uu)
    txx = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,16]
    txx_w = uvpnode2wnode(txx)
    sigma_na[str(StartTime + AvgIter*i)] = (uu_w - u_w*u_w - txx_w)
    
    v = xr.open_dataarray(path_na+'Data_Momentum_'+str(StartTime + AvgIter*i)+'.nc').data[:,:,0:50,1]
    v_w = uvpnode2wnode(v)
    
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

z3D = np.zeros((nx,ny,10),order='F')

for i in range(0,nx):
    for j in range(0,ny):
        z3D[i,j,:] = np.arange(0,10)*dz*zi

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

#%%Computing the scaling relations from Marc and Iva for unstable and stable stratification

zeta_vec_u = np.array([-1e2,-0.5e2,-1e1,-0.5e1,-1e0,-0.5e0,-1e-1,-5e-2,-1e-2,-5e-3,-1e-3,-5e-4,-1e-4])
zeta_vec_s = abs(zeta_vec_u[::-1])
y_b_vec = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
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

#%%
zeta_tot_a = np.array([])
zeta_tot_na = np.array([])
phi_a = np.array([])
phi_na = np.array([])
for i in range(0,30):
    zeta_a = (z3D/(L_rav_a[str(StartTime + AvgIter*i)][:,:,:10]*(zi))).flatten()
    zeta_na = (z3D/(L_rav_na[str(StartTime + AvgIter*i)][:,:,:10]*(zi))).flatten()
    zeta_tot_a = np.concatenate((zeta_tot_a,zeta_a))
    zeta_tot_na = np.concatenate((zeta_tot_na,zeta_na))
    phi_a = np.concatenate((phi_a,(np.sqrt(sigma_a[str(StartTime + AvgIter*i)])/ustar_a[str(StartTime + AvgIter*i)])[:,:,:10].flatten()))
    phi_na = np.concatenate((phi_na,(np.sqrt(sigma_na[str(StartTime + AvgIter*i)])/ustar_na[str(StartTime + AvgIter*i)])[:,:,:10].flatten()))

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
    
p1=axs1.scatter_density(abs(zeta_tot_a[(zeta_tot_a<0) & (phi_a>0)]),phi_a[(zeta_tot_a<0) & (phi_a>0)],cmap='hot_r')
p2=axs2.scatter_density(abs(zeta_tot_a[(zeta_tot_a>0) & (phi_a>0)]),phi_a[(zeta_tot_a>0) & (phi_a>0)],cmap='hot_r')
p3=axs3.scatter_density(abs(zeta_tot_na[(zeta_tot_na<0) & (phi_na>0)]),phi_na[(zeta_tot_na<0) & (phi_na>0)],cmap='hot_r')
p4=axs4.scatter_density(abs(zeta_tot_na[(zeta_tot_na>0) & (phi_na>0)]),phi_na[(zeta_tot_na>0) & (phi_na>0)],cmap='hot_r')

axs1.set_ylim(0,10)
axs2.set_ylim(0,10)
axs3.set_ylim(0,10)
axs4.set_ylim(0,10)

axs1.set_xscale('log')
axs2.set_xscale('log')
axs3.set_xscale('log')
axs4.set_xscale('log')

for i in range(len(y_b_vec)):
    axs1.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    axs2.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))
    axs3.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    axs4.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))

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

axs1.text(0.6, 0.9,f"w/ Aniso", transform=axs1.transAxes, fontsize=12)
axs2.text(0.6, 0.9,f"w/ Aniso", transform=axs2.transAxes, fontsize=12)
axs3.text(0.6, 0.9,f"Classic", transform=axs3.transAxes, fontsize=12)
axs4.text(0.6, 0.9,f"Classic", transform=axs4.transAxes, fontsize=12)

# cbar = plt.colorbar(p1)
# cbar = plt.colorbar(p2)
# cbar = plt.colorbar(p3)
# cbar = plt.colorbar(p4)


# plt.savefig(path_fig+'sigmaU_vs_zL.png',dpi=300)
plt.show()



#%%Plot the scaling as a function of zeta as a 2D PDF
os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()
nbins = 100
idx = 40
# zeta_a = (z3D/(L_rav_a[str(StartTime + AvgIter*idx)]*(zi))).flatten()
# zeta_na = (z3D/(L_rav_na[str(StartTime + AvgIter*idx)]*(zi))).flatten()

pdf, xedges, yedges = np.histogram2d(abs(zeta_tot_a[(zeta_tot_a<0) & (phi_a>0)]),\
                                      phi_a[(zeta_tot_a<0) & (phi_a>0)],bins=nbins,density=True)

xcenters = 0.5 * (xedges[1:] + xedges[:-1])
ycenters = 0.5 * (yedges[1:] + yedges[:-1])
X, Y = np.meshgrid(xcenters, ycenters)

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
# fig = plt.figure()
# axs=fig.add_subplot(1,1,1,projection='scatter_density')
    
# axs.scatter_density(abs(zeta_tot_a[(zeta_tot_a>0)]),phi_a[(zeta_tot_a>0)],cmap='hot_r')
axs.set_ylim(0,10)
axs.set_xscale('log')
for i in range(len(y_b_vec)):
    axs.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
axs.contourf(X,Y,pdf.T,levels=100,cmap='hot_r')
# axs.scatter(abs((z3D/(L_a[str(StartTime + AvgIter*idx)]*zi)).flatten()[(zeta_a<0)]),\
#             (np.sqrt(sigma_a[str(StartTime + AvgIter*idx)])/ustar_a[str(StartTime + AvgIter*idx)]).flatten()[(zeta_a<0)])
# axs.contour(X,Y,pdf.T,levels=10,c='k')
# axs.plot([-1e-6], [0], alpha=0)
# axs.set_xscale('symlog',linthresh=0.1)
# axs.set_xlim(-1e2,-1e-6)
axs.set_xlim(1e-4,120)
axs.invert_xaxis()
# axs.set_ylim(0,10)
# axs.autoscale(False)
# axs.set_xlabel(r'$\zeta$',fontsize=15)
# axs.set_ylabel(r'$\phi$',fontsize=15)
# axs.set_xticks([-1e2, -1e1, -1e0, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5, -1e-6])
# axs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

plt.show()

#%%Density plot using Gaussian KDE

import seaborn as sns
from matplotlib.colors import Normalize

x_zeta = abs(zeta_tot_a[(zeta_tot_a>0) & (phi_a>0)])
y_phi = phi_a[(zeta_tot_a>0) & (phi_a>0)]

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
h[h==0] = np.nan
img.set_array(h.T.ravel())
img.set_array(img.get_array()/np.nanmax(h))

# sns.kdeplot(x=x_sample,y=y_sample,fill=True,cmap='hot_r',bw_adjust=0.5)

plt.ylim(0,10)

plt.xscale('log')

for i in range(len(y_b_vec)):
    plt.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    # axs2.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))
    # axs3.semilogx(-zeta_vec_u,phi_u_fit_u[:,i],c=cmap(y_b_vec[i]))
    # axs4.semilogx(zeta_vec_s,phi_u_fit_s[:,i],c=cmap(y_b_vec[i]))

plt.xlim(1e-4,120)

plt.gca().invert_xaxis()

plt.xlabel(r'$-\zeta$',fontsize=15)
plt.ylabel(r'$\phi_u$',fontsize=15)
plt.title('No Anisotropy',fontsize=15)

plt.colorbar(img)

plt.show()








































