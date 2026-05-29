#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:38:25 2025

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
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from read_checkpoint_SC import read_checkpoint

from read_checkpoint_SC import read_checkpoint_sfc

#%%#inputs

sim = '64x3_1hr_2wnode_tzzP'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/'

nx = 64
ny = 64
nz = 64
lx = 0.5*np.pi
ly = 0.5*np.pi
lz = 0.5
dx = lx/nx
dy = ly/ny
dz = lz/nz

# T_STC = 300 #320; %298.15; %[K], temperature scale
dt = 0.1 #05; %0.000005;
zi = 500.0
uscale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

#%%

simPath = path

#%%Import variables


var = ['avgU','avgV', 'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3', 'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC']
    
var2D = ['avgUstar']
    
varS = ['avgT','avgT2','avgUT','avgVT','avgWT','avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus','avg_ds']

varS2D = ['avgWstar','avgL','avgPHIm','avgPSIm','avgPHIh','avgPSIh','avgSFCval','avgSFCflux']

varA = ['avgXB','avgYB']#,'avgPHIM','avgPHIH','avgPSIM','avgPSIH','avgL3D','avgustar3D','avgSCF3D']
    
dataM = xr.open_dataarray(path+'Data_Momentum.nc')
dataM2D = xr.open_dataarray(path+'Data_Momentum_2D.nc')
dataS = xr.open_dataarray(path+'Data_Scalar.nc')
dataS2D = xr.open_dataarray(path+'Data_Scalar_2D.nc')
dataA = xr.open_dataarray(path+'Data_Anisotropy.nc')

data_mom = dict()
data_mom_2D = dict()
data_sc = dict()
data_sc_2D = dict()
data_aniso = dict()

for i in range(len(var)):
    data_mom[var[i]] = dataM.data[:,:,:,i]
    
for i in range(len(var2D)):
    data_mom_2D[var2D[i]] = dataM2D.data[:,:,i]

for i in range(len(varS)):
    data_sc[varS[i]] = dataS.data[:,:,:,i]

for i in range(len(varS2D)):
    data_sc_2D[varS2D[i]] = dataS2D.data[:,:,i]
    
for i in range(len(varA)):
    data_aniso[varA[i]] = dataA.data[:,:,:,i]
    
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

#%%Compute anisotropy

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering, Anisotropy

uu = data_mom['avgU2'] - data_mom['avgU']*data_mom['avgU'] - data_mom['avgtxx']
vv = data_mom['avgV2'] - data_mom['avgV']*data_mom['avgV'] - data_mom['avgtyy']
ww = wnode2uvpnode(data_mom['avgW2'] - data_mom['avgW']*data_mom['avgW']) + data_mom['avgtzz']
uw = wnode2uvpnode(data_mom['avgUW']) - data_mom['avgU']*wnode2uvpnode(data_mom['avgW']) - wnode2uvpnode(data_mom['avgtxz'])
uv = data_mom['avgUV'] - data_mom['avgU']*data_mom['avgV'] - data_mom['avgtxy']
vw = wnode2uvpnode(data_mom['avgVW']) - data_mom['avgV']*wnode2uvpnode(data_mom['avgW']) - wnode2uvpnode(data_mom['avgtyz'])

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%%Plotting the Lumley Triangle xB vs yB
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
# import cmasher as cmr

# cmap_LT = cmr.get_sub_cmap('seismic_r', 0.5, 1)

#Tower Traj as a function of height -------------------------------------------------------------------------------------------------------

# tmpxB = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
# tmpyB = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

# for i in range(len(coord)):
#     loc = coord[i]
    
#     tmpxB[i,:] = np.reshape(xB_1D,(Nx,Ny,Nz))[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
#     tmpyB[i,:] = np.reshape(yB_1D,(Nx,Ny,Nz))[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    
# xB_median = np.nanmedian(tmpxB,axis=(0))
# yB_median = np.nanmedian(tmpyB,axis=(0))

# xB = np.reshape(xB_1D,(Nx,Ny,Nz))[5,5,5:]
# yB = np.reshape(yB_1D,(Nx,Ny,Nz))[5,5,5:]

#Filter 3D points where P-D is roughly 0 ---------------------------------------------------------------------------------------------------

# terms_bdg = np.load(pathOUT + 'TKE_terms.npy',allow_pickle='TRUE').item()

# tmpDIS = copy.deepcopy((terms_bdg['totdis'])[:,:,5:]) # copy.deepcopy(np.nanmean(terms_bdg['totdis'],axis=(1))) 
# tmpRES = copy.deepcopy((terms_bdg['prod'] - terms_bdg['totdis'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['prod']) - (terms_bdg['totdis']),axis=(1))) 

# # tmpRES[(abs(tmpRES)<1)] = 0 #float('nan')

# # terms = [((tmpRES)/abs(tmpDIS))*100]
# terms = [(tmpRES)]

# yB = np.reshape(yB_1D,(Nx,Ny,Nz)); xB = np.reshape(xB_1D,(Nx,Ny,Nz))
# yB[(dist[:,:,:]<0)] = float('nan'); xB[(dist[:,:,:]<0)] = float('nan')

# lvl1 = 1
# lvl2 = 2

# tmpxB = xB[:,:,5:][(dist[:,:,5:]>=39*lvl1) & (dist[:,:,5:]<39*lvl2)].flatten()
# tmpyB = yB[:,:,5:][(dist[:,:,5:]>=39*lvl1) & (dist[:,:,5:]<39*lvl2)].flatten()
# tmpZ = dist[:,:,5:][(dist[:,:,5:]>=39*lvl1) & (dist[:,:,5:]<39*lvl2)].flatten()/39

# nbins = 20
# x = tmpxB
# y = tmpyB

# k = gaussian_kde([x,y])
# xi, yi = np.mgrid[
#     x.min():x.max():nbins*1j,
#     y.min():y.max():nbins*1j
# ]
# z_i = k(np.vstack([
#     xi.flatten(),
#     yi.flatten()
# ])).reshape(xi.shape)


# Plotting the actual triangle----------------------------------------------------------------------------------------------------------------------

fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
# plot a triangle
xc = np.array([0, 1, 0.5])
yc = np.array([0, 0, np.sqrt(3)*0.5])
for i in np.arange(3):
    ip1 = (i+1)%3
    axs.plot([xc[i], xc[ip1]], [yc[i], yc[ip1]], 'k', linewidth=2)
# add grid
nsp = 5
lc = np.abs(xc[1]-xc[0])
dc = lc/nsp
cl = np.zeros([3*(nsp-1), 3])
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        cl[k,i] = dc * (j+1)
        cl[k,ip1] = dc * (nsp-j-1)
xl = np.dot(xc, cl.transpose())
yl = np.dot(yc, cl.transpose())
nl = xl.size
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        kp = (ip1 * (nsp-1) + nsp - j - 2) % nl
        axs.plot([xl[k], xl[kp]], [yl[k], yl[kp]], '--k', linewidth=0.75)
# plain strain limit
c1ps = np.array([2/3, 1/3, 0])
x1ps = np.dot(xc, c1ps.transpose())
y1ps = np.dot(yc, c1ps.transpose())
c2ps = np.array([0, 0, 1])
x2ps = np.dot(xc, c2ps.transpose())
y2ps = np.dot(yc, c2ps.transpose())
axs.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)
# add labels
labels = ['2-comp axi', '1-comp', 'Isotropic']
lbpx = xc
dshift = 0.05
lbpy = [yc[0]-dshift*lc, yc[1]-dshift*lc, yc[2]+dshift*lc]
for i in np.arange(3):
    axs.text(lbpx[i], lbpy[i], labels[i], ha='center', fontsize=12)
label_side1 = 'Prolate'
label_side2 = 'Oblate'
label_side3 = 'Two-component'
axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')

#-----------------------------------------------------------------------------------------------------------------------------------------

# z = z_uvp
# z_on_h = z/(39/zi)

# ### - Plot option 1
# # axs.pcolormesh(xi, yi, z_i, cmap='hot_r')
# # sc = axs.scatter(tmpxB,tmpyB,s=1,alpha=0.5)#,c=tmpZ,cmap='jet')
# # axs.pcolormesh(xi, yi, z_i, cmap='hot_r')
# # axs.contour(xi,yi,z_i)

# ### - Plot option 2
# # for i in range(len(coord)):
# #     sc = axs.scatter(tmpxB[i,21:],tmpyB[i,21:],s=1,c=z_on_h[21:Nz_SLayer],cmap='jet',alpha=0.5)
# #     # sc = axs.plot(tmpxB[i,:],tmpyB[i,:])
# #     # sc = axs.plot(tmpxB[i,21:],tmpyB[i,21:],c='r')
# # axs.plot(xB_median[21:],yB_median[21:],'k',lw=2)
# h1 = 39
# ### - Plot option 3
# for i in range(1,55):
#     # lev1 = i
#     # lev2 = i+1
    
    
#     xB = np.reshape(xB_1D,(Nx,Ny,Nz))[(dist>h1) & (dist<h1+10)].flatten()
#     yB = np.reshape(yB_1D,(Nx,Ny,Nz))[(dist>h1) & (dist<h1+10)].flatten()
    
#     h1 = h1+10
    
#     nbins = 20
#     x = xB
#     y = yB
    
#     k = gaussian_kde([x,y])
#     xi, yi = np.mgrid[
#     x.min():x.max():nbins*1j,
#     y.min():y.max():nbins*1j
#     ]
#     z_i = k(np.vstack([
#     xi.flatten(),
#     yi.flatten()
#     ])).reshape(xi.shape)
#     # Find the point with the highest probability
#     max_idx = np.argmax(z_i)
#     x_max, y_max = xi.ravel()[max_idx], yi.ravel()[max_idx]
    
#     # Add a marker for the highest probability point
#     axs.plot(x_max, y_max, 'ro', markersize=2, label="Max Probability")
    
#     # Draw contour for the extremes (e.g., 95th percentile of the KDE field)
#     contour_levels = [np.percentile(z_i, 95)]
#     contour = axs.contour(xi, yi, z_i, levels=contour_levels, colors='k')
    
#     # Create a mask to color the area within the contour
#     # cmap = ListedColormap(['lightblue'])
#     jet_cmap = plt.get_cmap('hot_r')  # Get the full jet colormap
#     cmap_section = ListedColormap(jet_cmap(np.linspace(i*10e-2, i*10e-2, 256)))  # Select a section of the colormap
#     axs.contourf(xi, yi, z_i, levels=contour_levels + [z_i.max()], cmap=cmap_section, alpha=0.6)

yb_median = np.zeros((nz))
xb_median = np.zeros((nz))
yb_std = np.zeros((nz))
xb_std = np.zeros((nz))

for i in range(0,nz):
    # yb_median[i] = np.median(data_aniso['avgYB'][:,:,i])
    # xb_median[i] = np.median(data_aniso['avgXB'][:,:,i])
    # yb_std[i] = np.std(data_aniso['avgYB'][:,:,i])
    # xb_std[i] = np.std(data_aniso['avgXB'][:,:,i])
    
    yb_median[i] = np.median(yB[:,:,i])
    xb_median[i] = np.median(xB[:,:,i])
    yb_std[i] = np.std(yB[:,:,i])
    xb_std[i] = np.std(xB[:,:,i])
    
sc = axs.scatter(xb_median,yb_median,c=np.arange(0,nz)*dz+dz/2,cmap='jet')

axs.axhline(0.38,0,2,c='k',ls='-.')
axs.axhline(0.36,0,2,c='k',ls='-.')
axs.set_xlim(-0.2,1.2)
axs.set_ylim(-0.1,1)
axs.set_xlabel(r'$x_B$',fontsize=15)
axs.set_ylabel(r'$y_B$',fontsize=15)
xticks = [0,0.5,1]
yticks = [0,0.38,np.sqrt(3)/2]
ylabels = ['0','0.38',r'$\sqrt{3}/2$']
axs.set_xticks(xticks)
axs.set_yticks(yticks,labels=ylabels)
cbar = plt.colorbar(sc,label=r'$z$')

plt.show()

# axs.set_title(f'valley',fontsize=15)

# plt.savefig(pathOUT +'Figures/'+'LumleyTri_traj_hieght.png',dpi=300,facecolor='white', edgecolor='white')

#%%Pcolor plot of anisotropy

fig,axs = plt.subplots(1,1,figsize=(10,6),tight_layout=True)

# p=axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,data_aniso['avgYB'][40,:,:].T,cmap = ColorAnisotropy())#,vmin=0,vmax=np.sqrt(3)/2)
# sc=axs.contour(np.arange(0,nx)*dx,np.arange(0,nz)*dz,(data_aniso['avgYB'][40,:,:]).T,levels=[0.38],colors=['black'])
p=axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz,yB[40,:,:].T,cmap = ColorAnisotropy())#,vmin=0,vmax=np.sqrt(3)/2)
sc=axs.contour(np.arange(0,nx)*dx,np.arange(0,nz)*dz,(yB[40,:,:]).T,levels=[0.38],colors=['black'])
# p=axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz + dz/2,data_mom['avgW'][40,:,:].T,cmap='jet')

axs.set_xlabel('x',fontsize=15)
axs.set_ylabel('z',fontsize=15)

cbar = plt.colorbar(p,label='yB')

plt.show()












































