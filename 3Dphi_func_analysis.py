#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:26:12 2025

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

sim = 'test_L3D'
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
                    'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz','avgdudz','avgdvdz','avgNut','avgCs','avgBetaSC','avgL3D']
    
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
    
#%%Check yB for negative values or greater than sqrt(3)/2

points = 0

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            if data_aniso['avgXB'][i,j,k]<0:
                points += 1
                print('Number of points:',points)
                print(i,j,k)
                
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

#%%Pcolor plots

fig,axs = plt.subplots(1,1,tight_layout=True)
# p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,nz)*dz+dz/2,data_mom['avgV'][30,:,:].T,cmap='jet')
# p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,data_mom['avgU'][:,:,30].T,cmap='Greens')
p = axs.pcolormesh(np.arange(0,nx)*dx,np.arange(0,ny)*dy,data_mom_2D['avgUstar'].T,cmap='jet')#,vmin=0.0008,vmax=0.0012)
c = plt.colorbar(p)
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_title(sim)
plt.show()

#%%Plot vertical profiles

Ruw = (data_mom['avgUW']) - uvpnode2wnode(data_mom['avgU'])*(data_mom['avgW'])
#Dispersive stress, Dxz at w nodes!!
U_disp = np.zeros((nx,ny,nz),'d',order='F') #compute dispersive velocity for U interpolated at w nodes!
W_disp = np.zeros((nx,ny,nz),'d',order='F')
for k in range(0,nz):
    U_disp[:,:,k] = (data_mom['avgU'][:,:,k])-np.mean(data_mom['avgU'][:,:,k],axis=(0,1))
    W_disp[:,:,k] = (data_mom['avgW'][:,:,k])-np.mean(data_mom['avgW'][:,:,k],axis=(0,1))
    
D_xz = -np.mean(uvpnode2wnode(U_disp)*W_disp,axis=(0,1))
Tw = -Ruw + (data_mom['avgtxz']) + D_xz

Rwt = data_sc['avgWT'] - uvpnode2wnode(data_sc['avgT'])*data_mom['avgW']
Ttheta = -Rwt + data_sc['avgWT_sgs']

fig,axs = plt.subplots(1,1,figsize=(6,6),tight_layout=True)

axs.plot(np.mean((data_mom['avgtxz']),axis=(0,1)),np.arange(0,nz)*dz,c='k')
axs.plot(np.mean(-Ruw,axis=(0,1)),np.arange(0,nz)*dz,c='r')
axs.plot(D_xz,np.arange(0,nz)*dz,c='b')
axs.plot(np.mean(Tw,axis=(0,1)),np.arange(0,nz)*dz,c='g')

# axs.plot(np.mean(data_mom['avgU'],axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='k')
# axs.plot(np.mean(data_mom['avgV'],axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='g')
# axs.plot(np.mean(data_mom['avgW'],axis=(0,1)),np.arange(0,nz)*dz,c='r')

# axs.plot(np.mean(data_sc['avgWT_sgs'],axis=(0,1)),np.arange(0,nz)*dz,c='k')
# axs.plot(np.mean(-Rwt,axis=(0,1)),np.arange(0,nz)*dz,c='g')
# axs.plot(np.mean(Ttheta,axis=(0,1)),np.arange(0,nz)*dz,c='r')

axs.set_xlabel('Var',fontsize=15)
axs.set_ylabel(r'$z/h_C$',fontsize=15)
axs.set_ylim(0,nz*dz)
axs.set_title(sim)
plt.show()

#%%Compute z/L
zeta = ((dz*zi)/2)/(data_sc_2D['avgL']*zi)

#%%Interpolate the txz and tyz stresses

def wnode2uvpnode_tij(phi):
    phi_c = np.zeros(phi.shape)
    phi_c[:,:,0] = phi[:,:,0]
    phi_c[:, :, 1:-1] = 0.5*(phi[:, :, 1:-1]+phi[:, :, 2:])
    phi_c[:, :, -1] = phi[:, :, -1]
    return phi_c
    
#%% Calculate the Reynolds Stresses:
    
items_bdg = ('adv_h', 'adv_v', 'prod_h', 'prod_v', 'uturb_h', 'uturb_v',
  'pturb_h', 'pturb_v', 'prod_dudz', 'canopy', 'dissip', 'sum')

# Initialization
terms_ptb = dict()
terms_drv = dict()
terms_bdg = dict.fromkeys(items_bdg, None)

wn_x = 2*np.pi*np.fft.rfftfreq(nx, dx)
wn_y = 2*np.pi*np.fft.rfftfreq(ny, dy)

# Interpolate u and v to w node
u_h = uvpnode2wnode(data_mom['avgU'])
v_h = uvpnode2wnode(data_mom['avgV'])


terms_ptb['u2_t'] = data_mom['avgU2'] - data_mom['avgU']**2 - data_mom['avgtxx']
terms_ptb['uv_t'] = data_mom['avgUV'] - data_mom['avgU']*data_mom['avgV'] - data_mom['avgtxy']
terms_ptb['uw_t'] = data_mom['avgUW'] - u_h*data_mom['avgW']
# terms_ptb['uw_t'] = wnode2uvpnode(data_mom['avgUW']) - data_mom['avgU']*wnode2uvpnode(data_mom['avgW']) - wnode2uvpnode_tij(data_mom['avgtxz'])
terms_ptb['uw_t'] = wnode2uvpnode(terms_ptb['uw_t']) - wnode2uvpnode(data_mom['avgtxz'])

terms_ptb['v2_t'] = data_mom['avgV2'] - data_mom['avgV']**2 - data_mom['avgtyy']
terms_ptb['vw_t'] = data_mom['avgVW'] - v_h*data_mom['avgW'] 
terms_ptb['vw_t'] = wnode2uvpnode(terms_ptb['vw_t']) - wnode2uvpnode(data_mom['avgtyz'])

terms_ptb['w2_t'] = data_mom['avgW2'] -data_mom['avgW']**2 
terms_ptb['w2_t'] = wnode2uvpnode(terms_ptb['w2_t']) + data_mom['avgtzz']

terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2
terms_ptb['tke_SGS'] = (data_mom['avgtxx']+data_mom['avgtyy']+data_mom['avgtzz']) / 2

#%%Plot profiles of Reynolds stresses and SGS components

l1 = ['u2_t','uv_t','uw_t','v2_t','vw_t','w2_t']
l2 = ['avgtxx','avgtxy','avgtxz','avgtyy','avgtyz','avgtzz']

data_mom['avgtxz'] = wnode2uvpnode(data_mom['avgtxz'])
data_mom['avgtyz'] = wnode2uvpnode(data_mom['avgtyz'])

fig,axs = plt.subplots(1,6,figsize=(16,8),tight_layout=True)

for i in range(len(axs)):
    
    # axs[i].plot(np.mean(terms_ptb[l1[i]],axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='k',label=l1[i])
    axs[i].plot(np.mean((-data_mom[l2[i]]),axis=(0,1)),np.arange(0,nz)*dz + dz/2,c='k',ls='--',label=l2[i])

    axs[i].axvline(0,c='k')
    axs[i].set_ylim(0,nz*dz)
    axs[i].legend()

plt.show()

#%%

checkpnt = read_checkpoint('/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/output_checkpoint/',[216000],nx,ny,nz)

#%%#%%Compute Reynolds stresses using the on-the-fly averages

uu = checkpnt['uu_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['u_new'][:,:,:nz]
vv = checkpnt['vv_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]
ww = checkpnt['ww_new'][:,:,:nz] - checkpnt['w_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]
uv = checkpnt['uv_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['v_new'][:,:,:nz]
uw = checkpnt['uw_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]
vw = checkpnt['vw_new'][:,:,:nz] - checkpnt['v_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz]

#%%Compute anisotropy using the checkpoint velocity
os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy

[xB,yB,lamba3] = Anisotropy(nx,ny,nz,terms_ptb['u2_t'],terms_ptb['v2_t'],terms_ptb['w2_t'],terms_ptb['uv_t'],terms_ptb['uw_t'],terms_ptb['vw_t'])
# [xB_c,yB_c,lamba3_c] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%%
points = 0

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            if yB[i,j,k]<0:
                points += 1
                print('Number of points:',points)
                print(i,j,k)
                
#%% Pcolor of Anisotropy

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz-1)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 20 #int(nz/2)
zslice = 20

# tmp = data_mom['avgYB'] - yB_c
tmp = data_aniso['avgYB'] - yB_c
# tmp = yB_c

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:-1].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%%
# tmp = terms_ptb['uw_t'] - (uw)
tmp = wnode2uvpnode(data_mom['avgW']) - (checkpnt['w_new'][:,:,:nz])
# tmp = (wnode2uvpnode(data_mom['avgUW']) - (data_mom['avgU'])*wnode2uvpnode(data_mom['avgW'])) - (checkpnt['uw_new'][:,:,:nz] - checkpnt['u_new'][:,:,:nz]*checkpnt['w_new'][:,:,:nz])
# tmp = wnode2uvpnode(data_mom['avgW2'])

x_ax = np.arange(0,nx)*dx
y_ax = np.arange(0,ny)*dy
z_ax = np.arange(0,nz-1)*dz
X, Y = np.meshgrid(x_ax,z_ax)
yslice = 20 #int(nz/2)
zslice = 30

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(8,6))

plt1 = axs.pcolormesh(x_ax,z_ax,tmp[:,yslice,:-1].T,cmap= 'coolwarm')
# plt1 = axs.pcolormesh(x_ax,y_ax,tmp[:,:,zslice].T,cmap= 'coolwarm')

axs.set_title(f'{yslice}')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
# plt.savefig(figPath+'fdzavg.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

#%% Fit curve to finer mesh and interpolate

def fit_curve_and_integrate(vertical_profile, z_coarse, z_fine):
    
    # x_coarse = np.linspace(2.5/2, 2.5*len(vertical_profile) + 2.5/2, len(vertical_profile))
    
    psi = np.zeros((len(vertical_profile))-1)
    
    coefficients = np.polyfit(z_coarse, vertical_profile, deg=9)
    polynomial = np.poly1d(coefficients)
    
    # x_fine = np.linspace(2.5/2, 2.5*len(vertical_profile) + 2.5/2, num_points)
    
    y_fine = polynomial(z_fine)
    
    for i in range(len(vertical_profile)-1):
        int_point = z_coarse[i+1]  # Second point on the coarse mesh
        idx_point = np.searchsorted(z_fine, int_point)
        
        x_integral = z_fine[:idx_point]
        y_integral = y_fine[:idx_point]
        
        psi[i] = np.trapz(y_integral, x_integral)
    
    return psi,y_fine

psiM = np.zeros((nx,ny,10))
z_coarse = np.zeros((nx,ny,11))
y_fine = np.zeros((nx,ny,51))
z_fine_ = np.zeros((nx,ny,51))
vertical_profile = np.zeros((nx,ny,11))
# z_coarse = np.array([0,dz/2,dz/2 + dz, dz/2+2*dz,dz/2 + 3*dz,dz/2 + 4*dz,dz/2 + 5*dz,dz/2 + 6*dz,dz/2 + 7*dz,dz/2 + 8*dz,dz/2 + 9*dz])

num_additional_points = 4

# Create the finer mesh vector
# z_fine = np.array([])

# for i in range(len(z_coarse) - 1):
#     # Add the current point
#     z_fine = np.append(z_fine, z_coarse[i])
    
#     # Generate additional points between the current and next point
#     z_fine = np.append(z_fine, np.linspace(z_coarse[i], z_coarse[i+1], num_additional_points + 2)[1:-1])

# # Add the last point
# z_fine = np.append(z_fine, z_coarse[-1])

for i in range(0,nx):
    for j in range(0,ny):
        
        z_coarse[i,j,:] = np.array([1e-10,dz/2,dz/2 + dz, dz/2+2*dz,dz/2 + 3*dz,dz/2 + 4*dz,dz/2 + 5*dz,dz/2 + 6*dz,dz/2 + 7*dz,dz/2 + 8*dz,dz/2 + 9*dz])
        for k in range(1,len(z_coarse[i,j,:])):
            z_coarse[i,j,k] = -z_coarse[i,j,k]/data_aniso['avgL3D'][i,j,k-1]
        
        z_fine = np.array([])
        for k in range(len(z_coarse[i,j,:]) - 1):
            # Add the current point
            z_fine = np.append(z_fine, z_coarse[i,j,k])
            # Generate additional points between the current and next point
            z_fine = np.append(z_fine, np.linspace(z_coarse[i,j,k], z_coarse[i,j,k+1], num_additional_points + 2)[1:-1])
        z_fine = np.append(z_fine, z_coarse[i,j,-1])
        z_fine_[i,j,:] = z_fine
        
        vertical_profile[i,j,1:] = data_aniso['avgPHIM'][i,j,0:10]
        # vertical_profile = np.insert(vertical_profile, 0 , 1)
        vertical_profile[i,j,1:] = (1 - vertical_profile[i,j,1:])/z_coarse[i,j,1:]
        psiM[i,j,:],y_fine[i,j,:] = fit_curve_and_integrate(vertical_profile[i,j,:],z_coarse[i,j,:],z_fine)

#%%Plotfit curves

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.scatter(vertical_profile[nx//2,ny//2,:],z_coarse[nx//2,ny//2,:],c='k')
axs.plot(y_fine[nx//2,ny//2,:],z_fine_[nx//2,ny//2,:])
# axs.scatter(y_fine[nx//2,ny//2,:],z_fine,c='r')
axs.set_xlabel('Phi')
axs.set_ylabel('z')
axs.set_xlim(-1,2)

plt.show()






































