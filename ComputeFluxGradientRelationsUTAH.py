#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:58:35 2025

@author: u1450851
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

#%%Set some simulation parameters

Nx = 128
Ny = 128
Nz = 128
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

zi = 1000
uscale = 0.4
Tscale = 290
x = np.arange(0,Nx)*dx
y = np.arange(0,Ny)*dy
z_uvp = np.arange(0,Nz)*dz + dz/2
z_w = np.arange(0,Nz)*dz

kvonk = 0.4
g_hat = 9.81*zi/(uscale**2)

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

#%%Set the path to the simulation and data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

# sim = 'Tpatch800_a288m290s2_1ms_a'
sim = 'patch128_aniso_fix'
path_to_data = path + sim + '/data/'

mom3D = xr.open_dataarray(path_to_data + 'Momentum3D/Data_Momentum_120000.nc')
mom2D = xr.open_dataarray(path_to_data + 'Momentum2D/Data_Momentum_2D_120000.nc')
sc3D = xr.open_dataarray(path_to_data + 'Scalar3D/Data_Scalar_120000.nc')
sc2D = xr.open_dataarray(path_to_data + 'Scalar2D/Data_Scalar_2D_120000.nc')
# aniso = xr.open_dataarray(path_to_data + 'Anisotropy/Data_Anisotropy_' + str(nRAV_start[i]) + '.nc')

#%%Plot surface temperature pattern

Tsfc = sc2D[:,:,-2].values

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,(Tsfc*Tscale).T,cmap='hot_r',vmin=285,vmax=295)
axs.set_xlabel(r"$x/z_i$",fontsize=15)
axs.set_ylabel(r"$y/z_i$",fontsize=15)
axs.set_title(r"Surface T", fontsize=12)
cbar = plt.colorbar(p)

plt.show()

#%%Plot surface parameters

ustar = mom2D[:,:,0].values
wT = sc2D[:,:,-1].values
L = sc2D[:,:,1].values
zoverL = (dz/2)/L
T = np.mean(sc3D[:,:,0,0].values)
L_rd = -(np.mean(ustar)**3)*T/(kvonk*g_hat*np.mean(wT))

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,4))

p = axs[0].pcolormesh(x,y,ustar.T,cmap='jet',vmin=0,vmax=1)
axs[0].set_xlabel(r"$x/z_i$",fontsize=15)
axs[0].set_ylabel(r"$y/z_i$",fontsize=15)
axs[0].set_title(r"$u_*$ - " + f"RD = {np.mean(ustar) :.03f}",fontsize=15)
cbar = plt.colorbar(p)

p = axs[1].pcolormesh(x,y,wT.T,cmap='jet',vmin=-5e-3,vmax=5e-3)
axs[1].set_xlabel(r"$x/z_i$",fontsize=15)
axs[1].set_ylabel(r"$y/z_i$",fontsize=15)
axs[1].set_title(r"$w'T'$ - " + f"RD = {np.mean(wT) :.2E}",fontsize=15)
cbar = plt.colorbar(p)

p = axs[2].pcolormesh(x,y,(zoverL).T,cmap='jet',vmin=-10,vmax=10)
axs[2].set_xlabel(r"$x/z_i$",fontsize=15)
axs[2].set_ylabel(r"$y/z_i$",fontsize=15)
axs[2].set_title(r"$L$ - " + f"RD = {L_rd :.2E}",fontsize=15)
cbar = plt.colorbar(p)

fig.suptitle(sim,fontsize=15)

plt.show()

#%%Compute the pdf of the surface parameters

import seaborn as sns

plt.figure()

sns.kdeplot(wT.flatten())
plt.title(sim,fontsize=15)
# plt.xscale('log')

plt.show()

#%%Plot the surface scaling functions

phiM = sc2D[:,:,2].values
psiM = sc2D[:,:,3].values
phiH = sc2D[:,:,4].values
psiH = sc2D[:,:,5].values

fig,axs = plt.subplots(1,4,tight_layout=True,figsize=(12,4),sharey=True)

p = axs[0].pcolormesh(x,y,phiM.T,cmap='jet',vmin=0,vmax=5)
# axs[0].plot(x[42],y[44],'ko')
axs[0].set_xlabel(r"$x/z_i$",fontsize=15)
axs[0].set_ylabel(r"$y/z_i$",fontsize=15)
axs[0].set_title(r"$\phi_M$",fontsize=15)
cbar = plt.colorbar(p)

p = axs[1].pcolormesh(x,y,psiM.T,cmap='jet',vmin=-10,vmax=2)
axs[1].set_xlabel(r"$x/z_i$",fontsize=15)
# axs[1].set_ylabel(r"$y/z_i$",fontsize=15)
axs[1].set_title(r"$\psi_M$",fontsize=15)
cbar = plt.colorbar(p)

p = axs[2].pcolormesh(x,y,phiH.T,cmap='jet',vmin=0,vmax=5)
axs[2].set_xlabel(r"$x/z_i$",fontsize=15)
# axs[2].set_ylabel(r"$y/z_i$",fontsize=15)
axs[2].set_title(r"$\phi_H$",fontsize=15)
cbar = plt.colorbar(p)

p = axs[3].pcolormesh(x,y,psiH.T,cmap='jet',vmin=-10,vmax=5)
axs[3].set_xlabel(r"$x/z_i$",fontsize=15)
# axs[3].set_ylabel(r"$y/z_i$",fontsize=15)
axs[3].set_title(r"$\psi_H$",fontsize=15)
cbar = plt.colorbar(p)

fig.suptitle(sim,fontsize=15)

plt.show()

#%%Compute the pdf of the surface parameters

import seaborn as sns

plt.figure()

sns.kdeplot(psiM.flatten())
plt.title(sim,fontsize=15)

plt.show()

#%%Compute the velocity gradient

ustar3D = (((mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2] - mom3D[:,:,:,20])**2 + \
          (mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2] - mom3D[:,:,:,21])**2)**0.25).values
    
heatflux3D = (sc3D[:,:,:,4] - uvpnode2wnode(sc3D[:,:,:,0])*mom3D[:,:,:,2] - sc3D[:,:,:,7]).values

L3D = -(wnode2uvpnode(ustar3D)**3)*sc3D[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D))

zoverL3D = np.ones((Nx,Ny,Nz),order='F')*z_uvp/L3D

u = (mom3D[:,:,:,0].values)
v = (mom3D[:,:,:,1].values)
dudz = wnode2uvpnode(mom3D[:,:,:,22].values)
dvdz = wnode2uvpnode(mom3D[:,:,:,23].values)
meanDUDZ = (u*dudz + v*dvdz)/np.sqrt(u**2 + v**2)

phiM3D = (0.4*(np.ones((Nx,Ny,Nz),order="F")*z_uvp)/wnode2uvpnode(ustar3D))*meanDUDZ

phiM_stable = phiM3D[:,:,0][(zoverL3D[:,:,0]>0)]
phiM_unstable = phiM3D[:,:,0][(zoverL3D[:,:,0]<0)]
zeta_stable = zoverL3D[:,:,0][(zoverL3D[:,:,0]>0)]
zeta_unstable = zoverL3D[:,:,0][(zoverL3D[:,:,0]<0)]

#%%Plot the surface level ustar,wT,zoverL computed using the RAV data, to compare to the LES outputs

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(10,4))

p = axs[0].pcolormesh(x,y,abs(ustar3D[:,:,0] - ustar).T,cmap='jet',vmin=0,vmax=1)
axs[0].set_xlabel(r"$x/z_i$",fontsize=15)
axs[0].set_ylabel(r"$y/z_i$",fontsize=15)
# axs[0].set_title(r"$u_*$ - " + f"RD = {np.mean(ustar) :.03f}",fontsize=15)
cbar = plt.colorbar(p)

p = axs[1].pcolormesh(x,y,abs(heatflux3D[:,:,0] - wT).T,cmap='jet',vmin=-5e-3,vmax=5e-3)
axs[1].set_xlabel(r"$x/z_i$",fontsize=15)
axs[1].set_ylabel(r"$y/z_i$",fontsize=15)
# axs[1].set_title(r"$w'T'$ - " + f"RD = {np.mean(wT) :.2E}",fontsize=15)
cbar = plt.colorbar(p)

p = axs[2].pcolormesh(x,y,abs(zoverL3D[:,:,0] - zoverL).T,cmap='jet',vmin=-10,vmax=10)
axs[2].set_xlabel(r"$x/z_i$",fontsize=15)
axs[2].set_ylabel(r"$y/z_i$",fontsize=15)
# axs[2].set_title(r"$L$ - " + f"RD = {L_rd :.2E}",fontsize=15)
cbar = plt.colorbar(p)

fig.suptitle(sim,fontsize=15)

plt.show()

#%%Plot the pdf of phiM3D

import seaborn as sns

plt.figure()

sns.kdeplot(phiM3D.flatten())
plt.title(sim,fontsize=15)

plt.show()

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

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Compute anisotropy with RAV data

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from Anisotropy_Functions import Anisotropy

Rxx = mom3D[:,:,:,4].values - mom3D[:,:,:,0].values*mom3D[:,:,:,0].values
Ryy = mom3D[:,:,:,5].values - mom3D[:,:,:,1].values*mom3D[:,:,:,1].values
Rzz = wnode2uvpnode(mom3D[:,:,:,6].values - mom3D[:,:,:,2].values*mom3D[:,:,:,2].values)
Rxy = mom3D[:,:,:,13].values - mom3D[:,:,:,0].values*mom3D[:,:,:,1].values
Rxz = wnode2uvpnode(mom3D[:,:,:,14].values - uvpnode2wnode(mom3D[:,:,:,0].values)*mom3D[:,:,:,2].values)
Ryz = wnode2uvpnode(mom3D[:,:,:,15].values - uvpnode2wnode(mom3D[:,:,:,1].values)*mom3D[:,:,:,2].values)

xb,yb,lambda3 = Anisotropy(Nx,Ny,Nz,Rxx,Ryy,Rzz,Rxy,Rxz,Ryz)

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
    phi_u_fit_u[:,i] = ((1 - 3*zeta_vec_u)**(1/3))*a_u
    phi_u_fit_s[:,i] = ((1 + 3*zeta_vec_s)**(c_s))*a_s
    print(c_s)

#%%Classic scalings for sigmaU

phiU_c_u = 2.55*(1-3*zeta_vec_u)**(1/3)
phiU_c_s = 2.06*np.ones(len(zeta_vec_u))

#%%Compute ustar,heatflux and L

ustar3D = (((mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2] - mom3D[:,:,:,20])**2 + \
          (mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2] - mom3D[:,:,:,21])**2)**0.25).values

heatflux3D = (sc3D[:,:,:,4] - uvpnode2wnode(sc3D[:,:,:,0])*mom3D[:,:,:,2] - sc3D[:,:,:,7]).values

L3D = -(wnode2uvpnode(ustar3D)**3)*sc3D[:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D))

zoverL3D = np.ones((Nx,Ny,Nz))*z_uvp/L3D

#%%Compute sigmaU

vstart = 0
vend = 1
vlevel = 0

sigmaU = np.sqrt(mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0]).values/wnode2uvpnode(ustar3D)

#Compute a(yb) and then normalize phiU with it to check data scatter collapse

log_yb = np.log10(yb)

a_yb_u = 0.784*log_yb**0 - 2.582*log_yb
a_yb_s = 2.332*yb**0 - 2.047*yb + 2.672*yb**2

sigmaU_u = sigmaU[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]<0)]#/a_yb_u[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]<0)]
sigmaU_s = sigmaU[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]>0)]#/a_yb_s[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]>0)]
zeta_u = zoverL3D[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]<0)]
zeta_s = zoverL3D[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]>0)]
yb_u = yb[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]<0)]
yb_s = yb[:,:,vstart:vend][(zoverL3D[:,:,vstart:vend]>0)]

#%%Plot the phiU vs zeta and scaling curves

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
    
axs[0].semilogx(-zeta_vec_u,phiU_c_u,c='k')
axs[1].semilogx(zeta_vec_s,phiU_c_s,c='k')

# axs[0].scatter(abs(zeta_u),sigmaU_u,s=0.1,alpha=0.1)
# axs[1].scatter(abs(zeta_s),sigmaU_s,s=0.1,alpha=0.1)

axs[0].hist2d(abs(zeta_u),sigmaU_u,bins=[np.logspace(np.log10(abs(zeta_u).min()), np.log10(abs(zeta_u).max()), 100),\
                                          np.linspace((sigmaU_u.min()),(sigmaU_u.max()), 500)],cmap='hot_r')
axs[1].hist2d(abs(zeta_s),sigmaU_s,bins=[np.logspace(np.log10(abs(zeta_s).min()), np.log10(abs(zeta_s).max()), 100),\
                                          np.linspace((sigmaU_s.min()),(sigmaU_s.max()), 500)],cmap='hot_r')


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

fig.suptitle(sim,fontsize=15)
    
plt.show()


#%%Compute the scalings for w velocity

zeta_vec_u = np.array([-1e2,-0.5e2,-1e1,-0.5e1,-1e0,-0.5e0,-1e-1,-5e-2,-1e-2,-5e-3,-1e-3,-5e-4,-1e-4])
zeta_vec_s = abs(zeta_vec_u[::-1])
y_b_vec = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
phi_w_fit_s = np.zeros((len(zeta_vec_s),len(y_b_vec)))
phi_w_fit_u = np.zeros((len(zeta_vec_u),len(y_b_vec)))
coef_a_u = [1.119,-0.019,-0.065,0.028]
coef_a_s = [0.953,0.188,2.253]
coef_c_s = [0.208,-1.935,6.183,-7.485,3.077]

for i in range(len(y_b_vec)):
    a_u=0
    a_s=0
    c_s=0
    for j in range(len(coef_a_u)):
        tmp_a_u = coef_a_u[j]*(y_b_vec[i])**(j)
        a_u = tmp_a_u + a_u
    for j in range(len(coef_a_s)):
        tmp_a_s = coef_a_s[j]*(y_b_vec[i])**(j)
        a_s = tmp_a_s + a_s
    for j in range(len(coef_c_s)):
        tmp_c_s = coef_c_s[j]*(y_b_vec[i])**(j)
        c_s = tmp_c_s + c_s
    phi_w_fit_u[:,i] = ((1 - 3*zeta_vec_u)**(1/3))*a_u
    phi_w_fit_s[:,i] = ((1 + 3*zeta_vec_s)**(c_s))*a_s

#%%Compute sigmaW

vlevel = 0

sigmaW = wnode2uvpnode(np.sqrt(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2]).values/(ustar3D))

#Compute a(yb) and then normalize phiU with it to check data scatter collapse

a_yb_u = 1.119*yb**0 - 0.019*yb**1 - 0.065*yb**2 + 0.028*yb**3
a_yb_s = 0.953*yb**0 + 0.188*yb**1 + 2.253*yb**2

sigmaW_u = sigmaW[:,:,:][(zoverL3D[:,:,:]<0)]#/a_yb_u[:,:,:][(zoverL3D[:,:,:]<0)]
sigmaW_s = sigmaW[:,:,:][(zoverL3D[:,:,:]>0)]#/a_yb_s[:,:,:][(zoverL3D[:,:,:]>0)]
zeta_u = zoverL3D[:,:,:][(zoverL3D[:,:,:]<0)]
zeta_s = zoverL3D[:,:,:][(zoverL3D[:,:,:]>0)]
    
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
    axs[0].semilogx(-zeta_vec_u,phi_w_fit_u[:,i],c=cmap(y_b_vec[i]))
    axs[1].semilogx(zeta_vec_s,phi_w_fit_s[:,i],c=cmap(y_b_vec[i]))

axs[0].hist2d(abs(zeta_u),sigmaW_u,bins=[np.logspace(np.log10(abs(zeta_u).min()), np.log10(abs(zeta_u).max()), 1000),\
                                          np.linspace((sigmaW_u.min()),(sigmaW_u.max()), 500)],cmap='hot_r')
axs[1].hist2d(abs(zeta_s),sigmaW_s,bins=[np.logspace(np.log10(abs(zeta_s).min()), np.log10(abs(zeta_s).max()), 1000),\
                                          np.linspace((sigmaW_s.min()),(sigmaW_s.max()), 500)],cmap='hot_r')

# axs[0].scatter(abs(zeta_u),sigmaW_u,s=0.5,alpha=0.1)
# axs[1].scatter(abs(zeta_s),sigmaW_s,s=0.5,alpha=0.1)

axs[0].set_ylim(0,10)
axs[0].set_xscale('log')
axs[0].set_xlim(1e-4,120)
axs[0].invert_xaxis()
axs[0].set_xlabel(r"$-\zeta$",fontsize=12)
axs[0].set_ylabel(r"$\phi_W$",fontsize=12)
axs[0].set_title(r"Unstable",fontsize=12)

axs[1].set_xscale('log')
axs[1].set_xlim(1e-4,120)
axs[1].set_xlabel(r"$\zeta$",fontsize=12)
axs[1].set_title(r"Stable",fontsize=12)

fig.suptitle(sim + f' - zlevel = {vlevel*dz*zi + dz*zi/2}m',fontsize=15)
    
plt.show()

#%%

#%% Temperature scaling

vlevel = 3

Tstar = -(sc3D[:,:,:,4].values - mom3D[:,:,:,2].values*uvpnode2wnode(sc3D[:,:,:,0].values) - sc3D[:,:,:,7].values)/ustar3D

sigmaT = np.sqrt(sc3D[:,:,:,1].values - sc3D[:,:,:,0].values*sc3D[:,:,:,0].values)/wnode2uvpnode(Tstar)

sigmaT_u = sigmaT[:,:,vlevel][(zoverL3D[:,:,vlevel]<0)]
sigmaT_s = sigmaT[:,:,vlevel][(zoverL3D[:,:,vlevel]>0)]
zeta_u = zoverL3D[:,:,vlevel][(zoverL3D[:,:,vlevel]<0)]
zeta_s = zoverL3D[:,:,vlevel][(zoverL3D[:,:,vlevel]>0)]

#%%Anisotropy adjusted scaling for temperature

zeta_vec_u = np.array([-1e2,-0.5e2,-1e1,-0.5e1,-1e0,-0.5e0,-1e-1,-5e-2,-1e-2,-5e-3,-1e-3,-5e-4,-1e-4])
zeta_vec_s = abs(zeta_vec_u[::-1])
y_b_vec = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
phi_T_fit_s = np.zeros((len(zeta_vec_s),len(y_b_vec)))
phi_T_fit_u = np.zeros((len(zeta_vec_u),len(y_b_vec)))
coef_a_u = [0.017,0.217]
coef_a_s = [0.607,-0.754]
coef_b_s = [-0.353,3.374,-8.544,6.297]
coef_c_s = [0.195,-1.857,5.042,-3.874]
coef_d_s = [0.0763,-1.004,2.836,-2.53]

for i in range(len(y_b_vec)):
    a_u=0
    a_s=0
    b_s=0
    c_s=0
    d_s=0
    for j in range(len(coef_a_u)):
        tmp_a_u = coef_a_u[j]*(y_b_vec[i])**(j)
        a_u = tmp_a_u + a_u
    for j in range(len(coef_a_s)):
        tmp_a_s = coef_a_s[j]*(y_b_vec[i])**(j)
        a_s = tmp_a_s + a_s
    for j in range(len(coef_b_s)):
        tmp_b_s = coef_b_s[j]*(y_b_vec[i])**(j)
        b_s = tmp_b_s + b_s
    for j in range(len(coef_c_s)):
        tmp_c_s = coef_c_s[j]*(y_b_vec[i])**(j)
        c_s = tmp_c_s + c_s
    for j in range(len(coef_d_s)):
        tmp_d_s = coef_d_s[j]*(y_b_vec[i])**(j)
        d_s = tmp_d_s + d_s
    phi_T_fit_u[:,i] = 1.07*(0.05 + abs(zeta_vec_u))**(-1/3) + (-1.14 + a_u*abs(zeta_vec_u)**(-9/10))*(1 - np.tanh(10*abs(zeta_vec_u)**(2/3)))
    phi_T_fit_s[:,i] = 10**(a_s + b_s*np.log10(zeta_vec_s) + c_s*(np.log10(zeta_vec_s))**2 + d_s*(np.log10(zeta_vec_s))**3)
    
#%%Plot

from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
import seaborn as sns
os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")
from Anisotropy_Functions import ColorAnisotropy
cmap = ColorAnisotropy()

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=False,figsize=(10,5))

# log_bin_x = np.logspace(np.log10(x_sample.min()), np.log10(x_sample.max()), 200)
# bin_y = np.linspace((y_sample.min()),(y_sample.max()), 200)

# h,xedge,yedge,img=axs[0].hist2d(x_sample,y_sample,bins=[log_bin_x,bin_y],cmap='viridis',density=False,norm=Normalize(vmin=0,vmax=1))
# h[h==0] = np.nan
# img.set_array(h.T.ravel())
# img.set_array(img.get_array()/np.nanmax(h))

for i in range(len(y_b_vec)):
    axs[0].plot(-zeta_vec_u,phi_T_fit_u[:,i],c=cmap(y_b_vec[i]))
    axs[1].plot(zeta_vec_s,phi_T_fit_s[:,i],c=cmap(y_b_vec[i]))

axs[0].hist2d(abs(zeta_u),abs(sigmaT_u),bins=[np.logspace(np.log10(abs(zeta_u).min()), np.log10(abs(zeta_u).max()), 200),\
                                          np.linspace(np.log10(abs(sigmaT_u).min()),np.log10(abs(sigmaT_u).max()), 200)],cmap='hot_r')
axs[1].hist2d(abs(zeta_s),abs(sigmaT_s),bins=[np.logspace(np.log10(abs(zeta_s).min()), np.log10(abs(zeta_s).max()), 20),\
                                          np.linspace(np.log10(abs(sigmaT_s).min()),np.log10(abs(sigmaT_s).max()), 20)],cmap='hot_r')

# axs[0].scatter(abs(zeta_u),abs(sigmaT_u),s=0.5,alpha=0.1)
# axs[1].scatter(abs(zeta_s),abs(sigmaT_s),s=0.5,alpha=0.1)

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlim(1e-4,120)
axs[0].set_ylim(1e-1,400)
axs[0].invert_xaxis()
axs[0].set_xlabel(r"$-\zeta$",fontsize=12)
axs[0].set_ylabel(r"$\phi_T$",fontsize=12)
axs[0].set_title(r"Unstable",fontsize=12)

axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlim(1e-4,120)
axs[1].set_ylim(1e-1,400)
axs[1].set_xlabel(r"$\zeta$",fontsize=12)
axs[1].set_title(r"Stable",fontsize=12)

fig.suptitle(sim + f' - zlevel = {vlevel*dz*zi + dz*zi/2}m',fontsize=15)
    
plt.show()

#%%Plot PDF of zeta

import seaborn as sns

plt.figure()

# sns.kdeplot(abs(zeta_u).flatten())
plt.hist(abs(zeta_s).flatten(),bins=np.logspace(np.log10(abs(zeta_s).min()), np.log10(abs(zeta_s).max()), 200))
plt.title(sim,fontsize=15)
plt.xscale('log')
# plt.xlim(0,1000)

plt.show()


#%%

plt.figure()

sns.kdeplot(abs(zeta_u).flatten(),)
# plt.hist(abs(sigmaU_u).flatten(),bins=np.logspace(np.log10(abs(sigmaU_u).min()), np.log10(abs(sigmaU_u).max()), 200))
plt.title(sim,fontsize=15)
# plt.xscale('log')
# plt.xlim(0,1000)

plt.show()




















































