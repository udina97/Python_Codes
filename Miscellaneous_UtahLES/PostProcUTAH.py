#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:27:54 2025

@author: u1450851
"""
#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc, Anisotropy

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

tscale = 290
uscale = 0.4
zi = 1000
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Import the data

sim = 'patch128_aniso_fix'
path = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/data/'
niter = 120000

dataM = xr.open_dataarray(path + 'Momentum3D/Data_Momentum_' + str(niter) + '.nc')
dataM2D = xr.open_dataarray(path + 'Momentum2D/Data_Momentum_2D_' + str(niter) + '.nc')
dataS = xr.open_dataarray(path + 'Scalar3D/Data_Scalar_' + str(niter) + '.nc')
dataS2D = xr.open_dataarray(path + 'Scalar2D/Data_Scalar_2D_' + str(niter) + '.nc')
dataA = xr.open_dataarray(path + 'Anisotropy/Data_Anisotropy_' + str(niter) + '.nc')

#%%Plot the surface temperature

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,(dataS2D[:,:,-2]*tscale).T,cmap='hot_r',vmin=285,vmax=295)
cbar = plt.colorbar(p)
axs.set_xlabel(r"$x/z_i$",fontsize=14)
axs.set_ylabel(r"$y/z_i$",fontsize=14)
axs.set_title(r"Surface Temp",fontsize=14)

plt.show()

#%%Flux gradient and the integrated functions at the surface

phiM = dataS2D.data[:,:,2]
phiH = dataS2D.data[:,:,4]
psiM = dataS2D.data[:,:,3]
psiH = dataS2D.data[:,:,5]


from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,4,tight_layout=True,figsize=(10,5))

kde = gaussian_kde((phiM[:,:]).flatten())
x_pdf = np.linspace(min((phiM[:,:]).flatten()), max((phiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(phiM[:,:]).T,cmap='bwr',vmin=0)
axs[1,0].hist((phiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((phiH[:,:]).flatten())
x_pdf = np.linspace(min((phiH[:,:]).flatten()), max((phiH[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(phiH[:,:]).T,cmap='bwr',vmin=0)
axs[1,1].hist((phiH[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiM[:,:]).flatten())
x_pdf = np.linspace(min((psiM[:,:]).flatten()), max((psiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,(psiM[:,:]).T,cmap='bwr')
axs[1,2].hist((psiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiH[:,:]).flatten())
x_pdf = np.linspace(min((psiH[:,:]).flatten()), max((psiH[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,3].pcolormesh(x,y,(psiH[:,:]).T,cmap='bwr')
axs[1,3].hist((psiH[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,3].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,3].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$\phi_M$', fontsize=14)
axs[1,1].set_xlabel(r'$\phi_H$', fontsize=14)
axs[1,2].set_xlabel(r'$\psi_M$', fontsize=14)
axs[1,3].set_xlabel(r'$\psi_H$', fontsize=14)

axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Plot phi vs z/L

zeta = np.logspace(-4,3,1000)
yb = np.linspace(0.1,0.8,20)
sc25_s = np.zeros((len(zeta),len(yb)))
sc25_u = np.zeros((len(zeta),len(yb)))
cmap = ColorAnisotropy()

for i in range(len(yb)):
    if yb[i]<0.6:
        a = 0.24-0.38*yb[i]
    else:
        a = 0.012
        
    b = 0.061
    c = 0.45 - 0.53*yb[i]
    n = -0.12 + 6.4*yb[i]
    sc25_u[:,i] = np.minimum((a + b*zeta**n)/(a+zeta**n) + c*zeta**(1/3),1)
    
    a = 0.76 + 1.5*yb[i]
    b = 6.3 - 4.3*yb[i]
    sc25_s[:,i] = np.maximum(a + b*zeta,1)

L = dataS2D.data[:,:,1]
zoverL = (dz/2)/L
zoverL_u = zoverL[(zoverL<0)]
zoverL_s = zoverL[(zoverL>0)]
phiM_u = phiM[(zoverL<0)]
phiM_s = phiM[(zoverL>0)]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.scatter(abs(zoverL_u),phiM_u,s=1,c='k')
for i in range(len(yb)):
    axs.plot(zeta,sc25_u[:,i],c=cmap(yb[i]))
axs.set_xscale('log')
axs.invert_xaxis()
axs.set_xlabel(r"$\zeta$",fontsize=14)
axs.set_ylabel(r"$\phi_M$",fontsize=14)
axs.set_ylim(0,2)

plt.show()

#%%Ustar and heatflux at the surface

ustar = dataM2D.data[:,:,0]
    
heatflux = dataS2D.data[:,:,-1]

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(8,6))

kde = gaussian_kde((ustar[:,:]).flatten())
x_pdf = np.linspace(min((ustar[:,:]).flatten()), max((ustar[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,ustar[:,:].T,cmap='bwr',vmin=0,vmax=1)
axs[1,0].hist((ustar[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((heatflux[:,:]).flatten())
x_pdf = np.linspace(min((heatflux[:,:]).flatten()), max((heatflux[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,heatflux[:,:].T,cmap='bwr',vmin=-0.001,vmax=0.001)
axs[1,1].hist((heatflux[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$u_*$ at sfc', fontsize=14)
axs[1,1].set_xlabel(r'$\overline{wT}$ at sfc', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)
axs[1,0].set_xlim(0,1)
axs[1,1].set_xlim(-0.001,0.002)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Anisotropy from the on-the-fly averaging

zslice = 0

xB = dataA.data[:,:,zslice,0]
yB = dataA.data[:,:,zslice,1]

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(9,7))

kde = gaussian_kde((xB).flatten())
x_pdf = np.linspace(min((xB).flatten()), max((xB).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,xB.T,cmap=ColorAnisotropy(),vmin=0,vmax=1)
axs[1,0].hist((xB).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((yB).flatten())
x_pdf = np.linspace(min((yB).flatten()), max((yB).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,yB.T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
axs[1,1].hist((yB).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$x_B$', fontsize=14)
axs[1,1].set_xlabel(r'$y_B$', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Compute the REynolds stresses and Anisotropy from RAV data

uu = dataM.data[:,:,:,4] - dataM.data[:,:,:,0]*dataM.data[:,:,:,0]
vv = dataM.data[:,:,:,5] - dataM.data[:,:,:,1]*dataM.data[:,:,:,1]
ww = wnode2uvpnode(dataM.data[:,:,:,6] - dataM.data[:,:,:,2]*dataM.data[:,:,:,2])
uv = dataM.data[:,:,:,13] - dataM.data[:,:,:,0]*dataM.data[:,:,:,1]
uw = wnode2uvpnode(dataM.data[:,:,:,14] - uvpnode2wnode(dataM.data[:,:,:,0])*dataM.data[:,:,:,2])
vw = wnode2uvpnode(dataM.data[:,:,:,15] - uvpnode2wnode(dataM.data[:,:,:,1])*dataM.data[:,:,:,2])

[xB_rav,yB_rav,lamba3] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%%Plot Anisotropy from RAV data

zslice = 0

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(9,7))

kde = gaussian_kde((xB_rav[:,:,zslice]).flatten())
x_pdf = np.linspace(min((xB_rav[:,:,zslice]).flatten()), max((xB_rav[:,:,zslice]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,xB_rav[:,:,zslice].T,cmap=ColorAnisotropy(),vmin=0,vmax=1)
axs[1,0].hist((xB_rav[:,:,zslice]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((yB_rav[:,:,zslice]).flatten())
x_pdf = np.linspace(min((yB_rav[:,:,zslice]).flatten()), max((yB_rav[:,:,zslice]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,yB_rav[:,:,zslice].T,cmap=ColorAnisotropy(),vmin=0,vmax=np.sqrt(3)/2)
axs[1,1].hist((yB_rav[:,:,zslice]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$x_B$', fontsize=14)
axs[1,1].set_xlabel(r'$y_B$', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Velocity field, vertical planes

vslice = 50

u = dataM.data[:,vslice,:,0]
v = dataM.data[:,vslice,:,1]
w = dataM.data[:,vslice,:,2]

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((u).flatten())
x_pdf = np.linspace(min((u).flatten()), max((u).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,z_uvp,u.T,cmap='bwr')
axs[1,0].hist((u).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((v).flatten())
x_pdf = np.linspace(min((v).flatten()), max((v).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,z_uvp,v.T,cmap='bwr')
axs[1,1].hist((v).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((w).flatten())
x_pdf = np.linspace(min((w).flatten()), max((w).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,z_w,w.T,cmap='bwr')
axs[1,2].hist((w).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$u$', fontsize=14)
axs[1,1].set_xlabel(r'$v$', fontsize=14)
axs[1,2].set_xlabel(r'$w$', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Velocity field, mean along y

u = np.mean(dataM.data[:,:,:,0],axis=1)
v = np.mean(dataM.data[:,:,:,1],axis=1)
w = np.mean(dataM.data[:,:,:,2],axis=1)

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((u).flatten())
x_pdf = np.linspace(min((u).flatten()), max((u).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,z_uvp,u.T,cmap='bwr')
axs[1,0].hist((u).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((v).flatten())
x_pdf = np.linspace(min((v).flatten()), max((v).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,z_uvp,v.T,cmap='bwr')
axs[1,1].hist((v).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((w).flatten())
x_pdf = np.linspace(min((w).flatten()), max((w).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,z_w,w.T,cmap='bwr')
axs[1,2].hist((w).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$u$', fontsize=14)
axs[1,1].set_xlabel(r'$v$', fontsize=14)
axs[1,2].set_xlabel(r'$w$', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Velocity field horizontal planes

hslice = 10

u = dataM.data[:,:,hslice,0]
v = dataM.data[:,:,hslice,1]
w = dataM.data[:,:,hslice,2]

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((u).flatten())
x_pdf = np.linspace(min((u).flatten()), max((u).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,u.T,cmap='bwr')
axs[1,0].hist((u).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((v).flatten())
x_pdf = np.linspace(min((v).flatten()), max((v).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,v.T,cmap='bwr')
axs[1,1].hist((v).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((w).flatten())
x_pdf = np.linspace(min((w).flatten()), max((w).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,w.T,cmap='bwr')
axs[1,2].hist((w).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$u$', fontsize=14)
axs[1,1].set_xlabel(r'$v$', fontsize=14)
axs[1,2].set_xlabel(r'$w$', fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%Plot variances,horizontal plane

hslice = 10

u_var = uu[:,:,hslice]
v_var = vv[:,:,hslice]
w_var = ww[:,:,hslice]

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((u_var).flatten())
x_pdf = np.linspace(min((u_var).flatten()), max((u_var).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,u_var.T,cmap='bwr')
axs[1,0].hist((u_var).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((v_var).flatten())
x_pdf = np.linspace(min((v_var).flatten()), max((v_var).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,v_var.T,cmap='bwr')
axs[1,1].hist((v_var).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((w_var).flatten())
x_pdf = np.linspace(min((w_var).flatten()), max((w_var).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,w_var.T,cmap='bwr')
axs[1,2].hist((w_var).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r"$\overline{u'u'}$", fontsize=14)
axs[1,1].set_xlabel(r"$\overline{v'v'}$", fontsize=14)
axs[1,2].set_xlabel(r"$\overline{w'w'}$", fontsize=14)
axs[0,0].set_ylabel(r"$y/z_i$",fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()


#%%Temperature and Temperature variance

zslice = 0

T = dataS.data[:,:,:,0]*tscale
TT = (dataS.data[:,:,:,1] - dataS.data[:,:,:,0]*dataS.data[:,:,:,0])*tscale**2

from scipy.stats import gaussian_kde

fig,axs = plt.subplots(2,2,tight_layout=True,figsize=(12,7))

kde = gaussian_kde((T[:,zslice,:]).flatten())
x_pdf = np.linspace(min((T[:,zslice,:]).flatten()), max((T[:,zslice,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,z_uvp,T[:,zslice,:].T,cmap='hot_r',vmin=285,vmax=288)
axs[1,0].hist((T[:,zslice,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((TT[:,:,zslice]).flatten())
x_pdf = np.linspace(min((TT[:,:,zslice]).flatten()), max((TT[:,:,zslice]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,TT[:,:,zslice].T,cmap='hot_r')
axs[1,1].hist((TT[:,:,zslice]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$T$', fontsize=14)
axs[1,1].set_xlabel(r"$\overline{T'T'}$", fontsize=14)
axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()

#%%TKE profile

tke = 0.5*(uu + vv + ww)

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(10,4))

p = axs.pcolormesh(x,z_uvp,tke[:,60,:].T,cmap='hot_r')
cbar = plt.colorbar(p,label='TKE')
axs.set_xlabel(r"$x/z_i$",fontsize=14)
axs.set_ylabel(r"$z/z_i$",fontsize=14)
fig.suptitle(sim,fontsize=14)

plt.show()

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(np.mean(tke,axis=(0,1)),z_uvp,c='k')
axs.set_xlabel(r"TKE",fontsize=14)
axs.set_ylabel(r"$z/z_i$",fontsize=14)
axs.set_ylim(0,z_uvp[-1])
fig.suptitle(sim,fontsize=14)

plt.show()




































