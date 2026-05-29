#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 15:00:29 2025

@author: u1450851
"""

#Import libraries

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
# sim = 'test_mag_L_ustar'
# sim = '800p_128_noaniso_1ms'
# sim = '800p_128_aniso_1ms'
sim = 'Tpatch800_a288m290s2_1ms_a'
# sim = 'Tpatch800_a288m290s2_1ms_na'
path_to_data = path + sim + '/data/'

nRAV_start = [100000,50000,86000,86410]
nRAV = 1
mom3D = dict(); mom2D = dict(); sc3D = dict(); sc2D = dict(); aniso = dict()

for i in range(0,nRAV):
    mom3D[str(nRAV_start[i])] = xr.open_dataarray(path_to_data + 'Momentum3D/Data_Momentum_' + str(nRAV_start[i]) + '.nc')
    mom2D[str(nRAV_start[i])] = xr.open_dataarray(path_to_data + 'Momentum2D/Data_Momentum_2D_' + str(nRAV_start[i]) + '.nc')
    sc3D[str(nRAV_start[i])] = xr.open_dataarray(path_to_data + 'Scalar3D/Data_Scalar_' + str(nRAV_start[i]) + '.nc')
    sc2D[str(nRAV_start[i])] = xr.open_dataarray(path_to_data + 'Scalar2D/Data_Scalar_2D_' + str(nRAV_start[i]) + '.nc')
    aniso[str(nRAV_start[i])] = xr.open_dataarray(path_to_data + 'Anisotropy/Data_Anisotropy_' + str(nRAV_start[i]) + '.nc')

#%%Plot the surface temperature map

tstep = 100000
Tsfc = sc2D[str(tstep)][:,:,-2].values

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,(Tsfc*Tscale).T,cmap='hot_r',vmin=285,vmax=295)
axs.set_xlabel(r"$x/z_i$",fontsize=15)
axs.set_ylabel(r"$y/z_i$",fontsize=15)
axs.set_title(r"Surface T", fontsize=12)
cbar = plt.colorbar(p)

plt.show()

#%%Plot the temperature at the first grid point

zslice = 0

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,(sc3D[str(tstep)][:,:,zslice,0]*Tscale).T,cmap='hot_r',vmin=285,vmax=295)
cbar = plt.colorbar(p)
axs.set_xlabel(r"$x/z_i$",fontsize=15)
axs.set_ylabel(r"$y/z_i$",fontsize=15)
axs.set_title(f"T at z-level = {zslice}",fontsize=15)

plt.show()

#%%Compute the deltaT between the surface and the first gridpoint
import seaborn as sns
deltaT = (Tsfc - sc3D[str(tstep)][:,:,0,0].values)*Tscale

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,y,deltaT.T,cmap='hot_r',vmin=-10,vmax=10)
axs.set_xlabel(r"$x/z_i$",fontsize=15)
axs.set_ylabel(r"$y/z_i$",fontsize=15)
axs.set_title('Tsfc - T(dz/2)',fontsize=15)
cbar = plt.colorbar(p)

plt.show()

#%%Plot ustar,wT,L at the surface

tstep = 100000
ustar = mom2D[str(tstep)][:,:,0].values
wT = sc2D[str(tstep)][:,:,-1].values
L = sc2D[str(tstep)][:,:,1].values
zoverL = (dz/2)/L
T = np.mean(sc3D[str(tstep)][:,:,0,0].values)
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

fig.suptitle(f'tstep = {tstep} - ' + sim,fontsize=15)

plt.show()

#%%Compute the pdf of the surface parameters

import seaborn as sns

plt.figure()

sns.kdeplot(ustar.flatten())

plt.show()

#%%Plot histogram

fig, axs = plt.subplots(1,1,tight_layout=True)

axs.hist(abs(zoverL).flatten(),bins=10000)
# axs.set_xlim(-1,1)
axs.set_xscale('log')
axs.set_yscale('log')

plt.show()

#%%Plot the surface scaling functions

# tstep = 50000
phiM = sc2D[str(tstep)][:,:,2].values
psiM = sc2D[str(tstep)][:,:,3].values
phiH = sc2D[str(tstep)][:,:,4].values
psiH = sc2D[str(tstep)][:,:,5].values

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

fig.suptitle(f'tstep = {tstep} - ' + sim,fontsize=15)

plt.show()

#%%Plot histogram of the scaling functions to see the range of common values

fig, axs = plt.subplots(1,1,tight_layout=True)

axs.hist(phiM.flatten(),bins=1000)
# axs.set_xlim(-1,1)

plt.show()

#%% plot horizontal slices of the velocity compoents

zslice = 1

u = mom3D[str(tstep)][:,:,zslice,0].values
v = mom3D[str(tstep)][:,:,zslice,1].values
w = mom3D[str(tstep)][:,:,zslice,2].values

fig,axs = plt.subplots(1,3,tight_layout=True,sharey=True,figsize=(10,4))

p1=axs[0].pcolormesh(x,y,u.T,cmap='jet')
axs[0].set_title('U')
cbar=plt.colorbar(p1)
p2=axs[1].pcolormesh(x,y,v.T,cmap='jet')
axs[1].set_title('V')
cbar=plt.colorbar(p2)
p3=axs[2].pcolormesh(x,y,w.T,cmap='jet')
axs[2].set_title('W')
cbar=plt.colorbar(p3)

for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=12)
    
axs[0].set_ylabel(r"$y/z_i$",fontsize=12)

fig.suptitle(f'tstep = {tstep} - ' + sim,fontsize=15)

plt.show()

#%%Plot vertical planes of the velocity

vslice = 60

u = mom3D[str(tstep)][:,vslice,:,0].values
v = mom3D[str(tstep)][:,vslice,:,1].values
w = mom3D[str(tstep)][:,vslice,:,2].values

fig,axs = plt.subplots(1,3,tight_layout=True,sharey=True,figsize=(10,4))

p1=axs[0].pcolormesh(x,z_uvp,u.T,cmap='jet')
axs[0].set_title('U')
cbar=plt.colorbar(p1)
p2=axs[1].pcolormesh(x,z_uvp,v.T,cmap='jet')
axs[1].set_title('V')
cbar=plt.colorbar(p2)
p3=axs[2].pcolormesh(x,z_w,w.T,cmap='jet')
axs[2].set_title('W')
cbar=plt.colorbar(p3)

for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=12)
    
axs[0].set_ylabel(r"$z/z_i$",fontsize=12)

fig.suptitle(f'tstep = {tstep} - ' + sim,fontsize=15)

plt.show()

#%%Plot anisotropy

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from Anisotropy_Functions import ColorAnisotropy

yB = aniso[str(tstep)][:,:,:,1].values
xB = aniso[str(tstep)][:,:,:,0].values

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True,figsize=(10,4))

p1=axs[0].pcolormesh(x,z_uvp,xB[:,vslice,:].T,cmap=ColorAnisotropy(),vmin = 0,vmax=1)
axs[0].set_title("xB",fontsize=12)
cbar=plt.colorbar(p1)
p2=axs[1].pcolormesh(x,z_uvp,yB[:,vslice,:].T,cmap=ColorAnisotropy(),vmin = 0,vmax=np.sqrt(3)/2)
axs[1].set_title("yB",fontsize=12)
cbar=plt.colorbar(p2)

axs[0].set_ylabel(r"$z/z_i$",fontsize=12)
for i in range(len(axs)):
    axs[i].set_xlabel(r"$x/z_i$",fontsize=12)

fig.suptitle(f'tstep = {tstep} - ' + sim,fontsize=15)

plt.show()

#%%Plot anisotropy scatter density on the Lumley Trieangle

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

z = z_uvp

### - Plot option 1 ----------------------------------------------------------------------------------------------------------------------------
# axs.pcolormesh(xi, yi, z_i, cmap='hot_r')
# sc = axs.scatter(tmpxB,tmpyB,s=1,alpha=0.5)#,c=tmpZ,cmap='jet')
# axs.pcolormesh(xi, yi, z_i, cmap='hot_r')
# axs.contour(xi,yi,z_i)

### - Plot option 2-------------------------------------------------------------------------------------------------------------------------
# for i in range(len(coord)):
#     sc = axs.scatter(tmpxB[i,21:],tmpyB[i,21:],s=1,c=z_on_h[21:Nz_SLayer],cmap='jet',alpha=0.5)
#     # sc = axs.plot(tmpxB[i,:],tmpyB[i,:])
#     # sc = axs.plot(tmpxB[i,21:],tmpyB[i,21:],c='r')
# axs.plot(xB_median[21:],yB_median[21:],'k',lw=2)

# ### - Plot option 3 ---------------------------------------------------------------------------------------------------------------------------
# h1 = 39
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


axs.hist2d(xB[:,:,1:10].flatten(),yB[:,:,1:10].flatten(),bins=100,cmap='hot_r')

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
# cbar = plt.colorbar(sc,label=r'$z/h_c$')

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

#%%Classic scalings for sigmaU

phiU_c_u = 2.55*(1-3*zeta_vec_u)**(1/3)
phiU_c_s = 2.06*np.ones(len(zeta_vec_u))

#%%Compute ustar,heatflux and L

ustar3D = (((mom3D[str(tstep)][:,:,:,14] - uvpnode2wnode(mom3D[str(tstep)][:,:,:,0])*mom3D[str(tstep)][:,:,:,2] - mom3D[str(tstep)][:,:,:,20])**2 + \
          (mom3D[str(tstep)][:,:,:,15] - uvpnode2wnode(mom3D[str(tstep)][:,:,:,1])*mom3D[str(tstep)][:,:,:,2] - mom3D[str(tstep)][:,:,:,21])**2)**0.25).values

heatflux3D = (sc3D[str(tstep)][:,:,:,4] - uvpnode2wnode(sc3D[str(tstep)][:,:,:,0])*mom3D[str(tstep)][:,:,:,2] - sc3D[str(tstep)][:,:,:,7]).values

L3D = -(wnode2uvpnode(ustar3D)**3)*sc3D[str(tstep)][:,:,:,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D))

zoverL3D = np.ones((Nx,Ny,Nz))*z_uvp/L3D

#%%Plot histogram of z over L

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.hist(abs(zoverL3D).flatten(),bins=10000)
axs.set_xscale('log')
axs.set_yscale('log')

plt.show()

#%% compute anisotropy using RAV data
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from Anisotropy_Functions import Anisotropy

Rxx = mom3D[str(tstep)][:,:,:,4].values - mom3D[str(tstep)][:,:,:,0].values*mom3D[str(tstep)][:,:,:,0].values
Ryy = mom3D[str(tstep)][:,:,:,5].values - mom3D[str(tstep)][:,:,:,1].values*mom3D[str(tstep)][:,:,:,1].values
Rzz = wnode2uvpnode(mom3D[str(tstep)][:,:,:,6].values - mom3D[str(tstep)][:,:,:,2].values*mom3D[str(tstep)][:,:,:,2].values)
Rxy = mom3D[str(tstep)][:,:,:,13].values - mom3D[str(tstep)][:,:,:,0].values*mom3D[str(tstep)][:,:,:,1].values
Rxz = wnode2uvpnode(mom3D[str(tstep)][:,:,:,14].values - uvpnode2wnode(mom3D[str(tstep)][:,:,:,0].values)*mom3D[str(tstep)][:,:,:,2].values)
Ryz = wnode2uvpnode(mom3D[str(tstep)][:,:,:,15].values - uvpnode2wnode(mom3D[str(tstep)][:,:,:,1].values)*mom3D[str(tstep)][:,:,:,2].values)

xb,yb,lambda3 = Anisotropy(Nx,Ny,Nz,Rxx,Ryy,Rzz,Rxy,Rxz,Ryz)

#%%Compute a(yb) and then normalize phiU with it to check data scatter collapse

log_yb = np.log10(yb)

a_yb_u = 0.784*log_yb**0 - 2.582*log_yb
a_yb_s = 2.332*yb**0 - 2.047*yb + 2.672*yb**2

#%%Compute sigmaU

vstart = 0
vend = 50

sigmaU = np.sqrt(mom3D[str(tstep)][:,:,:,4] - mom3D[str(tstep)][:,:,:,0]*mom3D[str(tstep)][:,:,:,0]).values/wnode2uvpnode(ustar3D)


sigmaU_u = sigmaU[:,:,10][(zoverL3D[:,:,10]<0)]/a_yb_u[:,:,10][(zoverL3D[:,:,10]<0)]
sigmaU_s = sigmaU[:,:,10][(zoverL3D[:,:,10]>0)]/a_yb_s[:,:,10][(zoverL3D[:,:,10]>0)]
zeta_u = zoverL3D[:,:,10][(zoverL3D[:,:,10]<0)]
zeta_s = zoverL3D[:,:,10][(zoverL3D[:,:,10]>0)]

#%%Plot a pcolormesh of sigmaU

# fig,axs = plt.subplots(1,1,tight_layout=True)

# p = axs.pcolormesh(x,y,sigmaU[:,:,0].T,cmap='viridis',vmin=0,vmax=10)
# axs.set_xlabel(r"$x/z_i$",fontsize=15)
# axs.set_ylabel(r"$y/z_i$",fontsize=15)
# cbar = plt.colorbar(p)

# plt.show()

#%%Plot histogram of zeta

# fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.hist(abs(zeta_u).flatten(),bins=10000)
# # axs.set_xscale('log')
# # axs.set_yscale('log')
# axs.set_xlim(-10,10)

# plt.show()

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
    
axs[0].semilogx(-zeta_vec_u,phiU_c_u,c='k')
axs[1].semilogx(zeta_vec_s,phiU_c_s,c='k')

axs[0].scatter(abs(zeta_u),sigmaU_u,s=0.1,alpha=0.1)
axs[1].scatter(abs(zeta_s),sigmaU_s,s=0.1,alpha=0.1)

# axs[0].hist2d(abs(zeta_u),sigmaU_u,bins=[np.logspace(np.log10(abs(zeta_u).min()), np.log10(abs(zeta_u).max()), 100),\
#                                           np.linspace((sigmaU_u.min()),(sigmaU_u.max()), 100)],cmap='hot_r')
# axs[1].hist2d(abs(zeta_s),sigmaU_s,bins=[np.logspace(np.log10(abs(zeta_s).min()), np.log10(abs(zeta_s).max()), 100),\
#                                           np.linspace((sigmaU_s.min()),(sigmaU_s.max()), 2000)],cmap='hot_r')


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
    phi_w_fit_u[:,i] = a_u*((1 - 3*zeta_vec_u)**(1/3))
    phi_w_fit_s[:,i] = a_s*((1 + 3*zeta_vec_s)**(c_s))

#%%Compute sigmaU

vlimit = 5

ustar3D = (((mom3D[str(tstep)][:,:,:vlimit,14] - uvpnode2wnode(mom3D[str(tstep)][:,:,:vlimit,0])*mom3D[str(tstep)][:,:,:vlimit,2] - mom3D[str(tstep)][:,:,:vlimit,20])**2 + \
          (mom3D[str(tstep)][:,:,:vlimit,15] - uvpnode2wnode(mom3D[str(tstep)][:,:,:vlimit,1])*mom3D[str(tstep)][:,:,:vlimit,2] - mom3D[str(tstep)][:,:,:vlimit,21])**2)**0.25).values

heatflux3D = (sc3D[str(tstep)][:,:,:vlimit,4] - uvpnode2wnode(sc3D[str(tstep)][:,:,:vlimit,0])*mom3D[str(tstep)][:,:,:vlimit,2] - sc3D[str(tstep)][:,:,:vlimit,7]).values

L3D = -(wnode2uvpnode(ustar3D)**3)*sc3D[str(tstep)][:,:,:vlimit,0].values/(g_hat*kvonk*wnode2uvpnode(heatflux3D))

zoverL3D = np.ones((Nx,Ny,vlimit))*z_uvp[:vlimit]/L3D

sigmaW = wnode2uvpnode(np.sqrt(mom3D[str(tstep)][:,:,:vlimit,6] - mom3D[str(tstep)][:,:,:vlimit,2]*mom3D[str(tstep)][:,:,:vlimit,2]).values/(ustar3D))

sigmaW_u = sigmaW[:,:,:vlimit][(zoverL3D[:,:,:vlimit]<0)]
sigmaW_s = sigmaW[:,:,:vlimit][(zoverL3D[:,:,:vlimit]>0)]
zeta_u = zoverL3D[:,:,:vlimit][(zoverL3D[:,:,:vlimit]<0)]
zeta_s = zoverL3D[:,:,:vlimit][(zoverL3D[:,:,:vlimit]>0)]
    
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

axs[0].hist2d(abs(zeta_u),sigmaW_u,bins=[np.logspace(np.log10(abs(zeta_u).min()), np.log10(abs(zeta_u).max()), 200),\
                                          np.linspace((sigmaW_u.min()),(sigmaW_u.max()), 200)],cmap='hot_r')
axs[1].hist2d(abs(zeta_s),sigmaW_s,bins=[np.logspace(np.log10(abs(zeta_s).min()), np.log10(abs(zeta_s).max()), 200),\
                                          np.linspace((sigmaW_s.min()),(sigmaW_s.max()), 200)],cmap='hot_r')

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
    
plt.show()














































