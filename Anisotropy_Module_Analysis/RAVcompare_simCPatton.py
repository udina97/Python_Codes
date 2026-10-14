#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:53:37 2026

@author: u1450851
"""

#%%Import libraries
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom, read_checkpoint_tke
from Anisotropy_Functions import ColorAnisotropy, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Simulation parameters

nx = 128
ny = 128
nz = 128

lx = 5120
ly = 5120
lz = 2048

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1024
zbl = 1099
uscale = 1
wstar = 2.05
Tscale = 300
qstar = 0.24
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load RAV data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 'simC_Patton2011'

step = 128000
time_ext = '30min'

cases = ['a','c']

mom3D = dict(); sc3D = dict(); tke3D = dict(); mom2D = dict(); sc2D = dict()

# aniso3D['c'] = xr.open_dataarray(path+sim+'/data/Anisotropy/Data_Anisotropy_'+str(time_ext)+'.nc').data
mom3D['c'] = xr.open_dataarray(path+sim+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
sc3D['c'] = xr.open_dataarray(path+sim+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
tke3D['c'] = xr.open_dataarray(path+sim+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
mom2D['c'] = xr.open_dataarray(path+sim+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
sc2D['c'] = xr.open_dataarray(path+sim+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

# aniso3D['a'] = xr.open_dataarray(path+sim+'_aniso'+'/data/Anisotropy/Data_Anisotropy_'+str(time_ext)+'.nc').data
mom3D['a'] = xr.open_dataarray(path+sim+'_aniso'+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
sc3D['a'] = xr.open_dataarray(path+sim+'_aniso'+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
tke3D['a'] = xr.open_dataarray(path+sim+'_aniso'+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
mom2D['a'] = xr.open_dataarray(path+sim+'_aniso'+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
sc2D['a'] = xr.open_dataarray(path+sim+'_aniso'+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

#Variable = ['u','v','w','p','u2','v2','w2','u3','v3','w3','u4','v4','w4',\
#            'uv','uw','vw','txx','tyy','tzz','txy','txz','tyz',\
#            'dudz','dvdz','nut','cs','tke_sgs']

# vardataS = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesSC),order='F'),\
#                         dims=('x','y','z','variable'), coords = {'variable':['avgT','avgT2',\
#                         'avgUT','avgVT','avgWT','avgUTT','avgVTT','avgWTT','avgDTDX','avgDTDY','avgDTDZ',\
#                         'avgDTDX2','avgDTDY2','avgDTDZ2','avgTUT_sgs','avgTVT_sgs','avgTWT_sgs',\
#                         'avgUT_sgs','avgVT_sgs','avgWT_sgs','avg_nus',\
#                         'avg_ds']}) 

# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
#                 dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
#                 'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})

# vardata = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariablesE),order='F'),\
#                     dims=('x', 'y','z','variable'), coords = {'variable':['avgE_w','avgE_uvp',\
#                     'avgRHS','avgProd','avgDiss','avgNut','avgNuh','avgKTX','avgKTY',\
#                     'avgKTZ']})

#%%Boundary layer height

# for i in cases:
#     dTdz = sc3D[i][:,:,:,10]
#     max_dTdz_level = np.argmax(dTdz, axis=2)
#     max_dTdz_height = (max_dTdz_level)*dz 
#     print(np.mean(max_dTdz_height))
    
def get_dphidz_uvp2w(phi, dz):
    import numpy as np
    dphidz = np.zeros(phi.shape)
    # dphidz[:,:,0] = phi[:,:,0]/(0.5*dz)
    dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
    dphidz[:,:,-1] = dphidz[:,:,-2]
    return dphidz

for i in cases:
    dTdz = get_dphidz_uvp2w(sc3D[i][:,:,:,0]*Tscale, dz)
    max_dTdz_level = np.argmax(dTdz, axis=2)
    max_dTdz_height = (max_dTdz_level)*dz 
    print(np.mean(max_dTdz_height))

#%%PDF of ustar and wstar

from scipy.stats import gaussian_kde

fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(10, 6))

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'
    
    # Variables displayed in the top row
    u_star = (mom2D[i][:, :, 0] * uscale).T
    w_star = (sc2D[i][:, :, 0] * uscale).T
    
    # Remove NaN/inf values before computing KDEs
    u_values = u_star[np.isfinite(u_star)]
    w_values = w_star[np.isfinite(w_star)]
    
    # Gaussian kernel-density estimates
    u_kde = gaussian_kde(u_values)
    w_kde = gaussian_kde(w_values)
    
    # Coordinates on which to evaluate each PDF
    u_grid = np.linspace(u_values.min(), u_values.max(), 1000)
    w_grid = np.linspace(w_values.min(), w_values.max(), 1000)

    axs[0].plot(u_grid, u_kde(u_grid), color="tab:blue", lw=2, ls=ls, label=i)
    axs[1].plot(w_grid, w_kde(w_grid), color="tab:orange", lw=2, ls=ls, label=i)
    
    axs[0].legend()
    
    print(f'Mean ustar for ' + sim + f': {np.mean(u_star)}')

axs[0].set_title(r"PDF of $u_*$", fontsize=14)
axs[1].set_title(r"PDF of $w_*$", fontsize=14)

axs[0].set_xlabel(r"$u_* \,[ms^{-1}]$", fontsize=12)
axs[1].set_xlabel(r"$w_* \,[ms^{-1}]$", fontsize=12)
axs[0].set_ylabel("Probability density", fontsize=12)

# fig.suptitle(sim, fontsize=16)
plt.show()

#%%Plot profiles of velocity components

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,6))

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'

    axs.plot(np.mean(mom3D[i][:,:,:,0]*uscale/wstar,axis=(0,1)),z_uvp,c='k',label='u - '+i,ls=ls)
    axs.plot(np.mean(mom3D[i][:,:,:,1]*uscale/wstar,axis=(0,1)),z_uvp,c='r',label='v - '+i,ls=ls)
    axs.plot(np.mean(mom3D[i][:,:,:,2]*uscale/wstar,axis=(0,1)),z_w,c='g',label='w - '+i,ls=ls)

axs.set_ylabel(r'$z$', fontsize=12)
axs.set_xlabel(r'$u_i/w_*$', fontsize=12)

axs.set_ylim(0,z_w[-1])

# fig.suptitle(sim, fontsize=16)
plt.show()

#%%Plot the streamwise velocity gradient



def get_dphidz_uvp2w(phi, dz):
    import numpy as np
    dphidz = np.zeros(phi.shape)
    # dphidz[:,:,0] = phi[:,:,0]/(0.5*dz)
    dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
    dphidz[:,:,-1] = dphidz[:,:,-2]
    return dphidz

def get_dphidz_uvp2w1D(phi, dz):
    import numpy as np
    dphidz = np.zeros(phi.shape)
    # dphidz[0] = phi[0]/(0.5*dz)
    dphidz[:-1] = (phi[1:]-phi[:-1]) / (dz)
    dphidz[-1] = dphidz[-2]
    return dphidz

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,6))

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'
        
    dudz = mom3D[i][:,:,:,22]*(uscale/zi)*(zbl/wstar)

    axs.plot(np.mean(dudz,axis=(0,1)),z_w/zbl,c='k',ls=ls)
    axs.plot(np.mean(get_dphidz_uvp2w(mom3D[i][:,:,:,0]*uscale, dz)*(zbl/wstar),axis=(0,1))[:-1],z_w[1:]/zbl,c='r',ls=ls)
    # axs.plot(get_dphidz_uvp2w1D(np.mean(mom3D[i][:,:,:,0]*uscale,axis=(0,1)), dz)[:-1]*(zbl/wstar),z_w[1:]/zbl,c='g')

axs.set_xlim(-0.5,8)
axs.set_ylim(0,1.5)
axs.set_xlabel(r'$\partial u /\partial z$', fontsize=12)
axs.set_ylabel(r'$z/z_i$',fontsize=12)

# fig.suptitle(sim, fontsize=16)
plt.show()

#%%Turbulence intensity

fig,axs = plt.subplots(1,1,tight_layout=True,sharey=True,figsize=(4,6))

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'

    # uu = mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0]
    # vv = mom3D[:,:,:,5] - mom3D[:,:,:,1]*mom3D[:,:,:,1]
    # ww = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2])
    
    uu_disp = (mom3D[i][:,:,:,0] - np.broadcast_to(np.mean(mom3D[i][:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2
    vv_disp = (mom3D[i][:,:,:,1] - np.broadcast_to(np.mean(mom3D[i][:,:,:,1],axis=(0,1)), (nx,ny,nz)))**2
    ww_disp = wnode2uvpnode((mom3D[i][:,:,:,2] - np.broadcast_to(np.mean(mom3D[i][:,:,:,2],axis=(0,1)), (nx,ny,nz)))**2)
    
    # tke_res = 0.5*(uu + vv + ww)
    tke_sgs = tke3D[i][:,:,:,1]
    # tke_disp = 0.5*(uu_disp + vv_disp + ww_disp)
    # tke_tot = tke_res + tke_sgs #+ tke_disp
    
    # U = np.sqrt(mom3D[:,:,:,0]**2 + mom3D[:,:,:,1]**2 + wnode2uvpnode(mom3D[:,:,:,2]**2))
    # I = np.sqrt((2/3)*tke_tot)/U
    
    uu1D = np.mean(mom3D[i][:,:,:,4],axis=(0,1)) - np.mean(mom3D[i][:,:,:,0]**2,axis=(0,1)) + np.mean(uu_disp,axis=(0,1))
    vv1D = np.mean(mom3D[i][:,:,:,5],axis=(0,1)) - np.mean(mom3D[i][:,:,:,1]**2,axis=(0,1)) + np.mean(vv_disp,axis=(0,1))
    ww1D = np.mean(wnode2uvpnode(mom3D[i][:,:,:,6]),axis=(0,1)) - np.mean(wnode2uvpnode(mom3D[i][:,:,:,2])**2,axis=(0,1)) + np.mean(ww_disp,axis=(0,1))
    tke1D = 0.5*(uu1D + vv1D + ww1D) + np.mean(tke_sgs,axis=(0,1))
    U1D = np.sqrt(np.mean(mom3D[i][:,:,:,0]**2,axis=(0,1)) + np.mean(mom3D[i][:,:,:,1]**2,axis=(0,1)) + np.mean(wnode2uvpnode(mom3D[i][:,:,:,2])**2, axis=(0,1)))
    # U1D = np.sqrt(np.mean(mom3D[i][:,:,:,0],axis=(0,1))**2 + np.mean(mom3D[i][:,:,:,1],axis=(0,1))**2 + np.mean(wnode2uvpnode(mom3D[i][:,:,:,2]), axis=(0,1))**2)
    # U1D = np.mean(np.sqrt(mom3D[i][:,:,:,0]**2 + mom3D[i][:,:,:,1]**2 + wnode2uvpnode(mom3D[i][:,:,:,2]**2)),axis=(0,1))
    I1D = np.sqrt((2/3)*tke1D)/U1D



    # axs[0].pcolormesh(x,z_uvp/zbl,I[:,ny//2,:].T,cmap='viridis',vmin=0,vmax=2)
    # axs[1].plot(np.mean(I,axis=(0,1)),z_uvp/zbl,c='k')
    axs.plot(I1D,z_uvp/zbl,c='k',ls=ls)

axs.set_ylabel(r'$z/z_i$',fontsize=12)
axs.set_xlabel(r'$I$')
axs.set_xlim(0,3)
axs.set_ylim(0)

# fig.suptitle(sim, fontsize=16)
plt.show()

#%%Plot temperature profile and heatflux

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'

    T = sc3D[i][:,:,:,0]
    T_w = uvpnode2wnode(T)
    T_w[:,:,0] = sc2D[i][:,:,-2]
    # wT_sgs = - sc3D[:,:,:,-2]*sc3D[:,:,:,10]
    # wT_sgs[:,:,0] = sc2D[:,:,-1]
    wT = np.mean(sc3D[i][:,:,:,4],axis=(0,1)) - np.mean(mom3D[i][:,:,:,2]*T_w,axis=(0,1)) #- sc3D[:,:,:,19]
    wT_sgs = - np.mean(sc3D[i][:,:,:,19],axis=(0,1))
    wT_disp = np.mean((mom3D[i][:,:,:,2] - np.broadcast_to(np.mean(mom3D[i][:,:,:,2],axis=(0,1)), (nx,ny,nz))) * (T_w - np.broadcast_to(np.mean(T_w,axis=(0,1)), (nx,ny,nz))),axis=(0,1))

    axs[0].plot(np.mean(T,axis=(0,1))*Tscale,z_uvp/zbl,c='k',ls=ls)
    axs[1].plot(wT*Tscale*uscale/qstar,z_w/zbl,c='orange',ls=ls)
    axs[1].plot(wT_sgs*Tscale*uscale/qstar,z_w/zbl,c='r',ls=ls)
    axs[1].plot(wT_disp*Tscale*uscale/qstar,z_w/zbl,c='g',ls=ls)
    axs[1].plot((wT + wT_sgs + wT_disp)*Tscale*uscale/qstar,z_w/zbl,c='k',ls=ls)

    
theta = np.select([z_uvp <= 974.0, (z_uvp > 974.0) & (z_uvp < 1074.0), z_uvp >= 1074.0], [300.0, 300.0 + (z_uvp - 974.0) * 0.08, 308.0 + (z_uvp - 1074.0) * 0.003])
axs[0].plot(theta,z_uvp/zbl,c='k',ls='-.')
axs[1].axvline(0,c='k',ls='--')

axs[0].set_ylim(0,1.5)
axs[0].set_xlabel(r'$\theta [K]$', fontsize=12)
axs[0].set_ylabel(r'$z/z_i$', fontsize=12)
axs[1].set_xlabel(r"$\overline{w'T'}/Q_*$", fontsize=12)

# fig.suptitle(sim)

plt.show()

#%%Plot TKE and temperature variance profile

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

for i in cases:
    if i == 'a':
        ls='-'
    else:
        ls='--'

    uu1D = np.mean(mom3D[i][:,:,:,4],axis=(0,1)) - np.mean(mom3D[i][:,:,:,0]*mom3D[i][:,:,:,0],axis=(0,1))
    vv1D = np.mean(mom3D[i][:,:,:,5],axis=(0,1)) - np.mean(mom3D[i][:,:,:,1]*mom3D[i][:,:,:,1],axis=(0,1))
    ww1D = np.mean(wnode2uvpnode(mom3D[i][:,:,:,6]),axis=(0,1)) - np.mean(wnode2uvpnode(mom3D[i][:,:,:,2])*wnode2uvpnode(mom3D[i][:,:,:,2]),axis=(0,1))
    TT = np.mean(sc3D[i][:,:,:,1],axis=(0,1)) - np.mean(sc3D[i][:,:,:,0]*sc3D[i][:,:,:,0],axis=(0,1))
    TT_disp = np.mean((sc3D[i][:,:,:,0] - np.broadcast_to(np.mean(sc3D[i][:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2,axis=(0,1))
    uu_disp = np.mean((mom3D[i][:,:,:,0] - np.broadcast_to(np.mean(mom3D[i][:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2,axis=(0,1))
    vv_disp = np.mean((mom3D[i][:,:,:,1] - np.broadcast_to(np.mean(mom3D[i][:,:,:,1],axis=(0,1)), (nx,ny,nz)))**2,axis=(0,1))
    ww_disp = np.mean(wnode2uvpnode((mom3D[i][:,:,:,2] - np.broadcast_to(np.mean(mom3D[i][:,:,:,2],axis=(0,1)), (nx,ny,nz)))**2),axis=(0,1))
    N = ((9.81/(uscale**2/zi))/sc3D[i][:,:,:,0])*wnode2uvpnode(sc3D[i][:,:,:,10])
    delta = ((1.5*dx*1.5*dy*dz)**(1/3))/zi
    z3D_uvp = np.broadcast_to(z_uvp, (nx,ny,nz))
    lf = np.where(N<=0, np.minimum(delta,2.2*z3D_uvp), np.minimum(np.minimum(delta,2.2*z3D_uvp),0.76*(np.sqrt(tke3D[i][:,:,:,1])/np.sqrt(N))))
    gamma = -(2*sc3D[i][:,:,:,10]*(-sc3D[i][:,:,:,19])*lf)/(2.02*np.sqrt(tke3D[i][:,:,:,1]))
    
    tke_res = 0.5*(uu1D + vv1D + ww1D)
    tke_sgs = np.mean(tke3D[i][:,:,:,1],axis=(0,1))
    tke_disp = 0.5*(uu_disp + vv_disp + ww_disp)



    axs[0].plot((tke_res)*(uscale**2)/(wstar**2),z_uvp/zbl,c='orange',ls=ls)
    axs[0].plot((tke_sgs)*(uscale**2)/(wstar**2),z_uvp/zbl,c='r',ls=ls)
    axs[0].plot((tke_disp)*(uscale**2)/(wstar**2),z_uvp/zbl,c='g',ls=ls)
    axs[0].plot((tke_disp + tke_sgs + tke_res)*(uscale**2)/(wstar**2),z_uvp/zbl,c='k',ls=ls)
    axs[1].plot((TT)*(Tscale**2)/(qstar/wstar),z_uvp/zbl, c='orange',ls=ls)
    axs[1].plot((TT_disp)*(Tscale**2)/(qstar/wstar),z_uvp/zbl, c='g',ls=ls)
    axs[1].plot(np.mean(gamma,axis=(0,1))*(Tscale**2)/(qstar/wstar),z_uvp/zbl,c='r',ls=ls)
    axs[1].plot(((TT_disp + TT + np.mean(gamma,axis=(0,1))))*(Tscale**2)/(qstar/wstar),z_uvp/zbl, c='k',ls=ls)

axs[0].set_ylim(0,1.5)
axs[0].set_xlabel(r'$\overline{E}/w_*^2$', fontsize=12)
axs[0].set_ylabel(r'$z/z_i$', fontsize=12)
axs[1].set_xlabel(r"$\overline{\theta'^2}/\theta_*$",fontsize=12)

# fig.suptitle(sim)

plt.show()

#%%Plot profiles of vertical and horizontal velocity variance

ww_res = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]*mom3D[:,:,:,2])
ww_sgs = - mom3D[:,:,:,18] + (2/3)*tke3D[:,:,:,1]
ww_disp = wnode2uvpnode((mom3D[:,:,:,2] - np.broadcast_to(np.mean(mom3D[:,:,:,2],axis=(0,1)), (nx,ny,nz)))**2)
uh_res = mom3D[:,:,:,4] - mom3D[:,:,:,0]*mom3D[:,:,:,0] + mom3D[:,:,:,5] - mom3D[:,:,:,1]*mom3D[:,:,:,1] 
uh_sgs = - mom3D[:,:,:,16] - mom3D[:,:,:,17] + (4/3)*tke3D[:,:,:,1]
uh_disp = (mom3D[:,:,:,0] - np.broadcast_to(np.mean(mom3D[:,:,:,0],axis=(0,1)), (nx,ny,nz)))**2 + (mom3D[:,:,:,1] - np.broadcast_to(np.mean(mom3D[:,:,:,1],axis=(0,1)), (nx,ny,nz)))**2

fig,axs = plt.subplots(1,2,tight_layout=True,sharey=True)

axs[0].plot(np.mean(ww_res,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='orange')
axs[0].plot(np.mean(ww_sgs,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='r')
axs[0].plot(np.mean(ww_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='g')
axs[0].plot(np.mean(ww_res + ww_sgs + ww_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='k')
axs[1].plot(np.mean(uh_res,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='orange')
axs[1].plot(np.mean(uh_sgs,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='r')
axs[1].plot(np.mean(uh_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='g')
axs[1].plot(np.mean(uh_res + uh_sgs + uh_disp,axis=(0,1))*uscale/(wstar**2),z_uvp/zbl,c='k')

axs[0].set_ylim(0,1.1)
axs[0].set_ylabel(r'$z/z_i$',fontsize=12)
axs[0].set_xlabel(r"$\overline{w'^2}/w_*^2$", fontsize=12)
axs[1].set_xlabel(r"$\overline{U'^2}/w_*^2$", fontsize=12)

fig.suptitle(sim)

plt.show()

































































