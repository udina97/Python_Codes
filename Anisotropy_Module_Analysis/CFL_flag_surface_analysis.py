#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:42:21 2025

@author: u1450851
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

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

def load_scalar_2D(Nx,Ny,NumVariables_2D_SC,StartTime,path,vardata):

    import numpy as np
    import xarray as xr
    import struct


    print('Defining arrays')


    # vardata = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D_SC),order='F'),\
    #                 dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
    #                 'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})
    
    tmpdata = np.zeros((NumVariables_2D_SC*Nx*Ny),'d') #This is a temporal variable used as a bridge.
    sumdata = np.zeros((NumVariables_2D_SC*Nx*Ny),'d') #This is a temporal variable used as a bridge.
    tmp = np.zeros((Nx*Ny, 1),'d')  #This is a temporal variable used as a bridge.



    if (StartTime < 10000):
        FileName = 'RAV_00000'+ str(StartTime) + '_SC_2D_CFL.dat'
    elif (StartTime < 100000) and (StartTime >= 10000):
        FileName = 'RAV_0000'+ str(StartTime) + '_SC_2D_CFL.dat'
    elif (StartTime < 1000000) and (StartTime >= 100000):
        FileName = 'RAV_000'+ str(StartTime) + '_SC_2D_CFL.dat'
    else:
        FileName = 'RAV_00'+ str(StartTime) + '_SC_2D_CFL.dat'
        
    f = open(path+FileName,"rb")

    #Next we read the data and we dump it all, as strings into a new variable.
    print('Done reading the binari data for file = ', FileName)


    for i in range(0,NumVariables_2D_SC):

        data_raw = f.read(Nx*Ny*8)

        n0 = (i)*(Nx*Ny)
        nf = (i+1)*(Nx*Ny)

        #Next we read the strings from above into double-type numbers, that are stored in data.
        tmpdata[n0:nf] = np.array(struct.unpack('d'*(Nx*Ny), data_raw))

    #Now we can delete the main data matrix uploaded initially to free some memory
    del data_raw
    f.close()

    sumdata = sumdata + tmpdata

    
    print('Done adding the data')
    
    for i in range(0,NumVariables_2D_SC):
        print('Variable =', i)
        n0 = (i)*(Nx*Ny)
        nf = (i+1)*(Nx*Ny)
        tmp = sumdata[n0:nf]

        vardata[:,:,i] = np.reshape(tmp,[Nx,Ny],'F')


    del i,tmp


    return(vardata)

#%%Set the path to the simulation and data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'

# sim = 'Tpatch800_a288m290s2_1ms_a'
# sim = 'Tpatch800_a288m290s2_1ms_na'
# sim = 'Tpatch800_a285m290s5_1ms_a_v2'
sim = 'Filt_Tpatch_m290s5a285_1ms_na'
path_to_data = path + sim + '/data/'

itr = 200735

sc2D = xr.DataArray(np.ones(shape = (Nx,Ny,8),order='F'),\
                dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
                'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})
    
vardata = xr.DataArray(np.ones(shape = (Nx,Ny,8),order='F'),\
                dims=('x','y','variable'), coords = {'variable':['Mav_wstar','Mav_L',\
                'Mav_phi_m','Mav_psi_m','Mav_phi_h','Mav_psi_h','Mav_sfcval','Mav_sfcflux']})

sc2D = load_scalar_2D(Nx,Ny,8,itr,path+sim+'/output_RAV/',vardata)

#%%

Tsfc = sc2D[:,:,-2].values*Tscale
phiM = sc2D[:,:,2].values
phiH = sc2D[:,:,4].values
psiM = sc2D[:,:,3].values
psiH = sc2D[:,:,5].values
sfcFLUX = sc2D[:,:,-1].values
L = sc2D[:,:,1].values
zeta = (dz/2)/L

#%%

from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,4,tight_layout=True,figsize=(10,5))

kde = gaussian_kde((phiM[:,:,]).flatten())
x_pdf = np.linspace(min((phiM[:,:]).flatten()), max((phiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(phiM[:,:]).T,cmap='bwr')
axs[1,0].hist((phiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((phiH[:,:,]).flatten())
x_pdf = np.linspace(min((phiH[:,:]).flatten()), max((phiH[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(phiH[:,:]).T,cmap='bwr')
axs[1,1].hist((phiH[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiM[:,:,]).flatten())
x_pdf = np.linspace(min((psiM[:,:]).flatten()), max((psiM[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,(psiM[:,:]).T,cmap='bwr')
axs[1,2].hist((psiM[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((psiH[:,:,]).flatten())
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

#%%

from scipy.stats import gaussian_kde

zlevel = 0

fig,axs = plt.subplots(2,3,tight_layout=True,figsize=(10,5))

kde = gaussian_kde((L[:,:]).flatten())
x_pdf = np.linspace(min((L[:,:]).flatten()), max((L[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,0].pcolormesh(x,y,(L[:,:]).T,cmap='bwr')
axs[1,0].hist((L[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,0].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,0].set_xlabel(r'$x/z_i$',fontsize=14)

# kde = gaussian_kde((Tsfc[:,:]).flatten())
# x_pdf = np.linspace(min((Tsfc[:,:]).flatten()), max((Tsfc[:,:]).flatten()), 1000)
# pdf = kde(x_pdf)
p = axs[0,1].pcolormesh(x,y,(Tsfc[:,:]).T,cmap='bwr')
axs[1,1].hist((Tsfc[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
# axs[1,1].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,1].set_xlabel(r'$x/z_i$',fontsize=14)

kde = gaussian_kde((sfcFLUX[:,:]).flatten())
x_pdf = np.linspace(min((sfcFLUX[:,:]).flatten()), max((sfcFLUX[:,:]).flatten()), 1000)
pdf = kde(x_pdf)
p = axs[0,2].pcolormesh(x,y,(sfcFLUX[:,:]).T,cmap='bwr')
axs[1,2].hist((sfcFLUX[:,:]).flatten(), bins=50, density=True, alpha=0.4, label="Histogram")
axs[1,2].plot(x_pdf, pdf, 'r-', label="KDE PDF")
cbar = plt.colorbar(p)
axs[0,2].set_xlabel(r'$x/z_i$',fontsize=14)

axs[1,0].set_xlabel(r'$L$', fontsize=14)
axs[1,1].set_xlabel(r'$\theta_s$', fontsize=14)
axs[1,2].set_xlabel(r"$\overline{w'\theta'}$", fontsize=14)

axs[0,0].set_ylabel(r'$y/z_i$',fontsize=14)

for i in range(1):
    axs[0,i+1].set_yticklabels([])

fig.suptitle(sim,fontsize=14)

plt.show()














































