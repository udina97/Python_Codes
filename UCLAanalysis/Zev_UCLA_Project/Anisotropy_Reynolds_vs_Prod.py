#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:23:13 2023

@author: u1450851
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
#plt.rcParams['figure.dpi'] = 300

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/')

from Stats import ReynoldsStress
from Anisotropy_Functions_BEN import Anisotropy, Anisotropy_Clustering, Anisotropy_Clustering_Prod, ColorAnisotropy, phi_m
from functions import UCLA_npy_to_netCDF

#%%

# ProjectOn_uvp(): function that projects the w-node data on uvp-node data
##############################################################################

def ProjectOn_uvp(Nz,Ny,Nx,var3D):
    "Calling the ReynoldsStress() function"

    import numpy as np

    var3D_uvp = np.zeros((Nz,Ny,Nx),'d',order='F')

    #Projecting the 3D variable onto the uvp nodes.
    for k in range(0,Nz-1):
        var3D_uvp[k,:,:]=0.5*(var3D[k,:,:] + var3D[k+1,:,:])

    var3D_uvp[-1,:,:] = var3D[-1,:,:]

    return(var3D_uvp)

# ProjectOn_w(): function that projects the uvp-node data on w-node data
##############################################################################

def ProjectOn_w(Nz,Ny,Nx,var3D):
    "Calling the ReynoldsStress() function"

    import numpy as np

    var3D_w = np.zeros((Nz,Ny,Nx),'d',order='F')
    var3D_w[0,:,:] = 0

    #Projecting the 3D variable onto the w nodes.
    for k in range(1,Nz):
        var3D_w[k,:,:]=0.5*(var3D[k-1,:,:] + var3D[k,:,:])

    return(var3D_w)

#%% Domain Parameters

Nx = 320
Ny = 160
Nz = 260
Nz_SLayer = int(Nz)

zi = 520 #length scale for non-dimensionalization

lx = 2000/zi
ly = 1000/zi
lz = 520/zi

zi = 520 #length scale for non-dimensionalization
height = 20 # Canopy height in grid points.
canopyH = 39/zi #Height of the canopy non-dim
kappa = 0.4

dx = lx/(Nx)
dy = ly/(Ny)
dz = lz/(Nz)

x = (np.arange(0,Nx)*dx)
x_m = (np.arange(0,Nx)*dx)*zi #in meters
y = (np.arange(0,Ny)*dy)
z = (np.arange(0,Nz)*dz)
z_m = (np.arange(0,Nz)*dz)*zi #in meters
# z = (np.arange(0,Nz)*dz) + dz/2 #Vertical height non-dimensional with zi. First grid point of dudz is not at z = 0, but dz/2.
z_on_h = z/canopyH
z_d = (z - (30.2/zi)) #(z-d)

#%% Load Data
# To load the data I use a function I created called UCLA_npy_to_netCDF, I shared that with you too, Obviously you can upload the 
# data in any other way you prefer

path_in = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/data/amazon_canopy_flat_3D/'
path_out = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/UCLAdata/anisotropy/flat/'

data = UCLA_npy_to_netCDF(path_in,path_out,Nx,Ny,Nz,'false')  #This uploads the data from the NetCDF files, that keep the xarray form.
data = data.transpose('x','y','z','variable') # I transpose the data from (Nz,Ny,Nx) to (Nx,Ny,Nz)
# content of data: 0. txx, 1. tyy, 2. tzz, 3. txy, 4. txz, 5. tyz, 6. u, 7. v, 8. w, 9. uu, 10. vv, 11. ww, 12. uv, 13. uw, 14. vw

#%%Compute the derivatives of velocities

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
  dphidz[:, :, :-1] = (phi[:, :, 1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[ :, :, 1:] = 0.5*(phi_c[ :, :, :-1] + phi_c[ :, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[ :, :, :-1] = 0.5*(phi_h[ :, :, :-1]+phi_h[ :, :, 1:])
  phi_c[ :, :, -1] = phi_c[ :, :, -2]
  return phi_c

wn_x = 2*np.pi*np.fft.rfftfreq(Nx, dx)
wn_y = 2*np.pi*np.fft.rfftfreq(Ny, dy)

dUdx = get_dphidx(data.data[:,:,:,6], wn_x)
dUdy = get_dphidy(data.data[:,:,:,6], wn_y)
dUdz = get_dphidz(data.data[:,:,:,6], dz)
dUdz = wnode2uvpnode(dUdz)
dVdx = get_dphidx(data.data[:,:,:,7], wn_x)
dVdy = get_dphidy(data.data[:,:,:,7], wn_y)
dVdz = get_dphidz(data.data[:,:,:,7], dz)
dVdz = wnode2uvpnode(dVdz)
dWdx = get_dphidx(data.data[:,:,:,8], wn_x)
dWdx = wnode2uvpnode(dWdx)
dWdy = get_dphidy(data.data[:,:,:,8], wn_y)
dWdy = wnode2uvpnode(dWdy)
dWdz = get_dphidz(data.data[:,:,:,8], dz)

#%% Computing the Reynolds Stress


# Computing the Reynolds Stress Tensor :

Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})
    
Prod = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,9),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Pxx','Pxy',\
                        'Pxz','Pyx','Pyy','Pyz','Pzx','Pzy','Pzz']})

# If you upload the data differently from what I do you might have to change the line below
Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,6],data.data[:,:,:,7],data.data[:,:,:,8],data.data[:,:,:,9],\
                         data.data[:,:,:,10],data.data[:,:,:,11],data.data[:,:,:,12],data.data[:,:,:,13],data.data[:,:,:,14])

Prod[:,:,:,0] = Rstress.data[:,:,:,0]*dUdx
Prod[:,:,:,1] = Rstress.data[:,:,:,3]*dUdy
Prod[:,:,:,2] = Rstress.data[:,:,:,4]*dUdz
Prod[:,:,:,3] = Rstress.data[:,:,:,3]*dVdx
Prod[:,:,:,4] = Rstress.data[:,:,:,1]*dVdy
Prod[:,:,:,5] = Rstress.data[:,:,:,5]*dVdz
Prod[:,:,:,6] = Rstress.data[:,:,:,4]*dWdx
Prod[:,:,:,7] = Rstress.data[:,:,:,5]*dWdy
Prod[:,:,:,8] = Rstress.data[:,:,:,2]*dWdz

#Vertical profiles of the shear stress:
    
fig, axs = plt.subplots(nrows=2,ncols=3)
plt.ion()

l = 0
Rij_names = list(Rstress.coords['variable'].data[:])

for m in range(0,2):
    for n in range(0,3):
        
        
        print(l)

        y = z_d/canopyH
        # y = z_on_h
    
        if (l < 4):
            Rij = np.mean(Rstress.data[:,:,:,l],axis=(0,1))
        else:
            Rij = - np.mean(Rstress.data[:,:,:,l],axis=(0,1))
        
        SGSij = np.mean(data.data[:,:,:,0+l],axis=(0,1))
        Tauij = Rij + SGSij
        
        #PLotting arguments:
            
        axs[m,n].plot(Rij,y,color='gray',linestyle='--')
        axs[m,n].plot(SGSij,y,color='gray',linestyle='-.')
        axs[m,n].plot(Tauij,y,color='black',linestyle='-')

        axs[m,n].set_ylim(y[0], y[-1])
        #axs[m,n].axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
        #axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
        
     
        axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
        axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
        axs[m,n].minorticks_on()

        axs[m,n].set_xscale('linear')
        #axs[m,n].set_title(f'Case = {case}')
        axs[m,n].set_xlabel(f'{Rij_names[l]}')
        
        l = l+1
    
    
axs[0,0].set_ylabel(r'$(z-d)/h$')
axs[1,0].set_ylabel(r'$(z-d)/h$')

plt.tight_layout()
plt.show()

# Vertical profile of the vertical shear stress together:
    
Dxz = np.mean((data.data[:,:,:,6]*data.data[:,:,:,8]),axis=(0,1)) - (np.mean(data.data[:,:,:,6],axis=(0,1))*np.mean(data.data[:,:,:,8],axis=(0,1)))
Dyz = np.mean((data.data[:,:,:,7]*data.data[:,:,:,8]),axis=(0,1)) - (np.mean(data.data[:,:,:,7],axis=(0,1))*np.mean(data.data[:,:,:,8],axis=(0,1)))

Dw =  np.sqrt(Dxz**2 + Dyz**2)
Rw = np.mean((np.sqrt(Rstress.data[:,:,:,4]**2 + Rstress.data[:,:,:,5]**2)),axis=(0,1))
SGSw = np.mean((np.sqrt(data.data[:,:,:,4]**2 + data.data[:,:,:,5]**2)),axis=(0,1))
    
tau_wall1D = Rw + SGSw + Dw

    
fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

axs.plot(tau_wall1D,y,color='black',linestyle='-')
# axs.hlines((canopyH-(dispH[case]/zi))/canopyH,0,4.5,colors='gray',linestyles='-')
axs.set_ylim(y[0], y[-1])
axs.set_xlim(0, 4.5)

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xlabel(r'$\tau_w$')
axs.set_ylabel(r'$(z-d)/h$')

#%% Computing turbulence anisotropy and clustering:

anisotropy_compute = 'false'    
# path = '/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_flat_3D/'

# Decide whether to compute turbulence anisotropy and clustering from scratch or read it from file.

if (anisotropy_compute == 'true'):
    
    [xB,yB,AnisType_1D] = Anisotropy_Clustering_Prod(Nx,Ny,Nz_SLayer,Rstress,-Prod)

    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)


    Anisotropy_clustering = xr.DataArray(np.zeros(shape = (Nx*Ny*Nz_SLayer,3),order='F'),\
                        dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D']})
        
    Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D 

    os.chdir(path_out)
    Anisotropy_clustering.to_netcdf('Anisotropy_clustering_Prod3.nc')
    
else:
        
    Anisotropy_clustering = xr.open_dataarray(path_out + 'Anisotropy_clustering.nc')

    xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
    yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
    AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 
    
    xB = np.reshape(xB_1D,(Nx,Ny,Nz_SLayer))
    yB = np.reshape(yB_1D,(Nx,Ny,Nz_SLayer))
    
    Anisotropy_clustering_Prod = xr.open_dataarray(path_out + 'Anisotropy_clustering_Prod3.nc')

    xB_1D_P = np.copy(Anisotropy_clustering_Prod.data[:,0]) 
    yB_1D_P = np.copy(Anisotropy_clustering_Prod.data[:,1]) 
    AnisType_1D_P = np.copy(Anisotropy_clustering_Prod.data[:,2]) 
    
    xB_P = np.reshape(xB_1D_P,(Nx,Ny,Nz_SLayer))
    yB_P = np.reshape(yB_1D_P,(Nx,Ny,Nz_SLayer))

#------------ End of the IF statement.

cmap = ColorAnisotropy()    

z3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,Nx):
        for j in range(0,Ny):
            z3D[i,j,k] = z_d[k]
    
    
z3D_1D = np.ndarray.flatten(z3D)


#%%

fig,axs = plt.subplots(1,1,constrained_layout=True,figsize=(6,3))
plt1=axs.pcolormesh(x,z[0:Nz_SLayer],np.transpose(yB_P[:,int(Ny/2),:]),cmap=cmap,vmin = 0, vmax = np.sqrt(3)/2)
plt.colorbar(plt1)

axs.axhline(canopyH,color='black',ls='--')
axs.set_ylim([0,1])
axs.set_xlim([0,x[-1]])
# axs.locator_params(axis='both',nbins=7)
axs.set_xlabel(r'x/$z_i$')
axs.set_ylabel(r'z/$z_i$')
axs.set_title(r'yB(x,$y_{Ny/2}$,z)')



















































