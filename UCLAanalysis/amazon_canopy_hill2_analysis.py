#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 01:58:31 2023

@author: benjamin
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.dpi'] = 300

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/')

from Stats import ReynoldsStress
from Anisotropy_Functions_BEN import Anisotropy, Anisotropy_Clustering, ColorAnisotropy
from UCLA_npy_to_netCDF import UCLA_npy_to_netCDF

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
Nz = 290
Nz_SLayer = int(Nz/2)

zi = 580 #length scale for non-dimensionalization

lx = 2000/zi
ly = 1000/zi
lz = 580/zi

zi = 580 #length scale for non-dimensionalization
height = 20 # Canopy height in grid points.
canopyH = 39/zi #Height of the canopy in meters.
kappa = 0.4

dx = lx/(Nx)
dy = ly/(Ny)
dz = lz/(Nz)

x = (np.arange(0,Nx)*dx)
y = (np.arange(0,Ny)*dy)
z = (np.arange(0,Nz)*dz)
z_slayer = (np.arange(0,Nz_SLayer)*dz)
# z = (np.arange(0,Nz)*dz) + dz/2 #Vertical height non-dimensional with zi. First grid point of dudz is not at z = 0, but dz/2.
z_on_h = z/canopyH

#%% Load Data

path = '/scratch/general/nfs1/u1450851/UCLAdata/data/amazon_canopy_hill2h_3D/'

data = UCLA_npy_to_netCDF(path,Nx,Ny,Nz,'false')  #This uploads the data from the NetCDF files, that keep the xarray form.
data = data.transpose('x','y','z','variable')
# diss  = np.load(path + 'dissip.npy')

#%% Importing the topography
from scipy.io import loadmat

topo = loadmat('/scratch/general/nfs1/u1450851/UCLAdata/data/amazon_canopy_hill2h_3D/matlab/topo.mat')

H = 100
L = 250
X = np.arange(0,Nx)*(2000/320)

z_tpg = ((H/2)*np.cos((np.pi/(2*L))*X + np.pi) + (H/2))/zi


# figure = plt.figure()
# plt.plot(x,z_tpg)

#%% Computing the Reynolds Stress


# Computing the Reynolds Stress Tensor and Sensible Heat fluxes:

Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,6],data.data[:,:,:,7],data.data[:,:,:,8],data.data[:,:,:,9],\
                         data.data[:,:,:,10],data.data[:,:,:,11],data.data[:,:,:,12],data.data[:,:,:,13],data.data[:,:,:,14])


#Vertical profiles of the shear stress:
    
fig, axs = plt.subplots(nrows=2,ncols=3)
plt.ion()

l = 0
Rij_names = list(Rstress.coords['variable'].data[:])

for m in range(0,2):
    for n in range(0,3):
        
        
        print(l)

        # y = z_d/canopyH
        y = z_on_h
    
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

#%%
mask = np.full((Nx,Nz),np.nan,'d',order='F')
for i in range(0,Nx):
    for k in range(0,Nz):
        if k*dz<=z_tpg[i]:
            mask[i,k] = 1


figure = plt.figure()
plt.pcolormesh(x,z,np.transpose(Rstress.data[:,60,:,0]),cmap='coolwarm')
plt.pcolormesh(x,z,np.transpose(mask),cmap='gray')
plt.plot(x,z_tpg,color='black')
plt.plot(x,z_tpg+canopyH,color='black',ls='--')



#%% Computing turbulence anisotropy and clustering:

anisotropy_compute = 'false'    
path = '/scratch/general/nfs1/u1450851/UCLAdata/data/amazon_canopy_hill2h_3D/'

# Decide whether to compute turbulence anisotropy and clustering from scratch or read it from file.

if (anisotropy_compute == 'true'):
    
    [xB,yB,AnisType_1D] = Anisotropy_Clustering(Nx,Ny,Nz_SLayer,Rstress)

    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)


    Anisotropy_clustering = xr.DataArray(np.zeros(shape = (Nx*Ny*Nz_SLayer,3),order='F'),\
                        dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D']})
        
    Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D 

    os.chdir(path)
    Anisotropy_clustering.to_netcdf('Anisotropy_clustering.nc')
    
else:
        
    Anisotropy_clustering = xr.open_dataarray(path + 'Anisotropy_clustering.nc')

    xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
    yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
    AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 
    
    xB = np.reshape(xB_1D,(Nx,Ny,Nz_SLayer))
    yB = np.reshape(yB_1D,(Nx,Ny,Nz_SLayer))

#------------ End of the IF statement.

cmap = ColorAnisotropy()    

z3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,Nx):
        for j in range(0,Ny):
            z3D[i,j,k] = z_on_h[k]
    
    
z3D_1D = np.ndarray.flatten(z3D)
# phi_M_1D = np.ndarray.flatten(phi_m_3D)
   
#%% Plot vertical and horizontal slices of xB and yB

yB_yavg=np.mean(yB[:,:,:],axis=1)

figure = plt.figure()
plt1=plt.pcolormesh(np.transpose(yB_yavg),cmap = cmap, shading='flat')
plt.pcolormesh(np.transpose(mask[:,0:Nz_SLayer]),cmap='gray')
plt.plot(np.arange(0,Nx),z_tpg*zi/(dz*zi)+height,color='black',ls='--')
plt.colorbar(plt1)
plt.xticks(np.arange(0,Nx),np.around(x,2))
plt.yticks(np.arange(0,Nz_SLayer),np.around(z_slayer,2))
plt.locator_params(axis='both',nbins=7)
plt.xlabel('x/zi', fontsize=15)
plt.ylabel('z/zi', fontsize=15)
plt.title('yB -- y_averaged', fontsize=15)

#%% Vertical profiles of sigma_u, sigma_v, sigma_w, Shera stress, dudz

pnt1 = [80, 100]
pnt2 = [120, 100]
pnt3 = [160, 100]

pnts = [pnt1, pnt2, pnt3]

sigma_u_pnt1 = np.zeros((Nz)); sigma_u_pnt2 = np.zeros((Nz)); sigma_u_pnt3 = np.zeros((Nz))
sigma_v_pnt1 = np.zeros((Nz)); sigma_v_pnt2 = np.zeros((Nz)); sigma_v_pnt3 = np.zeros((Nz))
sigma_w_pnt1 = np.zeros((Nz)); sigma_w_pnt2 = np.zeros((Nz)); sigma_w_pnt3 = np.zeros((Nz))

sigma_u_pnt1 = np.mean(Rstress.data[pnt1[0],:,:,0],axis=0) + np.mean(data.data[pnt1[0],:,:,0],axis=0)
sigma_u_pnt2 = np.mean(Rstress.data[pnt2[0],:,:,0],axis=0) + np.mean(data.data[pnt2[0],:,:,0],axis=0)
sigma_u_pnt3 = np.mean(Rstress.data[pnt3[0],:,:,0],axis=0) + np.mean(data.data[pnt3[0],:,:,0],axis=0)

sigma_v_pnt1 = np.mean(Rstress.data[pnt1[0],:,:,1],axis=0) + np.mean(data.data[pnt1[0],:,:,1],axis=0)
sigma_v_pnt2 = np.mean(Rstress.data[pnt2[0],:,:,1],axis=0) + np.mean(data.data[pnt2[0],:,:,1],axis=0)
sigma_v_pnt3 = np.mean(Rstress.data[pnt3[0],:,:,1],axis=0) + np.mean(data.data[pnt3[0],:,:,1],axis=0)

sigma_w_pnt1 = np.mean(Rstress.data[pnt1[0],:,:,2],axis=0) + np.mean(data.data[pnt1[0],:,:,2],axis=0)
sigma_w_pnt2 = np.mean(Rstress.data[pnt2[0],:,:,2],axis=0) + np.mean(data.data[pnt2[0],:,:,2],axis=0)
sigma_w_pnt3 = np.mean(Rstress.data[pnt3[0],:,:,2],axis=0) + np.mean(data.data[pnt3[0],:,:,2],axis=0)


figure, ax = plt.subplots(1,3,sharey=True)
ax[0].plot(sigma_u_pnt1,z,color='red',label='Point 1')
ax[0].plot(sigma_u_pnt2,z,color='green',label='Point 2')
ax[0].plot(sigma_u_pnt3,z,color='blue',label='Point 3')
ax[1].plot(sigma_v_pnt1,z,color='red',label='Point 1')
ax[1].plot(sigma_v_pnt2,z,color='green',label='Point 2')
ax[1].plot(sigma_v_pnt3,z,color='blue',label='Point 3')
ax[2].plot(sigma_w_pnt1,z,color='red',label='Point 1')
ax[2].plot(sigma_w_pnt2,z,color='green',label='Point 2')
ax[2].plot(sigma_w_pnt3,z,color='blue',label='Point 3')

ax[0].set_ylabel('z/zi',fontsize=15)

ax[0].set_xlabel(r'$\sigma_{u}$', fontsize=15)
ax[1].set_xlabel(r'$\sigma_{v}$', fontsize=15)
ax[2].set_xlabel(r'$\sigma_{w}$', fontsize=15)

ax[0].legend(fontsize=8, loc='upper right')
ax[1].legend(fontsize=8, loc='upper right')
ax[2].legend(fontsize=8, loc='upper right')

#%%Coooooolor the sigma profiles based on turbulence anisotropy

figure = plt.figure()
plt.scatter(sigma_u_pnt1[0:Nz_SLayer],z[0:Nz_SLayer],c=yB_yavg[pnt1[0],:],cmap=cmap)
plt.scatter(sigma_u_pnt2[0:Nz_SLayer],z[0:Nz_SLayer],c=yB_yavg[pnt2[0],:],cmap=cmap)
plt.scatter(sigma_u_pnt3[0:Nz_SLayer],z[0:Nz_SLayer],c=yB_yavg[pnt3[0],:],cmap=cmap)
# plt.axhline(canopyH,color='black',ls='--')
plt.colorbar()
plt.ylabel('z/zi')
plt.xlabel(r'$\sigma_{u}$')



#%% Plot of Tau_wall_xz averaged in x,y direction

#Reynolds stress, Rxz at w nodes!!

U_w = np.zeros((nz,ny,nx),'d',order='F') #interpolate U at the w nodes
for k in range(1,nz):
    U_w[k,:,:]=0.5*(u[k-1,:,:] + u[k,:,:])
U_w[0,:,:] = 0 #velocity is 0 at the surface

R_xz = (uw - (U_w*w))

#Sub-grid scale stress, Txz already at w nodes!!
tau_sgs_xz = txz
   
#Dispersive stress, Dxz at w nodes!!
U_w_disp = np.zeros((nz,ny,nx),'d',order='F') #compute dispersive velocity for U interpolated at w nodes!
W_disp = np.zeros((nz,ny,nx),'d',order='F')
for k in range(0,nz):
    # breakpoint()
    U_w_disp[k,:,:] = np.squeeze(U_w[k,:,:])-np.mean(U_w[k,:,:],axis=(0,1))
    W_disp[k,:,:] = np.squeeze(w[k,:,:])-np.mean(w[k,:,:],axis=(0,1))
    
D_xz = np.mean(U_w_disp*W_disp,axis=(1,2))

tau_wall = np.mean(R_xz, axis=(1,2)) + np.mean(tau_sgs_xz, axis=(1,2)) + D_xz #Twall defined at w nodes!!

fig,ax = plt.subplots()
ax.plot(tau_wall,z,'k',label=r'$\tau_{wall}$')
ax.plot(np.mean(R_xz, axis=(1,2)),z,'--g',label=r'$R_{XZ}$')
ax.plot(np.mean(tau_sgs_xz, axis=(1,2)),z,'--r',label=r'$\tau_{SGS,XZ}$')
ax.plot(D_xz,z,'--b',label=r'$D_{XZ}$')
ax.axhline(hc,ls='--', color='black',lw=0.5)
ax.axhline(0,ls='-', color='black',lw=0.5)
ax.axvline(0,ls='-', color='black',lw=0.5)
ax.set_xlabel(r'$Stress$')
ax.set_ylabel(r'z [m]')
ax.set_title(r'Vertical profile of $<\tau_{wall}>_{xy}$')
ax.legend()

#%%

#Reynolds stress, Rxz at w nodes!!

V_w = np.zeros((nz,ny,nx),'d',order='F') #interpolate U at the w nodes
for k in range(1,nz):
    V_w[k,:,:]=0.5*(v[k-1,:,:] + v[k,:,:])
V_w[0,:,:] = 0 #velocity is 0 at the surface

R_yz = (vw - (V_w*w))

#Sub-grid scale stress, Txz already at w nodes!!
tau_sgs_yz = tyz
   
#Dispersive stress, Dxz at w nodes!!
V_w_disp = np.zeros((nz,ny,nx),'d',order='F') #compute dispersive velocity for U interpolated at w nodes!
W_disp = np.zeros((nz,ny,nx),'d',order='F')
for k in range(0,nz):
    # breakpoint()
    V_w_disp[k,:,:] = np.squeeze(V_w[k,:,:])-np.mean(V_w[k,:,:],axis=(0,1))
    W_disp[k,:,:] = np.squeeze(w[k,:,:])-np.mean(w[k,:,:],axis=(0,1))
    
D_yz = np.mean(V_w_disp*W_disp,axis=(1,2))

tau_wall = np.mean(R_yz, axis=(1,2)) + np.mean(tau_sgs_yz, axis=(1,2)) + D_yz #Twall defined at w nodes!!

fig,ax = plt.subplots()
ax.plot(tau_wall,z,'k',label=r'$\tau_{wall}$')
ax.plot(np.mean(R_yz, axis=(1,2)),z,'--g',label=r'$R_{YZ}$')
ax.plot(np.mean(tau_sgs_yz, axis=(1,2)),z,'--r',label=r'$\tau_{SGS,YZ}$')
ax.plot(D_yz,z,'--b',label=r'$D_{YZ}$')
ax.axhline(hc,ls='--', color='black',lw=0.5)
ax.axhline(0,ls='-', color='black',lw=0.5)
ax.axvline(0,ls='-', color='black',lw=0.5)
ax.set_xlabel(r'$Stress$')
ax.set_ylabel(r'z [m]')
ax.set_title(r'Vertical profile of $<\tau_{wall}>_{xy}$')
ax.legend()

#%%

fig,ax = plt.subplots()
im1 = ax.pcolormesh(x,z,w[:,10,:],shading='gouraud',cmap='coolwarm')
cbar1 = fig.colorbar(im1, ax=ax, shrink=0.8,location='top',orientation='horizontal',pad=0.1)
ax.set_xlabel(r'x [m]')
ax.set_ylabel(r'z [m]')
ax.set_title(r'2D plot of u at y = %i' %10)




























    