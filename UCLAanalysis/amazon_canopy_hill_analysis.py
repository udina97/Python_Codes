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
from Anisotropy_Functions_BEN import Anisotropy, Anisotropy_Clustering, ColorAnisotropy, phi_m
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
Nz = 270
Nz_SLayer = int(Nz/2)

zi = 540 #length scale for non-dimensionalization

lx = 2000/zi
ly = 1000/zi
lz = 540/zi

zi = 540 #length scale for non-dimensionalization
height = 20 # Canopy height in grid points.
canopyH = 39/zi #Height of the canopy in meters and then non-dimensionalized.
kappa = 0.4 #vonKarman

dx = lx/(Nx)
dy = ly/(Ny)
dz = lz/(Nz)

x = (np.arange(0,Nx)*dx)
x_m = (np.arange(0,Nx)*dx)*zi #in meters
y = (np.arange(0,Ny)*dy)
z = (np.arange(0,Nz)*dz)
z_m = (np.arange(0,Nz)*dz)*zi #in meters
z_slayer = (np.arange(0,Nz_SLayer)*dz)
# z = (np.arange(0,Nz)*dz) + dz/2 #Vertical height non-dimensional with zi. First grid point of dudz is not at z = 0, but dz/2.
z_on_h = z/canopyH

#%% Load Data

path = '/scratch/general/nfs1/u1450851/UCLAdata/data/amazon_canopy_hill_3D/'

data = UCLA_npy_to_netCDF(path,Nx,Ny,Nz,'false')  #This uploads the data from the NetCDF files, that keep the xarray form.
data = data.transpose('x','y','z','variable') #reshape the data from (Nz,Ny,Nx) to (Nx,Ny,Nz)
# diss  = np.load(path + 'dissip.npy')

#%% Importing the topography
from scipy.io import loadmat

topo = loadmat('/scratch/general/nfs1/u1450851/UCLAdata/data/amazon_canopy_hill_3D/matlab/topo.mat')
z_tpg = np.mean(np.transpose(topo['z_tpg']),axis=1)/zi


figure = plt.figure()
plt.plot(x,z_tpg)

#%% Computing the Reynolds Stress


# Computing the Reynolds Stress Tensor:

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
        axs[m,n].axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
        axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
        
     
        axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
        axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
        axs[m,n].minorticks_on()

        axs[m,n].set_xscale('linear')
        #axs[m,n].set_title(f'Case = {case}')
        axs[m,n].set_xlabel(f'{Rij_names[l]}')
        
        l = l+1
    
    
axs[0,0].set_ylabel(r'$z/h$')
axs[1,0].set_ylabel(r'$z/h$')

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
axs.hlines(1,0,4.5,colors='gray',linestyles='-')
axs.hlines(3,0,4.5,colors='gray',linestyles='-')
axs.set_ylim(y[0], y[-1])
axs.set_xlim(0, 4.5)

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xlabel(r'$\tau_w$')
axs.set_ylabel(r'$z/h$')

#%% Create a mask to mask the topography in pcolormesh plots
mask = np.full((Nx,Nz),np.nan,'d',order='F')
for i in range(0,Nx):
    for k in range(0,Nz):
        if k*dz<=z_tpg[i]:
            mask[i,k] = 1


figure = plt.figure()
im=plt.pcolormesh(x_m,z_m,np.transpose(Rstress.data[:,60,:,0]),cmap='coolwarm')
plt.pcolormesh(x_m,z_m,np.transpose(mask),cmap='gray')
plt.plot(x_m,z_tpg*zi,color='black')
plt.plot(x_m,z_tpg*zi+canopyH*zi,color='black',ls='--')
plt.xlabel('x [m]', fontsize=15)
plt.ylabel('z [m]', fontsize=15)
plt.title('Rxx', fontsize=15)
plt.colorbar(im)



#%% Computing turbulence anisotropy and clustering:

anisotropy_compute = 'false'    
path = '/scratch/general/nfs1/u1450851/UCLAdata/data/amazon_canopy_hill_3D/'

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

z3D = np.zeros((Nx,Ny,Nz),'d',order='F')

for k in range(0,Nz):
    for i in range(0,Nx):
        for j in range(0,Ny):
            # z3D[i,j,k] = z_on_h[k]
            z3D[i,j,k] = (topo['Z_gd'][k,j,i] - 1)/zi
    
z2D = np.mean(z3D[:,:,0:Nz_SLayer],axis=1)

#%%Create a vector that tells me the grisd point where the topography ends
z_grid = np.zeros(Nx,'d',order='F')
for i in range(0,Nx):
    for k in range(0,Nz_SLayer):
        if z2D[i,k] > 0:
            z_grid[i] = k
            break
        
#%%Compute phi_M, to compute it we need vertical derivatives of u and v, to be computed post-process

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / dz
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

dUdz = get_dphidz(data.data[:,:,:,6], dz)
dVdz = get_dphidz(data.data[:,:,:,7], dz)

[phi_m_3D,U,ustar,z_ustar,tmp_ustar] = phi_m(Nx,Ny,Nz_SLayer,height,z,z_grid,data.data[:,:,:,6],data.data[:,:,:,7],dUdz,dVdz, \
                  Rstress.data[:,:,0:Nz_SLayer,4],data.data[:,:,0:Nz_SLayer,4],Rstress.data[:,:,0:Nz_SLayer,5],data.data[:,:,0:Nz_SLayer,5])


z3D_1D = np.ndarray.flatten(z3D[:,:,0:Nz_SLayer])
phi_M_1D = np.ndarray.flatten(phi_m_3D)
phi_m_2D = np.mean(phi_m_3D,axis=1)

#%%Compute ustar

Rxz_raw = Rstress.data[:,:,:,4]
SGSxz_raw = data.data[:,:,:,4]
Ryz_raw = Rstress.data[:,:,:,5]
SGSyz_raw = data.data[:,:,:,5]

#%%
Rxz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
SGSxz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
Ryz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')
SGSyz = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

# tmp_ustar = np.zeros((Nx,Ny,Nz),'d',order='F')

for i in range(0,Nx):
    for j in range(0,Ny):
        tmp_Rxz = Rxz_raw[i,j,:][z3D[i,j,:] >= 0]
        tmp_SGSxz = SGSxz_raw[i,j,:][z3D[i,j,:] >= 0]
        tmp_Ryz = Ryz_raw[i,j,:][z3D[i,j,:] >= 0]
        tmp_SGSyz = SGSyz_raw[i,j,:][z3D[i,j,:] >= 0]
        Rxz[i,j,:] = tmp_Rxz[0:Nz_SLayer]
        SGSxz[i,j,:] = tmp_SGSxz[0:Nz_SLayer]
        Ryz[i,j,:] = tmp_Ryz[0:Nz_SLayer]
        SGSyz[i,j,:] = tmp_SGSyz[0:Nz_SLayer]

tmp_ustar = (((-Rxz+SGSxz)**2 + (-Ryz + SGSyz)**2)**(1/4))

ustar = np.amax(tmp_ustar,axis=2) #ustar is a 2D matrix of the max values of the shear stress.

figure = plt.figure()
plt.plot(np.mean(tmp_ustar,axis=(0,1)),z_m[0:Nz_SLayer],color='red')
plt.axhline(canopyH*zi,color='black',ls='--')
plt.xlabel(r'$<u^{*}>$',fontsize=15)
plt.ylabel(r'z [m]',fontsize=15)

figure = plt.figure()
plt.pcolormesh(x,y,np.transpose(ustar),cmap='Greys',shading='gouraud')
plt.xlabel('x',fontsize=15)
plt.ylabel('y',fontsize=15)
plt.title('ustar',fontsize=15)
plt.colorbar()

#%%Compute Shear Stress
Txz = - Rstress.data[:,:,:,4] + data.data[:,:,:,4]
Tyz = - Rstress.data[:,:,:,5] + data.data[:,:,:,5]
Rw = np.sqrt(Rstress.data[:,:,:,4]**2 + Rstress.data[:,:,:,5]**2)
SGSw = np.sqrt(data.data[:,:,:,4]**2 + data.data[:,:,:,5]**2)
Tw = Rw + SGSw

Txz_2D = np.mean(Txz,axis=1)
Tyz_2D = np.mean(Tyz,axis=1)
Rw_2D = np.mean(Rw,axis=1)
SGSw_2D = np.mean(SGSw,axis=1)
Tw_2D = np.mean(Tw,axis=1)

Txz_1D = np.ndarray.flatten(Txz[:,:,0:Nz_SLayer])
Tyz_1D = np.ndarray.flatten(Tyz[:,:,0:Nz_SLayer])
Rw_1D = np.ndarray.flatten(Rw[:,:,0:Nz_SLayer])
SGSw_1D = np.ndarray.flatten(SGSw[:,:,0:Nz_SLayer])
Tw_1D = np.ndarray.flatten(Tw[:,:,0:Nz_SLayer])

#%% Plot vertical and horizontal slices of xB and yB

yB_yavg=np.mean(yB[:,:,:],axis=1)


figure = plt.figure()
plt1=plt.contourf(x_m,z_m[0:Nz_SLayer],np.transpose(yB_yavg),cmap=cmap)
# plt1=plt.pcolormesh(np.transpose(yB_yavg),cmap = cmap, shading='flat')
plt.contourf(x_m,z_m[0:Nz_SLayer],np.transpose(mask[:,0:Nz_SLayer]),cmap='gray')
plt.plot(x_m,z_tpg*zi+canopyH*zi,color='black',ls='--')
plt.plot(x_m,z_tpg*zi,color='black')
plt.colorbar(plt1)
# plt.xticks(np.arange(0,Nx),np.around(x,2))
# plt.yticks(np.arange(0,Nz_SLayer),np.around(z_slayer,2))
plt.locator_params(axis='both',nbins=7)
plt.xlabel('x [m]', fontsize=15)
plt.ylabel('z [m]', fontsize=15)
plt.title('yB -- y_averaged', fontsize=15)

# figure = plt.figure()
# plt1=plt.contourf(x_m,z_m[0:Nz_SLayer],np.transpose(phi_m_2D),cmap=cmap)
# # plt1=plt.pcolormesh(np.transpose(yB_yavg),cmap = cmap, shading='flat')
# plt.contourf(x_m,z_m[0:Nz_SLayer],np.transpose(mask[:,0:Nz_SLayer]),cmap='gray')
# plt.plot(x_m,z_tpg*zi+canopyH*zi,color='black',ls='--')
# plt.plot(x_m,z_tpg*zi,color='black')
# plt.colorbar(plt1)
# # plt.xticks(np.arange(0,Nx),np.around(x,2))
# # plt.yticks(np.arange(0,Nz_SLayer),np.around(z_slayer,2))
# plt.locator_params(axis='both',nbins=7)
# plt.xlabel('x [m]', fontsize=15)
# plt.ylabel('z [m]', fontsize=15)
# plt.title(r'$\phi_{M}$ -- y_averaged', fontsize=15)

#%% Vertical profiles of sigma_u, sigma_v, sigma_w, phi_m

pnt1 = [80] #x location corresponding to hill crest
pnt2 = [120] #x location corresponding to hill slope
pnt3 = [160] #x location corresponding to valley floor

sigma_u = Rstress.data[:,:,:,0] + data.data[:,:,:,0]
sigma_v = Rstress.data[:,:,:,1] + data.data[:,:,:,1]
sigma_w = Rstress.data[:,:,:,2] + data.data[:,:,:,2]

for i in range(0,Nx):
    for j in range(0,Ny):
        sigma_u[i,j,:] = sigma_u[i,j,:]/ustar[i,j] 
        sigma_v[i,j,:] = sigma_v[i,j,:]/ustar[i,j] 
        sigma_w[i,j,:] = sigma_w[i,j,:]/ustar[i,j] 

sigma_u_2D = np.mean(sigma_u,axis=1)
sigma_v_2D = np.mean(sigma_v,axis=1)
sigma_w_2D = np.mean(sigma_w,axis=1)

sigma_u_1D = np.ndarray.flatten(sigma_u[:,:,0:Nz_SLayer])
sigma_v_1D = np.ndarray.flatten(sigma_v[:,:,0:Nz_SLayer])
sigma_w_1D = np.ndarray.flatten(sigma_w[:,:,0:Nz_SLayer])

sigma_u_pnt1 = sigma_u_2D[pnt1[0],0:Nz_SLayer][z2D[pnt1[0],:] >= 0]
sigma_u_pnt2 = sigma_u_2D[pnt2[0],0:Nz_SLayer][z2D[pnt2[0],:] >= 0]
sigma_u_pnt3 = sigma_u_2D[pnt3[0],0:Nz_SLayer][z2D[pnt3[0],:] >= 0]

sigma_v_pnt1 = sigma_v_2D[pnt1[0],0:Nz_SLayer][z2D[pnt1[0],:] >= 0]
sigma_v_pnt2 = sigma_v_2D[pnt2[0],0:Nz_SLayer][z2D[pnt2[0],:] >= 0]
sigma_v_pnt3 = sigma_v_2D[pnt3[0],0:Nz_SLayer][z2D[pnt3[0],:] >= 0]

sigma_w_pnt1 = sigma_w_2D[pnt1[0],0:Nz_SLayer][z2D[pnt1[0],:] >= 0]
sigma_w_pnt2 = sigma_w_2D[pnt2[0],0:Nz_SLayer][z2D[pnt2[0],:] >= 0]
sigma_w_pnt3 = sigma_w_2D[pnt3[0],0:Nz_SLayer][z2D[pnt3[0],:] >= 0]

figure, ax = plt.subplots(1,3,sharey=True)
ax[0].plot(sigma_u_pnt1,z_m[0:len(sigma_u_pnt1)],color='red',label='Crest')
ax[0].plot(sigma_u_pnt2,z_m[0:len(sigma_u_pnt2)],color='green',label='Slope')
ax[0].plot(sigma_u_pnt3,z_m[0:len(sigma_u_pnt3)],color='blue',label='Floor')
ax[1].plot(sigma_v_pnt1,z_m[0:len(sigma_v_pnt1)],color='red',label='Crest')
ax[1].plot(sigma_v_pnt2,z_m[0:len(sigma_v_pnt2)],color='green',label='Slope')
ax[1].plot(sigma_v_pnt3,z_m[0:len(sigma_v_pnt3)],color='blue',label='Floor')
ax[2].plot(sigma_w_pnt1,z_m[0:len(sigma_w_pnt1)],color='red',label='Crest')
ax[2].plot(sigma_w_pnt2,z_m[0:len(sigma_w_pnt2)],color='green',label='Slope')
ax[2].plot(sigma_w_pnt3,z_m[0:len(sigma_w_pnt3)],color='blue',label='Floor')

ax[0].axhline(canopyH*zi,color='black',ls='--')
ax[1].axhline(canopyH*zi,color='black',ls='--')
ax[2].axhline(canopyH*zi,color='black',ls='--')

ax[0].set_ylabel('z [m]',fontsize=15)

ax[0].set_xlabel(r'$\sigma_{u}$', fontsize=15)
ax[1].set_xlabel(r'$\sigma_{v}$', fontsize=15)
ax[2].set_xlabel(r'$\sigma_{w}$', fontsize=15)

ax[0].legend(fontsize=8, loc='upper right')
ax[1].legend(fontsize=8, loc='upper right')
ax[2].legend(fontsize=8, loc='upper right')

#----------vertical profiles at 3 locations of phi_M

phi_m_pnt1 = phi_m_2D[pnt1[0],:][z2D[pnt1[0],:] >= 0]
phi_m_pnt2 = phi_m_2D[pnt2[0],:][z2D[pnt2[0],:] >= 0]
phi_m_pnt3 = phi_m_2D[pnt3[0],:][z2D[pnt3[0],:] >= 0]

figure, ax = plt.subplots()
ax.plot(phi_m_pnt1,z_m[0:len(phi_m_pnt1)],color='red',label='Crest')
ax.plot(phi_m_pnt2,z_m[0:len(phi_m_pnt2)],color='green',label='Slope')
ax.plot(phi_m_pnt3,z_m[0:len(phi_m_pnt3)],color='blue',label='Floor')

ax.axhline(canopyH*zi,color='black',ls='--')
ax.axvline(1.5,color='black',ls='--')

ax.set_ylabel('z [m]',fontsize=15)

ax.set_xlabel(r'$\phi_{M}$', fontsize=15)

ax.legend(fontsize=8, loc='upper right')

#%%Color the profiles based on turbulence anisotropy

yB_yavg=np.mean(yB[:,:,:],axis=1)
#Shift the vertical profiles for each x location to remove the topography
yB_yavg_shift = []
phi_m_2D_shift = []

for i in range(0,Nx):
    yB_yavg_shift.append(yB_yavg[i,:][z2D[i,:] >= 0])
    phi_m_2D_shift.append(phi_m_2D[i,:][z2D[i,:] >= 0])


# figure = plt.figure()
# for i in range(0,Nx):
#     sigma_u_i = sigma_w_2D[i,0:Nz_SLayer][z2D[i,:] >= 0]
#     plt.scatter(sigma_u_i[0:len(yB_yavg_shift[i])],z_m[0:len(yB_yavg_shift[i])],s=5,c=yB_yavg_shift[i],cmap=cmap)

# # plt.scatter(sigma_u_pnt1[0:len(yB_yavg_shift[pnt1[0]])],z[0:len(yB_yavg_shift[pnt1[0]])],c=yB_yavg_shift[pnt1[0]],cmap=cmap)
# # plt.scatter(sigma_u_pnt2[0:len(yB_yavg_shift[pnt2[0]])],z[0:len(yB_yavg_shift[pnt2[0]])],c=yB_yavg_shift[pnt2[0]],cmap=cmap)
# # plt.scatter(sigma_u_pnt3[0:len(yB_yavg_shift[pnt3[0]])],z[0:len(yB_yavg_shift[pnt3[0]])],c=yB_yavg_shift[pnt3[0]],cmap=cmap)

# plt.axhline(canopyH*zi,color='black',ls='--')
# plt.colorbar()
# plt.ylabel('z [m]', fontsize=15)
# plt.xlabel(r'$\sigma_{W}$', fontsize=15)

figure = plt.figure()
for i in range(0,Nx):
    phi_m_i = phi_m_2D[i,:][z2D[i,:] >= 0]
    plt.scatter(phi_m_i[0:len(yB_yavg_shift[i])],z_m[0:len(yB_yavg_shift[i])],s=5,c=yB_yavg_shift[i],cmap=cmap)

plt.axhline(canopyH*zi,color='black',ls='--')
plt.axvline(1,color='black',ls='--')
plt.colorbar()
plt.ylabel('z [m]', fontsize=15)
plt.xlabel(r'$\phi_{M}$', fontsize=15)

#%%Plot profiles of phi_m clustered based on turbulence anisotropy

from scipy.stats import kde

Nclusters = 10 #Number of clusters used to group the anisotorpy.

fig, axs = plt.subplots(nrows=3,ncols=3)
plt.ion()

#Loop through all the clusters (1 to 9) organized in a 3x3 subplot..
cluster = 0
for m in range(0,3):
    for n in range(0,3):

        cluster = cluster + 1

        tmp_z3D = z3D_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
        tmp_phi_u = phi_M_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
        # tmp_sigma_u = sigma_w_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
        tmp_yB = yB_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
        # breakpoint()
        if (cluster < 9):
            # x = tmp_phi_u
            x = tmp_phi_u
            y = tmp_z3D*zi
    

            #Developing a "2D Density plot" to better visualize where there are more points.
            tmp = np.array([x,y])
            nbins = 40

            # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
            k = kde.gaussian_kde(tmp)
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            density = k(np.vstack([xi.flatten(), yi.flatten()]))
 
            # plot the density with shading
            #axs.set_title('2D Density with shading')
            axs[m,n].pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')


        #Overlay the median on the denisty plot:
        phiM_median = np.zeros(Nz_SLayer,dtype='float')
        # sigmaU_median = np.zeros(Nz_SLayer,dtype='float')
        yB_median = np.zeros(Nz_SLayer,dtype='float')


        #First we downselect data based on the yB clustering developed earlier.
        #tmp_z3D = z3D_1D[(AnisType_1D == cluster)]
        #tmp_phi_u = phi_M_1D[(AnisType_1D == cluster)]
    
        for i in range(0,Nz_SLayer-1):
            phiM_median[i] = np.median(tmp_phi_u[(tmp_z3D >= z[i]) & (tmp_z3D < z[i+1])])
            # sigmaU_median[i] = np.median(tmp_sigma_u[(tmp_z3D >= z[i]) & (tmp_z3D < z[i+1])])
            yB_median[i] = np.median(tmp_yB[(tmp_z3D >= z[i]) & (tmp_z3D < z[i+1])])     
    

        y_ax = z_m[0:Nz_SLayer]
    
    
        sc = axs[m,n].scatter(phiM_median[0:-1],y_ax[0:-1],s=5,marker='o',c = yB_median[0:-1],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

#------------------------------------------------------------------------------


        axs[m,n].set_xlim(-1, 8)
        axs[m,n].set_ylim(y_ax[0], y_ax[-1])
        axs[m,n].axhline(y = canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='--')
        axs[m,n].axhline(y = 2*canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='-.')
        axs[m,n].axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')
        
     
        axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
        axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
        axs[m,n].minorticks_on()
    

        axs[m,n].set_xscale('linear')
        axs[m,n].set_title(f'cluster = {cluster}')
    
    
cbar = plt.colorbar(sc)
    
axs[0,0].set_ylabel(r'$z [m]$')
axs[1,0].set_ylabel(r'$z [m]$')
axs[2,0].set_ylabel(r'$z [m]$')

axs[2,0].set_xlabel(r'$\phi_M$')
axs[2,1].set_xlabel(r'$\phi_M$')
axs[2,2].set_xlabel(r'$\phi_M$')


plt.tight_layout()
plt.show()

#%% Graphical Representation of the results with Clustering in a single subplot:

import matplotlib.ticker


fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

# tmp_z3D = z3D_1D[z3D_1D >= 0]
# tmp_phi_m = phi_M_1D[z3D_1D >= 0]
# # tmp_sigma_u = sigma_w_1D[z3D_1D >= 0]
# tmp_yB = yB_1D[z3D_1D >= 0]

# x = tmp_phi_m
# # x = tmp_sigma_u
# y = tmp_z3D*zi

# #Developing a "2D Density plot" to better visualize where there are more points.
# tmp = np.array([x,y])
# nbins = 10

# # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
# k = kde.gaussian_kde(tmp)
# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
# density = k(np.vstack([xi.flatten(), yi.flatten()]))
 
# # plot the density with shading
# axs.pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')

#Median of the gradient profiles as a function of cluster and height. 
#For a fixed cluster (e.g. cluster = 4), we average all values of the gradient
#at a specific height.
 
phiU_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')
# sigmaU_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')
yB_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')

# for i in range(0,Nx):
#     phi_m_i = phi_m_2D[i,:][z2D[i,:] >= 0]
#     axs.scatter(phi_m_i[0:len(yB_yavg_shift[i])],z_m[0:len(yB_yavg_shift[i])],s=5,c=yB_yavg_shift[i],cmap='Greys')

for cluster in range(1,Nclusters):

    #First we downselect data based on the yB clustering developed earlier.
    tmp_phi_u = phi_M_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
    # tmp_sigma_u = Tw_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
    tmp_yB = yB_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
    tmp_z3D = z3D_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]

    for k in range(0,Nz_SLayer-1):            
        
        phiU_median[cluster,k] = np.median(tmp_phi_u[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
        # sigmaU_median[cluster,k] = np.median(tmp_sigma_u[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
        yB_median[cluster,k] = np.median(tmp_yB[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
        
    axs.scatter(phiU_median[cluster,0:-1],(z_m[0:Nz_SLayer-1]),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs.set_xlim(-1,8)
axs.set_ylim(y_ax[0], y_ax[Nz_SLayer-1])
axs.axhline(y = canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='--')
axs.axhline(y = 3*canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='-.')

axs.set_xscale('linear')
cbar = plt.colorbar(sc)
axs.set_ylabel(r'$z [m]$')
axs.set_xlabel(r'$\phi_M$')
# axs.set_title(f'{cases[num]}')

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')

plt.tight_layout()
plt.show()

#%%

from scipy.stats import kde

phi_m_3D_1D = np.ndarray.flatten(phi_m_3D)

fig, axs = plt.subplots()
plt.ion()

tmp_z3D = z3D_1D[z3D_1D >= 0]
tmp_phi_m = phi_m_3D_1D[z3D_1D >= 0]
# tmp_sigma_u = sigma_w_1D[z3D_1D >= 0]
tmp_yB = yB_1D[z3D_1D >= 0]

x = tmp_phi_m
# x = tmp_sigma_u
y = tmp_z3D*zi

#Developing a "2D Density plot" to better visualize where there are more points.
tmp = np.array([x,y])
nbins = 10

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(tmp)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
density = k(np.vstack([xi.flatten(), yi.flatten()]))
 
# plot the density with shading
axs.pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')
# axs.plot(np.mean(sigma_v[:,:,0:Nz_SLayer],axis=(0,1)),z_m[0:Nz_SLayer],color='red')

phiU_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')
yB_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')

for cluster in range(1,Nclusters):

    #First we downselect data based on the yB clustering developed earlier.
    tmp_phi_u = phi_m_3D_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
    # tmp_sigma_u = sigma_w_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
    tmp_yB = yB_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
    tmp_z3D = z3D_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]

    for k in range(0,Nz_SLayer-1):            
        
        phiU_median[cluster,k] = np.median(tmp_phi_u[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
        # sigmaU_median[cluster,k] = np.median(tmp_sigma_u[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
        yB_median[cluster,k] = np.median(tmp_yB[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
        
    axs.scatter(sigmaU_median[cluster,0:-1],(z_m[0:Nz_SLayer-1]),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs.set_xlim(-1, 8)
axs.axhline(y = canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='--')
axs.axhline(y = 2*canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='-.')
axs.axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()

axs.set_xscale('linear')
    
axs.set_ylabel(r'$z [m]$')

axs.set_xlabel(r'$\phi_M$')

plt.tight_layout()
plt.show()

#%%

# figure,axs = plt.subplots()

# for i in range(0,Nx):
#     phi_m_i = phi_m_2D[i,:][z2D[i,:] >= 0]
#     axs.plot(phi_m_i[0:len(yB_yavg_shift[i])],z_m[0:len(yB_yavg_shift[i])],color='grey',alpha=0.5,zorder=1)
    
# phiU_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')
# yB_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')

# for cluster in range(1,Nclusters):

#     #First we downselect data based on the yB clustering developed earlier.
#     tmp_phi_u = phi_M_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
#     tmp_yB = yB_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]
#     tmp_z3D = z3D_1D[(AnisType_1D == cluster) & (z3D_1D >= 0)]

#     for k in range(0,Nz_SLayer-1):            
        
#         phiU_median[cluster,k] = np.median(tmp_phi_u[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
#         yB_median[cluster,k] = np.median(tmp_yB[(tmp_z3D >= z[k]) & (tmp_z3D < z[k+1])]) 
        
#     axs.scatter(phiU_median[cluster,0:-1],(z_m[0:Nz_SLayer-1]),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2,zorder=2)
    
# axs.axhline(y = canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='--')
# axs.axhline(y = 3*canopyH*zi, xmin=-10, xmax=8,color='gray',linestyle='-.')
# axs.axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')
# axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
# axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
# axs.minorticks_on()
# axs.set_xscale('linear')
    
# axs.set_ylabel(r'$z [m]$')

# axs.set_xlabel(r'$\phi_M$')

# plt.tight_layout()
# plt.show()

# plt.colorbar()



























    