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
import matplotlib as mpl
#plt.rcParams['figure.dpi'] = 300

os.chdir('/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/Ben_Analysis/')

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

Nx = 376
Ny = 376
Nz = 270
Nz_SLayer = int(Nz)
Ny_Slice = int(Ny/2)

zi = 540 #length scale for non-dimensionalization

lx = 3000/zi
ly = 3000/zi
lz = 540/zi

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

path = '/uufs/chpc.utah.edu/common/home/calaf-group3/UCLA_data/data/amazon_canopy_real_3D/'

data = UCLA_npy_to_netCDF(path,Nx,Ny,Nz,'true')  #This uploads the data from the NetCDF files, that keep the xarray form.
data = data.transpose('x','y','z','variable') # I transpose the data from (Nz,Ny,Nx) to (Nx,Ny,Nz)
# content of data: 0. txx, 1. tyy, 2. tzz, 3. txy, 4. txz, 5. tyz, 6. u, 7. v, 8. w, 9. uu, 10. vv, 11. ww, 12. uv, 13. uw, 14. vw


    #%% Importing the topography
from scipy.io import loadmat

topo = loadmat('/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_real_3D/topo.mat')
# z_tpg = np.mean(np.transpose(topo['z_tpg']),axis=1)/zi
topo_slice = topo['Z_gd'][:,Ny_Slice,:]
z_tpg = topo['z_tpg'][Ny_Slice,:]

dist = topo['Z_gd']
dist = np.transpose(dist,(2,1,0))

figure = plt.figure()
plt.plot(x,z_tpg)

#Set 0 values below the topography, to facilitate computing the stress and anisitropy
for n in range(0,15):
    tmp = np.copy(data.data[:,:,:,n])
    tmp[(dist<0)] = float("nan")
    tmp2 = np.copy(tmp)
    data.data[:,:,:,n] = tmp2


#%% Computing the Reynolds Stress


# Computing the Reynolds Stress Tensor :

Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

# If you upload the data differently from what I do you might have to change the line below
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

path = '/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_real_3D/'

# Decide whether to compute turbulence anisotropy and clustering from scratch or read it from file.

if (anisotropy_compute == 'true'):
    
    [xB,yB,AnisType_1D] = Anisotropy_Clustering(Nx,Ny,Nz_SLayer,Rstress)
    
    #yB[(dist<0)] = float("nan")
    #xB[(dist<0)] = float("nan")

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


#%% Plot vertical slices of yB

# yB_yavg=np.mean(yB[:,:,:],axis=1)

path_fig = '/Users/mcalaf/Documents/Utah/Research/Collaborative_Research/Gaby-Marcelo/BichengTKE/amazon_canopy_real_3D/Figures/'


fig,axs = plt.subplots(1,1,constrained_layout=True,figsize=(6,3))
plt1=axs.pcolormesh(x,z[0:Nz_SLayer],np.transpose(yB[:,int(Ny/2),:]),cmap=cmap,vmin = 0, vmax = np.sqrt(3)/2)
plt.colorbar(plt1)

#axs.set_clim(vmin=0, vmax=np.sqrt(3)/2)
axs.plot(x,z_tpg/zi+canopyH,color='black',ls='--')
axs.set_ylim([0,1])
axs.set_xlim([0,x[-1]])
# axs.locator_params(axis='both',nbins=7)
axs.set_xlabel(r'x/$z_i$')
axs.set_ylabel(r'z/$z_i$')
axs.set_title(r'yB(x,$y_{Ny/2}$,z)')

plt.savefig(path_fig+'yB_real.png',dpi=300,facecolor='white',edgecolor='white')


#%% Compute the height with respect the topography locally at each grid point:
'''
z3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,Nx):
        for j in range(0,Ny):
            z3D[i,j,k] = z_d[k]
    
    
z3D_1D = np.ndarray.flatten(z3D)
'''

z_grid = np.zeros((Nx,Ny),'d',order='F')

for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz_SLayer):
            if topo['Z_gd'][k,j,i] > 0:
                z_grid[i,j] = k
                break
    
z3D = np.full((Nx,Ny,Nz),np.nan,'d',order='F')

# for k in range(0,Nz_SLayer):
for i in range(0,Nx):
    for j in range(0,Ny):
        # z3D[i,j,k] = z_on_h[k]
        # z3D[i,j,k] = (topo['Z_gd'][k,j,i])/zi
        if int(z_grid[i,j]) > 0:
            z3D[i,j,int(z_grid[i,j]):] = z_d[0:-(int(z_grid[i,j]))]
        else:
            #z3D[i,j,:] = z_d
            z3D[i,j,:] = float("nan")
        
z3D_1D = np.ndarray.flatten(z3D)



#%%Compute phi_M, to compute it we need vertical derivatives of u and v, to be computed post-process

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / dz
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

dUdz = get_dphidz(data.data[:,:,:,6], dz)
dVdz = get_dphidz(data.data[:,:,:,7], dz)


[phi_m_3D,U,ustar,z_ustar,tmp_ustar] = phi_m(Nx,Ny,Nz,height,z3D,z_grid,data.data[:,:,:,6],data.data[:,:,:,7],dUdz,dVdz,
                  Rstress.data[:,:,:,4],data.data[:,:,:,4],Rstress.data[:,:,:,5],data.data[:,:,:,5])


phi_M_1D = np.ndarray.flatten(phi_m_3D)
phi_m_2D = np.mean(phi_m_3D,axis=1) #phi_m averaged in y direction
phi_m_xy = np.mean(phi_m_3D,axis=(0,1)) #phi_m averaged in x-y direction


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


#%% Vertical profiles of sigma_u, sigma_v, sigma_w, phi_m

#pnt1 = [80] #x location corresponding to hill crest
#pnt2 = [120] #x location corresponding to hill slope
#pnt3 = [160] #x location corresponding to valley floor

sigma_u = Rstress.data[:,:,:,0] + data.data[:,:,:,0]
sigma_v = Rstress.data[:,:,:,1] + data.data[:,:,:,1]
sigma_w = Rstress.data[:,:,:,2] + data.data[:,:,:,2]

#normalize with ustar
for i in range(0,Nx):
    for j in range(0,Ny):
        sigma_u[i,j,:] = sigma_u[i,j,:]/ustar[i,j] 
        sigma_v[i,j,:] = sigma_v[i,j,:]/ustar[i,j] 
        sigma_w[i,j,:] = sigma_w[i,j,:]/ustar[i,j] 

#average in y direction
sigma_u_2D = np.mean(sigma_u,axis=1)
sigma_v_2D = np.mean(sigma_v,axis=1)
sigma_w_2D = np.mean(sigma_w,axis=1)

#create the 1D vector from the 3D field
sigma_u_1D = np.ndarray.flatten(sigma_u[:,:,0:Nz_SLayer])
sigma_v_1D = np.ndarray.flatten(sigma_v[:,:,0:Nz_SLayer])
sigma_w_1D = np.ndarray.flatten(sigma_w[:,:,0:Nz_SLayer])

# sigma_u_pnt1 = sigma_u_2D[pnt1[0],0:Nz_SLayer]
# sigma_u_pnt2 = sigma_u_2D[pnt2[0],0:Nz_SLayer]
# sigma_u_pnt3 = sigma_u_2D[pnt3[0],0:Nz_SLayer]

# sigma_v_pnt1 = sigma_v_2D[pnt1[0],0:Nz_SLayer]
# sigma_v_pnt2 = sigma_v_2D[pnt2[0],0:Nz_SLayer]
# sigma_v_pnt3 = sigma_v_2D[pnt3[0],0:Nz_SLayer]

# sigma_w_pnt1 = sigma_w_2D[pnt1[0],0:Nz_SLayer]
# sigma_w_pnt2 = sigma_w_2D[pnt2[0],0:Nz_SLayer]
# sigma_w_pnt3 = sigma_w_2D[pnt3[0],0:Nz_SLayer]

# figure, ax = plt.subplots(1,3,sharey=True)
# ax[0].plot(sigma_u_pnt1,z_m[0:len(sigma_u_pnt1)],color='red',label='Crest')
# ax[0].plot(sigma_u_pnt2,z_m[0:len(sigma_u_pnt2)],color='green',label='Slope')
# ax[0].plot(sigma_u_pnt3,z_m[0:len(sigma_u_pnt3)],color='blue',label='Floor')
# ax[1].plot(sigma_v_pnt1,z_m[0:len(sigma_v_pnt1)],color='red',label='Crest')
# ax[1].plot(sigma_v_pnt2,z_m[0:len(sigma_v_pnt2)],color='green',label='Slope')
# ax[1].plot(sigma_v_pnt3,z_m[0:len(sigma_v_pnt3)],color='blue',label='Floor')
# ax[2].plot(sigma_w_pnt1,z_m[0:len(sigma_w_pnt1)],color='red',label='Crest')
# ax[2].plot(sigma_w_pnt2,z_m[0:len(sigma_w_pnt2)],color='green',label='Slope')
# ax[2].plot(sigma_w_pnt3,z_m[0:len(sigma_w_pnt3)],color='blue',label='Floor')

# ax[0].axhline(canopyH*zi,color='black',ls='--')
# ax[1].axhline(canopyH*zi,color='black',ls='--')
# ax[2].axhline(canopyH*zi,color='black',ls='--')

# ax[0].set_ylabel('z [m]',fontsize=15)

# ax[0].set_xlabel(r'$\sigma_{u}$', fontsize=15)
# ax[1].set_xlabel(r'$\sigma_{v}$', fontsize=15)
# ax[2].set_xlabel(r'$\sigma_{w}$', fontsize=15)

# ax[0].legend(fontsize=8, loc='upper right')
# ax[1].legend(fontsize=8, loc='upper right')
# ax[2].legend(fontsize=8, loc='upper right')

#----------vertical profiles at 3 locations of phi_M

# phi_m_pnt1 = phi_m_2D[pnt1[0],:]
# phi_m_pnt2 = phi_m_2D[pnt2[0],:]
# phi_m_pnt3 = phi_m_2D[pnt3[0],:]

# figure, ax = plt.subplots()
# ax.plot(phi_m_xy,z_m[0:Nz_SLayer],color='red',label='Crest')
# # ax.plot(phi_m_pnt2,z_m[0:len(phi_m_pnt2)],color='green',label='Slope')
# # ax.plot(phi_m_pnt3,z_m[0:len(phi_m_pnt3)],color='blue',label='Floor')

# ax.axhline(canopyH*zi,color='black',ls='--')
# ax.axvline(1.5,color='black',ls='--')

# ax.set_ylabel('z [m]',fontsize=15)

# ax.set_xlabel(r'$\phi_{M}$', fontsize=15)

# ax.legend(fontsize=8, loc='upper right')

#%%Color the profiles based on turbulence anisotropy

#Sigma u: ---------------------------------------------------------------------

fig,axs = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))
norm = mpl.colors.BoundaryNorm(np.arange(0,1,0.1),cmap.N)
for i in range(0,Nx):
     sigma_u_i = sigma_u[i,int(Ny/2),0:Nz_SLayer]
     im=axs.scatter(sigma_u_i[0:Nz_SLayer],z_d[0:Nz_SLayer]/canopyH,s=5,c=yB[i,int(Ny/2),:],cmap=cmap,norm=norm)

axs.axhline(((canopyH*zi-30.2)/zi)/canopyH,color='black',ls='--')
fig.colorbar(im)
axs.set_ylabel('(z-d)/h')
axs.set_xlabel(r'$\sigma_{u}/u_*$')
axs.set_xlim([0,5])
axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()

plt.savefig(path_fig+'SigmaU_prof_anis.png',dpi=300,facecolor='white',edgecolor='white')

#Sigma v: ---------------------------------------------------------------------

fig,axs = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))
norm = mpl.colors.BoundaryNorm(np.arange(0,1,0.1),cmap.N)
for i in range(0,Nx):
     sigma_u_i = sigma_v[i,int(Ny/2),0:Nz_SLayer]
     im=axs.scatter(sigma_u_i[0:Nz_SLayer],z_d[0:Nz_SLayer]/canopyH,s=5,c=yB[i,int(Ny/2),:],cmap=cmap,norm=norm)

axs.axhline(((canopyH*zi-30.2)/zi)/canopyH,color='black',ls='--')
fig.colorbar(im)
axs.set_ylabel('(z-d)/h')
axs.set_xlabel(r'$\sigma_{v}/u_*$')
axs.set_xlim([0,4])
axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()

plt.savefig(path_fig+'SigmaV_prof_anis.png',dpi=300,facecolor='white',edgecolor='white')

#Sigma w: ---------------------------------------------------------------------

fig,axs = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))
norm = mpl.colors.BoundaryNorm(np.arange(0,1,0.1),cmap.N)
for i in range(0,Nx):
     sigma_u_i = sigma_w[i,int(Ny/2),0:Nz_SLayer]
     im=axs.scatter(sigma_u_i[0:Nz_SLayer],z_d[0:Nz_SLayer]/canopyH,s=5,c=yB[i,int(Ny/2),:],cmap=cmap,norm=norm)

axs.axhline(((canopyH*zi-30.2)/zi)/canopyH,color='black',ls='--')
fig.colorbar(im)
axs.set_ylabel('(z-d)/h')
axs.set_xlabel(r'$\sigma_{w}/u_*$')
axs.set_xlim([0,4])
axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()

plt.savefig(path_fig+'SigmaW_prof_anis.png',dpi=300,facecolor='white',edgecolor='white')

#PhiM : ---------------------------------------------------------------------

fig,axs = plt.subplots(1,1,constrained_layout=True,figsize=(6,6))
norm = mpl.colors.BoundaryNorm(np.arange(0,1,0.1),cmap.N)
for i in range(0,Nx):
    phi_m_i = phi_m_3D[i,int(Ny/2),:]
    im=axs.scatter(phi_m_i[0:Nz_SLayer],z_d[0:Nz_SLayer]/canopyH,s=5,c=yB[i,int(Ny/2),:],cmap=cmap, norm=norm)

axs.axhline(((canopyH*zi-30.2)/zi)/canopyH,color='black',ls='--')
fig.colorbar(im)
axs.set_ylabel('(z-d)/h')
axs.set_xlabel(r'$\phi_{M}$')
axs.set_xlim([-2,5])
axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()

plt.savefig(path_fig+'phiM_prof_anis.png',dpi=300,facecolor='white',edgecolor='white')


#%%Plot profiles of sigma clustered based on turbulence anisotropy

from scipy.stats import kde

Nclusters = 10 #Number of clusters used to group the anisotorpy.

fig, axs = plt.subplots(nrows=3,ncols=3,constrained_layout=True)
norm = mpl.colors.BoundaryNorm(np.arange(0,1,0.1),cmap.N)
plt.ion()

#Loop through all the clusters (1 to 9) organized in a 3x3 subplot..
cluster = 0
for m in range(0,3):
    for n in range(0,3):

        cluster = cluster + 1

        tmp_z3D = np.copy(z3D_1D[(AnisType_1D == cluster)])
        tmp_phi_u = np.copy(phi_M_1D[(AnisType_1D == cluster)])
        # tmp_sigma_u = sigma_w_1D[(AnisType_1D == cluster)]
        tmp_yB = np.copy(yB_1D[(AnisType_1D == cluster)])
        
        #Remove NaN values:
        tmp_z3D = tmp_z3D[~np.isnan(tmp_phi_u)]
        tmp_yB = tmp_yB[~np.isnan(tmp_phi_u)]
        tmp_phi_u = tmp_phi_u[~np.isnan(tmp_phi_u)]
                
        #tmp_z3D = tmp_z3D[~np.isnan(tmp_sigm_u)]
        #tmp_yB = tmp_yB[~np.isnan(tmp_sigma_u)]
        #tmp_sigma_u = tmp_z3D[np.isfinite(tmp_sigma_u)]
        
        # breakpoint()
        if (cluster < 9):
            
            x = tmp_phi_u
            y = tmp_z3D/canopyH
            

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
            phiM_median[i] = np.median(tmp_phi_u[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])
            yB_median[i] = np.median(tmp_yB[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])     
    

        y_ax = z_d[0:Nz_SLayer]/canopyH
    
    
        sc = axs[m,n].scatter(phiM_median[0:-1],y_ax[0:-1],s=5,marker='o',c = yB_median[0:-1],alpha=0.7,cmap=cmap,norm=norm)

#------------------------------------------------------------------------------


        axs[m,n].set_xlim(-1, 5)
        axs[m,n].set_ylim(y_ax[0], y_ax[-1])
        axs[m,n].axhline(y = (canopyH-30.2/zi)/canopyH, xmin=-10, xmax=8,color='gray',linestyle='--')
        axs[m,n].axhline(y = (2*canopyH-30.2/zi)/canopyH, xmin=-10, xmax=8,color='gray',linestyle='-.')
        # axs[m,n].axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')
        
     
        axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
        axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
        axs[m,n].minorticks_on()
    

        axs[m,n].set_xscale('linear')
        axs[m,n].set_title(f'cluster = {cluster}')
    
    
cbar = fig.colorbar(sc)
cbar.ax.tick_params(labelsize=6)
    
axs[0,0].set_ylabel(r'$(z-d)/h$')
axs[1,0].set_ylabel(r'$(z-d)/h$')
axs[2,0].set_ylabel(r'$(z-d)/h$')

axs[2,0].set_xlabel(r'$\phi_M$')
axs[2,1].set_xlabel(r'$\phi_M$')
axs[2,2].set_xlabel(r'$\phi_M$')

#plt.tight_layout()

plt.savefig(path_fig+'phiM_clusters.png',dpi=300,facecolor='white',edgecolor='white')


#%% Create the plots with shading and above the mean or median profiles
from scipy.stats import kde

variable_1D = phi_M_1D #change to sigma_u_1D,...
variable_3D = phi_m_3D #sigma_u

#variable_1D = sigma_w_1D #change to sigma_u_1D,...
#variable_3D = sigma_w #sigma_u



fig, axs = plt.subplots(1,1,figsize=(4,4))
norm = mpl.colors.BoundaryNorm(np.arange(0,1,0.1),cmap.N)
plt.ion()

#x = variable_1D #phi_M_1D #change to sigma_u_1D,...
#y = z3D_1D/canopyH

y = (z3D_1D[~np.isnan(z3D_1D)])/canopyH
x = variable_1D[~np.isnan(z3D_1D)]


#Developing a "2D Density plot" to better visualize where there are more points.
tmp = np.array([x,y])
nbins = 20

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(tmp)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
density = k(np.vstack([xi.flatten(), yi.flatten()]))
 
# plot the density with shading
axs.pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')

axs.plot(np.mean(variable_3D[:,:,0:Nz_SLayer],axis=(0,1)),z_d[0:Nz_SLayer]/canopyH,color='k') #x-y averaged profile

#Uncomment to Plot the median profiles----------------------------------------------------------------------
sigmaU_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')
yB_median = np.zeros((Nclusters,Nz_SLayer),dtype='float')

for cluster in range(1,Nclusters):

     #First we downselect data based on the yB clustering developed earlier.
     tmp_sigma_u = variable_1D[(AnisType_1D == cluster)]
     tmp_yB = yB_1D[(AnisType_1D == cluster)]
     tmp_z3D = z3D_1D[(AnisType_1D == cluster)]

     for k in range(0,Nz_SLayer-1):            
        
         sigmaU_median[cluster,k] = np.median(tmp_sigma_u[(tmp_z3D >= z_d[k]) & (tmp_z3D < z_d[k+1])]) 
         yB_median[cluster,k] = np.median(tmp_yB[(tmp_z3D >= z_d[k]) & (tmp_z3D < z_d[k+1])]) 
        
     im = axs.scatter(sigmaU_median[cluster,0:-1],(z_d[0:Nz_SLayer-1]/canopyH),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,norm=norm)

#axs.set_xlim(0, 5) #For the variances
axs.set_xlim(-2, 5) #For Phi_m

axs.axhline(y = (canopyH-30.2/zi)/canopyH, xmin=-10, xmax=8,color='gray',linestyle='--')
axs.axhline(y = (2*canopyH-30.2/zi)/canopyH, xmin=-10, xmax=8,color='gray',linestyle='-.')
# axs.axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
fig.colorbar(im)

axs.set_xscale('linear')
    
axs.set_ylabel(r'$(z-d)/h$')

axs.set_xlabel(r'$\phi_M$')
#axs.set_xlabel(r'$\sigma_w/u_*$')
#axs.set_xlabel(r'$\sigma_w/u_*$')

plt.tight_layout()

plt.savefig(path_fig+'phiM_mean_shading.png',dpi=300,facecolor='white',edgecolor='white')
#plt.savefig(path_fig+'Sigma_w_mean_shading.png',dpi=300,facecolor='white',edgecolor='white')


























    
