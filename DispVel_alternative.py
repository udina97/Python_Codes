#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:21:46 2023

@author: benjamin
"""

#%%Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import scipy as sp
from scipy.integrate import simps,trapz

os.chdir("/scratch/general/nfs1/u1450851/research/python_codes/")


from Stats import  ReynoldsStress, ReynoldsFlux
from Analysis import Fddx, Fddy, ddz3_w, ddz3_uv, dealias1, dealias2

#%% Loading the averaged Post-Processed Output_RAV data from the NetCDF files:

os.chdir("/scratch/general/nfs1/u1450851/research/caseUg1")
#os.chdir("/Volumes/Toshiba-HardDrive/SurfHeterogeneities/Paper_ParamDispFluxes/D800v3/case_Ug9/output_NetCDF/")
#os.chdir("/Volumes/Toshiba-HardDrive/SurfHeterogeneities/Paper_ParamDispFluxes/D200v1/case_Ug1/output_NetCDF/")

data = xr.open_dataarray('Data_Momentum.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
dataS = xr.open_dataarray('Data_Scalar.nc')
data_2D = xr.open_dataarray('Data_Momentum_2D.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
dataS_2D = xr.open_dataarray('Data_Scalar_2D.nc')

#%% We need to load the Raw 2D data for the surface temperature patchiness.

# os.chdir("/media/benjamin/b4648b0d-026f-4f15-9ffe-3a41a55386ef/PhD/Research/case_Ug1/output_NetCDF/")

# SurfaceTemp = xr.open_dataarray('SurfaceTempD800v3.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
#SurfaceTemp = xr.open_dataarray('SurfaceTempD400.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
#SurfaceTemp = xr.open_dataarray('SurfaceTempD200.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.


#%% Defining the main parameters of the simulations and uploaded files:

NumVariables = 26
NumVariablesSC = 10

Nx = 256
Ny = 256
Nz = 256
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz
z_i = 1
x = np.arange(0,Nx)*dx
y = np.arange(0,Nx)*dy
z = np.arange(0,Nz)*dz

#%% Compute the dispersive velocities using the velocity field from the LES model

uLES = data.data[:,:,:,0]   #average velocity in x direction
vLES = data.data[:,:,:,1]   #average velocity in y direction
wLES = data.data[:,:,:,2]   #average velocity in z direction

uLES_disp = np.zeros((Nx,Ny,Nz),'d',order='F')
vLES_disp = np.zeros((Nx,Ny,Nz),'d',order='F')
wLES_disp = np.zeros((Nx,Ny,Nz),'d',order='F')

for k in range(0,Nz):
    uLES_disp[:,:,k] = np.squeeze(uLES[:,:,k])-np.mean(uLES[:,:,k],axis=(0,1))
    vLES_disp[:,:,k] = np.squeeze(vLES[:,:,k])-np.mean(vLES[:,:,k],axis=(0,1))
    wLES_disp[:,:,k] = np.squeeze(wLES[:,:,k])-np.mean(wLES[:,:,k],axis=(0,1))
    
h = 13

fig, axes=plt.subplots(1,3, figsize=(8,4), constrained_layout=True)
im1 = axes[0].pcolormesh(x,y,np.transpose(uLES_disp[:,:,h]),shading='gouraud', cmap='coolwarm')
im2 = axes[1].pcolormesh(x,y,np.transpose(vLES_disp[:,:,h]),shading='gouraud', cmap='coolwarm')
im3 = axes[2].pcolormesh(x,y,np.transpose(wLES_disp[:,:,h]),shading='gouraud', cmap='coolwarm')

cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8,location='top',orientation='horizontal')
cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8,location='top',orientation='horizontal')
cbar3 = fig.colorbar(im3, ax=axes[2], shrink=0.8,location='top',orientation='horizontal')

axes[0].set_ylabel(r'$y/z_i$')
axes[0].set_xlabel(r'$x/z_i$'); axes[1].set_xlabel(r'$x/z_i$'); axes[2].set_xlabel(r'$x/z_i$');

axes[0].set_title(r'$\overline{u}- <\overline{u}>$')
axes[1].set_title(r'$\overline{v}- <\overline{v}>$')
axes[2].set_title(r'$\overline{w}- <\overline{w}>$')

#compute the dispersive fluxes using the dispersive velocities computed from LES data

uvLES_disp = np.mean((uLES_disp*vLES_disp),axis = (0,1))
uwLES_disp = np.mean((uLES_disp*wLES_disp),axis = (0,1))
vwLES_disp = np.mean((vLES_disp*wLES_disp),axis = (0,1))

fig, axes=plt.subplots(1,3, figsize=(8,4), constrained_layout=True)
axes[0].plot(uvLES_disp,z,'-k')
axes[1].plot(uwLES_disp,z,'-k')
axes[2].plot(vwLES_disp,z,'-k')

axes[0].set_ylabel(r'$z/z_i$')
axes[0].set_xlabel(r'$\tau_{xy}-Disp$'); axes[1].set_xlabel(r'$\tau_{xz}-Disp$'); axes[2].set_xlabel(r'$\tau_{yz}-Disp$');

#%% Compute the buoyancy term

u_scale = 0.45 #[m/s]
zi = 1000 #[m]
g = 9.81 #[m/s^2]
g_star = g * (zi/(u_scale**2)) #Non-dimensional g.

T = dataS.data[:,:,:,0]
DeltaT = np.zeros((Nx,Ny,Nz),'d',order='F')
DeltaT_filt = np.zeros((Nx,Ny,Nz),'d',order='F')
Tavg =  np.mean(T,axis=(0,1))    #compute the average temperature across the whole horizontal domain

sigma_y = 0
sigma_x = sigma_y

# Apply gaussian filter
sigma = [sigma_y, sigma_x]

for k in range(0,Nz):
    DeltaT[:,:,k] =  (T[:,:,k] - Tavg[k])/Tavg[k]
    DeltaT_filt[:,:,k] = sp.ndimage.gaussian_filter(DeltaT[:,:,k], sigma, mode='mirror')

Beta = -g_star*DeltaT_filt

#%%Integrate the buoyancy term to obtain an approximated pressure field

P_aprox = np.zeros((Nx,Ny,Nz),'d', order= 'F')
P_aprox_disp = np.zeros((Nx,Ny,Nz),'d', order= 'F')

P_aprox[:,:,0] = Beta[:,:,0]*(dz/2)

for k in range(1,Nz-1):
    zbottom = k-1
    ztop = k+1  
    P_aprox[:,:,k] = simps(Beta[:,:,zbottom:ztop],x=z[zbottom:ztop],axis=-1,even='first')

for k in range(0,Nz):
    P_aprox_disp[:,:,k] = P_aprox[:,:,k] - np.mean(P_aprox[:,:,k],axis=(0,1))
    
    
#%% Compute pressure field using the pressure from the LES model

pLES = data.data[:,:,:,3]   #pressure field output from LES

Txx_sgs = data.data[:,:,:,16]    #components of the sub-grid scale stress tensor
Tyy_sgs = data.data[:,:,:,17]
Tzz_sgs = data.data[:,:,:,18]

UU = data.data[:,:,:,4]   #not sure what these are, averages of average velocities
VV = data.data[:,:,:,5]
WW = data.data[:,:,:,6]

WW_interp = np.zeros((Nx,Ny,Nz),'d',order='F')     #we interpolate the vertical velocity on the u,v nodes
for k in range(0,Nz-1):
        WW_interp[:,:,k] = (WW[:,:,k] + WW[:,:,k+1])/2

p0 = pLES - (1/3)*(Txx_sgs + Tyy_sgs + Tzz_sgs) - (1/2)*(UU+VV+WW_interp)

p0_filt = np.zeros((Nx,Ny,Nz),'d',order='F')

sigma_y = 6
sigma_x = sigma_y

# Apply gaussian filter
sigma = [sigma_y, sigma_x]

for k in range(0,Nz):
    p0_filt[:,:,k] = sp.ndimage.gaussian_filter(p0[:,:,k], sigma, mode='mirror')
    

fig,ax = plt.subplots(1,3,constrained_layout=True)
im1 = ax[0].pcolormesh(x,y,np.transpose(pLES[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-10,vmax=10)
im2 = ax[1].pcolormesh(x,y,np.transpose(p0[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-20,vmax=10)
im3 = ax[2].pcolormesh(x,y,np.transpose(P_aprox[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-1,vmax=1)

cbar1 = fig.colorbar(im1, ax=ax[0], shrink=0.8,location='top',orientation='horizontal')
cbar2 = fig.colorbar(im2, ax=ax[1], shrink=0.8,location='top',orientation='horizontal')
cbar3 = fig.colorbar(im3, ax=ax[2], shrink=0.8,location='top',orientation='horizontal')

for i in range(len(ax)):
    ax[i].set_xlabel(r'$x/z_i$',fontsize=15)

ax[0].set_ylabel(r'$y/z_i$',fontsize=15)

ax[0].set_title(r'$P_{LES}$'); ax[1].set_title(r'$P_0$'); ax[2].set_title(r'$P_{0,filt}$')

plt.figure()
plt.plot(np.mean(p0,(0,1)),z,label='$<P_{Corrected}>_{xy}$')
plt.plot(np.mean(pLES,(0,1)),z,label='$<P_{LES}>_{xy}$')
plt.plot(np.mean(P_aprox,(0,1)),z,label='$<P_{aprox}>_{xy}$')
plt.title('Pressure field')
plt.legend()

#%% Compute and plot the horizontal pressure gradients

order = 1

dpdx_0 = np.zeros((Nx,Ny,Nz),'d',order='F')
dpdy_0 = np.zeros((Nx,Ny,Nz),'d',order='F')

dpdx_0 = Fddx(Nx,Ny,Nz,p0,order,0,Lx,Nx)
dpdy_0 = Fddy(Nx,Ny,Nz,p0,order,0,Ly,Ny)

# for k in range(0,Nz):
#     for j in range(0,Ny):
#         for i in range(1,Nx):
#             dpdx_0[i,j,k] = (1/dx)*(p0[i,j,k] - p0[i-1,j,k])
            
#     # breakpoint()
            
# for k in range(0,Nz):
#     for j in range(1,Ny):
#         for i in range(0,Nx):
#             dpdy_0[i,j,k] = (1/dy)*(p0[i,j,k] - p0[i,j-1,k])
            
#     # breakpoint()

# # breakpoint()
# #We use the horizontal periodicity of the domain to compute the last derivative.
# dpdx_0[0,:,:] = (1/dx)*(p0[0,:,:] - p0[-1,:,:])
# dpdy_0[:,0,:] = (1/dy)*(p0[:,0,:] - p0[:,-1,:])
# dpdx_0[0,:,:] = 0
# dpdy_0[:,0,:] = 0 

fig,ax = plt.subplots(1,2,constrained_layout=True)
im1 = ax[0].pcolormesh(x,y,np.transpose(dpdx_0[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-10,vmax=10)
im2 = ax[1].pcolormesh(x,y,np.transpose(dpdy_0[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-10,vmax=10)

cbar1 = fig.colorbar(im1, ax=ax[0], shrink=0.8,location='top',orientation='horizontal')
cbar2 = fig.colorbar(im2, ax=ax[1], shrink=0.8,location='top',orientation='horizontal')

ax[0].set_xlabel(r'$x/z_i$',fontsize=15); ax[1].set_xlabel(r'$x/z_i$',fontsize=15)
ax[0].set_ylabel(r'$y/z_i$',fontsize=15)

ax[0].set_title(r'$\frac{dP}{dx}$'); ax[1].set_title(r'$\frac{dP}{dy}$')

#%% Compute the Reynolds Stress at original grid.

Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress(Nx,Ny,Nz,data.data[:,:,:,0],data.data[:,:,:,1],data.data[:,:,:,2],data.data[:,:,:,4],\
                         data.data[:,:,:,5],data.data[:,:,:,6],data.data[:,:,:,13],data.data[:,:,:,14],data.data[:,:,:,15])

#Reynolds stresses computed previously

Txx = Rstress.data[:,:,:,0] #Erased minus sign
Tyy = Rstress.data[:,:,:,1] #Erased minus sign
Tzz = Rstress.data[:,:,:,2] #Erased minus sign
Txy = Rstress.data[:,:,:,3] #Erased minus sign
Txz = Rstress.data[:,:,:,4] #Erased minus sign
Tyz = Rstress.data[:,:,:,5] #Erased minus sign

#sub-grid scale stresses, output from the LES model

Txx_sgs = data.data[:,:,:,16]
Tyy_sgs = data.data[:,:,:,17]
Tzz_sgs = data.data[:,:,:,18]
Txy_sgs = data.data[:,:,:,19]
Txz_sgs = -data.data[:,:,:,20] #Added minus sign
Tyz_sgs = -data.data[:,:,:,21] #Added minus sign

W_interp = np.zeros((Nx,Ny,Nz),'d',order='F')
WW_interp = np.zeros((Nx,Ny,Nz),'d',order='F')
for k in range(0,Nz-1):
        WW_interp[:,:,k] = (WW[:,:,k] + WW[:,:,k+1])/2
        W_interp[:,:,k] = (wLES[:,:,k] + wLES[:,:,k+1])/2
        

#%% Integrate the momentum equations and continuity equation to retrieve dispersive velocities

nz = 50

# First of all we need to initialize the velocity fields:
#-----------------------------------------------------------------------------

# The velocity fields are initialized as a slab of three vertical levels so one can compute vertical gradients at the central level.     
u = np.zeros((Nx,Ny,nz),'d',order='F'); u_new = np.zeros((Nx,Ny,nz),'d',order='F')
# v = np.zeros((Nx,Ny,nz),'d',order='F'); v_new = np.zeros((Nx,Ny,nz),'d',order='F')
# w = np.zeros((Nx,Ny,nz),'d',order='F'); w_new = np.zeros((Nx,Ny,nz),'d',order='F')

v = vLES[:,:,0:nz]
w = wLES[:,:,0:nz]

U_interp = np.zeros((Nx,Ny,nz),'d',order='F')
V_interp = np.zeros((Nx,Ny,nz),'d',order='F')

AdvecX = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecX_old = np.zeros((Nx,Ny,nz),'d',order='F')
# AdvecY = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecY_old = np.zeros((Nx,Ny,nz),'d',order='F')

# dudxdvdy = np.zeros((Nx,Ny,nz),'d',order='F'); dudxdvdy_old = np.zeros((Nx,Ny,nz),'d',order='F')

# ustar0 = data_2D.data[:,:,0]
# phi_m = dataS_2D.data[:,:,2]
# vk = 0.41

#Beginning of the recursive loop:
TotalDiff = 10
tot_dif_vect = []
it = 0
it_vect= []

while (TotalDiff > 1e-4):
# while (it < 100):
    
    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,nz,6),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                            'Rzz','Rxy','Rxz','Ryz']})
    
    for k in range(1,nz):
            U_interp[:,:,k] = (u[:,:,k] + u[:,:,k-1])/2
            V_interp[:,:,k] = (v[:,:,k] + v[:,:,k-1])/2
    
    # dudx_new = np.zeros((Nx,Ny,nz),'d',order='F'); dvdy_new = np.zeros((Nx,Ny,nz),'d',order='F')

    #For x-momentum and y-momentum
    tmp1x = np.zeros((Nx,Ny,nz),'d',order='F'); tmp1y = np.zeros((Nx,Ny,nz),'d',order='F')
    tmp2x = np.zeros((Nx,Ny,nz),'d',order='F'); tmp2y = np.zeros((Nx,Ny,nz),'d',order='F')
    tmp3x = np.zeros((Nx,Ny,nz),'d',order='F'); tmp3y = np.zeros((Nx,Ny,nz),'d',order='F')
    tmpz = np.zeros((Nx,Ny,Nz),'d',order='F')
    
    for k in range(0,nz): #Here we dealias the  corresponding products:
        
        Upad = dealias1(Nx,Ny,u[:,:,k]) 
        Vpad = dealias1(Nx,Ny,v[:,:,k])
        Wpad = dealias1(Nx,Ny,w[:,:,k])
        
        #For x-momentum Eq:
        Uinterp_pad = dealias1(Nx,Ny,U_interp[:,:,k]) 
        
        tmp1_pad = Upad*Upad #At uvp-node
        tmp2_pad = Upad*Vpad #At uvp-node
        tmp3_pad = Uinterp_pad*Wpad #This is done at w-node. The vertical derivative puts the result in uvp-node.
        #--------------------------------
        
        #Going back in the dealiasing:
        tmp1x[:,:,k] = dealias2(Nx,Ny,tmp1_pad[:,:])
        tmp2x[:,:,k] = dealias2(Nx,Ny,tmp2_pad[:,:])
        tmp3x[:,:,k] = dealias2(Nx,Ny,tmp3_pad[:,:])
        
        #For y-momentum equation:
        Vinterp_pad = dealias1(Nx,Ny,V_interp[:,:,k])
        
        tmp1_pad = Upad*Vpad
        tmp2_pad = Vpad*Vpad
        tmp3_pad = Vinterp_pad*Wpad
        
        tmp1y[:,:,k] = dealias2(Nx,Ny,tmp1_pad[:,:])
        tmp2y[:,:,k] = dealias2(Nx,Ny,tmp2_pad[:,:])
        tmp3y[:,:,k] = dealias2(Nx,Ny,tmp3_pad[:,:])
        
        #For z component:
        tmpz_pad = Wpad*Wpad
        
        tmpz[:,:,k] = dealias2(Nx,Ny,tmpz_pad[:,:]) 
        
    order = 1
    
    Rstress = ReynoldsStress(Nx,Ny,nz,u,v,w,tmp1x,tmp2y,tmpz,tmp2x,tmp3x,tmp3y)

    #Reynolds stresses computed previously

    Txx = Rstress.data[:,:,:,0] #Erased minus sign
    Tyy = Rstress.data[:,:,:,1] #Erased minus sign
    Tzz = Rstress.data[:,:,:,2] #Erased minus sign
    Txy = Rstress.data[:,:,:,3] #Erased minus sign
    Txz = Rstress.data[:,:,:,4] #Erased minus sign
    Tyz = Rstress.data[:,:,:,5] #Erased minus sign

    #For the x-momentum equation:
    dUUdx = Fddx(Nx,Ny,nz,tmp1x,order,0,Lx,Nx)
    dUVdy = Fddy(Nx,Ny,nz,tmp2x,order,0,Ly,Ny)
    dUWdz = ddz3_w(Nx,Ny,nz,dz,tmp3x)
    
    Term1x = - (dUUdx + dUVdy + dUWdz) #non-linear advection term
    Term2x = - Fddx(Nx,Ny,nz,p0,order,0,Lx,Nx) #pressure gradient
    Term4x =  (Fddx(Nx,Ny,nz,Txx,order,0,Lx,Nx) + Fddy(Nx,Ny,nz,Txy,order,0,Ly,Ny) + ddz3_w(Nx,Ny,nz,dz,Txz)) #Reynolds stresses
    Term5x = - (Fddx(Nx,Ny,nz,Txx_sgs,order,0,Lx,Nx) + Fddy(Nx,Ny,nz,Txy_sgs,order,0,Ly,Ny) + ddz3_w(Nx,Ny,nz,dz,-Txz_sgs)) #sub-grid scale stresses
    
    #For the y-momentum Eq:
    dUVdx = Fddx(Nx,Ny,nz,tmp1y,order,0,Lx,Nx)
    dVVdy = Fddy(Nx,Ny,nz,tmp2y,order,0,Ly,Ny)
    dVWdz = ddz3_w(Nx,Ny,nz,dz,tmp3y)
    
    Term1y = - (dUVdx + dVVdy + dVWdz) #non-linear advection term
    Term2y = - Fddy(Nx,Ny,nz,p0,order,0,Ly,Ny) #pressure gradient
    Term4y =  (Fddx(Nx,Ny,nz,Txy,order,0,Lx,Nx) + Fddy(Nx,Ny,nz,Tyy,order,0,Ly,Ny) + ddz3_w(Nx,Ny,nz,dz,Tyz)) #Reynolds stresses
    Term5y = - (Fddx(Nx,Ny,nz,Txy_sgs,order,0,Lx,Nx) + Fddy(Nz,Ny,nz,Tyy_sgs,order,0,Ly,Ny) + ddz3_w(Nx,Ny,nz,dz,-Tyz_sgs)) #sub-grid scale stresses
    
    #Time advancement:
    #-----------------------------------------------------------------------------
    dt = 1e-6

    # Integrate the x, y components of the momentum equation
    for k in range(0,nz):
        AdvecX[:,:,k] = Term1x[:,:,k] + Term2x[:,:,k] + Term4x[:,:,k] + Term5x[:,:,k]
        u_new[:,:,k] = u[:,:,k] + dt*(1.5*AdvecX[:,:,k] - 0.5*AdvecX_old[:,:,k])  #We use second order Adam-Bashforth.    
        
        # AdvecY[:,:,k] = Term1y[:,:,k] + Term2y[:,:,k] + Term4y[:,:,k] + Term5y[:,:,k]
        # v_new[:,:,k] = v[:,:,k] + dt*(1.5*AdvecY[:,:,k] - 0.5*AdvecY_old[:,:,k])  #We use second order Adam-Bashforth.   
        # breakpoint()

    #Integrate conservation of mass to obtain W.
    # dudx_new = Fddx(Nx,Ny,nz,u_new,order,0,Lx,Nx)
    # dvdy_new = Fddy(Nx,Ny,nz,v_new,order,0,Ly,Ny)
    
    # dudxdvdy = - dudx_new - dvdy_new

    # w_new[:,:,0] = 0 #At the surface the vertical velocity is zero.
    
    # for k in range(0,nz-1):
    #     w_new[:,:,k+1] = w_new[:,:,k] + dz*(1.5*dudxdvdy[:,:,k] - 0.5*dudxdvdy_old[:,:,k]) 


    # Compute convergence on the computed velocity components:
    
    diff1 = np.max(np.abs((u_new - u)))   
    # diff2 = np.max(np.abs((v_new - v)))   
    # diff3 = np.max(np.abs((w_new - w)))   

    TotalDiff = np.max([diff1])   
    tot_dif_vect.append(TotalDiff)

    #Re-set the old velocity with the new velocity:

    u = np.copy(u_new); 
    # v = np.copy(v_new); 
    # w = np.copy(w_new)
    AdvecX_old = np.copy(AdvecX); 
    # AdvecY_old = np.copy(AdvecY)
    # dudxdvdy_old = np.copy(dudxdvdy)
    
    it_vect.append(it)
    it += 1 
    
    # breakpoint()

    # print(it,TotalDiff)
    print(it)

#%% Plot uLES and u to compare the two fields

h = 10

fig,ax = plt.subplots(2,3,constrained_layout=True)
im1 = ax[0,0].pcolormesh(x,y,np.transpose(uLES[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=np.min(uLES[:,:,h]),vmax=np.max(uLES[:,:,h]))
im2 = ax[0,1].pcolormesh(x,y,np.transpose(vLES[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=np.min(vLES[:,:,h]),vmax=np.max(vLES[:,:,h]))
im3 = ax[0,2].pcolormesh(x,y,np.transpose(wLES[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=np.min(wLES[:,:,h]),vmax=np.max(wLES[:,:,h]))
im4 = ax[1,0].pcolormesh(x,y,np.transpose(u[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.1,vmax=0.1)
im5 = ax[1,1].pcolormesh(x,y,np.transpose(v[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.01,vmax=0.01)
im6 = ax[1,2].pcolormesh(x,y,np.transpose(w[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.05,vmax=0.05)

cbar1 = fig.colorbar(im1,ax=ax[0,0],shrink=0.8,location='top',orientation='horizontal')
cbar2 = fig.colorbar(im2,ax=ax[0,1],shrink=0.8,location='top',orientation='horizontal')
cbar3 = fig.colorbar(im3,ax=ax[0,2],shrink=0.8,location='top',orientation='horizontal')
cbar4 = fig.colorbar(im4,ax=ax[1,0],shrink=0.8,location='top',orientation='horizontal')
cbar5 = fig.colorbar(im5,ax=ax[1,1],shrink=0.8,location='top',orientation='horizontal')
cbar6 = fig.colorbar(im6,ax=ax[1,2],shrink=0.8,location='top',orientation='horizontal')

ax[1,0].set_xlabel(r'$x/z_i$',fontsize=10)
ax[1,1].set_xlabel(r'$x/z_i$',fontsize=10)
ax[1,2].set_xlabel(r'$x/z_i$',fontsize=10)
    
ax[0,0].set_ylabel(r'$y/z_i$',fontsize=10)
ax[1,0].set_ylabel(r'$y/z_i$',fontsize=10)

ax[0,0].set_title(r'$u_{LES}$',fontsize=10)
ax[0,1].set_title(r'$v_{LES}$',fontsize=10)
ax[0,2].set_title(r'$w_{LES}$',fontsize=10)
ax[1,0].set_title(r'$u_{mod}$',fontsize=10)
ax[1,1].set_title(r'$v_{mod}$',fontsize=10)
ax[1,2].set_title(r'$w_{mod}$',fontsize=10)

#%% Plot the dispersive velocity field from the modeled velocity

u_disp = np.zeros((Nx,Ny,nz),'d',order='F')
v_disp = np.zeros((Nx,Ny,nz),'d',order='F')
w_disp = np.zeros((Nx,Ny,nz),'d',order='F')

for k in range(0,nz):
    u_disp[:,:,k] = np.squeeze(u[:,:,k])-np.mean(u[:,:,k],axis=(0,1))
    v_disp[:,:,k] = np.squeeze(v[:,:,k])-np.mean(v[:,:,k],axis=(0,1))
    w_disp[:,:,k] = np.squeeze(w[:,:,k])-np.mean(w[:,:,k],axis=(0,1))
    

fig,ax = plt.subplots(2,3,constrained_layout=True)
im1 = ax[0,0].pcolormesh(x,y,np.transpose(uLES_disp[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=np.min(uLES_disp[:,:,h]),vmax=np.max(uLES_disp[:,:,h]))
im2 = ax[0,1].pcolormesh(x,y,np.transpose(vLES_disp[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=np.min(vLES_disp[:,:,h]),vmax=np.max(vLES_disp[:,:,h]))
im3 = ax[0,2].pcolormesh(x,y,np.transpose(wLES_disp[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=np.min(wLES_disp[:,:,h]),vmax=np.max(wLES_disp[:,:,h]))
im4 = ax[1,0].pcolormesh(x,y,np.transpose(u_disp[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.1,vmax=0.1)
im5 = ax[1,1].pcolormesh(x,y,np.transpose(v_disp[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.01,vmax=0.01)
im6 = ax[1,2].pcolormesh(x,y,np.transpose(w_disp[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.01,vmax=0.01)

cbar1 = fig.colorbar(im1,ax=ax[0,0],shrink=0.8,location='top',orientation='horizontal')
cbar2 = fig.colorbar(im2,ax=ax[0,1],shrink=0.8,location='top',orientation='horizontal')
cbar3 = fig.colorbar(im3,ax=ax[0,2],shrink=0.8,location='top',orientation='horizontal')
cbar4 = fig.colorbar(im4,ax=ax[1,0],shrink=0.8,location='top',orientation='horizontal')
cbar5 = fig.colorbar(im5,ax=ax[1,1],shrink=0.8,location='top',orientation='horizontal')
cbar6 = fig.colorbar(im6,ax=ax[1,2],shrink=0.8,location='top',orientation='horizontal')

ax[1,0].set_xlabel(r'$x/z_i$',fontsize=10)
ax[1,1].set_xlabel(r'$x/z_i$',fontsize=10)
ax[1,2].set_xlabel(r'$x/z_i$',fontsize=10)
    
ax[0,0].set_ylabel(r'$y/z_i$',fontsize=10)
ax[1,0].set_ylabel(r'$y/z_i$',fontsize=10)

ax[0,0].set_title(r'$\overline{u_{LES}}-<\overline{u_{LES}}>$',fontsize=10)
ax[0,1].set_title(r'$\overline{v_{LES}}-<\overline{v_{LES}}>$',fontsize=10)
ax[0,2].set_title(r'$\overline{w_{LES}}-<\overline{w_{LES}}>$',fontsize=10)
ax[1,0].set_title(r'$\overline{u_{mod}}-<\overline{u_{mod}}>$',fontsize=10)
ax[1,1].set_title(r'$\overline{v_{mod}}-<\overline{v_{mod}}>$',fontsize=10)
ax[1,2].set_title(r'$\overline{w_{mod}}-<\overline{w_{mod}}>$',fontsize=10)

#%% Compare vertical profile of dispersive fluxes using the LES velocities and those we modeled

uv_disp = np.mean(u_disp*v_disp,axis=(0,1))
uw_disp = np.mean(u_disp*w_disp,axis=(0,1))
vw_disp = np.mean(v_disp*w_disp,axis=(0,1))

fig, axes=plt.subplots(2,3, constrained_layout=True)
axes[0,0].plot(uvLES_disp[0:nz],z[0:nz],'-k')
axes[0,1].plot(uwLES_disp[0:nz],z[0:nz],'-k')
axes[0,2].plot(vwLES_disp[0:nz],z[0:nz],'-k')
axes[1,0].plot(uv_disp,z[0:nz],'-k')
axes[1,1].plot(uw_disp,z[0:nz],'-k')
axes[1,2].plot(vw_disp,z[0:nz],'-k')

axes[0,0].set_ylabel(r'$z/z_i$')
axes[1,0].set_ylabel(r'$z/z_i$')

axes[0,0].set_xlabel(r'$\tau_{xy}-Disp$'); axes[0,1].set_xlabel(r'$\tau_{xz}-Disp$'); axes[0,2].set_xlabel(r'$\tau_{yz}-Disp$');
axes[1,0].set_xlabel(r'$\tau_{mod,xy}-Disp$'); axes[1,1].set_xlabel(r'$\tau_{mod,xz}-Disp$'); axes[1,2].set_xlabel(r'$\tau_{mod,yz}-Disp$');


#%% Compute the budget-x and plot the different components of it

Budgetx = np.zeros((Nx,Ny,nz),'d',order='F')

for k in range(0,nz):
    for i in range(0,Nx):
        for j in range(0,Ny):
            Budgetx[i,j,k] = (Term1x[i,j,k] + Term2x[i,j,k] + Term4x[i,j,k] + Term5x[i,j,k] )



plt.figure()
plt.plot(Term1x[10,10,:], z[0:nz], Term2x[10,10,:], z[0:nz], 
         Term4x[10,10,:], z[0:nz], Term5x[10,10,:], z[0:nz],
         Budgetx[10,10,:], z[0:nz])
plt.legend(('- Adv', 'DP', 'Reynolds Stress', 'SGS stress', 'Budget'))
plt.xlabel('$(Mom. Eq. Term)/(u_*^2/z_i)$')
plt.title('x-momentum')
plt.show()

# Graphical Representation of the x-momentum Budget
#---------------------------------------

#---------------------------------------------------------------------------
fig, ax=plt.subplots(2,3)

#----------------------------------------
#Subplot (3,2,1) --> Term 1
a = np.mean(Term1x,(0,1))
error = np.std(Term1x,(0,1))
ax[0,0].plot(a, z[0:nz],label="$Term \,Ix$",linewidth=1)
ax[0,0].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,2) --> Term 2
a = np.mean(Term2x,(0,1))
error = np.std(Term2x,(0,1))
ax[0,1].plot(a, z[0:nz],label="$Term \,IIx$",linewidth=1)
ax[0,1].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,3) --> Term 4
a = np.mean(Term4x,(0,1))
error = np.std(Term4x,(0,1))
ax[0,2].plot(a, z[0:nz],label="$Term \,IVx$",linewidth=1)
ax[0,2].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,4) --> Term 5
a = np.mean(Term5x,(0,1))
error = np.std(Term5x,(0,1))
ax[1,0].plot(a, z[0:nz],label="$Term \,Vx$",linewidth=1)
ax[1,0].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
# #Subplot (3,2,5) --> Term 6
# x = np.mean(Term6x,(0,1))
# error = np.std(Term6x,(0,1))
# ax[1,1].plot(x, z_uvpT,label="$Term \,VIx$",linewidth=1)
# ax[1,1].fill_betweenx(z_uvpT,x-error,x+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,6) --> Term Budget
a = np.mean(Budgetx,(0,1))
error = np.std(Budgetx,(0,1))
ax[1,2].plot(a, z[0:nz],label="$Budget x$",linewidth=1)
ax[1,2].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#ax[0,0].set_xlim((-25,25))
#ax[0,1].set_xlim((-25,25))
#ax[0,2].set_xlim((-25,25))
#ax[1,0].set_xlim((-25,25))
#ax[1,1].set_xlim((-25,25))
#ax[1,2].set_xlim((-25,25))

ax[0,0].set_xlabel('- Advection'), ax[0,1].set_xlabel('Pressure'), ax[0,2].set_xlabel('Reynolds Stress')
ax[1,0].set_xlabel('SGS'), ax[1,2].set_xlabel('Budget')

plt.tight_layout()

#%% Compute the budget-y and plot the different components of it

Budgety = np.zeros((Nx,Ny,nz),'d',order='F')

for k in range(0,nz):
    for i in range(0,Nx):
        for j in range(0,Ny):
            Budgety[i,j,k] = (Term1y[i,j,k] + Term2y[i,j,k] + Term4y[i,j,k] + Term5y[i,j,k] )



plt.figure()
plt.plot(Term1y[10,10,:], z[0:nz], Term2y[10,10,:], z[0:nz], 
         Term4y[10,10,:], z[0:nz], Term5y[10,10,:], z[0:nz],
         Budgety[10,10,:], z[0:nz])
plt.legend(('- Adv', 'DP', 'Reynolds Stress', 'SGS stress', 'Budget'))
plt.xlabel('$(Mom. Eq. Term)/(u_*^2/z_i)$')
plt.title('y-momentum')
plt.show()

# Graphical Representation of the x-momentum Budget
#---------------------------------------

#---------------------------------------------------------------------------
fig, ax=plt.subplots(2,3)

#----------------------------------------
#Subplot (3,2,1) --> Term 1
a = np.mean(Term1y,(0,1))
error = np.std(Term1y,(0,1))
ax[0,0].plot(a, z[0:nz],label="$Term \,Iy$",linewidth=1)
ax[0,0].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,2) --> Term 2
a = np.mean(Term2y,(0,1))
error = np.std(Term2y,(0,1))
ax[0,1].plot(a, z[0:nz],label="$Term \,IIy$",linewidth=1)
ax[0,1].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,3) --> Term 4
a = np.mean(Term4y,(0,1))
error = np.std(Term4y,(0,1))
ax[0,2].plot(a, z[0:nz],label="$Term \,IVy$",linewidth=1)
ax[0,2].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,4) --> Term 5
a = np.mean(Term5y,(0,1))
error = np.std(Term5y,(0,1))
ax[1,0].plot(a, z[0:nz],label="$Term \,Vy$",linewidth=1)
ax[1,0].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#----------------------------------------
# #Subplot (3,2,5) --> Term 6
# x = np.mean(Term6x,(0,1))
# error = np.std(Term6x,(0,1))
# ax[1,1].plot(x, z_uvpT,label="$Term \,VIx$",linewidth=1)
# ax[1,1].fill_betweenx(z_uvpT,x-error,x+error,alpha=0.2)

#----------------------------------------
#Subplot (3,2,6) --> Term Budget
a = np.mean(Budgety,(0,1))
error = np.std(Budgety,(0,1))
ax[1,2].plot(a, z[0:nz],label="$Budget y$",linewidth=1)
ax[1,2].fill_betweenx(z[0:nz],a-error,a+error,alpha=0.2)

#ax[0,0].set_xlim((-25,25))
#ax[0,1].set_xlim((-25,25))
#ax[0,2].set_xlim((-25,25))
#ax[1,0].set_xlim((-25,25))
#ax[1,1].set_xlim((-25,25))
#ax[1,2].set_xlim((-25,25))

ax[0,0].set_xlabel('- Advection'), ax[0,1].set_xlabel('Pressure'), ax[0,2].set_xlabel('Reynolds Stress')
ax[1,0].set_xlabel('SGS'), ax[1,2].set_xlabel('Budget')

plt.tight_layout()

























