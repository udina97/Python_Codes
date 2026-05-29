#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:06:51 2023

@author: benjamin
"""

#%% Libraries and Functions

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
# from scipy.integrate import simps,trapz
import scipy as sp
import scipy.ndimage
import random as rand

# os.chdir("/media/benjamin/b4648b0d-026f-4f15-9ffe-3a41a55386ef/PhD/Research/Python_Codes/")

# from Stats import  ReynoldsStress, ReynoldsFlux
# from Analysis import Fddx, Fddy, ddz3_w, ddz3_uv, dealias1, dealias2

#%% Output data from the LES model

os.chdir("/scratch/general/nfs1/u1450851/research/caseUg1")

data = xr.open_dataarray('Data_Momentum.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
dataS = xr.open_dataarray('Data_Scalar.nc')
SurfaceTemp = xr.open_dataarray('SurfaceTempD800v3.nc') 
data_2D = xr.open_dataarray('Data_Momentum_2D.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
dataS_2D = xr.open_dataarray('Data_Scalar_2D.nc')

#%% Define main parameters 

NumVariables = 26
NumVariablesSC = 10

Nx = 256   #nodes in x-dir
Ny = 256   #nodes in y-dir
Nz = 256   #nodes in z-dir
Lx = 2*np.pi    #domain size x-dir
Ly = 2*np.pi    #domain size y-dir
Lz = 2          #domain size z-dir
dx = Lx/Nx    #grid cell size x-dir
dy = Ly/Ny    #grid cell size y-dir
dz = Lz/Nz    #grid cell size z-dir
z_i = 1    #inversion height
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
    
uvLESdisp = np.zeros((Nz), 'd', order='F')
for k in range(0,Nz):
    uvLESdisp[k] = np.mean(uLES_disp[:,:,k]*wLES_disp[:,:,k],axis=(0,1))
# #Plotting Results: ----------------------
# h = 3

# fig, axes=plt.subplots(1,3, figsize=(8,4), constrained_layout=True)
# im1 = axes[0].pcolormesh(x,y,np.transpose(uLES_disp[:,:,h]),shading='gouraud', cmap='coolwarm',vmin=-5,vmax = 5)
# im2 = axes[1].pcolormesh(x,y,np.transpose(vLES_disp[:,:,h]),shading='gouraud', cmap='coolwarm',vmin=-5,vmax = 5)
# im3 = axes[2].pcolormesh(x,y,np.transpose(wLES_disp[:,:,h]),shading='gouraud', cmap='coolwarm',vmin=-2,vmax = 2)

# cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8,location='top',orientation='horizontal')
# cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8,location='top',orientation='horizontal')
# cbar3 = fig.colorbar(im3, ax=axes[2], shrink=0.8,location='top',orientation='horizontal')

# axes[0].set_ylabel(r'$y/z_i$')
# axes[0].set_xlabel(r'$x/z_i$'); axes[1].set_xlabel(r'$x/z_i$'); axes[2].set_xlabel(r'$x/z_i$');

# axes[0].set_title(r'$\overline{u}- <\overline{u}>$')
# axes[1].set_title(r'$\overline{v}- <\overline{v}>$')
# axes[2].set_title(r'$\overline{w}- <\overline{w}>$')

# fig.suptitle(r'H = %i' %h)
    
#%% Compute the dispersive pressure field using the pressure from the LES model

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

p0_disp = np.zeros((Nx,Ny,Nz),'d',order='F')    #dispersive pressure

for k in range(0,Nz):
    p0_disp[:,:,k] = p0[:,:,k] - np.mean(p0[:,:,k],axis=(0,1))    #no squeeze?
    #p0_disp[:,:,k] = np.squeeze(p0[:,:,k]) - np.mean(p0[:,:,k],axis=(0,1))
    
#%% Filtering the pressure field to remove sharp edges:

p0_disp_filt = np.zeros((Nx,Ny,Nz),'d',order='F')

sigma_y = 6
sigma_x = sigma_y

# Apply gaussian filter
sigma = [sigma_y, sigma_x]

for k in range(0,Nz):
    p0_disp_filt[:,:,k] = sp.ndimage.gaussian_filter(p0_disp[:,:,k], sigma, mode='mirror')

# breakpoint()

#%%
# fig,ax=plt.subplots(1,3,figsize=(10,5))
# ax[0].plot(x,p0[0,:,3], label='les pressure')
# ax[1].plot(x,p0_disp[0,:,3], label='dispersive pressure')
# ax[2].plot(x,p0_disp_filt[0,:,3], label='filtered dispersive pressure')
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
#%% Compute and plot the horizontal pressure gradients

dpdx_0 = np.zeros((Nx,Ny,Nz),'d',order='F')
dpdy_0 = np.zeros((Nx,Ny,Nz),'d',order='F')

for k in range(0,Nz):
    for j in range(0,Ny):
        for i in range(1,Nx):
            dpdx_0[i,j,k] = (1/dx)*(p0_disp_filt[i,j,k] - p0_disp_filt[i-1,j,k])
            
    # breakpoint()
            
for k in range(0,Nz):
    for j in range(1,Ny):
        for i in range(0,Nx):
            dpdy_0[i,j,k] = (1/dy)*(p0_disp_filt[i,j,k] - p0_disp_filt[i,j-1,k])
            
    # breakpoint()

# breakpoint()
#We use the horizontal periodicity of the domain to compute the last derivative.
# dpdx_0[0,:,:] = (1/dx)*(p0_disp_filt[0,:,:] - p0_disp_filt[-1,:,:])
# dpdy_0[:,0,:] = (1/dy)*(p0_disp_filt[:,0,:] - p0_disp_filt[:,-1,:])
dpdx_0[0,:,:] = 0
dpdy_0[:,0,:] = 0 

# breakpoint()

#%%Integrate the momentum equations and continuity equation to retrieve dispersive velocities

nz = Nz
    
# First of all we need to initialize the velocity fields:
#-----------------------------------------------------------------------------

# The velocity fields are initialized as a slab of three vertical levels so one can compute vertical gradients at the central level.     
u = np.zeros((Nx,Ny,nz),'d',order='F'); u_new = np.zeros((Nx,Ny,nz),'d',order='F')
v = np.zeros((Nx,Ny,nz),'d',order='F'); v_new = np.zeros((Nx,Ny,nz),'d',order='F')
w = np.zeros((Nx,Ny,nz),'d',order='F'); w_new = np.zeros((Nx,Ny,nz),'d',order='F')

ustar0 = data_2D.data[:,:,0]
phi_m = dataS_2D.data[:,:,2]
vk = 0.41

# vect_vel_1 = []; vect_vel_2 = []; vect_vel_3 = []

vect_it = []
tot_dif = []

# r_x_1 = rand.randrange(0,Nx); r_x_2 = rand.randrange(0,Nx); r_x_3 = rand.randrange(0,Nx)
# r_y_1 = rand.randrange(0,Ny); r_y_2 = rand.randrange(0,Ny); r_y_3 = rand.randrange(0,Ny)

#Beginning of the recursive loop:
TotalDiff = 10
it = 0

# while (TotalDiff > 1e-4):
while (it < 500):

    #A) Advection Terms:    

    # Computing the horizontal derivatives in finite differences:
    #-----------------------------------------------------------------------------
    
    dudx = np.zeros((Nx,Ny,nz),'d',order='F'); dvdx = np.zeros((Nx,Ny,nz),'d',order='F')
    dudy = np.zeros((Nx,Ny,nz),'d',order='F'); dvdy = np.zeros((Nx,Ny,nz),'d',order='F')
    dudz = np.zeros((Nx,Ny,nz),'d',order='F'); dvdz = np.zeros((Nx,Ny,nz),'d',order='F')
        
    dudx_new = np.zeros((Nx,Ny,nz),'d',order='F')
    dvdy_new = np.zeros((Nx,Ny,nz),'d',order='F')
        
    AdvecX = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecX_old = np.zeros((Nx,Ny,nz),'d',order='F')
    AdvecY = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecY_old = np.zeros((Nx,Ny,nz),'d',order='F')
           
    d2udx2 = np.zeros((Nx,Ny,nz),'d',order='F'); d2vdx2 = np.zeros((Nx,Ny,nz),'d',order='F')
    d2udy2 = np.zeros((Nx,Ny,nz),'d',order='F'); d2vdy2 = np.zeros((Nx,Ny,nz),'d',order='F')
    d2udz2 = np.zeros((Nx,Ny,nz),'d',order='F'); d2vdz2 = np.zeros((Nx,Ny,nz),'d',order='F')
              
    ViscousTerm_x = np.zeros((Nx,Ny,nz),'d',order='F')
    ViscousTerm_y = np.zeros((Nx,Ny,nz),'d',order='F')

    for k in range(0,nz):
        for j in range(0,Ny):
            for i in range(1,Nx):
                dudx[i,j,k] = (1/dx)*(u[i,j,k]-u[i-1,j,k])
                dvdx[i,j,k] = (1/dx)*(v[i,j,k]-v[i-1,j,k])
                
        dudx[0,:,k] = (1/dx)*(u[0,:,k] - u[-1,:,k])
        dvdx[0,:,k] = (1/dx)*(v[0,:,k] - v[-1,:,k]) 
                
    for k in range(0,nz):
        for j in range(1,Ny):
            for i in range(0,Nx):
                dudy[i,j,k] = (1/dy)*(u[i,j,k]-u[i,j-1,k])
                dvdy[i,j,k] = (1/dy)*(v[i,j,k]-v[i,j-1,k])
    
        dudy[:,0,k] = (1/dy)*(u[:,0,k] - u[:,-1,k])
        dvdy[:,0,k] = (1/dy)*(v[:,0,k] - v[:,-1,k])
        
        # breakpoint()

    #Vertical derivatives:
    dudz[:,:,0] = phi_m
    dvdz[:,:,0] = phi_m
    
    for k in range(1,nz):
        dudz[:,:,k] = (1/dz)*(u[:,:,k] -  u[:,:,k-1])
        dvdz[:,:,k] = (1/dz)*(v[:,:,k] -  v[:,:,k-1])
        
        # breakpoint()
                             
    # breakpoint()
    
    #B) Fictius Viscosity
    
    for k in range(0,nz):
        for j in range(0,Ny-1):
            for i in range(1,Nx-1):
                d2udx2[i,j,k] = (1/(dx**2))*(u[i+1,j,k] -(2*u[i,j,k]) +u[i-1,j,k])
                d2vdx2[i,j,k] = (1/(dx**2))*(v[i+1,j,k] -(2*v[i,j,k]) +v[i-1,j,k])
                
        d2udx2[0,:,k] = (1/(dx**2))*(u[1,:,k] -(2*u[0,:,k]) + u[-1,:,k])
        d2vdx2[0,:,k] = (1/(dx**2))*(v[1,:,k] -(2*v[0,:,k]) + v[-1,:,k])
        
        d2udx2[-1,:,k] = (1/(dx**2))*(u[0,:,k] -(2*u[-1,:,k]) + u[-2,:,k])
        d2vdx2[-1,:,k] = (1/(dx**2))*(v[0,:,k] -(2*v[-1,:,k]) + v[-2,:,k])
                
    for k in range(0,nz):
        for j in range(1,Ny-1):
              for i in range(0,Nx-1):
                  d2udy2[i,j,k] = (1/(dy**2))*(u[i,j+1,k] -(2*u[i,j,k]) +u[i,j-1,k])
                  d2vdy2[i,j,k] = (1/(dy**2))*(v[i,j+1,k] -(2*v[i,j,k]) +v[i,j-1,k])
                
        #We use the horizontal periodicity of the domain to compute the last derivatives on Nx, and Ny.
        d2udy2[:,0,k] = (1/(dy**2))*(u[:,1,k] -(2*u[:,0,k]) + u[:,-1,k])
        d2vdy2[:,0,k] = (1/(dy**2))*(v[:,1,k] -(2*v[:,0,k]) + v[:,-1,k])
        
        d2udy2[:,-1,k] = (1/(dy**2))*(u[:,0,k] -(2*u[:,-1,k]) + u[:,-2,k])
        d2vdy2[:,-1,k] = (1/(dy**2))*(v[:,0,k] -(2*v[:,-1,k]) + v[:,-2,k])
    
    # breakpoint()
    #Vertical derivatives:
        
    for k in range(1,nz-1):
        for j in range(0,Ny):
            for i in range(0,Nx):
                d2udz2[i,j,k] = (1/(dz**2))*(u[i,j,k+1] -(2*u[i,j,k]) +u[i,j,k-1])
                d2vdz2[i,j,k] = (1/(dz**2))*(v[i,j,k+1] -(2*v[i,j,k]) +v[i,j,k-1])
    # d2udz2 = 0
    # d2vdz2 = 0
    nu = 1e-2
    
    ViscousTerm_x = nu*(d2udx2 + d2udy2 + d2udz2)
    ViscousTerm_y = nu*(d2vdx2 + d2vdy2 + d2vdz2)
    
    # ViscousTerm_x= ViscousTerm_y= 1e-5

    #Time advancement:
    #-----------------------------------------------------------------------------
    dt = 1e-6
    
    #Interpolate the product w*dudz and w*dvdz to the (u,v,p,T) nodes    
    wdudz_interp = np.zeros((Nx,Ny,nz), 'd',order='F')
    wdvdz_interp = np.zeros((Nx,Ny,nz), 'd',order='F')
    
    for k in range(0,nz-1):
        wdudz_interp[:,:,k] = (w[:,:,k]*dudz[:,:,k] + w[:,:,k+1]*dudz[:,:,k+1])/2
        wdvdz_interp[:,:,k] = (w[:,:,k]*dvdz[:,:,k] + w[:,:,k+1]*dvdz[:,:,k+1])/2
        
    # breakpoint()

    #Integrate the x, y components of the momentum equation
    for k in range(0,nz-1):
        AdvecX[:,:,k] = - u[:,:,k]*dudx[:,:,k] - v[:,:,k]*dudy[:,:,k] - wdudz_interp[:,:,k] - dpdx_0[:,:,k] + ViscousTerm_x[:,:,k]
        u_new[:,:,k] = u[:,:,k] + dt*(1.5*AdvecX[:,:,k] - 0.5*AdvecX_old[:,:,k])  #We use second order Adam-Bashforth.    
        
        AdvecY[:,:,k] = - u[:,:,k]*dvdx[:,:,k] - v[:,:,k]*dvdy[:,:,k] - wdvdz_interp[:,:,k] - dpdy_0[:,:,k] + ViscousTerm_y[:,:,k]
        v_new[:,:,k] = v[:,:,k] + dt*(1.5*AdvecY[:,:,k] - 0.5*AdvecY_old[:,:,k])  #We use second order Adam-Bashforth.   
        # breakpoint()

    #Integrate conservation of mass to obtain W.
    for k in range(0,nz-1):
        for j in range(0,Ny):
            for i in range(1,Nx):
                dudx_new[i,j,k] = (1/dx)*(u_new[i,j,k] - u_new[i-1,j,k])
                
    for k in range(0,nz-1):
        for j in range(1,Ny):
            for i in range(0,Nx):
                dvdy_new[i,j,k] = (1/dy)*(v_new[i,j,k] - v_new[i,j-1,k])
   
    dudx_new[0,:,:] = (1/dx)*(u_new[0,:,:] - u_new[-1,:,:])
    dvdy_new[:,0,:] = (1/dy)*(v_new[:,0,:] - v_new[:,-1,:])

    w_new[:,:,0] = 0 #At the surface the vertical velocity is zero.
    
    for k in range(1,nz-1):
        w_new[:,:,k] = w_new[:,:,k-1] - dz*(dudx_new[:,:,k-1] + dvdy_new[:,:,k-1]) 


    # Compute convergence on the computed velocity components:
    
    # max_val = 0
    
    diff1 = np.max(np.abs((u_new - u)))   
    diff2 = np.max(np.abs((v_new - v)))   
    diff3 = np.max(np.abs((w_new - w)))   
    # diff3 = np.zeros((Nx,Ny,nz), 'd',order='F')
    # for k in range(0,nz):
    #     for j in range(0,Ny):
    #         for i in range(0,Nx):
    #             diff3[i,j,k] = w_new[i,j,k] - w[i,j,k]
    #             if np.abs(diff3[i,j,k]) > max_val:
    #                 max_val=diff3[i,j,k]
    #                 a=[i,j,k]

    TotalDiff = np.max([diff1, diff2, diff3])   

    #Re-set the old velocity with the new velocity:

    u = np.copy(u_new); v = np.copy(v_new); w = np.copy(w_new)
    AdvecX_old = np.copy(AdvecX); AdvecY_old = np.copy(AdvecY)
    
    tot_dif.append(TotalDiff)
    vect_it.append(it)
    # vect_vel_1.append(u_new[r_x_1,r_y_1,3])
    # vect_vel_2.append(v_new[r_x_2,r_y_2,3])
    # vect_vel_3.append(w_new[r_x_3,r_y_3,3])
    
    it += 1 
    
    # breakpoint()

    # print(it,TotalDiff)
    # print(it)

#%%
uv_model = np.zeros((nz), 'd', order='F')
for k in range(0,nz):
    uv_model[k]=np.mean(u[:,:,k]*w[:,:,k],axis=(0,1))
    
#%%Plot the resulting approximated dispersive velocities

h = 3

fig, axes=plt.subplots(1,3, figsize=(8,4), constrained_layout=True)
im1 = axes[0].pcolormesh(x,y,np.transpose(u[:,:,h]),shading='gouraud', cmap='coolwarm')#,vmin=-0.1,vmax = 0.1
im2 = axes[1].pcolormesh(x,y,np.transpose(v[:,:,h]),shading='gouraud', cmap='coolwarm')#,vmin=-0.1,vmax = 0.1
im3 = axes[2].pcolormesh(x,y,np.transpose(w[:,:,h]),shading='gouraud', cmap='coolwarm')#,vmin=-0.001,vmax = 0.001

im1.set_clim(vmin=-0.0005, vmax=0.0005)
im2.set_clim(vmin=-0.0005, vmax=0.0005)
im3.set_clim(vmin=-0.0001, vmax=0.0001)

cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8,location='top',orientation='horizontal')
cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8,location='top',orientation='horizontal')
cbar3 = fig.colorbar(im3, ax=axes[2], shrink=0.8,location='top',orientation='horizontal')

axes[0].set_ylabel('$y/z_{i}$')
axes[0].set_xlabel('$x/z_{i}$'); axes[1].set_xlabel('$x/z_{i}$'); axes[2].set_xlabel('$x/z_{i}$');

axes[0].set_title('$u_{Ples}$'); axes[1].set_title('$v_{Ples}$');axes[2].set_title('$w_{Ples}$')
    
fig.suptitle(r'Sigma = %i' %sigma_y + ' -- H = %i' %h + ' -- it = %i' %it)

#%% Surface temperature

fig, ax = plt.subplots()
im1=ax.pcolormesh(x, y, SurfaceTemp, shading='gouraud', cmap='coolwarm')
cbar1 = fig.colorbar(im1, ax=ax, shrink=0.7,location='top',orientation='horizontal', pad=0.1)
im1.set_clim(vmin=280, vmax=300)
ax.set_xlabel(r'$x/z_i$')
ax.set_ylabel(r'$y/z_i$')
ax.set_title(r'$SurfaceTemp$')
    
#%% Velocity time series

diff = np.zeros(len(vect_it)-1)

for i in range(0,len(vect_it)-1):
    diff[i] = vect_vel_3[i+1]-vect_vel_3[i]

fig,ax = plt.subplots()

ax.scatter(vect_it[0:len(vect_it)-1],diff,1,'k',label='U')
# # plt.semilogy(vect_it,vect_vel_2,'-m',label='V')
# # plt.semilogy(vect_it,vect_vel_3,'-y',label='W')

ax.set_xlabel(r'Iteration Step')
ax.set_ylabel(r'U[i+1] - U[i]')
# ax.set_yscale('log')
# ax.set_xscale('log')

# plt.legend()