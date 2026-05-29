#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:42:15 2023

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
import pandas as pd

os.chdir("/scratch/general/nfs1/u1450851/research/python_codes/")


from Stats import  ReynoldsStress, ReynoldsFlux
from Analysis import Fddx, Fddy, ddz3_w, ddz3_uv, dealias1, dealias2


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

#%% Compute and plot the horizontal pressure gradients

dpdx_0 = np.zeros((Nx,Ny,Nz),'d',order='F')
dpdy_0 = np.zeros((Nx,Ny,Nz),'d',order='F')

for k in range(0,Nz):
    for j in range(0,Ny):
        for i in range(1,Nx):
            dpdx_0[i,j,k] = (1/dx)*(p0[i,j,k] - p0[i-1,j,k])
            
    # breakpoint()
            
for k in range(0,Nz):
    for j in range(1,Ny):
        for i in range(0,Nx):
            dpdy_0[i,j,k] = (1/dy)*(p0[i,j,k] - p0[i,j-1,k])
            
    # breakpoint()

# breakpoint()
#We use the horizontal periodicity of the domain to compute the last derivative.
dpdx_0[0,:,:] = (1/dx)*(p0[0,:,:] - p0[-1,:,:])
dpdy_0[:,0,:] = (1/dy)*(p0[:,0,:] - p0[:,-1,:])
# dpdx_0[0,:,:] = 0
# dpdy_0[:,0,:] = 0 

# breakpoint()

#%% Solve the momentum equations and continuity equation usign data from LES to evaluate relevance of neglected terms

neglect_x=np.zeros((Nx,Ny,Nz),'d',order='F'); neglect_y=np.zeros((Nx,Ny,Nz),'d',order='F'); neglect_z=np.zeros((Nx,Ny,Nz),'d',order='F')
wdudzLES_interp=np.zeros((Nx,Ny,Nz),'d',order='F'); wdvdzLES_interp=np.zeros((Nx,Ny,Nz),'d',order='F')
dudzLES=np.zeros((Nx,Ny,Nz),'d',order='F'); dvdzLES=np.zeros((Nx,Ny,Nz),'d',order='F')
phi_m = dataS_2D.data[:,:,2]

dudzLES[:,:,0] = phi_m
dvdzLES[:,:,0] = phi_m

for k in range(1,Nz):
    dudzLES[:,:,k] = (1/dz)*(uLES[:,:,k] -  uLES[:,:,k-1])
    dvdzLES[:,:,k] = (1/dz)*(vLES[:,:,k] -  vLES[:,:,k-1])
    
for k in range(0,Nz-1):
    wdudzLES_interp[:,:,k] = (wLES[:,:,k]*dudzLES[:,:,k] + wLES[:,:,k+1]*dudzLES[:,:,k+1])/2
    wdvdzLES_interp[:,:,k] = (wLES[:,:,k]*dvdzLES[:,:,k] + wLES[:,:,k+1]*dvdzLES[:,:,k+1])/2
    

for k in range(0,Nz):
    for j in range(0,Ny):
        for i in range(0,Nx):
            neglect_x[i,j,k] = uLES[i,j,k]*(1/dx)*(uLES[i,j,k]-uLES[i-1,j,k]) + vLES[i,j,k]*(1/dy)*(uLES[i,j,k]-uLES[i,j-1,k]) + wdudzLES_interp[i,j,k] + dpdx_0[i,j,k]

for k in range(0,Nz):
    for j in range(0,Ny):
        for i in range(0,Nx):
            neglect_y[i,j,k] = uLES[i,j,k]*(1/dx)*(vLES[i,j,k]-vLES[i-1,j,k]) + vLES[i,j,k]*(1/dy)*(vLES[i,j,k]-vLES[i,j-1,k]) + wdvdzLES_interp[i,j,k] + dpdy_0[i,j,k]

for k in range(1,Nz):
    for j in range(0,Ny):
        for i in range(0,Nx):
            neglect_z[i,j,k]=(1/dx)*(uLES[i,j,k]-uLES[i-1,j,k]) + (1/dy)*(vLES[i,j,k]-vLES[i,j-1,k]) + (1/dz)*(wLES[i,j,k]-wLES[i,j,k-1])

#%% Integrate the momentum equations and continuity equation to retrieve dispersive velocities

nz = 50

# First of all we need to initialize the velocity fields:
#-----------------------------------------------------------------------------

# The velocity fields are initialized as a slab of three vertical levels so one can compute vertical gradients at the central level.     
u = np.zeros((Nx,Ny,nz),'d',order='F'); u_new = np.zeros((Nx,Ny,nz),'d',order='F')
# v = np.zeros((Nx,Ny,nz),'d',order='F'); v_new = np.zeros((Nx,Ny,nz),'d',order='F')
# w = np.zeros((Nx,Ny,nz),'d',order='F'); w_new = np.zeros((Nx,Ny,nz),'d',order='F')

AdvecX = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecX_old = np.zeros((Nx,Ny,nz),'d',order='F')
# AdvecY = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecY_old = np.zeros((Nx,Ny,nz),'d',order='F')

# ustar0 = data_2D.data[:,:,0]
phi_m = dataS_2D.data[:,:,2]
# vk = 0.41

#Beginning of the recursive loop:
TotalDiff = 10
tot_dif_vect = []
it = 0
it_vect= []

while (TotalDiff > 1e-4):
# while (it < 500):

    #A) Advection Terms:    

    # Computing the horizontal derivatives in finite differences:
    #-----------------------------------------------------------------------------
    
    dudx = np.zeros((Nx,Ny,nz),'d',order='F'); dvdx = np.zeros((Nx,Ny,nz),'d',order='F')
    dudy = np.zeros((Nx,Ny,nz),'d',order='F'); dvdy = np.zeros((Nx,Ny,nz),'d',order='F')
    dudz = np.zeros((Nx,Ny,nz),'d',order='F'); dvdz = np.zeros((Nx,Ny,nz),'d',order='F')
        
    dudx_new = np.zeros((Nx,Ny,nz),'d',order='F')
    dvdy_new = np.zeros((Nx,Ny,nz),'d',order='F')
        
    # AdvecX = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecX_old = np.zeros((Nx,Ny,nz),'d',order='F')
    # AdvecY = np.zeros((Nx,Ny,nz),'d',order='F'); AdvecY_old = np.zeros((Nx,Ny,nz),'d',order='F')
           
    d2udx2 = np.zeros((Nx,Ny,nz),'d',order='F'); d2vdx2 = np.zeros((Nx,Ny,nz),'d',order='F')
    d2udy2 = np.zeros((Nx,Ny,nz),'d',order='F'); d2vdy2 = np.zeros((Nx,Ny,nz),'d',order='F')
    d2udz2 = np.zeros((Nx,Ny,nz),'d',order='F'); d2vdz2 = np.zeros((Nx,Ny,nz),'d',order='F')
              
    ViscousTerm_x = np.zeros((Nx,Ny,nz),'d',order='F')
    ViscousTerm_y = np.zeros((Nx,Ny,nz),'d',order='F')

    for k in range(0,nz):
        for j in range(0,Ny):
            for i in range(1,Nx):
                dudx[i,j,k] = (1/dx)*(u[i,j,k]-u[i-1,j,k])
                dvdx[i,j,k] = (1/dx)*(vLES[i,j,k]-vLES[i-1,j,k])
                
        dudx[0,:,k] = (1/dx)*(u[0,:,k] - u[-1,:,k])
        dvdx[0,:,k] = (1/dx)*(vLES[0,:,k] - vLES[-1,:,k]) 
                
    for k in range(0,nz):
        for j in range(1,Ny):
            for i in range(0,Nx):
                dudy[i,j,k] = (1/dy)*(u[i,j,k]-u[i,j-1,k])
                dvdy[i,j,k] = (1/dy)*(vLES[i,j,k]-vLES[i,j-1,k])
    
        dudy[:,0,k] = (1/dy)*(u[:,0,k] - u[:,-1,k])
        dvdy[:,0,k] = (1/dy)*(vLES[:,0,k] - vLES[:,-1,k])
        
        # breakpoint()

    #Vertical derivatives:
    dudz[:,:,0] = phi_m
    dvdz[:,:,0] = phi_m
    
    for k in range(1,nz):
        dudz[:,:,k] = (1/dz)*(u[:,:,k] -  u[:,:,k-1])
        dvdz[:,:,k] = (1/dz)*(vLES[:,:,k] -  vLES[:,:,k-1])
        
        # breakpoint()
                             
    # breakpoint()
    
    #B) Fictius Viscosity
    
    for k in range(0,nz):
        for j in range(0,Ny-1):
            for i in range(1,Nx-1):
                d2udx2[i,j,k] = (1/(dx**2))*(u[i+1,j,k] -(2*u[i,j,k]) +u[i-1,j,k])
                d2vdx2[i,j,k] = (1/(dx**2))*(vLES[i+1,j,k] -(2*vLES[i,j,k]) +vLES[i-1,j,k])
                
        d2udx2[0,:,k] = (1/(dx**2))*(u[1,:,k] -(2*u[0,:,k]) + u[-1,:,k])
        d2vdx2[0,:,k] = (1/(dx**2))*(vLES[1,:,k] -(2*vLES[0,:,k]) + vLES[-1,:,k])
        
        d2udx2[-1,:,k] = (1/(dx**2))*(u[0,:,k] -(2*u[-1,:,k]) + u[-2,:,k])
        d2vdx2[-1,:,k] = (1/(dx**2))*(vLES[0,:,k] -(2*vLES[-1,:,k]) + vLES[-2,:,k])
                
    for k in range(0,nz):
        for j in range(1,Ny-1):
              for i in range(0,Nx-1):
                  d2udy2[i,j,k] = (1/(dy**2))*(u[i,j+1,k] -(2*u[i,j,k]) +u[i,j-1,k])
                  d2vdy2[i,j,k] = (1/(dy**2))*(vLES[i,j+1,k] -(2*vLES[i,j,k]) +vLES[i,j-1,k])
                
        #We use the horizontal periodicity of the domain to compute the last derivatives on Nx, and Ny.
        d2udy2[:,0,k] = (1/(dy**2))*(u[:,1,k] -(2*u[:,0,k]) + u[:,-1,k])
        d2vdy2[:,0,k] = (1/(dy**2))*(vLES[:,1,k] -(2*vLES[:,0,k]) + vLES[:,-1,k])
        
        d2udy2[:,-1,k] = (1/(dy**2))*(u[:,0,k] -(2*u[:,-1,k]) + u[:,-2,k])
        d2vdy2[:,-1,k] = (1/(dy**2))*(vLES[:,0,k] -(2*vLES[:,-1,k]) + vLES[:,-2,k])
    
    # breakpoint()
    #Vertical derivatives:
        
    for k in range(1,nz-1):
        for j in range(0,Ny):
            for i in range(0,Nx):
                d2udz2[i,j,k] = (1/(dz**2))*(u[i,j,k+1] -(2*u[i,j,k]) +u[i,j,k-1])
                d2vdz2[i,j,k] = (1/(dz**2))*(vLES[i,j,k+1] -(2*vLES[i,j,k]) +vLES[i,j,k-1])
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
        wdudz_interp[:,:,k] = (wLES[:,:,k]*dudz[:,:,k] + wLES[:,:,k+1]*dudz[:,:,k+1])/2
        wdvdz_interp[:,:,k] = (wLES[:,:,k]*dvdz[:,:,k] + wLES[:,:,k+1]*dvdz[:,:,k+1])/2
        
    # breakpoint()

    # Integrate the x, y components of the momentum equation
    for k in range(0,nz-1):
        AdvecX[:,:,k] = - u[:,:,k]*dudx[:,:,k] - vLES[:,:,k]*dudy[:,:,k] - wdudz_interp[:,:,k] - dpdx_0[:,:,k] + ViscousTerm_x[:,:,k]
        u_new[:,:,k] = u[:,:,k] + dt*(1.5*AdvecX[:,:,k] - 0.5*AdvecX_old[:,:,k])  #We use second order Adam-Bashforth.    
        
        # AdvecY[:,:,k] = - uLES[:,:,k]*dvdx[:,:,k] - v[:,:,k]*dvdy[:,:,k] - wdvdz_interp[:,:,k] - dpdy_0[:,:,k] + ViscousTerm_y[:,:,k]
        # v_new[:,:,k] = v[:,:,k] + dt*(1.5*AdvecY[:,:,k] - 0.5*AdvecY_old[:,:,k])  #We use second order Adam-Bashforth.   
        # breakpoint()

    # #Integrate conservation of mass to obtain W.
    # for k in range(0,nz-1):
    #     for j in range(0,Ny):
    #         for i in range(1,Nx):
    #             dudx_new[i,j,k] = (1/dx)*(u_new[i,j,k] - u_new[i-1,j,k])
                
    # for k in range(0,nz-1):
    #     for j in range(1,Ny):
    #         for i in range(0,Nx):
    #             dvdy_new[i,j,k] = (1/dy)*(v_new[i,j,k] - v_new[i,j-1,k])
   
    # dudx_new[0,:,:] = (1/dx)*(u_new[0,:,:] - u_new[-1,:,:])
    # dvdy_new[:,0,:] = (1/dy)*(v_new[:,0,:] - v_new[:,-1,:])

    # w_new[:,:,0] = 0 #At the surface the vertical velocity is zero.
    
    # for k in range(1,nz-1):
    #     w_new[:,:,k] = w_new[:,:,k-1] - dz*(dudx[:,:,k-1] + dvdy[:,:,k-1]) 


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
im5 = ax[1,1].pcolormesh(x,y,np.transpose(vLES[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.1,vmax=0.1)
im6 = ax[1,2].pcolormesh(x,y,np.transpose(wLES[:,:,h]),shading='gouraud',cmap='coolwarm',vmin=-0.5,vmax=0.5)

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
    v_disp[:,:,k] = np.squeeze(vLES[:,:,k])-np.mean(vLES[:,:,k],axis=(0,1))
    w_disp[:,:,k] = np.squeeze(wLES[:,:,k])-np.mean(wLES[:,:,k],axis=(0,1))
    

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
