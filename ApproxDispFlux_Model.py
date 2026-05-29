#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 08:37:14 2023

@author: benjamin
"""

#%%Libraries and Functions

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

os.chdir("/scratch/general/nfs1/u1450851/research/python_codes")

from Stats import  ReynoldsStress, ReynoldsFlux
from Analysis import Fddx, Fddy, ddz3_w, ddz3_uv, dealias1, dealias2

os.chdir("/scratch/general/nfs1/u1450851/research/caseUg1")

data = xr.open_dataarray('Data_Momentum.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
dataS = xr.open_dataarray('Data_Scalar.nc')

#%%Define general parameters

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
y = np.arange(0,Ny)*dy
z = np.arange(0,Nz)*dz

#%%Compute the buoyancy term

u_scale = 0.45 #[m/s]
zi = 1000 #[m]
g = 9.81 #[m/s^2]
g_star = g * (zi/(u_scale**2)) #Non-dimensional g.

T = dataS.data[:,:,:,0]
DeltaT = np.zeros((Nx,Ny,Nz),'d',order='F')
Tavg =  np.mean(T,axis=(0,1))

for k in range(0,Nz):
    DeltaT[:,:,k] =  (T[:,:,k] - Tavg[k])/Tavg[k]

Beta = -g_star*DeltaT

#%%Integrate the buoyancy term to obtain an approximated pressure field

from scipy.integrate import simps,trapz

P_aprox = np.zeros((Nx,Ny,Nz),'d', order= 'F')
P_aprox_disp = np.zeros((Nx,Ny,Nz),'d', order= 'F')

P_aprox[:,:,0] = Beta[:,:,0]*dz

for k in range(1,Nz-1):
    zbottom = k
    ztop = k+1  
    P_aprox[:,:,k] = simps(Beta[:,:,zbottom:ztop],x=z[zbottom:ztop],axis=2,even='first') + P_aprox[:,:,0]
    #P_aprox[:,:,k] = trapz(Beta[:,:,zbottom:ztop],x=z[zbottom:ztop],axis=2) + P_aprox[:,:,k-1]

for k in range(0,Nz):
    P_aprox_disp[:,:,k] = P_aprox[:,:,k] - np.mean(P_aprox,axis=(0,1))

#%%Filter the approximated pressure field to remove sharp edges

import scipy as sp
import scipy.ndimage

P_filt = np.zeros((Nx,Ny,Nz),'d',order='F')

sigma_y = 10
sigma_x = 10

# Apply gaussian filter
sigma = [sigma_y, sigma_x]

for k in range(0,Nz):
    P_filt[:,:,k] = sp.ndimage.gaussian_filter(P_aprox_disp[:,:,k], sigma, mode='constant')
    
#%%Compute pressure gradients in horizontal direction needed for integration of momentum equantion

dpdx = np.zeros((Nx,Ny,Nz),dtype='float')
dpdy = np.zeros((Nx,Ny,Nz),dtype='float')
#dpdz = np.zeros((Nx,Ny,Nz),dtype='float')

for k in range(0,Nz):
    for j in range(1,Ny):
        for i in range(1,Nx):
            dpdx[i,j,k] = (1/dx)*(P_filt[i,j,k] - P_filt[i-1,j,k])
            dpdy[i,j,k] = (1/dy)*(P_filt[i,j,k] - P_filt[i,j-1,k])

#for k in range(1,Nz):
#    dpdz[:,:,k] = (1/dz)*(P_filt[:,:,k] -  P_filt[:,:,k-1])
                         
#dpdz[:,:,-1] = 0                               

#We use the horizontal periodicity of the domain to compute the last derivative.
dpdx[0,:,:] = (1/dx)*(P_filt[0,:,:] - P_filt[-1,:,:])
dpdy[:,0,:] = (1/dy)*(P_filt[:,0,:] - P_filt[:,-1,:])

#%%Integrate the momentum equations and continuity equation to retrieve dispersive velocities

nz = 20
iter=500

from numpy import random

time=np.zeros(iter)
vel=np.zeros(iter, dtype='float')
randX=random.randint(Nx)
randY=random.randint(Ny)
    
# First of all we need to initialize the velocity fields:
#-----------------------------------------------------------------------------

# The velocity fields are initialized as a slab of three vertical levels so one can compute vertical gradients at the central level.     
u = np.zeros((Nx,Ny,nz),dtype='float'); u_new = np.zeros((Nx,Ny,nz),dtype='float')
v = np.zeros((Nx,Ny,nz),dtype='float'); v_new = np.zeros((Nx,Ny,nz),dtype='float')
w = np.zeros((Nx,Ny,nz),dtype='float'); w_new = np.zeros((Nx,Ny,nz),dtype='float')


#Beginning of the recursive loop:
TotalDiff = 10
it = 0

#while (TotalDiff > 1e-5):
while (it < iter):

    #A) Advection Terms:    

    # Computing the horizontal derivatives in finite differences:
    #-----------------------------------------------------------------------------

    dudx = np.zeros((Nx,Ny,nz),dtype='float'); dvdx = np.zeros((Nx,Ny,nz),dtype='float')
    dudy = np.zeros((Nx,Ny,nz),dtype='float'); dvdy = np.zeros((Nx,Ny,nz),dtype='float')
    dudz = np.zeros((Nx,Ny,nz),dtype='float'); dvdz = np.zeros((Nx,Ny,nz),dtype='float')
    
    dudx_new = np.zeros((Nx,Ny,nz),dtype='float')
    dvdy_new = np.zeros((Nx,Ny,nz),dtype='float')
    
    
    AdvecX = np.zeros((Nx,Ny,nz),dtype='float'); AdvecX_old = np.zeros((Nx,Ny,nz),dtype='float')
    AdvecY = np.zeros((Nx,Ny,nz),dtype='float'); AdvecY_old = np.zeros((Nx,Ny,nz),dtype='float')
    
    
    d2udx2 = np.zeros((Nx,Ny,nz),dtype='float'); d2vdx2 = np.zeros((Nx,Ny,nz),dtype='float')
    d2udy2 = np.zeros((Nx,Ny,nz),dtype='float'); d2vdy2 = np.zeros((Nx,Ny,nz),dtype='float')
    d2udz2 = np.zeros((Nx,Ny,nz),dtype='float'); d2vdz2 = np.zeros((Nx,Ny,nz),dtype='float')
    
       
    ViscousTerm_x = np.zeros((Nx,Ny,nz),dtype='float')
    ViscousTerm_y = np.zeros((Nx,Ny,nz),dtype='float')


    for k in range(0,nz):
        for j in range(1,Ny):
            for i in range(1,Nx):
            
                dudx[i,j,k] = (1/dx)*(u[i,j,k] - u[i-1,j,k])
                dvdx[i,j,k] = (1/dx)*(v[i,j,k] - v[i-1,j,k])
            
                dudy[i,j,k] = (1/dy)*(u[i,j,k] - u[i,j-1,k])
                dvdy[i,j,k] = (1/dy)*(v[i,j,k] - v[i,j-1,k])
                
                    
        #We use the horizontal periodicity of the domain to compute the last derivatives on Nx, and Ny.
        dudx[0,:,k] = (1/dx)*(u[0,:,k] - u[-1,:,k])
        dvdx[0,:,k] = (1/dx)*(v[0,:,k] - v[-1,:,k]) 
    
        dudy[:,0,k] = (1/dy)*(u[:,0,k] - u[:,-1,k])
        dvdy[:,0,k] = (1/dy)*(v[:,0,k] - v[:,-1,k])

    #Vertical derivatives:
    for k in range(1,nz):
        dudz[:,:,k] = (1/dz)*(u[:,:,k] -  u[:,:,k-1])
        dvdz[:,:,k] = (1/dz)*(v[:,:,k] -  v[:,:,k-1])
                             
    dudz[:,:,0] = 1
    dvdz[:,:,0] = 1      
        

    #B) Fictius Viscosity
    
    for k in range(0,nz):
        for j in range(1,Ny-1):
            for i in range(1,Nx-1):
            
                d2udx2[i,j,k] = (1/(dx**2))*(u[i+1,j,k] -(2*u[i,j,k]) + u[i-1,j,k])
                d2vdx2[i,j,k] = (1/(dx**2))*(v[i+1,j,k] -(2*v[i,j,k]) + v[i-1,j,k])
                
                d2udy2[i,j,k] = (1/(dy**2))*(u[i,j+1,k] -(2*u[i,j,k]) + u[i,j-1,k])
                d2vdy2[i,j,k] = (1/(dy**2))*(v[i,j+1,k] -(2*v[i,j,k]) + v[i,j-1,k])
        
        #We use the horizontal periodicity of the domain to compute the last derivatives on Nx, and Ny.
        d2udx2[0,:,k] = (1/(dx**2))*(u[1,:,k] -(2*u[0,:,k]) + u[-1,:,k])
        d2vdx2[0,:,k] = (1/(dx**2))*(v[1,:,k] -(2*v[0,:,k]) + v[-1,:,k])
        
        d2udx2[-1,:,k] = (1/(dx**2))*(u[0,:,k] -(2*u[-1,:,k]) + u[-2,:,k])
        d2vdx2[-1,:,k] = (1/(dx**2))*(v[0,:,k] -(2*v[-1,:,k]) + v[-2,:,k])
    
        d2udy2[:,0,k] = (1/(dy**2))*(u[:,1,k] -(2*u[:,0,k]) + u[:,-1,k])
        d2vdy2[:,0,k] = (1/(dy**2))*(v[:,1,k] -(2*v[:,0,k]) + v[:,-1,k])
        
        d2udy2[:,-1,k] = (1/(dy**2))*(u[:,0,k] -(2*u[:,-1,k]) + u[:,-2,k])
        d2vdy2[:,-1,k] = (1/(dy**2))*(v[:,0,k] -(2*v[:,-1,k]) + v[:,-2,k])
        
    #Vertical derivatives:
    d2udz2 = 0
    d2vdz2 = 0
    nu = 1e-2
    
    ViscousTerm_x = nu*(d2udx2 + d2udy2 + d2udz2)
    ViscousTerm_y = nu*(d2vdx2 + d2vdy2 + d2vdz2)


    #Time advancement:
    #-----------------------------------------------------------------------------
    dt = 1e-6

    #Integrate the x, y components of the momentum equation
    for k in range(0,nz):
        AdvecX[:,:,k] = u[:,:,k]*dudx[:,:,k] + v[:,:,k]*dudy[:,:,k] + w[:,:,k]*dudz[:,:,k] + dpdx[:,:,k] + ViscousTerm_x[:,:,k]
        u_new[:,:,k] = u[:,:,k] + dt*(1.5*AdvecX[:,:,k] - 0.5*AdvecX_old[:,:,k])  #We use second order Adam-Bashforth.    
        
        AdvecY[:,:,k] = u[:,:,k]*dvdx[:,:,k] + v[:,:,k]*dvdy[:,:,k] + w[:,:,k]*dvdz[:,:,k] + dpdy[:,:,k] + ViscousTerm_y[:,:,k]
        v_new[:,:,k] = v[:,:,k] + dt*(1.5*AdvecY[:,:,k] - 0.5*AdvecY_old[:,:,k])  #We use second order Adam-Bashforth.   


    #Integrate conservation of mass to obtain W.
    for k in range(0,nz):
        for j in range(1,Ny):
            for i in range(1,Nx):
           
                dudx_new[i,j,k] = (1/dx)*(u_new[i,j,k] - u_new[i-1,j,k])        
                dvdy_new[i,j,k] = (1/dy)*(v_new[i,j,k] - v_new[i,j-1,k])
            
   
    dudx_new[0,:,:] = (1/dx)*(u_new[0,:,:] - u_new[-1,:,:])
    dvdy_new[:,0,:] = (1/dy)*(v_new[:,0,:] - v_new[:,-1,:])

    w_new[:,:,0] = 0 #At the surface the vertical velocity is zero.
    
    for k in range(1,nz):
        w_new[:,:,k] = w_new[:,:,k-1] - dz*(dudx_new[:,:,k] + dvdy_new[:,:,k]) 


    # Compute convergence on the computed velocity components:
    
    diff1 = np.max((u_new - u))   
    diff2 = np.max((v_new - v))   
    diff3 = np.max((w_new - w)) 

    TotalDiff = np.max([diff1, diff2, diff3])   

    #Re-set the old velocity with the new velocity:

    u = np.copy(u_new); v = np.copy(v_new); w = np.copy(w_new)
    AdvecX_old = np.copy(AdvecX); AdvecY_old = np.copy(AdvecY)
    
    time[it]=it
    vel[it]=u[randX,randY,13]
    
    it += 1 

    print(it)
    
#%%Plot iteration time step vs u

plt.semilogy(time,abs(vel),'-k')
plt.xlabel(r'Iteration')
plt.ylabel(r'U')
    
#%%Plot the resulting approximated dispersive velocities

h = 13

fig, axes=plt.subplots(1,3, figsize=(8,4), constrained_layout=True)
im1 = axes[0].pcolormesh(x,y,np.transpose(u[:,:,h]),shading='gouraud', cmap='coolwarm')#,vmin=-0.1,vmax = 0.1
im2 = axes[1].pcolormesh(x,y,np.transpose(v[:,:,h]),shading='gouraud', cmap='coolwarm')#,vmin=-0.1,vmax = 0.1
im3 = axes[2].pcolormesh(x,y,np.transpose(w[:,:,h]),shading='gouraud', cmap='coolwarm')#,vmin=-0.001,vmax = 0.001

#im1.set_clim(vmin=-0.1, vmax=0.1)
#im2.set_clim(vmin=-0.1, vmax=0.1)
#im3.set_clim(vmin=-0.1, vmax=0.1)

cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8,location='top',orientation='horizontal')
cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8,location='top',orientation='horizontal')
cbar3 = fig.colorbar(im3, ax=axes[2], shrink=0.8,location='top',orientation='horizontal')


axes[0].set_ylabel('$y/z_{i}$')
axes[0].set_xlabel('$x/z_{i}$'); axes[1].set_xlabel('$x/z_{i}$'); axes[2].set_xlabel('$x/z_{i}$');

axes[0].set_title('u'); axes[1].set_title('v');axes[2].set_title('w')