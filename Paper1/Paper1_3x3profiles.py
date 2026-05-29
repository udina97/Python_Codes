#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:27:40 2025

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf

#%%Paths to data

pathFig = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/'
directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'

flat = 'simflat_256x256x384_out5hr_v2'
atto = 'simATTO_256x256x384_full_out5hr'
sin = 'simbicheng_hill_256x256x384_out5hr'

pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/' 

path_flat = pathOUT + flat + '/'
path_sin = pathOUT + sin + '/'
path_atto = pathOUT + atto + '/'

#%%Loading Topo data

zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py

tpath_flat = directory + flat + '/output/'
tpath_sin = directory + sin + '/output/'
tpath_atto = directory + atto + '/output/'

# Read parameter file for ta1_field
with open(tpath_flat + 'ta1_field/parameters.txt', 'r') as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

Nx = int(param[0])
Ny = int(param[1])
Nz = int(param[2])
Lx = param[3]
Ly = param[4]
Lz = param[5]
dx = param[6]
dy = param[7]
dz = param[8]
mpiProc = int(param[11])
nzTot = Nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

# Assuming you have a function build_phi to load phi data from a file
phi_flat = build_phi(tpath_flat + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_flat, iintf_flat = build_intf(phi_flat, dz)
phi_sin = build_phi(tpath_sin + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_sin, iintf_sin = build_intf(phi_sin, dz)
phi_atto = build_phi(tpath_atto + 'phi_functions/', Nx, Ny, nzTot, mpiProc)
intf_atto, iintf_atto = build_intf(phi_atto, dz)

intf = {
        'flat' : intf_flat,
        'sin' : intf_sin,
        'atto' : intf_atto
        }

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
Nz_SLayer = 300 #int(Nz/2)

zeds = np.zeros((Nx,Ny,Nz))

for i in range(0,Nx):
    for j in range(0,Ny):
        zeds[i,j,:] = np.arange(0,Nz)*(dz*zi)
    
dist_flat = copy.deepcopy(zeds); dist_sin = copy.deepcopy(zeds); dist_atto = copy.deepcopy(zeds)

for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz):
            dist_flat[i,j,k] = dist_flat[i,j,k] - intf_flat[i,j]*zi
            dist_sin[i,j,k] = dist_sin[i,j,k] - intf_sin[i,j]*zi
            dist_atto[i,j,k] = dist_atto[i,j,k] - intf_atto[i,j]*zi

dist = {
        'flat' : dist_flat,
        'sin_min' : dist_sin,
        'atto_min' : dist_atto,
        'sin_max' : dist_sin,
        'atto_max' : dist_atto
        }

#%%Import data

import pickle

var = ['u','v','w','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','dudx','dudy','dudz',\
       'dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2','txx','tyy','tzz','txy','txz','tyz']

with open(path_flat+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_flat = dict()
for i in range(len(var)):
    data_flat[var[i]] = np.mean(data[var[i]],axis=3)

with open(path_sin+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_sin = dict()
for i in range(len(var)):
    data_sin[var[i]] = np.mean(data[var[i]],axis=3)

with open(path_atto+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
data_atto = dict()
for i in range(len(var)):
    data_atto[var[i]] = np.mean(data[var[i]],axis=3)

velocity = {
    'flat':data_flat,
    'sin_min':data_sin,
    'sin_max':data_sin,
    'atto_min':data_atto,
    'atto_max':data_atto
    }

#%%Import anisotropy data
       
Aniso_data_flat = xr.open_dataarray(path_flat + 'Anisotropy_clustering_'+flat+'_v2.nc')
Aniso_data_sin = xr.open_dataarray(path_sin + 'Anisotropy_clustering_'+sin+'_v2.nc')
Aniso_data_atto = xr.open_dataarray(path_atto + 'Anisotropy_clustering_'+atto+'_v2.nc')

aniso_flat = dict(); aniso_sin = dict(); aniso_atto = dict()
aniso_var = ['xB','yB','type']

for i in range(len(aniso_var)):
    aniso_flat[aniso_var[i]] = np.reshape(np.copy(Aniso_data_flat.data[:,i]),(Nx,Ny,Nz)) 
    aniso_sin[aniso_var[i]] = np.reshape(np.copy(Aniso_data_sin.data[:,i]),(Nx,Ny,Nz)) 
    aniso_atto[aniso_var[i]] = np.reshape(np.copy(Aniso_data_atto.data[:,i]),(Nx,Ny,Nz)) 

for i in range(len(aniso_var)):
    aniso_flat[aniso_var[i]][(dist_flat[:,:,:]<0)] = float('nan')
    aniso_sin[aniso_var[i]][(dist_sin[:,:,:]<0)] = float('nan')
    aniso_atto[aniso_var[i]][(dist_atto[:,:,:]<0)] = float('nan')

aniso = {
    'flat' : aniso_flat,
    'sin_min' : aniso_sin,
    'atto_min' : aniso_atto,
    'sin_max' : aniso_sin,
    'atto_max' : aniso_atto
    }

#%%Select coordinates
import random
os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex

N_twrs = 100
Nz_SLayer = 300

coord_atto_max = find_coordinates(intf['atto'],N_twrs,'max') #min,max
coord_sin_max = find_coordinates(intf['sin'],N_twrs,'max') #min,max
coord_atto_min = find_coordinates(intf['atto'],N_twrs,'min') #min,max
coord_sin_min = find_coordinates(intf['sin'],N_twrs,'min') #min,max
coord_flat = []
for i in range(0,N_twrs):
    coord_flat.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))

coord = {
    'flat' : coord_flat,
    'sin_min' : coord_sin_min,
    'atto_min' : coord_atto_min,
    'sin_max' : coord_sin_max,
    'atto_max' : coord_atto_max
    }

LAD = [0.15655190, 0.20633190, 0.24492203, 0.28024144, 0.33267326, 0.33145316, 0.32065714, 0.28729650, 0.25240169, 0.17358901, 0.11740349, 0.064294815, \
    0.041340224, 0.023756023, 0.013134912, 0.013107183] #ATTO nz=384
    
dataNAN_flat = copy.deepcopy(data_flat); dataNAN_sin = copy.deepcopy(data_sin); dataNAN_atto = copy.deepcopy(data_atto)

for i in range(len(var)):
    dataNAN_flat[var[i]][(dist_flat<0)] = float("nan")
    dataNAN_sin[var[i]][(dist_sin<0)] = float("nan")
    dataNAN_atto[var[i]][(dist_atto<0)] = float("nan")

dataNAN = {
    'flat' : dataNAN_flat,
    'sin_min' : dataNAN_sin,
    'atto_min' : dataNAN_atto,
    'sin_max' : dataNAN_sin,
    'atto_max' : dataNAN_atto
        }
    
d_dim_flat = compute_d_twr(dataNAN_flat,coord['flat'],dist['flat'],height,dz,zi,u_scale,LAD,'ATTO')
d_dim_sin_min = compute_d_twr(dataNAN_sin,coord['sin_min'],dist['sin_min'],height,dz,zi,u_scale,LAD,'ATTO')
d_dim_sin_max = compute_d_twr(dataNAN_sin,coord['sin_max'],dist['sin_max'],height,dz,zi,u_scale,LAD,'ATTO')
d_dim_atto_min = compute_d_twr(dataNAN_atto,coord['atto_min'],dist['atto_min'],height,dz,zi,u_scale,LAD,'ATTO')
d_dim_atto_max = compute_d_twr(dataNAN_atto,coord['atto_max'],dist['atto_max'],height,dz,zi,u_scale,LAD,'ATTO')

d_dim = {
    'flat': d_dim_flat,
    'sin_min' : d_dim_sin_min,
    'sin_max' : d_dim_sin_max,
    'atto_min' : d_dim_atto_min,
    'atto_max' : d_dim_atto_max
    }

#%%Functions

from scipy.optimize import curve_fit     

"""
Define a function that will fit the mean velocity with a logarithmic fit
"""
def log_fit(z, a, b):
    return a*np.log(b*z)


"""
Define a function that will compute z0hi, and u_star using a logarithmic fit on the mean velocity profile.
"""
def compute_ustar(Nz_SLayer,z_d,u,v,twr=False):    

    U = np.sqrt(u**2 + v**2)
    U_mean = U[0:Nz_SLayer]

    #Data used for the Inertial logarithmic fit above the RSL. --------------
    #------------------------------------------------------------------------
    level1 = 20 #70
    level2 = 30 #95
    level3 = 70
    level4 = 90

    u1 = U_mean[level1] #This the velocity @ z-d/zi = 0.253
    u2 = U_mean[level2] #This the velocity @ z-d/zi = 0.350
    u3 = U_mean[level3]
    u4 = U_mean[level4]
    
    z_data = np.array([z_d[level1], z_d[level2], z_d[level3], z_d[level4]])#
    U_data = np.array([u1, u2, u3, u4])


    coefs, pcov = curve_fit(log_fit, z_data, U_data)
    u_fit = coefs[0]*np.log(coefs[1]*(z_d[0:Nz_SLayer]))

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    z0hi = (1/coefs[1]) #Normalized values of 'z0hi', hence z0hi/zi.
    ustar = U_mean[level1]/((1/kappa)*np.log(z_d[level1]/z0hi)) #Normalized values of 'ustar', hence u*/uscale.


    return(z0hi,ustar,U_mean,U_data,z_data,u_fit)

#%%Plot 3x3 figure containing velocity profile, reynolds stress, velocity gradient


cases = ['flat','sin_min','sin_max','atto_min','atto_max']
ls = ['-','--','-','--','-']
SH_terms = ['uw','vw']
labels = ["(a)", "(b)", "(c)", "(d)", "(e)",'(f)','(g)','(h)','(i)']  # Subplot labels
l = 0

fig,axs = plt.subplots(3,3,figsize=(8,12),tight_layout=True,sharey=True)
plt.subplots_adjust(hspace=0.5) 

for m in range(0,3):
    for n in range(0,3):
        
        if m==0:
            for j in range(len(cases)):
                
                tmp = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
                ustar = np.zeros((N_twrs))
                Ruw = (velocity[cases[j]]['uw'] - velocity[cases[j]]['u']*velocity[cases[j]]['w'] - velocity[cases[j]]['txz'])*(u_scale**2)
                Rvw = (velocity[cases[j]]['vw'] - velocity[cases[j]]['v']*velocity[cases[j]]['w'] - velocity[cases[j]]['tyz'])*(u_scale**2)
            
                for k in range(len(coord[cases[j]])):
                    loc = coord[cases[j]][k]
                    ustar[k] = ((Ruw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2 + \
                                (Rvw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2)**(1/4)
                    tmp[k,:] = np.sqrt((velocity[cases[j]][var[0]][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                          int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])**2 + \
                                       (velocity[cases[j]][var[1]][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                          int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])**2 + \
                                       (velocity[cases[j]][var[2]][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                          int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])**2)*(u_scale/ustar[k])
            
                if j==0:
                    axs[m,0].plot(np.nanmean(tmp,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                elif j==1 or j==2:
                    axs[m,1].plot(np.nanmean(tmp,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                else:
                    axs[m,2].plot(np.nanmean(tmp,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                    
            axs[m,n].set_xlabel(r'$\overline{U}/u_{*}$',fontsize=13)
            axs[m,n].set_xlim(-0.1,15)
            
        elif m==1:
            for j in range(len(cases)):
                
                tmpSH = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
                ustar = np.zeros((N_twrs))
                Ruw = (velocity[cases[j]]['uw'] - velocity[cases[j]]['u']*velocity[cases[j]]['w'] - velocity[cases[j]]['txz'])*(u_scale**2)
                Rvw = (velocity[cases[j]]['vw'] - velocity[cases[j]]['v']*velocity[cases[j]]['w'] - velocity[cases[j]]['tyz'])*(u_scale**2)
                shear = {
                    'uw' : Ruw,
                    'vw' : Rvw}
            
                for k in range(len(coord[cases[j]])):
                    loc = coord[cases[j]][k]
                    ustar[k] = ((Ruw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2 + \
                                (Rvw[loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+16])**2)**(1/4)
                    tmpSH[k,:] = np.sqrt((shear[SH_terms[0]][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                                      int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])**2 + \
                                          (shear[SH_terms[1]][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]): \
                                                                      int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer])**2)/(ustar[k]**2)
            
                if j==0:
                    axs[m,0].plot(np.nanmean(tmpSH,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                elif j==1 or j==2:
                    axs[m,1].plot(np.nanmean(tmpSH,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                else:
                    axs[m,2].plot(np.nanmean(tmpSH,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                    
            axs[m,n].set_xlabel(r"$\sqrt{\overline{u'w'}^2+\overline{v'w'}^2}/u_{*}^{2}$",fontsize=13)
            axs[m,n].set_xlim(-0.1,2)
            
        else:
            for j in range(len(cases)):
                os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

                from Anisotropy_Functions import phi_m_loc

                kappa = 0.4
                phi_m_2D = np.zeros((len(coord[cases[j]]),Nz_SLayer))
                z2D = np.zeros((len(coord[cases[j]]),Nz_SLayer),'d',order='F')
                z0hi = np.zeros((len(coord[cases[j]])),'d',order='F')
                ustar = np.zeros((len(coord[cases[j]])),'d',order='F')
                z_over_d = np.zeros((len(coord[cases[j]]),Nz_SLayer),'d',order='F')
                
                for k in range(len(coord[cases[j]])):
                    loc = coord[cases[j]][k]
                
                    z = z_uvp
                    z_d = (z - ((d_dim[cases[j]][k])/zi))
                    z2D[k,:] = z_d[0:Nz_SLayer]
                    z_over_d[k,:] = ((z_uvp[0:Nz_SLayer]*zi)-d_dim[cases[j]][k])/39 
                    
                    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN[cases[j]]['u'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]):],\
                                                                            dataNAN[cases[j]]['v'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]):],True)
                    
                    phi_m_2D[k,:] = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,dataNAN[cases[j]]['u'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]):],\
                                          dataNAN[cases[j]]['v'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]):],\
                                              dataNAN[cases[j]]['dudz'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]):],\
                                                  dataNAN[cases[j]]['dvdz'][loc[0],loc[1],int(np.where(dist[cases[j]][loc[0],loc[1],:]>0)[0][0]):],ustar[k])

                if j==0:
                    axs[m,0].plot(np.nanmean(phi_m_2D,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                elif j==1 or j==2:
                    axs[m,1].plot(np.nanmean(phi_m_2D,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                else:
                    axs[m,2].plot(np.nanmean(phi_m_2D,axis=(0)),np.arange(0,Nz_SLayer)*dz/(39/zi) + ((dz/2)/(39/zi)),label='U',linewidth=2, c='k', ls=ls[j])
                    
            axs[m,n].set_xlabel(r'$\phi_M$',fontsize=13)
            axs[m,n].set_xlim(-0.5,2)

for m in range(0,3):
    for n in range(0,3):
        axs[m,n].set_ylim(0,12)
        axs[m,n].grid()
        axs[m,n].tick_params(axis='both', which='major', labelsize=12)
        axs[m,n].axhline(1,ls='--',c='k')
        axs[m,n].text(0.13, 0.98, labels[l], transform=axs[m,n].transAxes, fontsize=12, va="top", ha="right")
        l += 1
        
axs[0,0].set_ylabel(r'$z/h_C$',fontsize=15), axs[0,0].set_title(r'Flat',fontsize=15)
axs[1,0].set_ylabel(r'$z/h_C$',fontsize=15), axs[0,1].set_title(r'Sinusoidal',fontsize=15)
axs[2,0].set_ylabel(r'$z/h_C$',fontsize=15), axs[0,2].set_title(r'ATTO',fontsize=15)

# plt.savefig(pathFig+'3x3profiles_12_tij.png',dpi=300,facecolor='None', edgecolor='None')
plt.show()

#%%














































