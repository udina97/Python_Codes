#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 07:01:10 2024

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

#%% Load data

import pickle

var = ['u','v','w','p','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','txx','txy','txz','tyy','tyz',\
        'tzz','dxx','dxy','dxz','dyy','dyz','dzz','fdx','fdy','fdz','dpdx','dpdy','dpdz','pdudx','pdvdy','pdwdz','pu','pv','pw','dudx','dudy',\
        'dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2',\
        'utxx','utxy','utxz','vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','ufdx','vfdy','wfdz','utyy','utzz','vtxx','vtzz','wtxx','wtyy']

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

#%%Import anisotropy data
       
Aniso_data_flat = xr.open_dataarray(path_flat + 'Anisotropy_clustering_'+flat+'.nc')
Aniso_data_sin = xr.open_dataarray(path_sin + 'Anisotropy_clustering_'+sin+'.nc')
Aniso_data_atto = xr.open_dataarray(path_atto + 'Anisotropy_clustering_'+atto+'.nc')

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

#%%Select random towers and compute the dispalcement height for each

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

#%%Plot profile of the velocity gradient

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m_loc

cases = ['flat','sin_min','sin_max','atto_min','atto_max']

for c in range(len(cases)):
    
    xB_1D = aniso[cases[c]]['xB'].flatten()
    yB_1D = aniso[cases[c]]['yB'].flatten()
    AnisType_1D = aniso[cases[c]]['type'].flatten()

    kappa = 0.4
    phi_m_2D = np.zeros((len(coord[cases[c]]),Nz_SLayer))
    z2D = np.zeros((len(coord[cases[c]]),Nz_SLayer),'d',order='F')
    xB_2D = np.zeros((len(coord[cases[c]]),Nz),'d',order='F')
    yB_2D = np.zeros((len(coord[cases[c]]),Nz),'d',order='F')
    AnisType_2D = np.zeros((len(coord[cases[c]]),Nz),'d',order='F')
    z0hi = np.zeros((len(coord[cases[c]])),'d',order='F')
    ustar = np.zeros((len(coord[cases[c]])),'d',order='F')
    z_over_d = np.zeros((len(coord[cases[c]]),Nz_SLayer),'d',order='F')
    
    for i in range(len(coord[cases[c]])):
        loc = coord[cases[c]][i]
    
        z = z_uvp
        z_d = (z - ((d_dim[cases[c]][i])/zi))
        z2D[i,:] = z_d[0:Nz_SLayer]
        z_over_d[i,:] = ((z_uvp[0:Nz_SLayer]*zi)-d_dim[cases[c]][i])/39 
        
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
        
        [z0hi[i],ustar[i],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN[cases[c]]['u'][loc[0],loc[1],int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):],\
                                                                dataNAN[cases[c]]['v'][loc[0],loc[1],int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):],True)
        
        phi_m_2D[i,:] = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,dataNAN[cases[c]]['u'][loc[0],loc[1],int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):],\
                              dataNAN[cases[c]]['v'][loc[0],loc[1],int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):],\
                                  dataNAN[cases[c]]['dudz'][loc[0],loc[1],int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):],\
                                      dataNAN[cases[c]]['dvdz'][loc[0],loc[1],int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):],ustar[i])
        
        xB_2D[i,:] = xB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],:]
        yB_2D[i,:] = yB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],:]
        AnisType_2D[i,:] = AnisType_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],:]
    
    xB_SLayer = np.zeros((len(coord[cases[c]]),Nz_SLayer),'d',order='F')
    yB_SLayer = np.zeros((len(coord[cases[c]]),Nz_SLayer),'d',order='F')
    AnisType_SLayer = np.zeros((len(coord[cases[c]]),Nz_SLayer),'d',order='F')
    
    for i in range(len(coord[cases[c]])):
        loc = coord[cases[c]][i]
        xB_SLayer[i,:] = xB_2D[i,int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        yB_SLayer[i,:] = yB_2D[i,int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
        AnisType_SLayer[i,:] = AnisType_2D[i,int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[cases[c]][loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    
    z2D_1D = np.ndarray.flatten(z2D)
    yB_2D_1D = np.ndarray.flatten(yB_SLayer)
    AnisType_2D_1D = np.ndarray.flatten(AnisType_SLayer)
    phi_M_1D = np.ndarray.flatten(phi_m_2D)
    
    cmap = ColorAnisotropy() 
    
    Nclusters = 9 #Number of clusters used to group the anisotorpy.
    
    #Overlay the median on the denisty plot:
    phiM_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); phiM_median.fill(np.NaN)
    yB_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); yB_median.fill(np.NaN)
    
    y_ax = z_uvp[0:Nz_SLayer]/(39/zi)
    
    cluster = 0
    for m in range(0,3):
        for n in range(0,3):
            
            cluster = cluster + 1
            
            print(f'cluster = {cluster}')
    
            tmp_z3D = z2D_1D[(AnisType_2D_1D == cluster)]
            tmp_phi_u = phi_M_1D[(AnisType_2D_1D == cluster)]
            tmp_yB = yB_2D_1D[(AnisType_2D_1D == cluster)]
            
            NumPoints = np.size(tmp_yB) #Total number of points in a given cluster. This should be larger than 100 to do statistics.
            
            print(f'Total number of points in cluster = {NumPoints}')
    
        
            for i in range(0,Nz_SLayer-1):
                phiM_median[cluster-1,i] = np.median(tmp_phi_u[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])
                yB_median[cluster-1,i] = np.median(tmp_yB[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])     
        
    
            print(f'Done with cluster {cluster}')
    
    fig2, axs = plt.subplots(nrows=1,ncols=1, figsize=(4,6))
    plt.ion()
     
    for cluster in range(1,Nclusters):
        #print(f'{np.count_nonzero(np.isnan(yB_median[cluster,0:-1]))}')
        if (np.count_nonzero(np.isnan(yB_median[cluster,0:-1])) < Nz_SLayer-1):
            sc = axs.scatter(phiM_median[cluster,0:-1],(y_ax[0:Nz_SLayer-1]),marker='o',c = yB_median[cluster,0:-1],cmap='Greys',vmin=0,vmax=np.sqrt(3)/2)
    
    axs.plot(np.nanmean(phi_m_2D[:,:],axis=(0)),(y_ax[0:Nz_SLayer]),linestyle='-',color='k')
    
    axs.set_xlim(-0.5, 2.5)
    axs.set_ylim(y_ax[0], 20)
    axs.axhline(y = (canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle=':')
    # axs.axhline(y = 3*(canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle='-.')
    
    axs.set_xscale('linear')
    # cbar = plt.colorbar(sc)
    axs.set_ylabel(r'$z/h_C$', fontsize=15)
    axs.set_xlabel(r'$\phi_M$',fontsize=15)
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.set_title(f'{cases[c]}',fontsize=15)
    
    axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
    axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
    
    plt.tight_layout()
    # plt.savefig(pathFig+f'{cases[c]}_PHIprofTWR.png',dpi=300,facecolor='None', edgecolor='None')
    
    plt.show()








































