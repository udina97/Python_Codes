#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:29:38 2023

@author: benjamin
"""

#%%

import os
import numpy as np

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from LES_check_functions import log_u_profile, check_mke, check_ustar, tau_wall_profile, plot_u_2D_xy, plot_u_2D_xz, plot_ustar_2D, plot_T_2D, plot_sfcval, plot_test

#%% Set path to DataMomentum.nc and running_diagnostics.txt files
sim = 'Homog/256/9ms/Aniso_A'
path_diag = '/scratch/general/nfs1/u1450851/LES_Sims/'+sim+'/'
# path_RAV = '/scratch/general/nfs1/u1450851/LES_Sims/anisotropy/output_RAV/'
# path_TKE = '/scratch/general/nfs1/u1450851/tke_test/output_tke_3D/'

#%% Define domain shape and grid resolution

Nx = 256
Ny = 256
Nz = 256
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2
nRAV = 1
Iter = 10000
u_scale = 0.4

#%%Plot the logarithmic profile of u


plot1 = log_u_profile(Nz, Lz, path_diag,'Data_Momentum.nc',u_scale)

#%% Plot time series of mke for convergence checking

plot2 = check_mke(path_diag,nRAV,Iter)

#%% Plot time series of ustar for convergence checking

plot3 = check_ustar(path_diag,nRAV,Iter)

#%% Plot vertical profile of <tau_wall>

plot4 = tau_wall_profile(Nx, Ny, Nz, Lz, path_diag,'Data_Momentum.nc')

#%% Plot 2D XY plane of u at fixes height

plot5 = plot_u_2D_xy(Nx, Ny, Lx, Ly, path_diag, 20,'Data_Momentum.nc')

#%% Plot 2D XZ plane of u at fixed y

plot6 = plot_u_2D_xz(Nx, Nz, Lx, Lz, path_diag, int(Nz/2),'Data_Momentum.nc')

#%% Plot 2D ustar output LES

plot7 = plot_ustar_2D(Nx, Ny, Lx, Ly, path_diag,'Data_Momentum_2D_2hr.nc')

#%% Plot 2D T output LES

plot8 = plot_T_2D(Nx, Ny, Nz, Lx, Ly, Lz, 50, path_diag,'Data_Scalar_2hr.nc')

#%% Plot Mav_sfcval

plot9 = plot_sfcval(Nx, Ny, Lx, Ly, path_diag,'Data_Scalar_2D_2hr.nc')

#%% Compare p and p_clean

plot10 = plot_test(Nx,Ny,Nz,Lx,Ly,path_diag,20)


































