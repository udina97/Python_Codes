# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 06:17:03 2025

@author: udina
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

os.chdir("C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\functions\\")

from Anisotropy_Functions import ColorAnisotropy, Anisotropy_Clustering
from Stats import  ReynoldsStress, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\UCLAanalysis\\Zev_UCLA_Project\\")

from functions import build_phi, build_intf

#%%#%%Paths to data

directory = 'E:\\PhD\\TopographyData\\'

sin = 'ATTO'

# pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/' 

path_sin = directory + sin + '\\'

#%%#%%Loading Topo data

zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py

Nx = 256
Ny = 256
Nz = 384
Lx = 2.880
Ly = 2.880
Lz = 0.960
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz
mpiProc = 32
nzTot = Nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

#%%#%%Import data

import pickle

# # var = ['u','v','w','p','uu','vv','ww','uuu','vvv','www','dudx','dvdy','dwdz','uv','uw','vw','txx','tyy','tzz','txy','txz','tyz',\
# #        'dudz','dvdz','dudy','dvdx']

var_TKE = ['u','v','w','p','uu','vv','ww','uv','uw','vw','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
                       'txx','tyy','tzz','txy','txz','tyz','uuu','vvu','wwu','uuv','vvv','wwv','uuw','vvw','www',\
                       'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz','vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
                       'pu','pv','pw','dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']

with open(path_sin+'data.pkl', 'rb') as f:
# Load the data from the pickle file
    data = pickle.load(f)
    
# data_sin = dict()
# for i in range(len(var)):
#     data_sin[var[i]] = np.mean(data[var[i]],axis=3)

# del data

#%%Save data to netcdf like Giulia's data structure
NumVariables = 64
dataTKE = xr.DataArray(np.ones(shape = (Nx,Ny,Nz-5,NumVariables),order='F'),\
                       dims=('x','y','z','variable'), coords = {'variable':['u','v','w','p','uu','vv','ww','uv','uw','vw','dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
                                              'txx','tyy','tzz','txy','txz','tyz','uuu','vvu','wwu','uuv','vvv','wwv','uuw','vvw','www',\
                                              'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz','vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
                                              'pu','pv','pw','dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']})
    
for i in range(0,NumVariables):
    dataTKE.data[:,:,:,i] = np.mean(data[var_TKE[i]],axis=(3))[:,:,5:]
    print(f'Done with {var_TKE[i]}')

del data

dataTKE.to_netcdf(path_sin+'dataTKE.nc')