#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:47:15 2025

@author: u1450851
"""

#Libraries and Functions
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/')

from functions import load_3d_UCLAdata, load_UCLAnpy_files, load_2d_UCLAdata

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/")

from Anisotropy_Functions import ColorAnisotropy, phi_m, Anisotropy_Clustering, phi_m_loc
from Stats import  ReynoldsStress, DispFluct, ReynoldsStressUVP, ReynoldsStressUVP_SGS

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Anisotropy/')
from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,compute_d_twr,Twr_TKE_Multi,find_coordinates,box_plot,ChameckiIndex,compute_ustar_twr

#%% Defining the path to the files:

#My data

# directory = '/scratch/general/nfs1/u1450851/LES_Sims/'
directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'
# cases = ['amazon_canopy_flat_3D','amazon_canopy_hill05h_3D','amazon_canopy_hill2h_3D','amazon_canopy_hill_3D','amazon_canopy_real_3D']
# cases = ['simflat_256x256x384_out5hr']
# cases = ['simflat_256x256x384_001_1hr']
# cases = ['simflat_256x256x384_out5hr_v2']
# cases = ['simflat_256x256x384_out30min']
# cases = ['simATTO1_256x256x384_v2_out5hr']
# cases = ['simATTO2_256x256x384_out5hr']
# cases = ['simATTO12_256x256x384_out5hr']
cases = ['simATTO_256x256x384_full_out5hr']
# cases = ['simATTO_256x256x384_full_out30min']
# cases = ['simbicheng_hill_256x256x384_out5hr']
# cases = ['simbicheng_hill_256x256x384_out30min']

num = 0

path = directory + cases[num] + '/'+'output/ta1_field/'
pathOUT = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/'+cases[num]+'/'
pathTurb = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Turb_Proj/'

if os.path.exists(pathOUT):
    print(f"The forlder '{pathOUT}' aready exists.")
else:
    os.makedirs(pathOUT)
    print(f"Created folder '{pathOUT}'.")

os.chdir(path)

#%% Defining the main parameters of the simulations

anisotropy_compute = 'false' #If 'True' turb. Anisotropy and clustering will be computed from scratch, otherwise read from saved file.

sim = cases[num]

pcnt3 = 180000 #18000; %Important! used in get_var 
avg_time = pcnt3*1 #total timesteps averaged  Important! used in get_var
startavg = 1
endavg = 1 # max is avg_time/pcnt3
incskip = 0 #should be 0 unless simulation incomplete (# of avgs incompleted)

completed_sim = True
infinite_geom = True
# slice_profiles = False
# shifting_z = False
# tke_budget_flag = True
# derivatives_flag = True
# PCON_flag = False
# PCON_TEMP_flag = False
# v_fracflag = False
# SAVEAS_flag = False
# clear_var_flag = False

T_STC = 305 #320; %298.15; %[K], temperature scale
dt = 0.1 #05; %0.000005;
zi = 1000.0
u_scale = 0.4 #set equal to whatever is in parameters.py
textsize = 20
wbase = 1000

plot_profile = 0
plot_color = 0

# Input and output paths
ipath = path
opath = directory + cases[num]+ '/output/'

# Read parameter file for ta1_field
with open(ipath + 'parameters.txt', 'r') as param_file:
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
nt = int(param[9])
actLaunch = int(param[10])
mpiProc = int(param[11])
Re = param[12]
Ri = param[13]
Pr = param[14]
alpha = param[15]
SGS = param[16]
S_FLAG = param[17]
ibm = param[18]
rot = param[19]
nzTot = Nz  # nz * mpiProc
z_shift = 4.5 * dz  # IBM surface vertical shift
canopyH = (39+z_shift*zi)/zi

# Build normalized axes
x = np.arange(0, Nx) * dx
y = np.arange(0, Ny) * dy
z_uvp = np.arange(0, nzTot) * dz + 0.5 * dz
z_w = np.arange(0, nzTot) * dz

# if shifting_z:
#     # Modify axes to move 0 to IBM surface
#     z_uvp = z_uvp[5:]
#     z_uvp = z_uvp - z_uvp[0]
#     z_w = z_w[5:]
#     z_w = z_w - z_w[0]

# Build interface and solidity of IBM
if not completed_sim:
    intf = np.zeros((Nx, Ny))
    iintf = np.round(np.zeros((Nx, Ny)))
    # nt = len(ke) // pcnt3
elif ibm == 1:
    # Assuming you have a function build_phi to load phi data from a file
    phi = build_phi(ipath + '../phi_functions/', Nx, Ny, nzTot, mpiProc)
    intf, iintf = build_intf(phi, dz)
else:
    intf = np.zeros((Nx, Ny))
    iintf = np.round(np.zeros((Nx, Ny)))

if ibm:
    phi_uv = np.zeros_like(phi)
    for k in range(Nz - 1):
        phi_uv[:, :, k] = (phi[:, :, k] + phi[:, :, k + 1]) / 2.0

# if PCON_flag and v_fracflag:
#     v_frac = get_var(opath + 'cut_cell/', 'v_frac', Nx, Ny, nzTot, 1, iintf, mpiProc)
#     a_cut = get_var(opath + 'cut_cell/', 'a_cut', Nx, Ny, nzTot, 1, iintf, mpiProc)

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
kappa = 0.4
Nz_SLayer = 300 #int(Nz/2)
NumCases = 1 #Total number of study cases.

#%% Loading the topography data:

zeds = np.zeros((Nx,Ny,Nz))

for i in range(0,Nx):
    for j in range(0,Ny):
        zeds[i,j,:] = np.arange(0,Nz)*(dz*zi)
    
dist = copy.deepcopy(zeds)

for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi
            
#%%Import variables

import pickle

save_data = False
load_data = True

var = ['u','v','w','p','uu','uv','uw','vv','vw','ww','uuu','uuv','uuw','vvv','vvu','vvw','wwu','wwv','www','txx','txy','txz','tyy','tyz',\
       'tzz','dxx','dxy','dxz','dyy','dyz','dzz','fdx','fdy','fdz','dpdx','dpdy','dpdz','pdudx','pdvdy','pdwdz','pu','pv','pw','dudx','dudy',\
       'dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz','dudx2','dudy2','dudz2','dvdx2','dvdy2','dvdz2','dwdx2','dwdy2','dwdz2',\
       'utxx','utxy','utxz','vtxy','vtyy','vtyz','wtxz','wtyz','wtzz','ufdx','vfdy','wfdz','utyy','utzz','vtxx','vtzz','wtxx','wtyy']
    
if load_data:
    with open(pathOUT+'data.pkl', 'rb') as f:
    # Load the data from the pickle file
        data = pickle.load(f)
        
    data_tavg = dict()
    for i in range(len(var)):
        data_tavg[var[i]] = np.mean(data[var[i]],axis=3)

if save_data:
    with open(pathOUT+'data.pkl', 'wb') as f:
        pickle.dump(data, f)

dataNAN = copy.deepcopy(data_tavg)

for i in range(len(var)):
    dataNAN[var[i]][(dist<0)] = float("nan")

#%%Compute Stresses
    
Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})
    
RstressTij = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStressUVP(Nx,Ny,Nz,data_tavg['u'],data_tavg['v'],data_tavg['w'],data_tavg['uu'],\
                          data_tavg['vv'],data_tavg['ww'],data_tavg['uv'],data_tavg['uw'],data_tavg['vw'])
    
tij = ['txx','tyy','tzz','txy','txz','tyz']

for i in range(len(tij)):
    RstressTij.data[:,:,:,i] = Rstress.data[:,:,:,i] - data_tavg[tij[i]]

#%%Plot stress

fig,axs = plt.subplots(1,6,figsize=(12,4))

for i in range(len(axs)):
    axs[i].plot(np.mean(Rstress.data[:,:,:,i],axis=(0,1)),np.arange(0,nzTot)*dz+dz/2,c='k',label='Rij')
    axs[i].plot(np.mean(-data_tavg[tij[i]],axis=(0,1)),np.arange(0,nzTot)*dz+dz/2,c='r',label='tij')
    axs[i].plot(np.mean(RstressTij.data[:,:,:,i],axis=(0,1)),np.arange(0,nzTot)*dz+dz/2,c='g',label='RijTij')
    axs[i].legend()
    axs[i].set_xlabel('Rij')
    
axs[0].set_ylabel('z')
plt.show()

#%%Compute anisotropy

anisotropy_compute = False
#Anisotropy analysis:

    
if anisotropy_compute:
    
    [xB,yB,AnisType_1D,lambda3] = Anisotropy_Clustering(Nx,Ny,Nz,RstressTij)

    yB_1D = np.ndarray.flatten(yB)
    xB_1D = np.ndarray.flatten(xB)
    lambda3_1D = np.ndarray.flatten(lambda3)


    Anisotropy_clustering = xr.DataArray(np.zeros(shape = (Nx*Ny*Nz,4),order='F'),\
                        dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D','lambda3_1D']})
        
    Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D; Anisotropy_clustering[:,3] = lambda3_1D 

    os.chdir(pathOUT)
    Anisotropy_clustering.to_netcdf('Anisotropy_clustering_'+sim+'_v2.nc')
    
else:
        
    Anisotropy_clustering = xr.open_dataarray(pathOUT + 'Anisotropy_clustering_'+sim+'_DIAG.nc')

    xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
    yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
    AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 
            # lambda3_1D = np.copy(Anisotropy_clustering.data[:,3])

#%%Plot anisotropy
