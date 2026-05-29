#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:47:24 2024

@author: u1450851
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy
import math

os.chdir("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/UCLAanalysis/Zev_UCLA_Project/")

from functions import build_phi, build_intf, get_var

#%% Defining the path to the files:

#My data

# directory = '/scratch/general/nfs1/u1450851/LES_Sims/'
directory = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'
# cases = ['amazon_canopy_flat_3D','amazon_canopy_hill05h_3D','amazon_canopy_hill2h_3D','amazon_canopy_hill_3D','amazon_canopy_real_3D']
# cases = ['simflat_256x256x384_out5hr']
# cases = ['simflat_256x256x384_001_1hr']
cases = ['simflat_256x256x384_out5hr_v2']
# cases = ['simflat_256x256x384_out30min']
# cases = ['simATTO1_256x256x384_v2_out5hr']
# cases = ['simATTO2_256x256x384_out5hr']
# cases = ['simATTO12_256x256x384_out5hr']
# cases = ['simATTO_256x256x384_full_out5hr']
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

sim = cases[num]

pcnt3 = 180000 #18000; %Important! used in get_var 
avg_time = pcnt3*1 #total timesteps averaged  Important! used in get_var
startavg = 1
endavg = 1 # max is avg_time/pcnt3
incskip = 0 #should be 0 unless simulation incomplete (# of avgs incompleted)

completed_sim = True
infinite_geom = True
slice_profiles = False
shifting_z = False
tke_budget_flag = True
derivatives_flag = True
PCON_flag = False
PCON_TEMP_flag = False
v_fracflag = False
SAVEAS_flag = False
clear_var_flag = False

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

if shifting_z:
    # Modify axes to move 0 to IBM surface
    z_uvp = z_uvp[5:]
    z_uvp = z_uvp - z_uvp[0]
    z_w = z_w[5:]
    z_w = z_w - z_w[0]

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

if PCON_flag and v_fracflag:
    v_frac = get_var(opath + 'cut_cell/', 'v_frac', Nx, Ny, nzTot, 1, iintf, mpiProc)
    a_cut = get_var(opath + 'cut_cell/', 'a_cut', Nx, Ny, nzTot, 1, iintf, mpiProc)

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
            
            
#%% Plot the Topography in 3D

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(0, Nx*(dx*zi), dx*zi)
Y = np.arange(0, Ny*(dy*zi), dy*zi)
X, Y = np.meshgrid(X, Y)
Z = (iintf-4)*(dz*zi)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z.T, cmap='viridis',
                       linewidth=0, antialiased=False)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
# Customize the z axis.
ax.set_zlim(0, 150)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5, label='[m]')

plt.show()























































