#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:23:49 2024

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

pcnt3 = 36000 #18000; %Important! used in get_var 
avg_time = pcnt3*10 #total timesteps averaged  Important! used in get_var
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
dt = 0.01 #05; %0.000005;
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

#Canopy Parameters:

height = math.ceil(canopyH/dz) # Canopy height in grid points.
kappa = 0.4
Nz_SLayer = 300 #int(Nz/2)
NumCases = 1 #Total number of study cases.

#%% Loading the topography data:
    
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

zeds = np.zeros((Nx,Ny,Nz))

for i in range(0,Nx):
    for j in range(0,Ny):
        zeds[i,j,:] = np.arange(0,Nz)*(dz*zi)
    
dist = copy.deepcopy(zeds)

for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz):
            dist[i,j,k] = dist[i,j,k] - intf[i,j]*zi
            
#%%

nt_tot = nt * actLaunch - incskip
print('nt_tot is', nt_tot)

avgT = int(avg_time / pcnt3)
print('avgT is', avgT)

print('startavg is', startavg)
print('endavg is', endavg)

# timescale = zi / ustar
# timesteps = pcnt3 * (list(range(nt_tot - nt + 1, nt_tot + 1)))
# time = [t * timescale * dt / 60 for t in timesteps]

#%%Import Anisotropy

cmap = ColorAnisotropy()   

case = cases[num]
path = directory + cases[num] + '/'
os.chdir(path)

Anisotropy_clustering = xr.open_dataarray(pathOUT + 'Anisotropy_clustering_'+case+'.nc')
Anisotropy_clustering_DIAG = xr.open_dataarray(pathOUT + 'Anisotropy_clustering_'+case+'_DIAG.nc')

xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 
# lambda3_1D = np.copy(Anisotropy_clustering.data[:,3])

yB = np.reshape(yB_1D,(Nx,Ny,Nz)); xB = np.reshape(xB_1D,(Nx,Ny,Nz));
yB[(dist[:,:,:]<0)] = float('nan'); xB[(dist[:,:,:]<0)] = float('nan')

xB_1D_DIAG = np.copy(Anisotropy_clustering_DIAG.data[:,0]) 
yB_1D_DIAG = np.copy(Anisotropy_clustering_DIAG.data[:,1]) 
AnisType_1D_DIAG = np.copy(Anisotropy_clustering_DIAG.data[:,2]) 

yB_DIAG = np.reshape(yB_1D_DIAG,(Nx,Ny,Nz)); xB_DIAG = np.reshape(xB_1D_DIAG,(Nx,Ny,Nz));
yB_DIAG[(dist[:,:,:]<0)] = float('nan'); xB_DIAG[(dist[:,:,:]<0)] = float('nan')

#%%

ratio = yB_DIAG/yB
# ratio = yB_DIAG - yB

yslice = 100

fig,axs = plt.subplots(1,1,tight_layout=True)
p = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),ratio[:,yslice,5:].T,cmap=cmap)
axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$z/h_C$')
axs.set_title(f'$y = {yslice*dy*zi}m$')
cb = plt.colorbar(p,label=r'$\frac{y_{B,DIAG}}{y_{B}}$')

# fig, axs = plt.subplots(nrows=1,ncols=1)

# axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
# sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),np.transpose(yB[:,yslice,5:]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)

# cbar = plt.colorbar(sc,label='yB')
    
# axs.set_ylabel(r'$z/h$',fontsize=15)
# axs.set_xlabel(r'$x/z_i$',fontsize=15)
# axs.set_title(f'$y_B(y = {yslice*dy*zi}m)$')

# plt.tight_layout()

# plt.savefig(pathOUT +'Figures/'+'yB_ratio.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

#%%

lv1 = 39
lv2 = 39*20

ratio_subset = ratio[:,:,:][(dist[:,:,:]>lv1) & (dist[:,:,:]<lv2)]

fig, axs = plt.subplots(1,1,figsize=(10,5),tight_layout=True)

axs.hist(ratio_subset,bins=100,density=False,stacked=False,cumulative=False)
axs.axvline(np.nanmean(ratio_subset),c='k',ls='--',label='Mean')
axs.axvline(np.nanmedian(ratio_subset),c='k',ls='-.',label='Median')
axs.axvline(np.nanmean(ratio_subset)-np.std(ratio_subset),c='k',ls=':',label='+/- STD')
axs.axvline(np.nanmean(ratio_subset)+np.std(ratio_subset),c='k',ls=':')
axs.plot([],[],' ',label=f'Mean = {round(np.nanmean(ratio_subset),2)}')
axs.plot([],[],' ',label=f'Median = {round(np.nanmedian(ratio_subset),2)}')
# axs.axhline(1,c='k',ls='--')
axs.set_xlabel(r'$\frac{y_{B,DIAG}}{y_{B}}$',fontsize=12)
axs.set_ylabel(r'Counts',fontsize=12)
axs.set_title(f'All points between {lv1}m and {lv2}m')
axs.legend()
# axs.set_xlim(0,0.3)

# plt.savefig(pathOUT +'Figures/'+'yB_ratio_hist.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()





















































