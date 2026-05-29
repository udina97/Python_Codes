"""
Program Author: Marc Calaf.

This program uses Giulia's LES data, to compute vertical gradients 
of the mean velocity as a function of z/z* in neutral conditions, and turbulence anisotropy.
This program is a streamlined version of "Vegetation_FluxGradient_LES.py"

Date created: 22 April 2023
Last date modified: 22 April 2023

To Do: 

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

#%% Plot ke.txt for convergence checking

wbase = 1000
parentpath = directory + cases[num]+'/'
file_name = 'ke.txt'

# Load and import data
ke = np.genfromtxt(parentpath + file_name)
ts = np.linspace(0, len(ke) * wbase, len(ke))

# Create the plot
plt.figure()
plt.plot(ts, ke, 'k-', linewidth=2.5)
plt.xlabel('timesteps')
plt.ylabel('MKE')
plt.gca().set_facecolor('white')  # Set background color to white

# plt.savefig(figPath+'ke.png',dpi=300,facecolor='white', edgecolor='white')
# Show the plot
plt.show()

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
    nt = len(ke) // pcnt3
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
else:
    data = dict()
    data_tavg = dict()

    for i in range(len(var)):
        data[var[i]] = get_var(ipath,var[i],Nx,Ny,nzTot,nt_tot,iintf,mpiProc,avgT)
        data_tavg[var[i]] = np.mean(data[var[i]],axis=3)

if save_data:
    with open(pathOUT+'data.pkl', 'wb') as f:
        pickle.dump(data, f)

dataNAN = copy.deepcopy(data_tavg)

for i in range(len(var)):
    dataNAN[var[i]][(dist<0)] = float("nan")

#%%

# PLOTTING PCOLOR SLICES OF U,V,W TO CHECK THE VELOCITY FIELD
    
#%% pcolor of u,w

tmpU = copy.deepcopy(dataNAN['u'][:,:,5:])
tmpV = copy.deepcopy(dataNAN['v'][:,:,5:])
tmpW = copy.deepcopy(dataNAN['w'][:,:,5:])

yslice = 161

fig,axs = plt.subplots(1,3,figsize=(12,4),tight_layout=True)
p1 = axs[0].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),tmpU[:,yslice,:].T,cmap='coolwarm',shading='gouraud',vmin=0)
p2 = axs[1].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),tmpV[:,yslice,:].T,cmap='coolwarm',shading='gouraud')
p3 = axs[2].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),tmpW[:,yslice,:].T,cmap='coolwarm',shading='gouraud',vmin=-0.3,vmax=0.3)
axs[0].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs[1].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs[2].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs[0].axhline(canopyH/canopyH,ls='--',c='k'),axs[1].axhline(canopyH/canopyH,ls='--',c='k')
axs[0].set_xlabel(r'$x/z_i$'), axs[1].set_xlabel(r'$x/z_i$'), axs[2].set_xlabel(r'$x/z_i$')
axs[0].set_ylabel(r'$z/h_c$')
axs[0].set_title(r'U/$u_{scale}$ (yslice)'),axs[1].set_title(r'V/$u_{scale}$ (yslice)'), axs[2].set_title(r'W/$u_{scale}$ (yslice)')
cbar1 = plt.colorbar(p1); cbar2 = plt.colorbar(p2); cbar3 = plt.colorbar(p3)
fig.suptitle(f'yslice = {yslice*dy*zi}m')

#########################################################################################################################################

# xslice = 255

# fig,axs = plt.subplots(1,3,figsize=(12,4),tight_layout=True)
# p1 = axs[0].pcolormesh(np.arange(0,Ny)*dy,np.arange(0,Nz-5)*dz/(39/zi),tmpU[xslice,:,:].T,cmap='coolwarm',shading='gouraud',vmin=0)
# p2 = axs[1].pcolormesh(np.arange(0,Ny)*dy,np.arange(0,Nz-5)*dz/(39/zi),tmpV[xslice,:,:].T,cmap='coolwarm',shading='gouraud')
# p3 = axs[2].pcolormesh(np.arange(0,Ny)*dy,np.arange(0,Nz-5)*dz/(39/zi),tmpW[xslice,:,:].T,cmap='coolwarm',shading='gouraud',vmin=-0.3,vmax=0.3)
# axs[0].plot(np.arange(0,Ny)*dy,(intf[xslice,:]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs[1].plot(np.arange(0,Ny)*dy,(intf[xslice,:]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs[2].plot(np.arange(0,Ny)*dy,(intf[xslice,:]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# # axs[0].axhline(canopyH/canopyH,ls='--',c='k'),axs[1].axhline(canopyH/canopyH,ls='--',c='k')
# axs[0].set_xlabel(r'$y/z_i$'), axs[1].set_xlabel(r'$y/z_i$'), axs[2].set_xlabel(r'$y/z_i$')
# axs[0].set_ylabel(r'$z/h_c$')
# axs[0].set_title(r'U/$u_{scale}$ (xslice)'),axs[1].set_title(r'V/$u_{scale}$ (xslice)'), axs[2].set_title(r'W/$u_{scale}$ (xslice)')
# cbar1 = plt.colorbar(p1); cbar2 = plt.colorbar(p2); cbar3 = plt.colorbar(p3)
# fig.suptitle(f'xslice = {xslice*dx*zi}m')

############################################################################################################################################

# zslice = 40

# fig,axs = plt.subplots(1,3,figsize=(12,4),tight_layout=True)
# p1 = axs[0].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,tmpU[:,:,zslice],cmap='coolwarm',shading='gouraud')
# p2 = axs[1].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,tmpV[:,:,zslice],cmap='coolwarm',shading='gouraud')
# p3 = axs[2].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,tmpW[:,:,zslice],cmap='coolwarm',shading='gouraud')

# axs[0].set_xlabel(r'$x/z_i$'), axs[1].set_xlabel(r'$x/z_i$'), axs[2].set_xlabel(r'$x/z_i$')
# axs[0].set_ylabel(r'$y/z_i$'), axs[1].set_ylabel(r'$y/z_i$'), axs[2].set_ylabel(r'$y/z_i$')
# axs[0].set_title(r'U/$u_{scale}$ (zslice)'),axs[1].set_title(r'V/$u_{scale}$ (zslice)'), axs[2].set_title(r'W/$u_{scale}$ (zslice)')
# cbar1 = plt.colorbar(p1); cbar2 = plt.colorbar(p2); cbar3 = plt.colorbar(p3)
# fig.suptitle(f'zslice = {zslice*dz*zi}m')


# plt.savefig(pathOUT +'Figures/'+'U_W_pcolor_yslice.png',dpi=300,facecolor='white', edgecolor='white')

#%%Pcolor plot of U = sqrt(u^2+v^2)

tmpUsqrt = np.sqrt(dataNAN['u'][:,:,5:]**2 + dataNAN['v'][:,:,5:]**2)

# yslice = 255

# fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
# p1 = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),tmpUsqrt[:,yslice,:].T,cmap='coolwarm',shading='gouraud')
# axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs.set_xlabel(r'$x/z_i$')
# axs.set_ylabel(r'$z/h_c$')
# axs.set_title(r'U/$u_{scale}$ (yslice)')
# cbar1 = plt.colorbar(p1)
# fig.suptitle(f'yslice = {yslice*dy*zi}m')

#########################################################################################################################################

# xslice = 120

# fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
# p1 = axs.pcolormesh(np.arange(0,Ny)*dy,np.arange(0,Nz-5)*dz/(39/zi),tmpUsqrt[xslice,:,:].T,cmap='coolwarm',shading='gouraud')
# axs.plot(np.arange(0,Ny)*dy,(intf[xslice,:]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
# axs.set_xlabel(r'$y/z_i$')
# axs.set_ylabel(r'$z/h_c$')
# axs.set_title(r'U/$u_{scale}$ (xslice)')
# cbar1 = plt.colorbar(p1)
# fig.suptitle(f'xslice = {xslice*dx*zi}m')

############################################################################################################################################

zslice = 60

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
p1 = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,tmpUsqrt[:,:,zslice].T,cmap='coolwarm',shading='gouraud')
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$y/z_i$')
axs.set_title(r'U/$u_{scale}$ (zslice)')
cbar1 = plt.colorbar(p1)
fig.suptitle(f'zslice = {zslice*dz*zi}m')

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'Umag150m_atto.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()

#%% Compute the displacement height ww/ Giulia's routine

from scipy.integrate import trapz
import random

N_twrs = 100

LAD_ATTO = True #if false it uses the LAD for the flat 256x256x256 case
ATTO_twr = True #if True it carries out the analysis for a tower in the ATTO sims
Flat_twr = False #if True it does the tower analysis for the flat case
Flat_twr_multi = False #if True selects N_twrs at random locations in the flat case to carry out the "2D" analysis
#if all False it does the displacement height computation averaging the velocity filed in x and y (makes sense only for the flat case)

# LAD = [0, 0.026, 0.1785, 0.258, 0.19, 0.175, 0.2, 0.285, 0.295, 0.305, 0.308, 0.35, 0.374, 0.335, 0.334, 0.335, 0.305, 0.295, 0.285,\
#        0.325, 0.315, 0.284, 0.235, 0.175, 0.195, 0.145, 0.09, 0.075, 0.085, 0.055, 0.04, 0.035, 0.045, 0.01, 0.012, 0.018, 0.019, 0.026, 0.003]

if LAD_ATTO:
    LAD = [0.15655190, 0.20633190, 0.24492203, 0.28024144, 0.33267326, 0.33145316, 0.32065714, 0.28729650, 0.25240169, 0.17358901, 0.11740349, 0.064294815, \
        0.041340224, 0.023756023, 0.013134912, 0.013107183] #ATTO nz=384
else:
    LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344] #flat case nz=256

# LAD_interp = np.interp(np.linspace(LAD[1],len(LAD),20),np.linspace(LAD[1],len(LAD),len(LAD)-1),LAD[1:]) #interpolate to simulation veritacal resolution

if ATTO_twr:
    coord = find_coordinates(intf,N_twrs,'max') #min,max
    
    topo_twr = plot_topo_twr(zi,Nx,Ny,dx,dy,intf,coord,textsize)
    d_dim = compute_d_twr(dataNAN,coord,dist,height,dz,zi,u_scale,LAD,'ATTO')
        
elif Flat_twr_multi:
    
    coord = []
    for i in range(0,N_twrs):
        coord.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))
    
    topo_twr = plot_topo_twr(zi,Nx,Ny,dx,dy,intf,coord,textsize)
    d_dim = compute_d_twr(dataNAN,coord,dist,height,dz,zi,u_scale,LAD,'ATTO')
        
elif Flat_twr:
    # #tower analysis for flat case
    
    locx = random.randrange(0,Nx,1); locy = random.randrange(0,Ny,1)
    
    u = dataNAN['u'][locx,locy,5:]
    v = dataNAN['v'][locx,locy,5:]
    U = np.sqrt(u**2+v**2)
    
    prof2 = {}
    prof2['U'] = U #mean_U
    Z = np.linspace(1,height-5,16)*dz*zi - (dz*zi)/2
    VAR = prof2['U'][0:height-5]*u_scale
    Y = (VAR**2)*0.4*LAD
    d_dim = trapz(Y*Z,Z)/trapz(Y,Z)
    
else:
    u = dataNAN['u'][:,:,5:]
    v = dataNAN['v'][:,:,5:]
    U = np.sqrt(u**2+v**2)
    
    mean_U = np.nanmean(np.nanmean(U, axis=0), axis=0)
    mean_U = np.squeeze(mean_U)
    
    prof2 = {}
    prof2['U'] = mean_U
    Z = np.linspace(1,height-5,16)*dz*zi - (dz*zi)/2
    VAR = prof2['U'][0:height-5]*u_scale
    Y = (VAR**2)*0.4*LAD
    d_dim = trapz(Y*Z,Z)/trapz(Y,Z)
    
# plt.savefig(pathOUT +'Figures/'+'topo_valley_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')

#%%

#Displacement height for the different cases in meters.

case = cases[num]

dispH = {'simflat_256x256x384_out5hr':0,'simATTO1_256x256x384_v2_out5hr':0,'simATTO2_256x256x384_out5hr':0,\
         'simATTO12_256x256x384_out5hr':0,'simATTO_256x256x384_full_out5hr':0}
   

if ATTO_twr:
    rand_loc = random.randrange(0,len(coord),1)
    loc = coord[rand_loc]
    dispH[case] = d_dim[rand_loc]
    z = z_uvp
    z_on_h = z/(39/zi)
    z_d = (z - ((dispH[case])/zi))
    
elif Flat_twr:
    dispH[case] = d_dim
    z = z_uvp
    z_on_h = z/(39/zi)
    z_d = (z - ((dispH[case])/zi)) #(z-d)/h
    
else:
    dispH[case] = d_dim
    z = z_uvp
    z_on_h = z/(39/zi)
    z_d = (z - ((dispH[case])/zi)) #(z-d)/h

#Here we just initialize the following two dictionaries. The corresponding values will computed inside.

z0hi_dict = {'simflat_256x256x384_out5hr':0,'simATTO1_256x256x384_v2_out5hr':0,'simATTO2_256x256x384_out5hr':0,\
         'simATTO12_256x256x384_out5hr':0,'simATTO_256x256x384_full_out5hr':0} #in [m]

ustar_dict = {'simflat_256x256x384_out5hr':0,'simATTO1_256x256x384_v2_out5hr':0,'simATTO2_256x256x384_out5hr':0,\
         'simATTO12_256x256x384_out5hr':0,'simATTO_256x256x384_full_out5hr':0} #in [m/s]

#%%Plot mean profiles of velocity for the selected towers

u_twr = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
v_twr = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
w_twr = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

for i in range(len(coord)):
    u_twr[i,:] = dataNAN['u'][coord[i][0],coord[i][1],int(np.where(dist[coord[i][0],coord[i][1],:]>0)[0][0]):int(np.where(dist[coord[i][0],coord[i][1],:]>0)[0][0])+Nz_SLayer]
    v_twr[i,:] = dataNAN['v'][coord[i][0],coord[i][1],int(np.where(dist[coord[i][0],coord[i][1],:]>0)[0][0]):int(np.where(dist[coord[i][0],coord[i][1],:]>0)[0][0])+Nz_SLayer]
    w_twr[i,:] = dataNAN['w'][coord[i][0],coord[i][1],int(np.where(dist[coord[i][0],coord[i][1],:]>0)[0][0]):int(np.where(dist[coord[i][0],coord[i][1],:]>0)[0][0])+Nz_SLayer]
    
fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(4,6))
axs.plot(np.nanmean(u_twr,axis=0),z_uvp[:Nz_SLayer]/(39/zi),c='g',label='U')
axs.plot(np.nanmean(v_twr,axis=0),z_uvp[:Nz_SLayer]/(39/zi),c='r',label='V')
axs.plot(np.nanmean(w_twr,axis=0),z_uvp[:Nz_SLayer]/(39/zi),c='b',label='W')
axs.axhline(1,ls='--',c='k')
axs.set_xlabel(r'U/$u_*$ [-]')
axs.set_ylabel(r'z/$h_C$ [-]')
axs.set_xlim(-6,10)
axs.set_ylim(0,z_uvp[Nz_SLayer]/(39/zi))
axs.grid()
axs.legend()
axs.set_title('Peaks')

plt.show()

# plt.savefig(pathOUT +'Figures/'+'twr_mean_vel_prof_peaks.png',dpi=300,facecolor='white', edgecolor='white')

#%%

# ANALYSIS OF THE REYNOLDS STRESS COMPONENTS, VERTICAL PROFILES FOR TOWER AVERAGES, PRODUCTION TERMS FOR THE VARIANCES

#%% Plot profiles of the variances over the TKE

terms_ptb = dict()

terms_ptb['u2_t'] = dataNAN['uu'] - dataNAN['u']**2 + dataNAN['txx']
terms_ptb['uv_t'] = dataNAN['uv'] - dataNAN['u']*dataNAN['v'] + dataNAN['txy']
terms_ptb['uw_t'] = dataNAN['uw'] - dataNAN['u']*dataNAN['w'] + dataNAN['txz']

terms_ptb['v2_t'] = dataNAN['vv'] - dataNAN['v']**2 + dataNAN['tyy']
terms_ptb['vw_t'] = dataNAN['vw'] - dataNAN['v']*dataNAN['w'] + dataNAN['tyz']

terms_ptb['w2_t'] = dataNAN['ww'] -dataNAN['w']**2# + dataNAN['tzz']

terms_ptb['tke_SGS'] = (dataNAN['txx']+dataNAN['tyy']+dataNAN['tzz']) / 2
terms_ptb['tke'] = (terms_ptb['u2_t']+terms_ptb['v2_t']+terms_ptb['w2_t']) / 2


tmpTKE = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmpUU = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmpVV = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmpWW = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmpUW = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmpVW = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
tmpUV = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

for i in range(len(coord)):
    loc = coord[i]
    
    tmpTKE[i,:] = terms_ptb['tke'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmpUU[i,:] = terms_ptb['u2_t'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmpVV[i,:] = terms_ptb['v2_t'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmpWW[i,:] = terms_ptb['w2_t'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmpUW[i,:] = terms_ptb['uw_t'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmpVW[i,:] = terms_ptb['vw_t'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    tmpUV[i,:] = terms_ptb['uv_t'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    
z = z_uvp
z_on_h = z/(39/zi)

fig,axs = plt.subplots(1,3,figsize=(10,8),tight_layout=True)

# axs[0].plot(np.mean(tmpUU/tmpTKE,axis=(0)),z_on_h[:Nz_SLayer],c='k')
# axs[1].plot(np.mean(tmpVV/tmpTKE,axis=(0)),z_on_h[:Nz_SLayer],c='k')
# axs[2].plot(np.mean(tmpWW/tmpTKE,axis=(0)),z_on_h[:Nz_SLayer],c='k')
axs[0].plot(np.mean(terms_ptb['u2_t']/terms_ptb['tke'],axis=(0,1))[5:Nz_SLayer+5],z_on_h[0:Nz_SLayer],c='k')
axs[1].plot(np.mean(terms_ptb['v2_t']/terms_ptb['tke'],axis=(0,1))[5:Nz_SLayer+5],z_on_h[0:Nz_SLayer],c='k')
axs[2].plot(np.mean(terms_ptb['w2_t']/terms_ptb['tke'],axis=(0,1))[5:Nz_SLayer+5],z_on_h[0:Nz_SLayer],c='k')
for i in range(len(axs)):
    axs[i].set_xlim(-0.75,2)
    axs[i].grid(True)
    axs[i].set_ylim(z_on_h[0],z_on_h[Nz_SLayer])
    axs[i].axhline(1,-1,2,ls='-.',c='k')

axs[0].set_ylabel(r'$z/h_c$',fontsize='15')
# axs[0].set_xlabel(r"$\frac{\overline{u'w'}}{\overline{e}}$",fontsize='15'),axs[1].set_xlabel(r"$\frac{\overline{v'w'}}{\overline{e}}$",fontsize='15'),axs[2].set_xlabel(r"$\frac{\overline{u'v'}}{\overline{e}}$",fontsize='15')
axs[0].set_xlabel(r"$\frac{\overline{u'u'}}{\overline{e}}$",fontsize='15'),axs[1].set_xlabel(r"$\frac{\overline{v'v'}}{\overline{e}}$",fontsize='15'),axs[2].set_xlabel(r"$\frac{\overline{w'w'}}{\overline{e}}$",fontsize='15')
# axs[0].set_xlabel(r"$\overline{u'u'}$",fontsize='15'),axs[1].set_xlabel(r"$\overline{v'v'}$",fontsize='15'),axs[2].set_xlabel(r"$\overline{w'w'}$",fontsize='15')
# axs[0].set_xlabel(r"$\overline{u'v'}$",fontsize='15'),axs[1].set_xlabel(r"$\overline{u'w'}$",fontsize='15'),axs[2].set_xlabel(r"$\overline{v'w'}$",fontsize='15')

# fig.suptitle('Peaks Avg 100 Towers')

# plt.savefig(pathOUT +'Figures/'+'VelVar_prof_valley_bicheng.png',dpi=300,facecolor='white', edgecolor='white')

#%%Plot r from Ayet et al. 2020

fig, axs = plt.subplots(1,1,tight_layout=True)
axs.plot(np.nanmean(terms_ptb['w2_t']/terms_ptb['tke']-2/3,axis=(0,1))[5:],np.arange(0,Nz-5)*dz/(39/zi),c='k')
plt.show()

#%% Compute and plot sigma_w/ustar

z0hi,ustar = compute_ustar_twr(coord, dataNAN['u'], dataNAN['v'], Nz_SLayer, d_dim, zi, dz, dist)

prof = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
for i in range(len(coord)):
    loc = coord[i]
    prof[i,:] = copy.deepcopy(np.sqrt(terms_ptb['u2_t'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]))/(ustar[i])

fig,axs = plt.subplots(1,1,figsize=(4,8),tight_layout=True)
axs.plot(np.nanmedian(prof,axis=(0)), (np.arange(0,Nz_SLayer)*dz+0.5*dz)/(39/zi), c='r')
axs.fill_betweenx((np.arange(0,Nz_SLayer)*dz+0.5*dz)/(39/zi),np.nanmedian(prof,axis=(0)) - np.nanstd(prof,axis=(0)),\
                 np.nanmedian(prof,axis=(0)) + np.nanstd(prof,axis=(0)),alpha=0.5)
axs.axhline(1,-5,5,c='k',ls='--')
# axs.axvline(1,-5,5,c='k',ls='--')
# axs.set_xlim(0,8)
axs.set_ylim(0,2)
axs.set_xlabel(f'$\sigma_u/u*$',fontsize=15)
# axs.set_xlabel(r"$-\overline{u'w'}/u_{*}^{2}$",fontsize=15)
axs.set_ylabel(f'$z/h_C$',fontsize=15)
plt.show()

# plt.savefig(pathOUT +'Figures/'+'uw_prof_2_valley.png',dpi=300,facecolor='white', edgecolor='white')

#%% Plot sigma_w vs ustar (compute ustar as the sqrt of total shear stress so that it is a 3D field)
from scipy.stats import gaussian_kde

sigma_w = np.sqrt((terms_ptb['w2_t']))
ustar_3D = np.sqrt(np.sqrt(terms_ptb['uw_t']**2 + terms_ptb['vw_t']**2))
ustar_b_3D = np.sqrt(np.sqrt(terms_ptb['uw_t']**2))

height1 = 39 #in m
height2 = 10*height1

tmpustar = ustar_3D.flatten()[(dist.flatten()>height1) & (dist.flatten()<height2)]
tmpsigma = sigma_w.flatten()[(dist.flatten()>height1) & (dist.flatten()<height2)]

# fig, axs = plt.subplots(1,1,tight_layout=True)
# axs.plot(np.mean(ustar_3D,axis=(0,1))[5:],np.arange(0,Nz-5)*(dz)/(39/zi),c='k',label=r"$u_* = \sqrt[4]{\overline{u'w'}^2 + \overline{v'w'}^2}$")
# axs.plot(np.mean(ustar_b_3D,axis=(0,1))[5:],np.arange(0,Nz-5)*(dz)/(39/zi),c='r',label=r"$u_* = \sqrt[4]{\overline{u'w'}^2}$")
# axs.axhline(1,0,1,c='k',ls='--')
# axs.set_ylim(0,20)
# axs.set_xlabel(f'$u_*$',fontsize=15)
# axs.set_ylabel(f'z/H',fontsize=15)
# axs.legend()
# plt.show()

# fig, axs = plt.subplots(1,1,tight_layout=True)
# axs.scatter(tmpustar,tmpsigma,s=1,c='k')
# axs.set_xlabel(f'$u_*$',fontsize=15)
# axs.set_ylabel(f'$\sigma_w$',fontsize=15)
# axs.set_title(f'From {height}m to top.')
# axs.text(0.15, 0.95, f'r = {round(np.corrcoef(tmpustar,tmpsigma)[0,1],2)}',transform=axs.transAxes, fontsize=14,verticalalignment='top')
# plt.show()

nbins = 20
x_2 = tmpustar
y_2 = tmpsigma

k_2 = gaussian_kde([x_2,y_2])
xi_2, yi_2 = np.mgrid[
    x_2.min():x_2.max():nbins*1j,
    y_2.min():y_2.max():nbins*1j
]
z_i_2 = k_2(np.vstack([
    xi_2.flatten(),
    yi_2.flatten()
])).reshape(xi_2.shape)

fig, axs = plt.subplots(1,1,tight_layout=True)
axs.pcolormesh(xi_2,yi_2,z_i_2,cmap='hot_r')
axs.set_xlabel(f'$u_*$',fontsize=15)
axs.set_ylabel(f'$\sigma_w$',fontsize=15)
axs.set_title(f'From {height1}m to {height2}m.')
axs.text(0.15, 0.95, f'r = {round(np.corrcoef(tmpustar,tmpsigma)[0,1],2)}',transform=axs.transAxes, fontsize=14,verticalalignment='top')
plt.show()

# plt.savefig(pathOUT +'Figures/'+'ustar_vs_sigmaw_h1h2.png',dpi=300,facecolor='white', edgecolor='white')

#%% Save the data for the profiles of the stresses

# import pandas as pd 
# df = pd.DataFrame(np.mean(tmpVW,axis=(0)))
# df.to_csv(pathOUT + "CSV/VW_prof_flat.csv", header=False, index=False)

#%% pcolor plots of the Reynolds stress tensor components

yslice = 150

stress = 'w2_t'

fig,axs = plt.subplots(1,1,figsize=(10,6),tight_layout=True)
pc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),terms_ptb[stress][:,yslice,5:].T,cmap='coolwarm',vmin=0,vmax=2)
axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$z/h_c$')
axs.set_title(stress + f' - yslice = {yslice*dy*zi}m')
cbar = plt.colorbar(pc)


#%% Computing the Reynolds Stress

# Computing the Reynolds Stress Tensor and Sensible Heat fluxes:

Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress(Nx,Ny,Nz,dataNAN['u'],dataNAN['v'],dataNAN['w'],dataNAN['uu'],\
                         dataNAN['vv'],dataNAN['ww'],dataNAN['uv'],dataNAN['uw'],dataNAN['vw'])

        
#Vertical profiles of the shear stress:
    
fig, axs = plt.subplots(nrows=2,ncols=3)
plt.ion()

l = 0
Rij_names = list(Rstress.coords['variable'].data[:])
SGSij_names = ['txx','tyy','tzz','txy','txz','tyz']

for m in range(0,2):
    for n in range(0,3):
        
        
        print(l)

        y = z_d/(39/zi)
    
        if (l < 4):
            if ATTO_twr:
                Rij = Rstress.data[loc[0],loc[1],:,l] #for tower topography analysis
            elif Flat_twr:
                Rij = Rstress.data[locx,locx,:,l] #for tower flat analysis
            else:
                Rij = np.nanmean(Rstress.data[:,:,:,l],axis=(0,1))
            
        else:
            if ATTO_twr:
                Rij = -Rstress.data[loc[0],loc[1],:,l] #for tower topography analysis
            elif Flat_twr:
                Rij = -Rstress.data[locx,locx,:,l] #for tower flat analysis
            else:
                Rij = -np.nanmean(Rstress.data[:,:,:,l],axis=(0,1))
        
        if ATTO_twr:
            SGSij = dataNAN[SGSij_names[l]][loc[0],loc[1],:] #for tower topography analysis
        elif Flat_twr:
            SGSij = dataNAN[SGSij_names[l]][locx,locy,:] #for tower flat analysis
        else:
            SGSij = np.nanmean(dataNAN[SGSij_names[l]],axis=(0,1))
       
        Tauij = Rij + SGSij
        
        #PLotting arguments:
            
        axs[m,n].plot(Rij[5:],y[:-5],color='gray',linestyle='--')
        axs[m,n].plot(SGSij[5:],y[:-5],color='gray',linestyle='-.')
        axs[m,n].plot(Tauij[5:],y[:-5],color='black',linestyle='-')

        axs[m,n].set_ylim(y[0], y[-1])
        axs[m,n].axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
        axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
        
     
        axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
        axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
        axs[m,n].minorticks_on()

        axs[m,n].set_xscale('linear')
        #axs[m,n].set_title(f'Case = {case}')
        axs[m,n].set_xlabel(f'{Rij_names[l]}')
        
        l = l+1
    
    
axs[0,0].set_ylabel(r'$(z-d)/h$')
axs[1,0].set_ylabel(r'$(z-d)/h$')

plt.tight_layout()
plt.show()

# Vertical profile of the vertical shear stress together:
    
Dxz = np.nanmean((dataNAN['u']*dataNAN['w']),axis=(0,1)) - (np.nanmean(dataNAN['u'],axis=(0,1))*np.nanmean(dataNAN['w'],axis=(0,1)))
Dyz = np.nanmean((dataNAN['v']*dataNAN['w']),axis=(0,1)) - (np.nanmean(dataNAN['v'],axis=(0,1))*np.nanmean(dataNAN['w'],axis=(0,1)))

Dw =  np.sqrt(Dxz**2 + Dyz**2)
Rw = np.nanmean((np.sqrt(Rstress.data[:,:,:,4]**2 + Rstress.data[:,:,:,5]**2)),axis=(0,1))
SGSw = np.nanmean((np.sqrt(dataNAN['txz']**2 + dataNAN['tyz']**2)),axis=(0,1))
    
tau_wall1D = Rw + SGSw + Dw
    
fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

axs.plot(tau_wall1D[5:],y[:-5],color='black',linestyle='-', label=r"$\tau_w$")
axs.plot(Rw[5:],y[:-5],color='black',linestyle='--',label='$R_w$')
axs.plot(SGSw[5:],y[:-5],color='black',linestyle='-.',label='$SGS_w$')
axs.plot(Dw[5:],y[:-5],color='black',linestyle=':',label='$D_w$')
# axs.hlines((canopyH-(dispH[case]/zi))/canopyH,0,4.5,colors='gray',linestyles='-')
axs.set_ylim(y[0], y[-1])
axs.set_xlim(0, 4.5)
axs.legend()

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xlabel(r'$\tau_w$')
axs.set_ylabel(r'$(z-d)/h$')
    
#%%
level = 25

figure = plt.figure()
plt.semilogx(z_d,np.mean(data['u'],axis=(0,1)))
plt.semilogx(z_d[level],np.mean(data['u'],axis=(0,1))[level],'ok')
plt.xlabel('z')
plt.ylabel('u')

uhi = np.mean(data['u'],axis=(0,1))[level]
zh = z_d[level]
zhi = zh/np.exp(0.4*uhi)

#%% Compute the corresponding values of z0hi and u* above the canopy:

from scipy.optimize import curve_fit     

# Nz_SLayer = Nz-1
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
    # U_mean = np.nanmean(U[:,:,:Nz_SLayer],axis=(0,1))
    U_mean = U[:Nz_SLayer]

    #Data used for the Inertial logarithmic fit above the RSL. --------------
    #------------------------------------------------------------------------
    level1 = 70 #20 #70
    level2 = 95 #30 #95
    # level3 = 70
    # level4 = 90

    u1 = U_mean[level1] #This the velocity @ z-d/zi = 0.253
    u2 = U_mean[level2] #This the velocity @ z-d/zi = 0.350
    # u3 = U_mean[level3]
    # u4 = U_mean[level4]
    
    z_data = np.array([z_d[level1], z_d[level2]])#, z_d[level3], z_d[level4]])#
    U_data = np.array([u1, u2])#, u3, u4])


    coefs, pcov = curve_fit(log_fit, z_data, U_data)
    # pos_zd = np.maximum(z_d[0:Nz_SLayer], 1e-10)
    u_fit = coefs[0]*np.log(coefs[1]*(z_d[0:Nz_SLayer]))

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    z0hi = (1/coefs[1]) #Normalized values of 'z0hi', hence z0hi/zi.
    ustar = U_mean[level1]/((1/kappa)*np.log(z_d[level1]/z0hi)) #Normalized values of 'ustar', hence u*/uscale.


    return(z0hi,ustar,U_mean,U_data,z_data,u_fit)

if ATTO_twr:
    [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                                                        dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],True)
elif Flat_twr:
    [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'][locx,locy,int(np.where(dist[locx,locy,:]>0)[0][0]):],\
                                                            dataNAN['v'][locx,locy,int(np.where(dist[locx,locy,:]>0)[0][0]):],True)
else:
    [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'],dataNAN['v'])



print(f'The corresponding value of z0hi/zi is {z0hi}, and u*/uscale is {ustar}')


fig, ax=plt.subplots(1,2)

ax[0].plot(U_mean,z_d[0:Nz_SLayer]/(39/zi),'-k',label='$\overline{U}$')
ax[0].plot(U_data,z_data/(39/zi),'ok',label='$\overline{U}$ @ lvl 1,2')
ax[0].plot(u_fit,z_d[0:Nz_SLayer]/(39/zi),'--k',label='log fit')
# ax[0].hlines((canopyH-(dispH[case]/zi))/canopyH,0,30,colors='gray',linestyles='-')
ax[0].set_xlabel(r'$\overline{u}(z)/u_*$')
ax[0].set_ylabel(r'$\frac{z-d}{h}$')
ax[0].set_xlim([0, 30])
ax[0].set_ylim([z_d[0]/(39/zi), z_d[Nz_SLayer]/(39/zi)])
ax[0].legend()


ax[1].semilogx(z_d[0:Nz_SLayer],U_mean,'-k',label='$\overline{U}$')
#ax.plot(U_data,z_data/canopyH,'ok')
ax[1].semilogx(z_d[0:Nz_SLayer],u_fit,'--k',label='log fit')
ax[1].hlines(0,z_d[0],z_d[-1],colors='gray',linestyles='-')
ax[1].plot(z0hi,0,'ok',label='$z_{0,hi}$')
ax[1].set_ylabel(r'$\overline{u}(z)/u_*$')
ax[1].set_xlabel(r'$\frac{z-d}{z_i}$')
ax[1].legend()
#ax.set_xlim([0, 30])
#ax.set_ylim([z_d[0]/canopyH, z_d[Nz_SLayer]/canopyH])

plt.tight_layout()


#%% Compute the phi_m and plot both the derivative and mean wind speed:

# Nz_SLayer = int(Nz/2)

# Computing phiM --------------------
if ATTO_twr:
    phi_m_1D = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                      dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                          dataNAN['dudz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                              dataNAN['dvdz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],ustar)
elif Flat_twr:
    phi_m_1D = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'][locx,locy,:], dataNAN['v'][locx,locy,:], dataNAN['dudz'][locx,locy,:],\
                              dataNAN['dvdz'][locx,locy,:],ustar)
else:
    phi_m_3D = phi_m(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'][:,:,5:],dataNAN['v'][:,:,5:],dataNAN['dudz'][:,:,5:],dataNAN['dvdz'][:,:,5:],ustar)
    phi_m_1D = np.nanmedian(phi_m_3D,axis=(0,1))



## Plotting the Mean wind speed and the corresponding velocity gradient.

fig, ax=plt.subplots(1,1)

ax.plot(phi_m_1D,z_on_h[0:Nz_SLayer],'-k')
# ax.hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
ax.vlines(1,z_on_h[0],z_on_h[-1],colors='gray',linestyles='-')
ax.set_xlabel(r'$\phi_m(z)$')
ax.set_ylabel(r'$\frac{z-d}{h}$')
ax.set_xlim([-1, 2])
ax.set_ylim([z_on_h[0], z_on_h[Nz_SLayer]])



#%% Compute the velocity gradient for all study cases:

'''
#Stuff needed to filter a given signal:
    
from scipy.signal import butter,filtfilt
    
nyq = 2 * (1/dz)  # Nyquist Frequency
order = 2
cutoff = 0.2*(1/dz)
fs = dz

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#Low pass filter the derivative:
dydz_filt = butter_lowpass_filter(dydz, cutoff, fs, order)

'''

phiM_3D_all_cases = xr.DataArray(np.ones(shape = (Nx,Ny,Nz_SLayer,NumCases),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':[cases[num]]})

#Flag that determines whether we are pursuing an anisotorpy analysis.
anisotropy_analysis = 'true'
anisotropy_compute = 'false'

# fig, ax=plt.subplots(1,1)

for num in range(0,NumCases):

    case = cases[num]
    path = directory + cases[num] + '/'
    os.chdir(path)
    
    # print(path)
    
    Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                            dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                            'Rzz','Rxy','Rxz','Ryz']})

    Rstress = ReynoldsStressUVP(Nx,Ny,Nz,data_tavg['u'],data_tavg['v'],data_tavg['w'],data_tavg['uu'],\
                              data_tavg['vv'],data_tavg['ww'],data_tavg['uv'],data_tavg['uw'],data_tavg['vw'])#,\
                                  # data_tavg['txx'],data_tavg['tyy'],data_tavg['tzz'],data_tavg['txy'],data_tavg['txz'],data_tavg['tyz'])
    
    # Rstress.data[:,:,:,3] = np.zeros((Nx,Ny,Nz),'d',order='F')
    # Rstress.data[:,:,:,4] = np.zeros((Nx,Ny,Nz),'d',order='F')
    # Rstress.data[:,:,:,5] = np.zeros((Nx,Ny,Nz),'d',order='F')
    # [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'],dataNAN['v'])   
        
    # z0hi_dict[cases[num]] = z0hi
    # ustar_dict[cases[num]] = ustar    
        
    # #Compute phi_M:
    # #---------------------

    # phi_m_3D = phi_m(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'],dataNAN['v'],dataNAN['dudz'],dataNAN['dvdz'],ustar)

    # phiM_3D_all_cases[:,:,:,num] = phi_m_3D

    # phi_m_1D = np.nanmedian(phi_m_3D,axis=(0,1))
    
    # #ax.plot(phi_m_1D,z_d[0:Nz_SLayer]/canopyH,'-',label=case) #Plot of all PhiM
    # ax.plot(phi_m_1D,z_on_h[0:Nz_SLayer],'-',label=case) #Plot of all PhiM
    
    
    #Anisotropy analysis:
    if (anisotropy_analysis == 'true'):
        
        if (anisotropy_compute == 'true'):
            
            [xB,yB,AnisType_1D,lambda3] = Anisotropy_Clustering(Nx,Ny,Nz,Rstress)

            yB_1D = np.ndarray.flatten(yB)
            xB_1D = np.ndarray.flatten(xB)
            lambda3_1D = np.ndarray.flatten(lambda3)


            Anisotropy_clustering = xr.DataArray(np.zeros(shape = (Nx*Ny*Nz,4),order='F'),\
                                dims=('space','variable'), coords = {'variable':['xB_1D','yB_1D','AnisType_1D','lambda3_1D']})
                
            Anisotropy_clustering[:,0] = xB_1D; Anisotropy_clustering[:,1] = yB_1D; Anisotropy_clustering[:,2] = AnisType_1D; Anisotropy_clustering[:,3] = lambda3_1D 

            os.chdir(pathOUT)
            Anisotropy_clustering.to_netcdf('Anisotropy_clustering_'+case+'.nc')
            
        else:
                
            Anisotropy_clustering = xr.open_dataarray(pathOUT + 'Anisotropy_clustering_'+case+'_DIAG.nc')

            xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
            yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
            AnisType_1D = np.copy(Anisotropy_clustering.data[:,2]) 
            # lambda3_1D = np.copy(Anisotropy_clustering.data[:,3])

        #------------ End of the IF statement.
    #--- End of Anisotropy analysis.
    

#Saving the phiM_3D_all_cases variable to a file.
os.chdir(pathOUT)
# phiM_3D_all_cases.to_netcdf('phiM_3D_all_cases.nc')

#ax.hlines((canopyH-(dispH[case]/zi))/canopyH,-0.5,1.5,colors='gray',linestyles='-')
# ax.vlines(1,z_on_h[0],z_on_h[-1],colors='gray',linestyles='-')
# ax.set_xlabel(r'$\phi_m(z)$')
# ax.set_ylabel(r'$\frac{z}{h}$')
# #ax.set_xlim([-0.5, 1.5])
# ax.set_ylim([z_on_h[0], z_on_h[Nz_SLayer]])

# ax.axhline(y = 2.754, xmin=-10, xmax=8,color='gray',linestyle='-.') #Homogenous
# ax.axhline(y = 5.859, xmin=-10, xmax=8,color='gray',linestyle='-.') #g1200
# ax.axhline(y = 4.557, xmin=-10, xmax=8,color='gray',linestyle='-.') #g800
# ax.axhline(y = 3.255, xmin=-10, xmax=8,color='gray',linestyle='-.') #g400

#%%Plot yslice of yB

tmpYB = copy.deepcopy(np.reshape(yB_1D,(Nx,Ny,nzTot))[:,:,5:])
tmpYB[dist[:,:,5:]<0] = float('nan')
cmap = ColorAnisotropy()

yslice = 63

fig,axs = plt.subplots(1,1,figsize=(5,5),tight_layout=True)
p1 = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),tmpYB[:,yslice,:].T,cmap=cmap,shading='gouraud',vmin=0,vmax=np.sqrt(3)/2)
axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
axs.set_xlabel(r'$x/z_i$')
axs.set_ylabel(r'$z/h_C$')
axs.set_title(r'$y_{B,d}$ (yslice)')
cbar1 = plt.colorbar(p1)
fig.suptitle(f'yslice = {yslice*dy*zi}m')

# plt.savefig('/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper2/' + 'yB_d_atto.png',dpi=300,facecolor='None', edgecolor='None')

plt.show()


#%%Tower analysis using multiple towers

figure1, figure2, z0hi, ustar, z_over_d, z_over_d_m, phi_m_v2 = Twr_Anis_Multi(dataNAN,dist,xB_1D,yB_1D,AnisType_1D,Nx,Ny,Nz,Nz_SLayer,coord,d_dim,z_uvp,zi,canopyH,\
                                               pathOUT,'phi_cluster_valley','phi_median_valley',False)

#%%Plot phi vs z/disp_H

fig, axs = plt.subplots(1,1,tight_layout=True)

axs.plot(phi_m_v2,z_over_d_m,c='k')
axs.axvline(1,ls='--',c='k')
axs.axhline(9.5,ls='--',c='k')
axs.set_ylim(np.min(z_over_d),np.max(z_over_d_m))
axs.set_xlim(-0.50,2)
axs.set_ylabel('z/disp_H',fontsize=15)
axs.set_xlabel(r'$\phi_M$',fontsize=15)
axs.grid()

plt.show()

#%% Graphical Representation of the results with Clustering analysis, ploting individual cluster contributions:


from scipy import stats

cmap = ColorAnisotropy()    

#Create a 3D matrix that includes the corresponding heights.
z3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,Nx):
        for j in range(0,Ny):
            z3D[i,j,k] = z_d[k]

if ATTO_twr:
    z3D_1D = z3D[loc[0],loc[1],:]
    dist_1D = dist[loc[0],loc[1],:]
elif Flat_twr:
    z3D_1D = z3D[locx,locy,:]
    dist_1D = dist[locx,locy,:Nz_SLayer]
else:
    z3D_1D = np.ndarray.flatten(z3D)
    dist_1D = np.ndarray.flatten(dist[:,:,5:Nz_SLayer+5])


num = 0
case = cases[num]
print(f'The current study case is {cases[num]}')

#Read the phi_m data from the stored file: 
# phiM_3D_all_cases = xr.open_dataarray(opath + 'phiM_3D_all_cases.nc')
if ATTO_twr:
    [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                                                        dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):])
    phi_M_1D = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                      dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                          dataNAN['dudz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],\
                              dataNAN['dvdz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):],ustar)
elif Flat_twr:
    [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'][locx,locy,:], dataNAN['v'][locx,locy,:])
    phi_M_1D = phi_m_loc(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'][locx,locy,:], dataNAN['v'][locx,locy,:], dataNAN['dudz'][locx,locy,:], dataNAN['dvdz'][locx,locy,:],ustar)
else:
    [z0hi,ustar,U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_SLayer,z_d,dataNAN['u'],dataNAN['v'])
    phi_m_3D = phi_m(Nx,Ny,Nz_SLayer,z_d,dataNAN['u'][:,:,5:],dataNAN['v'][:,:,5:],dataNAN['dudz'][:,:,5:],dataNAN['dvdz'][:,:,5:],ustar)
    phi_M_1D = phi_m_3D.flatten()

# phi_M_1D = np.ndarray.flatten(phiM_3D_all_cases.data[:,:,:,num])

#Need to load the corresponding Anisotorpy file:
#------------------------------------------------------------------------------

os.chdir(path)

print(path)

Anisotropy_clustering = xr.open_dataarray(pathOUT + 'Anisotropy_clustering_'+case+'.nc')
xB_1D = np.reshape(np.copy(Anisotropy_clustering.data[:,0]),(Nx,Ny,Nz))[:,:,5:Nz_SLayer+5].flatten() 
yB_1D = np.reshape(np.copy(Anisotropy_clustering.data[:,1]),(Nx,Ny,Nz))[:,:,5:Nz_SLayer+5].flatten() 
AnisType_1D = np.reshape(np.copy(Anisotropy_clustering.data[:,2]),(Nx,Ny,Nz))[:,:,5:Nz_SLayer+5].flatten()

if ATTO_twr:
    xB_loc = xB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):]
    yB_loc = yB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):]
    AnisType_loc = AnisType_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):]
    xB_loc = xB_loc[:Nz_SLayer]; yB_loc = yB_loc[:Nz_SLayer]; AnisType_loc = AnisType_loc[:Nz_SLayer]
elif Flat_twr:
    xB_loc = xB_1D.reshape(Nx,Ny,Nz)[locx,locy,:]
    yB_loc = yB_1D.reshape(Nx,Ny,Nz)[locx,locy,:]
    AnisType_loc = AnisType_1D.reshape(Nx,Ny,Nz)[locx,locy,:]
    xB_loc = xB_loc[:Nz_SLayer]; yB_loc = yB_loc[:Nz_SLayer]; AnisType_loc = AnisType_loc[:Nz_SLayer]

#------------------------------------------------------------------------------

Nclusters = 9 #Number of clusters used to group the anisotorpy.

fig, axs = plt.subplots(nrows=3,ncols=3)
plt.ion()

#Loop through all the clusters (1 to 9) organized in a 3x3 subplot..

#Overlay the median on the denisty plot:
phiM_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); phiM_median.fill(np.NaN)
yB_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); yB_median.fill(np.NaN)

y_ax = z_on_h[0:Nz_SLayer]#/(39/zi)

cluster = 0
for m in range(0,3):
    for n in range(0,3):
        
        cluster = cluster + 1
        
        print(f'cluster = {cluster}')
        
        if ATTO_twr:
            tmp_z3D = z3D_1D[(AnisType_loc == cluster)]
            tmp_phi_u = phi_M_1D[(AnisType_loc == cluster)]
            tmp_yB = yB_loc[(AnisType_loc == cluster)]
        elif Flat_twr:
            tmp_z3D = z3D_1D[(AnisType_loc == cluster) & (dist_1D>0)]
            tmp_phi_u = phi_M_1D[(AnisType_loc == cluster) & (dist_1D>0)]
            tmp_yB = yB_loc[(AnisType_loc == cluster) & (dist_1D>0)]
        else:
            tmp_z3D = z3D_1D[(AnisType_1D == cluster) & (dist_1D>0)]
            tmp_phi_u = phi_M_1D[(AnisType_1D == cluster) & (dist_1D>0)]
            tmp_yB = yB_1D[(AnisType_1D == cluster) & (dist_1D>0)]
        
        NumPoints = np.size(tmp_yB) #Total number of points in a given cluster. This should be larger than 100 to do statistics.
        
        print(f'Total number of points in cluster = {NumPoints}')

        if (cluster <= 9 and NumPoints > 100):
            x = tmp_phi_u
            y = tmp_z3D/(39/zi)
    
            #Developing a "2D Density plot" to better visualize where there are more points.
            tmp = np.array([x,y])
            nbins = 40

            # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
            k = stats.gaussian_kde(tmp)
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            density = k(np.vstack([xi.flatten(), yi.flatten()]))
 
            # plot the density with shading
            #axs.set_title('2D Density with shading')
            axs[m,n].pcolormesh(xi, yi, density.reshape(xi.shape), shading='gouraud', cmap='binary')
 
        for i in range(0,Nz_SLayer-1):
            phiM_median[cluster-1,i] = np.median(tmp_phi_u[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])
            yB_median[cluster-1,i] = np.median(tmp_yB[(tmp_z3D >= z_d[i]) & (tmp_z3D < z_d[i+1])])     
    
        print(f'Done with cluster {cluster}')
        
        if (np.size(tmp_phi_u) > 0):
            sc = axs[m,n].scatter(phiM_median[cluster-1,0:-1],y_ax[0:-1],s=5,marker='o',c = yB_median[cluster-1,0:-1],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

#------------------------------------------------------------------------------

        axs[m,n].set_xlim(-0.5, 2.5)
        axs[m,n].set_ylim(y_ax[0], y_ax[-1])
        axs[m,n].axhline(y = 1, xmin=-10, xmax=8,color='gray',linestyle='--')
        axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
        axs[m,n].axvline(x =1, ymin = 0, ymax=13,color='gray',linestyle='--')
           
        axs[m,n].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
        axs[m,n].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
        axs[m,n].minorticks_on()
    
        axs[m,n].set_xscale('linear')
        axs[m,n].set_title(f'cluster = {cluster}')
    
cbar = plt.colorbar(sc)
    
axs[0,0].set_ylabel(r'$(z-d)/h$')
axs[1,0].set_ylabel(r'$(z-d)/h$')
axs[2,0].set_ylabel(r'$(z-d)/h$')

axs[2,0].set_xlabel(r'$\phi_M$')
axs[2,1].set_xlabel(r'$\phi_M$')
axs[2,2].set_xlabel(r'$\phi_M$')

plt.tight_layout()
plt.show()

# plt.savefig(path +'Figures/'+'dUdz_anisotropy_subplots_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')

#%% Graphical Representation of the results with Clustering in a single subplot:

import matplotlib.ticker


fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

#Median of the gradient profiles as a function of cluster and height. 
#For a fixed cluster (e.g. cluster = 4), we average all values of the gradient
#at a specific height.
 
for cluster in range(1,Nclusters):
    #print(f'{np.count_nonzero(np.isnan(yB_median[cluster,0:-1]))}')
    if (np.count_nonzero(np.isnan(yB_median[cluster,0:-1])) < Nz_SLayer-1):
        #sc = axs.scatter(phiM_median[cluster,0:-1],(z_d[0:Nz_SLayer-1]/canopyH),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
        sc = axs.scatter(phiM_median[cluster,0:-1],(z_on_h[0:Nz_SLayer-1]),marker='o',c = yB_median[cluster,0:-1],cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

# axs.plot(np.nanmean(phiM_3D_all_cases.data[:,:,:,num],axis=(0,1)),(z_on_h[0:Nz_SLayer]),linestyle='-',color='k')

axs.set_xlim(-0.5, 2.5)
#axs.set_ylim(y_ax[0], y_ax[Nz_SLayer-1])
axs.set_ylim(z_on_h[0], z_on_h[Nz_SLayer-1])
#axs.axhline(y = (canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle=':')
#axs.axhline(y = 3*(canopyH - (dispH[cases[num]]/zi))/canopyH, xmin=-10, xmax=8,color='gray',linestyle='-.')
axs.axhline(y = (canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle=':')
#axs.axhline(y = 3*(canopyH/canopyH), xmin=-10, xmax=8,color='gray',linestyle='-.')

axs.set_xscale('linear')
cbar = plt.colorbar(sc)
#axs.set_ylabel(r'$(z-d)/h$')
axs.set_ylabel(r'$z/h$')
axs.set_xlabel(r'$\phi_M$')
axs.set_title(f'{cases[num]}')

axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')

plt.tight_layout()
plt.show()

# plt.savefig(path +'Figures/'+'dUdz_anisotropy_allinone_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')

#%%Plotting the Lumley Triangle xB vs yB
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
# import cmasher as cmr

# cmap_LT = cmr.get_sub_cmap('seismic_r', 0.5, 1)

#Tower Traj as a function of height -------------------------------------------------------------------------------------------------------

# tmpxB = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
# tmpyB = np.zeros((N_twrs,Nz_SLayer),'d',order='F')

# for i in range(len(coord)):
#     loc = coord[i]
    
#     tmpxB[i,:] = np.reshape(xB_1D,(Nx,Ny,Nz))[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
#     tmpyB[i,:] = np.reshape(yB_1D,(Nx,Ny,Nz))[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    
# xB_median = np.nanmedian(tmpxB,axis=(0))
# yB_median = np.nanmedian(tmpyB,axis=(0))

# xB = np.reshape(xB_1D,(Nx,Ny,Nz))[5,5,5:]
# yB = np.reshape(yB_1D,(Nx,Ny,Nz))[5,5,5:]

#Filter 3D points where P-D is roughly 0 ---------------------------------------------------------------------------------------------------

# terms_bdg = np.load(pathOUT + 'TKE_terms.npy',allow_pickle='TRUE').item()

# tmpDIS = copy.deepcopy((terms_bdg['totdis'])[:,:,5:]) # copy.deepcopy(np.nanmean(terms_bdg['totdis'],axis=(1))) 
# tmpRES = copy.deepcopy((terms_bdg['prod'] - terms_bdg['totdis'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['prod']) - (terms_bdg['totdis']),axis=(1))) 

# # tmpRES[(abs(tmpRES)<1)] = 0 #float('nan')

# # terms = [((tmpRES)/abs(tmpDIS))*100]
# terms = [(tmpRES)]

# yB = np.reshape(yB_1D,(Nx,Ny,Nz)); xB = np.reshape(xB_1D,(Nx,Ny,Nz))
# yB[(dist[:,:,:]<0)] = float('nan'); xB[(dist[:,:,:]<0)] = float('nan')

# lvl1 = 1
# lvl2 = 2

# tmpxB = xB[:,:,5:][(dist[:,:,5:]>=39*lvl1) & (dist[:,:,5:]<39*lvl2)].flatten()
# tmpyB = yB[:,:,5:][(dist[:,:,5:]>=39*lvl1) & (dist[:,:,5:]<39*lvl2)].flatten()
# tmpZ = dist[:,:,5:][(dist[:,:,5:]>=39*lvl1) & (dist[:,:,5:]<39*lvl2)].flatten()/39

# nbins = 20
# x = tmpxB
# y = tmpyB

# k = gaussian_kde([x,y])
# xi, yi = np.mgrid[
#     x.min():x.max():nbins*1j,
#     y.min():y.max():nbins*1j
# ]
# z_i = k(np.vstack([
#     xi.flatten(),
#     yi.flatten()
# ])).reshape(xi.shape)


# Plotting the actual triangle----------------------------------------------------------------------------------------------------------------------

fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
# plot a triangle
xc = np.array([0, 1, 0.5])
yc = np.array([0, 0, np.sqrt(3)*0.5])
for i in np.arange(3):
    ip1 = (i+1)%3
    axs.plot([xc[i], xc[ip1]], [yc[i], yc[ip1]], 'k', linewidth=2)
# add grid
nsp = 5
lc = np.abs(xc[1]-xc[0])
dc = lc/nsp
cl = np.zeros([3*(nsp-1), 3])
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        cl[k,i] = dc * (j+1)
        cl[k,ip1] = dc * (nsp-j-1)
xl = np.dot(xc, cl.transpose())
yl = np.dot(yc, cl.transpose())
nl = xl.size
for i in np.arange(3):
    ip1 = (i+1)%3
    for j in np.arange(nsp-1):
        k = i * (nsp-1) + j
        kp = (ip1 * (nsp-1) + nsp - j - 2) % nl
        axs.plot([xl[k], xl[kp]], [yl[k], yl[kp]], '--k', linewidth=0.75)
# plain strain limit
c1ps = np.array([2/3, 1/3, 0])
x1ps = np.dot(xc, c1ps.transpose())
y1ps = np.dot(yc, c1ps.transpose())
c2ps = np.array([0, 0, 1])
x2ps = np.dot(xc, c2ps.transpose())
y2ps = np.dot(yc, c2ps.transpose())
axs.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)
# add labels
labels = ['2-comp axi', '1-comp', 'Isotropic']
lbpx = xc
dshift = 0.05
lbpy = [yc[0]-dshift*lc, yc[1]-dshift*lc, yc[2]+dshift*lc]
for i in np.arange(3):
    axs.text(lbpx[i], lbpy[i], labels[i], ha='center', fontsize=12)
label_side1 = 'Prolate'
label_side2 = 'Oblate'
label_side3 = 'Two-component'
axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')

#-----------------------------------------------------------------------------------------------------------------------------------------

z = z_uvp
z_on_h = z/(39/zi)

### - Plot option 1
# axs.pcolormesh(xi, yi, z_i, cmap='hot_r')
# sc = axs.scatter(tmpxB,tmpyB,s=1,alpha=0.5)#,c=tmpZ,cmap='jet')
# axs.pcolormesh(xi, yi, z_i, cmap='hot_r')
# axs.contour(xi,yi,z_i)

### - Plot option 2
# for i in range(len(coord)):
#     sc = axs.scatter(tmpxB[i,21:],tmpyB[i,21:],s=1,c=z_on_h[21:Nz_SLayer],cmap='jet',alpha=0.5)
#     # sc = axs.plot(tmpxB[i,:],tmpyB[i,:])
#     # sc = axs.plot(tmpxB[i,21:],tmpyB[i,21:],c='r')
# axs.plot(xB_median[21:],yB_median[21:],'k',lw=2)
h1 = 39
### - Plot option 3
for i in range(1,55):
    # lev1 = i
    # lev2 = i+1
    
    
    xB = np.reshape(xB_1D,(Nx,Ny,Nz))[(dist>h1) & (dist<h1+10)].flatten()
    yB = np.reshape(yB_1D,(Nx,Ny,Nz))[(dist>h1) & (dist<h1+10)].flatten()
    
    h1 = h1+10
    
    nbins = 20
    x = xB
    y = yB
    
    k = gaussian_kde([x,y])
    xi, yi = np.mgrid[
    x.min():x.max():nbins*1j,
    y.min():y.max():nbins*1j
    ]
    z_i = k(np.vstack([
    xi.flatten(),
    yi.flatten()
    ])).reshape(xi.shape)
    # Find the point with the highest probability
    max_idx = np.argmax(z_i)
    x_max, y_max = xi.ravel()[max_idx], yi.ravel()[max_idx]
    
    # Add a marker for the highest probability point
    axs.plot(x_max, y_max, 'ro', markersize=2, label="Max Probability")
    
    # Draw contour for the extremes (e.g., 95th percentile of the KDE field)
    contour_levels = [np.percentile(z_i, 95)]
    contour = axs.contour(xi, yi, z_i, levels=contour_levels, colors='k')
    
    # Create a mask to color the area within the contour
    # cmap = ListedColormap(['lightblue'])
    jet_cmap = plt.get_cmap('hot_r')  # Get the full jet colormap
    cmap_section = ListedColormap(jet_cmap(np.linspace(i*10e-2, i*10e-2, 256)))  # Select a section of the colormap
    axs.contourf(xi, yi, z_i, levels=contour_levels + [z_i.max()], cmap=cmap_section, alpha=0.6)


axs.axhline(0.38,0,2,c='k',ls='-.')
axs.axhline(0.36,0,2,c='k',ls='-.')
axs.set_xlim(-0.2,1.2)
axs.set_ylim(-0.1,1)
axs.set_xlabel(r'$x_B$',fontsize=15)
axs.set_ylabel(r'$y_B$',fontsize=15)
xticks = [0,0.5,1]
yticks = [0,0.38,np.sqrt(3)/2]
ylabels = ['0','0.38',r'$\sqrt{3}/2$']
axs.set_xticks(xticks)
axs.set_yticks(yticks,labels=ylabels)
# cbar = plt.colorbar(sc,label=r'$z/h_c$')

plt.show()

# axs.set_title(f'valley',fontsize=15)

# plt.savefig(pathOUT +'Figures/'+'LumleyTri_traj_hieght.png',dpi=300,facecolor='white', edgecolor='white')

#%%Plotting the third eigenvalue

yB = np.reshape(yB_1D,(Nx,Ny,Nz)); xB = np.reshape(xB_1D,(Nx,Ny,Nz)); lambda3 = np.reshape(lambda3_1D,(Nx,Ny,Nz))
yB[(dist[:,:,:]<0)] = float('nan'); xB[(dist[:,:,:]<0)] = float('nan'); lambda3[(dist[:,:,:]<0)] = float('nan')

yB_lol = (3*lambda3 + 1)*np.sqrt(3)/2
Bzz = terms_ptb['w2_t']/(terms_ptb['u2_t'] + terms_ptb['v2_t'] + terms_ptb['w2_t']) - 1/3

yslice=150

fig,axs = plt.subplots(1,1,figsize=(10,6),tight_layout=True)
pl = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),lambda3[:,yslice,5:].T,cmap='coolwarm',vmin=-0.30,vmax=0)
axs.set_xlabel('x/zi')
axs.set_ylabel('z/H')
cbar = plt.colorbar(pl)

#%% Combining the Anisotropy analysis with the TKE analysis: 

num = 0
case = cases[num]

print(f'The current study case is {cases[num]}')


#Need to load the corresponding Anisotorpy file:
#------------------------------------------------------------------------------

# path = directory + 'RAV_' + cases[num] +'/'
# os.chdir(path)

# print(path)

# Anisotropy_clustering = xr.open_dataarray(pathOUT + 'Anisotropy_clustering_'+case+'.nc')
# xB_1D = np.copy(Anisotropy_clustering.data[:,0]) 
# yB_1D = np.copy(Anisotropy_clustering.data[:,1]) 
# AnisType_1D = np.copy(Anisotropy_clustering.data[:,2])
#------------------------------------------------------------------------------

Nclusters = 9 #Number of clusters used to group the anisotorpy.

#Need to load the corresponding TKE file:
#------------------------------------------------------------------------------

# path_in = directory + '/RAV_' + cases[num] +'/TKE_RAV_Output/' 
# print(path_in)

terms_bdg = np.load(pathOUT + 'TKE_terms.npy',allow_pickle='TRUE').item()


#%% Analyze the TKE terms as a function of Turb. Anisotropy

#-------Set the TKE terms to study:--------------------------------------------
n = 4
labels = ['Production', 'Dissipation', 't-Transport', 'p-Transport','Prod-Dissip','Advection','Residual']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.
#keys = ['adv_h','adv_v','prod_h','prod_v','uturb_h','uturb_v','pturb_h','pturb_v','prod_dudz','canopy','dissip','sum']

terms = [terms_bdg['prod_v']+terms_bdg['prod_h'],-terms_bdg['dissip']-terms_bdg['canopy'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
         terms_bdg['pturb_h'] + terms_bdg['pturb_v'],(terms_bdg['prod']) - (terms_bdg['totdis']),
         terms_bdg['adv_v'] + terms_bdg['adv_h'],terms_bdg['res']]

TKE_term = terms[n]
TKE_term[(dist<0)] = float('nan')

z = z_uvp
z_on_h = z/(39/zi)

#Plot the TKE_term clustering multiple towers at a similar height

fig,fig2,z_over_d,z_over_d_m,tke_m = Twr_TKE_Multi(Nx, Ny, Nz, Nz_SLayer, z_on_h, coord, xB_1D, yB_1D, AnisType_1D, TKE_term, dist, Nclusters, labels, n,\
                         pathOUT,'prod_dis_cluster_valley','prod_dis_median_valley',z_uvp,zi,d_dim,False)
    
#%%Plot TKE term vs z/disp_H

fig, axs = plt.subplots(1,1,tight_layout=True)

axs.plot(tke_m,z_over_d_m,c='k')
axs.axvline(0,ls='--',c='k')
axs.axhline(4.5,ls='--',c='k')
axs.set_ylim(0,np.max(z_over_d_m))
# axs.set_xlim(-0.50,2)
axs.set_ylabel('z/disp_H',fontsize=15)
axs.set_xlabel(r'$TKE$',fontsize=15)
axs.grid()

plt.show()

#%%Plot sigma_w vs yB to check the dependence of sigma_w on yB

from scipy.stats import gaussian_kde

sigma_w = (np.sqrt(terms_ptb['w2_t']))
yB = np.reshape(yB_1D,(Nx,Ny,Nz))

height1 = 39
height2 = 700

dist_flat = dist.flatten()
yB_flat = yB.flatten()[(dist_flat>height1) & (dist_flat<height2)]
sigma_w_flat = sigma_w.flatten()[(dist_flat>height1) & (dist_flat<height2)]

nbins = 20
x_2 = yB_flat
y_2 = sigma_w_flat

k_2 = gaussian_kde([x_2,y_2])
xi_2, yi_2 = np.mgrid[
    x_2.min():x_2.max():nbins*1j,
    y_2.min():y_2.max():nbins*1j
]
z_i_2 = k_2(np.vstack([
    xi_2.flatten(),
    yi_2.flatten()
])).reshape(xi_2.shape)

fig, axs = plt.subplots(1,1,tight_layout=True)
axs.pcolormesh(xi_2,yi_2,z_i_2,cmap='hot_r')
axs.set_xlabel(f'$y_B$',fontsize=15)
axs.set_ylabel(f'$\sigma_w$',fontsize=15)
axs.set_title(f'From {height1}m to {height2}m. Diag Re.')
axs.text(0.15, 0.95, f'r = {round(np.corrcoef(yB_flat,sigma_w_flat)[0,1],2)}',transform=axs.transAxes, fontsize=14,verticalalignment='top')
plt.show()

# fig, axs = plt.subplots(1,1,figsize=(6,6),tight_layout=True)
# sc = axs.scatter(yB_flat,sigma_w_flat,s=1,c=dist_flat[(dist_flat>height1) & (dist_flat<height2)],cmap='hot_r')
# cbar = plt.colorbar(sc,label='z[m]')
# axs.set_xlabel('yB',fontsize=15)
# axs.set_ylabel(f'$\sigma_w$',fontsize=15)
# axs.set_title(f'From {height1}m to {height2}m.')
# axs.text(0.65, 0.95, f'r = {round(np.corrcoef(yB_flat,sigma_w_flat)[0,1],2)}',transform=axs.transAxes, fontsize=14,verticalalignment='top')
# plt.show()

# plt.savefig(pathOUT +'Figures/'+'yB_vs_sigmaw_diagRe_39to700_dens.png',dpi=300,facecolor='white', edgecolor='white')

#%% Plot profiles of P-D vs height

# Nz_SLayer = 50
from operator import add
from operator import sub

tmpDIS = copy.deepcopy(terms_bdg['totdis'])
tmpRES = copy.deepcopy((terms_bdg['prod']) - (terms_bdg['totdis']))

tmpRES[(abs(tmpRES)<1)] = 0
tmpRES[(dist[:,:,:]<0)] = float('nan')

level = 10

tmpRES = (tmpRES/abs(tmpDIS))*100

x_z = copy.deepcopy(dist)[:,:,:][(dist[:,:,:]>39) & (dist[:,:,:]<level*39)]
y_TKE = tmpRES[:,:,:][(dist[:,:,:]>39) & (dist[:,:,:]<level*39)]

TKE_median = []
TKE_std = []
z_mean = []
j=36
for i in range(0,200):
    if i == 0:
        TKE_median.append(np.mean(y_TKE[x_z<j]))
        TKE_std.append(np.std(y_TKE[x_z<j]))
        z_mean.append(1)
    else:
        TKE_median.append(np.mean(y_TKE[(x_z>j) & (x_z<j+3)]))
        TKE_std.append(np.std(y_TKE[(x_z>j) & (x_z<j+3)]))
        z_mean.append((j+j+3)/2)
    j+=3
    
z_mean[:] = [x / 39 for x in z_mean]

fig, axs = plt.subplots(1,1,figsize=(6,6),tight_layout=True)
axs.plot(z_mean,TKE_median,c='k',marker='o')
axs.fill_between(z_mean,list(map(sub,TKE_median,TKE_std)),list(map(add,TKE_median,TKE_std)),alpha=0.5)
axs.axhline(0,0,1,c='k',ls='-.')
axs.axvline(1,-2000,20000,c='k',ls='-.')
axs.axvline(2,-2000,20000,c='k',ls='-.')
axs.set_xlabel('z/H')
axs.set_ylabel('P-D')
axs.text(0.65, 0.95, f'r = {round(np.corrcoef(x_z.flatten(),y_TKE.flatten())[0,1],2)}',transform=axs.transAxes, fontsize=14,verticalalignment='top')
axs.set_title(r'h to' + f' {level}h')
# fig.suptitle(f'$y = {yslice*dy*zi}m$')

# plt.savefig(pathOUT +'Figures/'+'P_D_prof_corr_10.png',dpi=300,facecolor='white', edgecolor='white')


#%%
Rstress = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,6),order='F'),\
                        dims=('x','y','z','variable'), coords = {'variable':['Rxx','Ryy',\
                        'Rzz','Rxy','Rxz','Ryz']})

Rstress = ReynoldsStress(Nx,Ny,Nz,dataNAN['u'],dataNAN['v'],dataNAN['w'],dataNAN['uu'],\
                         dataNAN['vv'],dataNAN['ww'],dataNAN['uv'],dataNAN['uw'],dataNAN['vw'])

R_uw_twr = np.zeros((len(coord),Nz_SLayer)); R_vw_twr = np.zeros((len(coord),Nz_SLayer))
SGS_uw_twr = np.zeros((len(coord),Nz_SLayer)); SGS_vw_twr = np.zeros((len(coord),Nz_SLayer))
Tau_uw_twr = np.zeros((len(coord),Nz_SLayer)); Tau_vw_twr = np.zeros((len(coord),Nz_SLayer))
D_uw_twr = np.zeros((len(coord),Nz_SLayer)); D_vw_twr = np.zeros((len(coord),Nz_SLayer))
u_twr = np.zeros((len(coord),Nz_SLayer)); v_twr = np.zeros((len(coord),Nz_SLayer)); w_twr = np.zeros((len(coord),Nz_SLayer))
u2_t = np.zeros((len(coord),Nz_SLayer)); v2_t = np.zeros((len(coord),Nz_SLayer)); w2_t = np.zeros((len(coord),Nz_SLayer))
tke = np.zeros((len(coord),Nz_SLayer))

for i in range(len(coord)):
    loc = coord[i]
    u_twr[i,:] = dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    v_twr[i,:] = dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    w_twr[i,:] = dataNAN['w'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    u2_t[i,:] = dataNAN['uu'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer] \
        - dataNAN['u'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]**2
    v2_t[i,:] = dataNAN['vv'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer] \
        - dataNAN['v'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]**2
    w2_t[i,:] = dataNAN['ww'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer] \
        - dataNAN['w'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]**2
    tke[i,:] = (u2_t[i,:] + v2_t[i,:] + w2_t[i,:])/2
    R_uw_twr[i,:] = - Rstress.data[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer,4]
    R_vw_twr[i,:] = - Rstress.data[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer,5]
    SGS_uw_twr[i,:] = dataNAN['txz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    SGS_vw_twr[i,:] = dataNAN['tyz'][loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    Tau_uw_twr[i,:] = R_uw_twr[i,:] + SGS_uw_twr[i,:]
    Tau_vw_twr[i,:] = R_vw_twr[i,:] + SGS_vw_twr[i,:]

D_uw = np.mean(u_twr*w_twr,axis=(0)) - np.mean(u_twr,axis=(0))*np.mean(w_twr,axis=(0))
D_vw = np.mean(v_twr*w_twr,axis=(0)) - np.mean(v_twr,axis=(0))*np.mean(w_twr,axis=(0))

#Vertical profiles of the shear stress:
meanRuw = np.mean(R_uw_twr,axis=(0)); meanRvw = np.mean(R_vw_twr,axis=(0))
std_Ruw = np.std(R_uw_twr,axis=(0)); std_Rvw = np.std(R_vw_twr,axis=(0))
meanSGSuw = np.mean(SGS_uw_twr,axis=(0)); meanSGSvw = np.mean(SGS_vw_twr,axis=(0))
std_SGSuw = np.std(SGS_uw_twr,axis=(0)); std_SGSvw = np.std(SGS_vw_twr,axis=(0))
meanTKE = np.mean(tke,axis=(0))
stdTKE = np.std(tke,axis=(0))

fig, axs = plt.subplots(1,3,figsize=(10,8))
# plt.rcParams.update({'font.size': 16})
axs[0].plot(meanRuw,z_on_h[0:Nz_SLayer],color='red',label='$R_{xz}$')
axs[0].fill_betweenx(z_on_h[0:Nz_SLayer], meanRuw - std_Ruw, meanRuw + std_Ruw, alpha=0.2, color='red')
axs[0].plot(meanSGSuw,z_on_h[0:Nz_SLayer],color='green',label='$SGS_{xz}$')
axs[0].fill_betweenx(z_on_h[0:Nz_SLayer], meanSGSuw - std_SGSuw, meanSGSuw + std_SGSuw, alpha=0.2, color='green')
axs[0].plot(D_uw,z_on_h[0:Nz_SLayer],color='k',label='$D_{uw}$')
axs[1].plot(meanRvw,z_on_h[0:Nz_SLayer],color='red',label='$R_{yz}$')
axs[1].fill_betweenx(z_on_h[0:Nz_SLayer], meanRvw - std_Rvw, meanRvw + std_Rvw, alpha=0.2, color='red')
axs[1].plot(meanSGSvw,z_on_h[0:Nz_SLayer],color='green',label='$SGS_{yz}$')
axs[1].fill_betweenx(z_on_h[0:Nz_SLayer], meanSGSvw - std_SGSvw, meanSGSvw + std_SGSvw, alpha=0.2, color='green')
axs[1].plot(D_vw,z_on_h[0:Nz_SLayer],color='k',label='$D_{vw}$')
axs[2].plot(meanTKE,z_on_h[0:Nz_SLayer],color='k',label='TKE')
axs[2].fill_betweenx(z_on_h[0:Nz_SLayer], meanTKE - stdTKE, meanTKE + stdTKE, alpha=0.2, color='black')

# fig.suptitle('Valley')
for i in range(len(axs)):
    axs[i].set_ylim(z_on_h[0],z_on_h[Nz_SLayer])    
    axs[i].legend(loc='upper right')
    axs[i].grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
    axs[i].grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
    axs[i].axhline(1,-5,5,color='k',ls='--')
axs[0].set_title('UW'), axs[1].set_title('VW'), axs[2].set_title('TKE')
axs[0].set_xlabel('$R_{uw}$'), axs[1].set_xlabel('$R_{vw}$'), axs[2].set_xlabel('TKE')
axs[0].set_ylabel('z/h')

# plt.savefig(pathOUT +'Figures/'+'stress_prof_flat' +'.png',dpi=300,facecolor='white', edgecolor='white')


#%%

#-------Run the script --------------------------------------------

#------------------------------------------------------------------------------

#Create a 3D matrix that includes the corresponding heights.
z3D = np.zeros((Nx,Ny,Nz_SLayer),'d',order='F')

for k in range(0,Nz_SLayer):
    for i in range(0,Nx):
        for j in range(0,Ny):
            z3D[i,j,k] = z_on_h[k]
    
if ATTO_twr:
    z3D_1D = z3D[loc[0],loc[1],:]
    dist_1D = dist[loc[0],loc[1],:]
elif Flat_twr:
    z3D_1D = z3D[locx,locy,:]
    dist_1D = dist[locx,locy,:Nz_SLayer]
else:
    z3D_1D = np.ndarray.flatten(z3D)
    dist_1D = np.ndarray.flatten(dist[:,:,0:Nz_SLayer])

#----------------------------------------

if ATTO_twr:
    TKE_term_1D = TKE_term[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
elif Flat_twr:
    TKE_term_1D = TKE_term[locx,locy,0:Nz_SLayer]
else:
    TKE_term_1D = np.ndarray.flatten(TKE_term[:,:,0:Nz_SLayer])
    
if ATTO_twr:
    xB_loc = xB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):]
    yB_loc = yB_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):]
    AnisType_loc = AnisType_1D.reshape(Nx,Ny,Nz)[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):]
    xB_loc = xB_loc[:Nz_SLayer]; yB_loc = yB_loc[:Nz_SLayer]; AnisType_loc = AnisType_loc[:Nz_SLayer]
elif Flat_twr:
    xB_loc = xB_1D.reshape(Nx,Ny,Nz)[locx,locy,:]
    yB_loc = yB_1D.reshape(Nx,Ny,Nz)[locx,locy,:]
    AnisType_loc = AnisType_1D.reshape(Nx,Ny,Nz)[locx,locy,:]
    xB_loc = xB_loc[:Nz_SLayer]; yB_loc = yB_loc[:Nz_SLayer]; AnisType_loc = AnisType_loc[:Nz_SLayer]

#Overlay the median on the denisty plot:
TKE_term_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); TKE_term_median.fill(np.NaN)
yB_median = np.empty((Nclusters,Nz_SLayer),dtype='float'); yB_median.fill(np.NaN)

fig, axs = plt.subplots(nrows=1,ncols=1)
plt.ion()

cmap = ColorAnisotropy()  

cluster = 0
for cluster in range(1,10):
        
    print(f'cluster = {cluster}')
    if ATTO_twr:
        tmp_TKE_term_1D = TKE_term_1D[(AnisType_loc == cluster)]
        tmp_z3D = z3D_1D[(AnisType_loc == cluster)]
        tmp_yB = yB_loc[(AnisType_loc == cluster)]
    elif Flat_twr:
        tmp_TKE_term_1D = TKE_term_1D[(AnisType_loc == cluster) & (dist_1D>0)]
        tmp_z3D = z3D_1D[(AnisType_loc == cluster) & (dist_1D>0)]
        tmp_yB = yB_loc[(AnisType_loc == cluster) & (dist_1D>0)]
    else:
        tmp_TKE_term_1D = TKE_term_1D[(AnisType_1D == cluster) & (dist_1D>0)]
        tmp_z3D = z3D_1D[(AnisType_1D == cluster) & (dist_1D>0)]
        tmp_yB = yB_1D[(AnisType_1D == cluster) & (dist_1D>0)]
    
    for i in range(0,Nz_SLayer-1):
        TKE_term_median[cluster-1,i] = np.median(tmp_TKE_term_1D[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])
        yB_median[cluster-1,i] = np.median(tmp_yB[(tmp_z3D >= z_on_h[i]) & (tmp_z3D < z_on_h[i+1])])     
    
        
    print(f'Done with cluster {cluster}')
        
    if (np.size(tmp_TKE_term_1D) > 0):
        sc = axs.scatter(TKE_term_median[cluster-1,:],z_on_h[0:Nz_SLayer],s=10,marker='o',c = yB_median[cluster-1,:],alpha=0.7,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)


axs.plot(np.nanmean(TKE_term[:,:,0:Nz_SLayer],axis=(0,1)),z_on_h[0:Nz_SLayer],'-k')

#------------------------------------------------------------------------------


x_min = -100
x_max = 100

axs.set_xlim(x_min, x_max)
axs.set_ylim(z_on_h[0], z_on_h[Nz_SLayer])
axs.axhline(y = 1, xmin=x_min, xmax=x_max,color='gray',linestyle='--')
#axs[m,n].axhline(y = 3, xmin=-10, xmax=8,color='gray',linestyle='-.')
axs.axvline(x =0, ymin = 0, ymax=13,color='gray',linestyle='--')
        
     
axs.grid(which='major', axis='both',color='k', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xscale('linear')
    
cbar = plt.colorbar(sc)
    
axs.set_ylabel(r'$z/h$')
axs.set_xlabel(f'{labels[n]}')


plt.tight_layout()
plt.show()


# plt.savefig(path_in +'Figures/'+'Anisotropy_TKE_'+ labels[n] + '_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')




#%% New Colormap:
    
iva_colors_HEX = ["#410d00","#831901","#983e00","#b56601","#ab8437",
              "#b29f74","#7f816b","#587571","#596c72","#454f51"]

#Transform the HEX colors to RGB.
from PIL import ImageColor

iva_colors_RGB = np.zeros((np.size(iva_colors_HEX),3),dtype='int')

for i in range(0,np.size(iva_colors_HEX)):
    iva_colors_RGB[i,:] = ImageColor.getcolor(iva_colors_HEX[i], "RGB")

iva_colors_RGB = iva_colors_RGB[:,:]/(256)

#Transform the array of colors to a list of values. 
colors = iva_colors_RGB.tolist()
#----------------------------------------------------

#The next few lines create a new colormap using IVA's colors:
from matplotlib.colors import LinearSegmentedColormap,ListedColormap

inbetween_color_amount = 10

# the 10 is from the original 10 colors, the 4 is for R, G, B, A
newcolvals = np.zeros(shape=(10 * (inbetween_color_amount) - (inbetween_color_amount - 1), 3))

# add first one already
newcolvals[0] = colors[0]

for i, (rgba1, rgba2) in enumerate(zip(colors[:-1], np.roll(colors, -1, axis=0)[:-1])):
    for j, (p1, p2) in enumerate(zip(rgba1, rgba2)):
        flow = np.linspace(p1, p2, (inbetween_color_amount + 1))
        # discard first 1 since we already have it from previous iteration
        flow = flow[1:]
        newcolvals[ i * (inbetween_color_amount) + 1 : (i + 1) * (inbetween_color_amount) + 1, j] = flow
    
newcolvals

cmap = ListedColormap(newcolvals, name='from_list', N=None)

#%% pcolor test plots of TKE budget terms

tmpDIS = copy.deepcopy(terms_bdg['totdis'][:,:,5:])
tmpRES = copy.deepcopy(((terms_bdg['prod']) - (terms_bdg['totdis']))[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['prod']) - (terms_bdg['totdis']),axis=(1))) 
tmpTUR = copy.deepcopy((terms_bdg['uturb_h'] + terms_bdg['uturb_v'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['uturb_h'] + terms_bdg['uturb_v']),axis=(1))) 
tmpPRE = copy.deepcopy((terms_bdg['pturb_h'] + terms_bdg['pturb_v'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['pturb_h'] + terms_bdg['pturb_v']),axis=(1))) 
tmpADV = copy.deepcopy((terms_bdg['adv_v'] + terms_bdg['adv_h'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['adv_v'] + terms_bdg['adv_h']),axis=(1)))  

tmpDIS[(dist[:,:,5:]<0)] = float('nan')
tmpRES[(dist[:,:,5:]<0)] = float('nan')
tmpTUR[(dist[:,:,5:]<0)] = float('nan')
tmpPRE[(dist[:,:,5:]<0)] = float('nan')
tmpADV[(dist[:,:,5:]<0)] = float('nan')

tmpTUR[(abs(tmpTUR)<1)] = 0 #float('nan')
tmpPRE[(abs(tmpPRE)<1)] = 0 #float('nan')
tmpADV[(abs(tmpADV)<1)] = 0 #float('nan')
tmpRES[(abs(tmpRES)<1)] = 0 #float('nan')

labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

# terms = [tmpRES,tmpTUR,tmpPRE,tmpADV]

terms = [(tmpRES/abs(tmpDIS)),(tmpTUR/abs(tmpDIS)),\
         (tmpPRE/abs(tmpDIS)), (tmpADV/abs(tmpDIS))]

yslice = 150

fig,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

for i in range(0,len(axs)):
    for j in range(0,len(axs[0])):
        if i == 0:
            n = j
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            p2 = axs[i,j].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmp_TKE_term[:,yslice,:]).T,cmap='bwr',alpha=1,vmin=-1,vmax=1)
            # axs[i,j].plot(np.arange(0,Nx)*dx,iintf[:,yslice]*dz/(canopyH),color='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
            # axs[i,j].plot(np.arange(0,Nx)*dx,np.ones((Nx)),color='k')
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
        else:
            n = j+2
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            p2 = axs[i,j].pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmp_TKE_term[:,yslice,:]).T,cmap='bwr',alpha=1,vmin=-1,vmax=1)
            # axs[i,j].plot(np.arange(0,Nx)*dx,iintf[:,yslice]*dz/(canopyH),color='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
            # axs[i,j].plot(np.arange(0,Nx)*dx,np.ones((Nx)),color='k')
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
    
axs[0,0].set_ylabel(r'$z/h_c$',fontsize=15),axs[1,0].set_ylabel(r'$z/h_c$',fontsize=15)
axs[1,0].set_xlabel(r'$x/z_i$',fontsize=15),axs[1,1].set_xlabel(r'$x/z_i$',fontsize=15)
fig.suptitle(f'$y = {yslice*dy*zi}m$')

# plt.savefig(pathOUT +'Figures/'+'TKE_pcolor_y' + str(yslice) + '_ATTO.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()


#%% Scatter plot of yB,xB and TKE term
from scipy.stats import gaussian_kde
from operator import add
from operator import sub

yB = np.reshape(yB_1D,(Nx,Ny,Nz)); xB = np.reshape(xB_1D,(Nx,Ny,Nz))
yB[(dist[:,:,:]<0)] = float('nan'); xB[(dist[:,:,:]<0)] = float('nan')

tmpANIS = copy.deepcopy(yB)

tmpDIS = copy.deepcopy(terms_bdg['totdis'])
tmpRES = copy.deepcopy((terms_bdg['prod']) - (terms_bdg['totdis']))
tmpTUR = copy.deepcopy((terms_bdg['uturb_h'] + terms_bdg['uturb_v'])) 
tmpPRE = copy.deepcopy((terms_bdg['pturb_h'] + terms_bdg['pturb_v'])) 
tmpADV = copy.deepcopy((terms_bdg['adv_v'] + terms_bdg['adv_h']))  

labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

tmpRES[(abs(tmpRES)<1)] = 0

terms = [(tmpRES/abs(tmpDIS))*100,(tmpTUR/abs(tmpDIS))*100,\
         (tmpPRE/abs(tmpDIS))*100, (tmpADV/abs(tmpDIS))*100]

n = 0
yslice = 63
level = 3

TKE_term = terms[n]
tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
tmp_TKE_term[(dist[:,:,:]<0)] = float('nan')

x_yB = tmpANIS[:,:,:][(dist[:,:,:]>39) & (dist[:,:,:]<level*39)]
y_TKE = tmp_TKE_term[:,:,:][(dist[:,:,:]>39) & (dist[:,:,:]<level*39)]
# x_yB = np.nanmean(tmpANIS,axis=(1))[(dist[:,yslice,:]>39) & (dist[:,yslice,:]<level*39)]
# y_TKE = np.nanmean(tmp_TKE_term,axis=(1))[(dist[:,yslice,:]>39) & (dist[:,yslice,:]<level*39)]

# nbins = 100
# x = x_yB
# y = y_TKE

# k = gaussian_kde([x,y])
# xi, yi = np.mgrid[
#     x.min():x.max():nbins*1j,
#     y.min():y.max():nbins*1j
# ]
# z_i = k(np.vstack([
#     xi.flatten(),
#     yi.flatten()
# ])).reshape(xi.shape)

TKE_median = []
TKE_std = []
yB_mean = []
j=0.1
for i in range(0,28):
    if i == 0:
        TKE_median.append(np.nanmedian(y_TKE[x_yB<j]))
        TKE_std.append(np.std(y_TKE[x_yB<j]))
        yB_mean.append(0.05)
    else:
        TKE_median.append(np.nanmedian(y_TKE[(x_yB>j) & (x_yB<j+0.025)]))
        TKE_std.append(np.std(y_TKE[(x_yB>j) & (x_yB<j+0.025)]))
        yB_mean.append((j+j+0.025)/2)
    j+=0.025

fig, axs = plt.subplots(1,1,figsize=(6,6),tight_layout=True)
# sc = axs.scatter(x_yB,y_TKE,c=x_yB,cmap='Greys', vmin=0,vmax=np.sqrt(3)/2)
# axs.pcolormesh(xi, yi, z_i, cmap='hot_r')
# sc = axs.scatter(tmpANIS[:,yslice,:][(dist[:,yslice,:]>39) & (dist[:,yslice,:]<level*39)],tmp_TKE_term[:,yslice,:][(dist[:,yslice,:]>39) & (dist[:,yslice,:]<level*39)],c=tmpANIS[:,yslice,:][(dist[:,yslice,:]>39) & (dist[:,yslice,:]<level*39)],cmap=cmap, vmin=0,vmax=1)
axs.plot(yB_mean,TKE_median,c='k',marker='o')
axs.fill_between(yB_mean,list(map(sub,TKE_median,TKE_std)),list(map(add,TKE_median,TKE_std)),alpha=0.5)
axs.axhline(0,0,1,c='k',ls='-.')
axs.axvline(0.36,-2000,20000,c='k',ls='-.')
axs.axvline(0.38,-2000,20000,c='k',ls='-.')
axs.set_xlabel('Anisotropy Invariant')
axs.set_ylabel('TKE term')
axs.text(0.65, 0.95, f'r = {round(np.corrcoef(x_yB.flatten(),y_TKE.flatten())[0,1],2)}',transform=axs.transAxes, fontsize=14,verticalalignment='top')
# cbar = plt.colorbar(sc,label='yB')
axs.set_title(r'$\frac{' + f'{labels[n]}' + '}{|Dissip|}$ - h to' + f' {level}h')
# fig.suptitle(f'$y = {yslice*dy*zi}m$')

# plt.savefig(pathOUT+'/Figures/PD_yB_y' + str(yslice) + '_ATTO.png',dpi=300,facecolor='white', edgecolor='white')
# plt.savefig(pathOUT+'/Figures/PD_yB_3D_3_DIAG.png',dpi=300,facecolor='white', edgecolor='white')
# 
#%% Some Pcolor figures for yB,xB 

iva_colors = ["#410d00","#831901","#983e00","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]

yB = np.reshape(yB_1D,(Nx,Ny,Nz)); yB[(dist[:,:,:]<0)] = float('nan')
xB = np.reshape(xB_1D,(Nx,Ny,Nz)); xB[(dist[:,:,:]<0)] = float('nan')

tmpANIS = copy.deepcopy(yB[:,:,5:])

# height = 350
z = z_uvp
z_on_h = z/(39/zi)

###########################################################################
## yslices
###########################################################################

yslice = 100

fig, axs = plt.subplots(nrows=1,ncols=1)

# sc = axs.contourf(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),np.transpose(tmpANIS[:,yslice,:]),cmap=cmap,levels=[0,0.33,0.36,0.38,0.9],alpha=0.5)
# p = axs.contour(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(((terms_bdg['prod']) - (terms_bdg['totdis']))[:,yslice,:]),levels=[-100,-5,-1,1,5,100])
# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[:,yslice,:]),levels=[0,0.3,0.32,0.34,0.36,0.38,0.4,0.9],colors=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'black'])#vmin = 0, vmax = np.sqrt(3)/2)
# sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(xB[:,yslice,:]),cmap=cmap, levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
axs.plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
# axs.clabel(p, p.levels, inline=True, fontsize=10)
# sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),np.transpose(np.nanmean(tmpANIS,axis=(1))),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)
sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),np.transpose(tmpANIS[:,yslice,:]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)
# 
cbar = plt.colorbar(sc,label='yB')
    
axs.set_ylabel(r'$z/h$',fontsize=15)
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_title(f'$y_B(y = {yslice*dy*zi}m)$')

plt.tight_layout()
# plt.savefig(pathOUT + 'Figures/' + 'yB_y' + str(yslice) + '_pcolor.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

###########################################################################
## xslices
###########################################################################

# xslice = 200

# fig, axs = plt.subplots(nrows=1,ncols=1)

# sc = axs.contourf(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[xslice,:,:]),cmap=cmap,levels=[0,0.33,0.36,0.38,0.9],alpha=0.5)
# # p = axs.contour(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(((terms_bdg['prod']) - (terms_bdg['totdis']))[xslice,:,:]),levels=[-100,-4,-1,1,4,100],colors=['blue','green','white','yellow','red'])
# # sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[:,yslice,:]),levels=[0,0.3,0.32,0.34,0.36,0.38,0.4,0.9],colors=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'black'])#vmin = 0, vmax = np.sqrt(3)/2)
# # sc = axs.contourf(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(xB[:,yslice,:]),cmap=cmap, levels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# axs.plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
# axs.plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
# # axs.clabel(p, p.levels, inline=True, fontsize=10)
#sc = axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,np.transpose(yB[:,:,height]),cmap = cmap, shading = 'gouraud', vmin = 0, vmax = np.sqrt(3)/2)

# cbar = plt.colorbar(sc,label='yB')
    
# axs.set_ylabel(r'$z/h$',fontsize=15)
# axs.set_xlabel(r'$y/z_i$',fontsize=15)
# axs.set_title(f'$y_B(x = {xslice*dx*zi}m)$')

# plt.tight_layout()

# plt.savefig(pathTurb + 'xB.png',dpi=300,facecolor='white', edgecolor='white')
# plt.savefig(pathOUT + 'Figures/' + 'yB_y' + str(yslice) + '_contourf_DIAG.png',dpi=300,facecolor='white', edgecolor='white')

#%% Pcolor plots of TKE terms with contours of yB - yslices

yB = np.reshape(yB_1D,(Nx,Ny,Nz)); yB[(dist[:,:,:]<0)] = float('nan')
xB = np.reshape(xB_1D,(Nx,Ny,Nz)); xB[(dist[:,:,:]<0)] = float('nan')

tmpANIS = copy.deepcopy(yB[:,:,5:])

tmpDIS = copy.deepcopy((terms_bdg['totdis'])[:,:,5:]) # copy.deepcopy(np.nanmean(terms_bdg['totdis'],axis=(1))) 
tmpRES = copy.deepcopy((terms_bdg['prod'] - terms_bdg['totdis'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['prod']) - (terms_bdg['totdis']),axis=(1))) 
tmpTUR = copy.deepcopy((terms_bdg['uturb_h'] + terms_bdg['uturb_v'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['uturb_h'] + terms_bdg['uturb_v']),axis=(1))) 
tmpPRE = copy.deepcopy((terms_bdg['pturb_h'] + terms_bdg['pturb_v'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['pturb_h'] + terms_bdg['pturb_v']),axis=(1))) 
tmpADV = copy.deepcopy((terms_bdg['adv_v'] + terms_bdg['adv_h'])[:,:,5:]) # copy.deepcopy(np.nanmean((terms_bdg['adv_v'] + terms_bdg['adv_h']),axis=(1)))  

# tmpDIS[(abs(tmpRES)<1)] = float('nan')
tmpTUR[(abs(tmpTUR)<1)] = 0 #float('nan')
tmpPRE[(abs(tmpPRE)<1)] = 0 #float('nan')
tmpADV[(abs(tmpADV)<1)] = 0 #float('nan')
tmpRES[(abs(tmpRES)<1)] = 0 #float('nan')

labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

terms = [(tmpRES)/abs(tmpDIS),tmpTUR/abs(tmpDIS),\
         tmpPRE/abs(tmpDIS),(tmpADV)/abs(tmpDIS)]
    
# levels=[-100,-25,-1,1,25,100]
levels=[-100,-1,1,100]
# levels_2 = [0,0.39,0.43,0.50,1]
levels_2 = [0,0.38,np.sqrt(3)/2]
colors=['blue','white','red']

###########################################################################
## yslices
###########################################################################

yslice = 63

fig,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

for i in range(0,len(axs)):
    for j in range(0,len(axs[0])):
        if i == 0:
            n = j
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            tmp_TKE_term[(dist[:,:,5:]<0)] = float('nan')
            p2 = axs[i,j].contourf(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmp_TKE_term[:,yslice,:]*100).T,colors=colors,alpha=0.5,levels=levels,extend='both')#vmin=-10,vmax=10)
            p2.cmap.set_under('blue')
            p2.cmap.set_over('red')
            sc = axs[i,j].contour(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmpANIS[:,yslice,:]).T,levels=levels_2,colors=['blue', 'green'])
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
            axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
            fig.colorbar(p2)
            axs[i,j].set_title(r'$\frac{' + f'{labels[n]}' + '}{|Dissip|}$')
            # axs[i,j].set_ylim(0,20)
            # axs[i,j].set_xlim(0,1)
        else:
            n = j+2
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
            tmp_TKE_term[(dist[:,:,5:]<0)] = float('nan')
            p2 = axs[i,j].contourf(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmp_TKE_term[:,yslice,:]*100).T,colors=colors,alpha=0.5,levels=levels,extend='both')#vmin=-10,vmax=10)
            p2.cmap.set_under('blue')
            p2.cmap.set_over('red')
            sc = axs[i,j].contour(np.arange(0,Nx)*dx,np.arange(0,Nz-5)*dz/(39/zi),(tmpANIS[:,yslice,:]).T,levels=levels_2,colors=['blue', 'red', 'orange', 'green'])
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift+(39/zi))/(39/zi),ls='--',c='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,(intf[:,yslice]-z_shift)/(39/zi),ls='-',c='k')
            axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
            fig.colorbar(p2)
            axs[i,j].set_title(r'$\frac{' + f'{labels[n]}' + '}{|Dissip|}$')
            # axs[i,j].set_ylim(0,20)
            # axs[i,j].set_xlim(0,1)
    
axs[0,0].set_ylabel(r'$z/h$',fontsize=15),axs[1,0].set_ylabel(r'$z/h$',fontsize=15)
axs[1,0].set_xlabel(r'$x/z_i$',fontsize=15),axs[1,1].set_xlabel(r'$x/z_i$',fontsize=15)
fig.suptitle(f'$y = {yslice*dy*zi}m$')

# plt.savefig(pathOUT + 'Figures/TKE_yB_y' + str(yslice) + '_DIAG_corr.png',dpi=300,facecolor='white', edgecolor='white')
plt.show()

###########################################################################
## xslices
###########################################################################

# xslice = 222

# fig,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

# for i in range(0,len(axs)):
#     for j in range(0,len(axs[0])):
#         if i == 0:
#             n = j
#             TKE_term = terms[n]
#             tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
#             tmp_TKE_term[(dist<0)] = float('nan')
#             p2 = axs[i,j].contourf(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),(tmp_TKE_term[xslice,:,:]*100).T,levels=levels,colors=colors,alpha=0.5)#vmin=-5,vmax=5)
#             sc = axs[i,j].contour(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[xslice,:,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
#             axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
#             cbar2 = plt.colorbar(p2)
#             axs[i,j].set_title(f'{labels[n]}')
#         else:
#             n = j+2
#             TKE_term = terms[n]
#             tmp_TKE_term = copy.deepcopy(TKE_term)#/(terms_bdg['totdis'])
#             tmp_TKE_term[(dist<0)] = float('nan')
#             p2 = axs[i,j].contourf(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),(tmp_TKE_term[xslice,:,:]*100).T,levels=levels,colors=colors,alpha=0.4)#vmin=-5,vmax=5)
#             sc = axs[i,j].contour(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),np.transpose(yB[xslice,:,:]),levels=[0,0.33,0.36,0.38,0.9],colors=['blue', 'red', 'orange', 'green'])
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
#             axs[i,j].plot(np.arange(0,Nx)*dx,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
#             axs[i,j].clabel(sc, sc.levels, inline=True, fontsize=10)
#             cbar2 = plt.colorbar(p2)
#             axs[i,j].set_title(f'{labels[n]}')
    
# axs[0,0].set_ylabel(r'$z/h$',fontsize=15),axs[1,0].set_ylabel(r'$z/h$',fontsize=15)
# axs[1,0].set_xlabel(r'$y/z_i$',fontsize=15),axs[1,1].set_xlabel(r'$y/z_i$',fontsize=15)
# fig.suptitle(f'$x = {xslice*dx*zi}m$')

# plt.savefig(pathOUT + 'Figures/TKE_yB_y' + str(yslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

#%%Pcolor plots of Prod-Dis, Adv, P-Trans, T-Trans and yB as a function of anisotropy cluster - yslices

cluster = 8

yB = np.reshape(yB_1D,(Nx,Ny,Nz))

AnisType = np.reshape(AnisType_1D,(Nx,Ny,Nz))
tmp_yb = copy.deepcopy(yB)
tmp_yb[(dist<0) | (AnisType!=cluster)] = float('nan')

z = z_uvp
z_on_h = z/(39/zi)
yslice = 161

# fig1, ax = plt.subplots()
# contour = ax.contourf(x * zi, y * zi, intf.T * zi, 30, cmap='viridis')
# ax.axhline(yslice*dy*zi,0,3000,color='k',ls='--')
# plt.title(f'y = {yslice*dy*zi}m', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
# plt.xlabel('$x [m]$', usetex=True, fontsize=15)
# plt.ylabel('$y [m]$', usetex=True, fontsize=15)
# plt.colorbar(contour,label='Elevation [m]')
# ax.set_aspect('auto')
# plt.gca().set_facecolor('white')

# plt.savefig(pathOUT +'Figures/' + 'ATTOtopo_y' + str(yslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

plt.show()

fig2,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)

p1 = axs.pcolormesh(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_yb[:,yslice,:].T,cmap=cmap, vmin = 0, vmax = np.sqrt(3)/2)

axs.plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
axs.plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi),color='k')

cbar1 = plt.colorbar(p1,label='yB')
    
axs.set_ylabel(r'$z/h$')
axs.set_xlabel(r'$x/z_i$')
axs.set_title(f'$y = {yslice}$ - Cluster = {cluster}')

# plt.savefig(pathOUT +'Figures/' + 'yB_c' + str(cluster) + '_y' + str(yslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

# n = 3
# labels = ['Production', 'Dissipation', 't-Transport', 'p-Transport','Prod-Dissip','Advection','Residual']
labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

# terms = [terms_bdg['prod_v']+terms_bdg['prod_h'],-terms_bdg['dissip']-terms_bdg['canopy'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
#          terms_bdg['pturb_h'] + terms_bdg['pturb_v'],(terms_bdg['prod']) - (terms_bdg['totdis']),
#          terms_bdg['adv_v'] + terms_bdg['adv_h'],terms_bdg['res']]

terms = [(terms_bdg['prod']) - (terms_bdg['totdis']),terms_bdg['uturb_h'] + terms_bdg['uturb_v'], terms_bdg['pturb_h'] + terms_bdg['pturb_v'],\
         terms_bdg['adv_v'] + terms_bdg['adv_h']]


fig3,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

for i in range(0,len(axs)):
    for j in range(0,len(axs[0])):
        if i == 0:
            n = j
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)
            tmp_TKE_term[(dist<0) | (AnisType!=cluster)] = float('nan')
            p2 = axs[i,j].pcolormesh(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_TKE_term[:,yslice,:].T,cmap='coolwarm',vmin=-20,vmax=20)
            axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi),color='k')
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
        else:
            n = j+2
            TKE_term = terms[n]
            tmp_TKE_term = copy.deepcopy(TKE_term)
            tmp_TKE_term[(dist<0) | (AnisType!=cluster)] = float('nan')
            p2 = axs[i,j].pcolormesh(np.arange(0,Nx)*dx,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_TKE_term[:,yslice,:].T,cmap='coolwarm',vmin=-20,vmax=20)
            axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
            axs[i,j].plot(np.arange(0,Nx)*dx,intf[:,yslice]*zi/(canopyH*zi-z_shift*zi),color='k')
            cbar2 = plt.colorbar(p2)
            axs[i,j].set_title(f'{labels[n]}')
    
axs[0,0].set_ylabel(r'$z/h$'),axs[1,0].set_ylabel(r'$z/h$')
axs[1,0].set_xlabel(r'$x/z_i$'),axs[1,1].set_xlabel(r'$x/z_i$')
fig3.suptitle(f'$y = {yslice}$ - Cluster = {cluster}')

# plt.savefig(pathOUT +'Figures/' + 'TKE_c' + str(cluster) + '_y' + str(yslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

z_ax = z_uvp*zi/(canopyH*zi-z_shift*zi)

fig4,axs = plt.subplots(1,1,figsize=(6,9),tight_layout=True)

axs.axvline(0,0,30,color='k')
for i in range(len(labels)):
    n = i
    TKE_term = terms[n]
    tmp_TKE_term = copy.deepcopy(TKE_term)
    tmp_TKE_term[(dist<0) | (AnisType!=cluster)] = float('nan')
    axs.plot(np.nanmean(tmp_TKE_term[:,yslice,:],axis=(0)),z_uvp*zi/(canopyH*zi-z_shift*zi),label=f'{labels[n]}',linewidth=3)
axs.set_ylim(z_ax[0],z_ax[-1])
axs.set_xlabel('TKE term')
axs.set_ylabel('z/h')
axs.set_title(f'$y = {yslice}$ - Cluster = {cluster}')
axs.legend(loc='upper right')

# plt.savefig(pathOUT +'Figures/' + 'TKEprof_c' + str(cluster) + '_y' + str(yslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

#%%Pcolor plots of Prod-Dis, Adv, P-Trans, T-Trans and yB as a function of anisotropy cluster - xslices

cluster = 8

yB = np.reshape(yB_1D,(Nx,Ny,Nz))

AnisType = np.reshape(AnisType_1D,(Nx,Ny,Nz))
tmp_yb = copy.deepcopy(yB)
tmp_yb[(dist<0) | (AnisType!=cluster)] = float('nan')

z = z_uvp
z_on_h = z/(39/zi)
xslice = 110

fig1, ax = plt.subplots()
contour = ax.contourf(x * zi, y * zi, intf.T * zi, 30, cmap='viridis')
ax.axvline(xslice*dx*zi,0,3000,color='k',ls='--')
plt.title(f'x = {xslice}', fontsize=textsize, fontweight='bold', fontname='Arial', usetex=True)
plt.xlabel('$x$', usetex=True)
plt.ylabel('$y$', usetex=True)
plt.colorbar(contour)
ax.set_aspect('auto')
plt.gca().set_facecolor('white')

# plt.savefig(pathOUT +'Figures/' + 'ATTOtopo_x' + str(xslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

# plt.show()

# fig2,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)

# p1 = axs.pcolormesh(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_yb[xslice,:,:].T,cmap=cmap, vmin = 0, vmax = np.sqrt(3)/2)

# axs.plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
# axs.plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')

# cbar1 = plt.colorbar(p1,label='yB')
    
# axs.set_ylabel(r'$z/h$')
# axs.set_xlabel(r'$y/z_i$')
# axs.set_title(f'$x = {xslice}$ - Cluster = {cluster}')

# # plt.savefig(pathOUT +'Figures/' + 'yB_c' + str(cluster) + '_x' + str(xslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

# # n = 3
# # labels = ['Production', 'Dissipation', 't-Transport', 'p-Transport','Prod-Dissip','Advection','Residual']
# labels = ['Prod-Dissip','t-Transport', 'p-Transport','Advection']

# keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.

# # terms = [terms_bdg['prod_v']+terms_bdg['prod_h'],-terms_bdg['dissip']-terms_bdg['canopy'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
# #          terms_bdg['pturb_h'] + terms_bdg['pturb_v'],(terms_bdg['prod']) - (terms_bdg['totdis']),
# #          terms_bdg['adv_v'] + terms_bdg['adv_h'],terms_bdg['res']]

# terms = [(terms_bdg['prod']) - (terms_bdg['totdis']),terms_bdg['uturb_h'] + terms_bdg['uturb_v'], terms_bdg['pturb_h'] + terms_bdg['pturb_v'],\
#          terms_bdg['adv_v'] + terms_bdg['adv_h']]


# fig3,axs = plt.subplots(2,2,figsize=(10,8),tight_layout=True)

# for i in range(0,len(axs)):
#     for j in range(0,len(axs[0])):
#         if i == 0:
#             n = j
#             TKE_term = terms[n]
#             tmp_TKE_term = copy.deepcopy(TKE_term)
#             tmp_TKE_term[(dist<0) | (AnisType!=cluster)] = float('nan')
#             p2 = axs[i,j].pcolormesh(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_TKE_term[xslice,:,:].T,cmap='coolwarm',vmin=-20,vmax=20)
#             axs[i,j].plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
#             axs[i,j].plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
#             cbar2 = plt.colorbar(p2)
#             axs[i,j].set_title(f'{labels[n]}')
#         else:
#             n = j+2
#             TKE_term = terms[n]
#             tmp_TKE_term = copy.deepcopy(TKE_term)
#             tmp_TKE_term[(dist<0) | (AnisType!=cluster)] = float('nan')
#             p2 = axs[i,j].pcolormesh(np.arange(0,Ny)*dy,z_uvp*zi/(canopyH*zi-z_shift*zi),tmp_TKE_term[xslice,:,:].T,cmap='coolwarm',vmin=-20,vmax=20)
#             axs[i,j].plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi)+1,color='k')
#             axs[i,j].plot(np.arange(0,Ny)*dy,intf[xslice,:]*zi/(canopyH*zi-z_shift*zi),color='k')
#             cbar2 = plt.colorbar(p2)
#             axs[i,j].set_title(f'{labels[n]}')
    
# axs[0,0].set_ylabel(r'$z/h$'),axs[1,0].set_ylabel(r'$z/h$')
# axs[1,0].set_xlabel(r'$y/z_i$'),axs[1,1].set_xlabel(r'$y/z_i$')
# fig3.suptitle(f'$x = {xslice}$ - Cluster = {cluster}')

# # plt.savefig(pathOUT +'Figures/' + 'TKE_c' + str(cluster) + '_x' + str(xslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

# fig4,axs = plt.subplots(1,1,figsize=(6,9),tight_layout=True)

# z_ax = z_uvp*zi/(canopyH*zi-z_shift*zi)
# axs.axvline(0,0,30,color='k')
# for i in range(len(labels)):
#     n = i
#     TKE_term = terms[n]
#     tmp_TKE_term = copy.deepcopy(TKE_term)
#     tmp_TKE_term[(dist<0) | (AnisType!=cluster)] = float('nan')
#     axs.plot(np.nanmean(tmp_TKE_term[xslice,:,:],axis=(0)),z_uvp*zi/(canopyH*zi-z_shift*zi),label=f'{labels[n]}',linewidth=3)
# axs.set_ylim(z_ax[0],z_ax[-1])
# axs.set_xlabel('TKE term')
# axs.set_ylabel('z/h')
# axs.set_title(f'$x = {xslice}$ - Cluster = {cluster}')
# axs.legend(loc='upper right')

# plt.savefig(pathOUT +'Figures/' + 'TKEprof_c' + str(cluster) + '_x' + str(xslice) + '.png',dpi=300,facecolor='white', edgecolor='white')

#%%Computing the advection index from Chamecki et al 2023 an plotting vertical profile

import random

def ChameckiIndex(Nz_SLayer,coord,dist,Adv,TotDis):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    AdvTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    TotDisTwr = np.zeros((len(coord),Nz_SLayer),'d',order='F')
    Ia = np.zeros((Nz_SLayer),'d',order='F')
    
    for i in range(len(coord)):
        
        loc = coord[i]
        AdvTwr[i,:] = Adv[110,loc,int(np.where(dist[110,loc,:]>0)[0][0]):int(np.where(dist[110,loc,:]>0)[0][0])+Nz_SLayer]
        TotDisTwr[i,:] = TotDis[110,loc,int(np.where(dist[110,loc,:]>0)[0][0]):int(np.where(dist[110,loc,:]>0)[0][0])+Nz_SLayer]
        
    Ia = np.mean(np.abs(AdvTwr)/TotDisTwr,axis=0)
    
    return Ia


N_twrs = 5000

coord = []
for i in range(0,N_twrs):
    # coord.append((random.randrange(0,Nx,1),random.randrange(0,Ny,1)))
    coord.append((random.randrange(0,Nx,1)))
    
    
terms = [terms_bdg['prod_v']+terms_bdg['prod_h'],-terms_bdg['dissip']-terms_bdg['canopy'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
         terms_bdg['pturb_h'] + terms_bdg['pturb_v'],(terms_bdg['prod']) - (terms_bdg['totdis']),
         terms_bdg['adv_v'] + terms_bdg['adv_h'],terms_bdg['res']]

Adv = terms[5]
TotDis = -terms[1]

z = z_uvp
z_on_h = z/(39/zi)

Ia = ChameckiIndex(Nz_SLayer, coord, dist, Adv, TotDis)

fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
axs.plot(Ia[0:50],z_on_h[0:50])
axs.set_ylim(z_on_h[0],z_on_h[50])
axs.set_xlabel('Ia')
axs.set_ylabel('z/h')

# terms = [terms_bdg['prod_v']+terms_bdg['prod_h'],-terms_bdg['dissip']-terms_bdg['canopy'],terms_bdg['uturb_h'] + terms_bdg['uturb_v'],
#          terms_bdg['pturb_h'] + terms_bdg['pturb_v'],(terms_bdg['prod']) - (terms_bdg['totdis']),
#          terms_bdg['adv_v'] + terms_bdg['adv_h'],terms_bdg['res']]

# Adv = terms[5]
# TotDis = -terms[1]

# z = z_uvp
# z_on_h = z/(39/zi)

# Ia = ChameckiIndex(Nz_SLayer, coord, dist, Adv, TotDis)

# fig,axs = plt.subplots(1,1,figsize=(8,6),tight_layout=True)
# axs.plot(Ia[0:50],z_on_h[0:50])
# axs.set_ylim(z_on_h[0],z_on_h[50])
# axs.set_xlabel('Ia')
# axs.set_ylabel('z/h')

#%%Plot verical profiles of yB for selected towers
yB = np.reshape(yB_1D,(Nx,Ny,Nz)); xB = np.reshape(xB_1D,(Nx,Ny,Nz))
yB[(dist[:,:,:]<0)] = float('nan'); xB[(dist[:,:,:]<0)] = float('nan')
yB_Twr = np.zeros((N_twrs,Nz_SLayer),'d',order='F')
z = z_uvp
z_on_h = z/(39/zi)

for i in range(len(coord)):
    loc = coord[i]
    
    yB_Twr[i,:] = yB[loc[0],loc[1],int(np.where(dist[loc[0],loc[1],:]>0)[0][0]):int(np.where(dist[loc[0],loc[1],:]>0)[0][0])+Nz_SLayer]
    
h1 = np.min(z_over_d)
yB_m = []
for i in range(0,int(np.ceil(np.max(z_over_d)/0.5))):
    yB_m.append(np.median(yB_Twr.flatten()[(z_over_d.flatten()>h1) & (z_over_d.flatten()<h1+0.5)]))
    h1 = h1+0.5
fig,axs = plt.subplots(1,1,figsize=(8,5),tight_layout=True)

# axs.plot(np.mean(yB,axis=(0,1))[5:Nz_SLayer+5],z_on_h[0:Nz_SLayer])
axs.plot(yB_m,z_over_d_m,c='k')
# axs.plot(np.mean(yB_Twr,axis=(0)),z_on_h[0:Nz_SLayer])
axs.axvline(0.38,ls='--')
axs.axvline(0.36,ls='--')
axs.axhline(9.5,ls='--')
# axs.axhline(1,-3,3,c='k',ls='-')
# axs.axhline(2,-3,3,c='k',ls='--')
# axs.axhline(3,-3,3,c='k',ls='-.')
axs.set_ylim(np.min(z_over_d),np.max(z_over_d))
axs.set_xlim(0,np.sqrt(3)/2)
axs.set_xlabel('yB',fontsize=15)
axs.set_ylabel('z/disp_h',fontsize=15)
axs.grid()
plt.show()
# axs.set_title('Valley',fontsize=12)

# plt.savefig(pathOUT +'Figures/' + 'yB_prof_meantwr_valley_bicheng.png',dpi=300,facecolor='white', edgecolor='white')
# 
#%% #%% Gathering the desired TKE term for analysis as a function of yB cluster:


import seaborn as sns
import pandas as pd

#-------Set the TKE terms to study:--------------------------------------------
n = 4
labels = ['Production', 'Dissipation', 't-Transport', 'p-Transport','Prod-Dissip']

keys = list(terms_bdg) #Creates a list of the keys in ther terms_bdg dictionry.
#keys = ['adv_h','adv_v','prod_h','prod_v','uturb_h','uturb_v','pturb_h','pturb_v','prod_dudz','canopy','dissip','sum']

#TKE_term = terms_bdg['prod_v']
#TKE_term = -terms_bdg['dissip']
#TKE_term = terms_bdg['uturb_h'] + terms_bdg['uturb_v']
#TKE_term = terms_bdg['pturb_h'] + terms_bdg['pturb_v']
TKE_term = terms_bdg['prod'] - terms_bdg['totdis']

#%%------------------------------------------------------------------------------

level = 3

fig1,fig2 = box_plot(Nx, Ny, Nz, Nz_SLayer, TKE_term, coord, AnisType_1D, dist, labels, n, z_on_h, level, dz/(39/zi),\
                     pathOUT,'prod_dis_box_3canopyH_flat','prod_dis_pts_3canopyH_flat',False)

#%%
TKE_term_1D = np.ndarray.flatten(TKE_term[:,:,0:Nz_SLayer])

TKE_cluster = {}  #Note, I am not using a pd.DataFrame because in there all columns need to be the same length.
                  #I could  make the columns of same lengths by adding NaNs but that would increase the memory needs.

nPoints_Cluster = {} #dictionary that keeps track of the total number of points per cluster
Total_nPoints = 0
cluster = 0
for cluster in range(1,10):
    print(f'cluster = {cluster}')
    #tmp_TKE_term_1D = TKE_term_1D[(AnisType_1D == cluster)]
    cluster_name = 'cluster ' + str(cluster)
    TKE_cluster[cluster_name] = TKE_term_1D[(AnisType_1D == cluster) & (dist_1D>0)]
    nPoints_Cluster[cluster_name] = TKE_cluster[cluster_name].size
    Total_nPoints = Total_nPoints + nPoints_Cluster[cluster_name] 

df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in TKE_cluster.items()]))

    
#%% PLotting the Results as a Boxplot


fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(8, 5))

#Box Plot:
#axs.boxplot(TKE_cluster.values(),vert=False, notch = False, showfliers = False,
#            patch_artist = True,boxprops = dict(facecolor = "lightblue"),
#            medianprops = dict(color = "orange", linewidth = 1),
#            whiskerprops = dict(color = "red", linewidth = 1),
#            capprops = dict(color = "red", linewidth = 1))

#axs.set_yticklabels(TKE_cluster.keys())

#Seaborn Boxplot Plot:
cm = sns.color_palette( ["#410d00","#831901","#983e00","#b56601","#ab8437",
                         "#b29f74","#7f816b","#587571","#596c72"])

sns.boxplot(data=df,orient='h',palette=cm,saturation=1,width=0.6,linewidth=0.8,fliersize = 0.5)


x_min = -500
x_max = 500

axs.set_xlim(x_min, x_max)
axs.axvline(x =0, ymin = 0, ymax=9,color='gray',linestyle='--')
             
axs.grid(which='major', axis='both',color='grey', alpha = 0.3, linestyle='-')
axs.grid(which='minor', axis='both',color='grey', alpha = 0.2, linestyle='-')
axs.minorticks_on()
axs.set_xscale('linear')
    
#cbar = plt.colorbar(sc)
    
axs.set_ylabel(r'$cluster$')
axs.set_xlabel(f'{labels[n]}')


plt.tight_layout()
plt.show()


# plt.savefig(path_in +'Figures/'+ labels[n] + '_boxplot_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')


#%% PLotting the total number of points per cluster:

percentage_Cluster = (pd.Series(nPoints_Cluster.values())/Total_nPoints)*100

fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(6, 3))

axs.scatter(np.arange(1,10),percentage_Cluster,c=["#410d00","#831901","#983e00","#b56601","#ab8437",
                         "#b29f74","#7f816b","#587571","#596c72"])

axs.set_xlabel(r'$cluster$')
axs.set_ylabel(r'$N_i/N_{total}$')

plt.tight_layout()
plt.show()


# plt.savefig(path_in +'Figures/'+ labels[n] + '_Npoints_' + cases[num] +'.png',dpi=300,facecolor='white', edgecolor='white')
