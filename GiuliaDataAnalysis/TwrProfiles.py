# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 07:10:47 2025

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

#%%Set case and path to data


# Path to the data files:
path = 'E:\\PhD\\GiuliaData\\'
os.chdir(path)  

# cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','Hom_Amazon_9mps']
cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps']
case = 0

#%% Defining the main parameters of the simulations

Nx = 256
Ny = 256
Nz = 256
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 1
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

x = np.arange(0,Nx)*dx
y = np.arange(0,Ny)*dy
z_w = np.arange(0,Nz)*dz
z_uvp = np.arange(0,Nz)*dz + dz/2

zi = 1000
uscale = 0.313
Hcanopy = 39/zi
kappa = 0.4  # von Karman constant
LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
height = math.ceil(Hcanopy/dz)

Ntwr = 100
Nz_Slayer = 126

# Loading the 3D Momentum, 2D Momentum, and 3D TKE Budget data:

#For the sake of clarity, below I sepcify the variables included in "data" and "dataS":
#--------------------------------------------------------------------------------------------------
#data = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['avgU','avgV',\
#                        'avgW','avgP','avgU2','avgV2','avgW2','avgU3','avgV3',\
#                        'avgW3','avgU4','avgV4','avgW4','avgUV','avgUW','avgVW',\
#                        'avgtxx','avgtyy','avgtzz','avgtxy','avgtxz','avgtyz',\
#                        'avgdudz','avgdvdz','avgNut','avgCs']})



#data_2D = xr.DataArray(np.ones(shape = (Nx,Ny,NumVariables_2D),order='F'),\
#                dims=('x','y','variable'), coords = {'variable':['Mav_ustar']})


#data_TKE = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,NumVariables),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['u','v','w','p',\
#                        'uu','vv','ww','uv','uw','vw',\
#                        'dudx','dudy','dudz','dvdx','dvdy','dvdz','dwdx','dwdy','dwdz',\
#                        'txx','tyy','tzz','txy','txz','tyz',\
#                        'uuu','uvv','uww','vuu','vvv','vww','wuu','wvv','www',\
#                        'utxx','utyy','utzz','vtxx','vtyy','vtzz','wtxx','wtyy','wtzz',\
#                        'vtxy','wtxz','utxy','wtyz','utxz','vtyz',\
#                        'up','vp','wp',\
#                        'dxx','dyy','dzz','dxy','dxz','dyz','fdx','fdy','fdz','ufdx','vfdy','wfdz']}) 

#%%Load the data
    
data = xr.open_dataarray(cases[case]+'/Data_Momentum_4TKE.nc')
sfc = np.load(path+'input_txt_files\\'+cases[case]+'\\sfc.npy')

#%%Randomly select coordinates

coord_f = np.argwhere(sfc==1.6)
coord_p = np.argwhere(sfc==0)

idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)

sel_coord_f = coord_f[idx_f]
sel_coord_p = coord_p[idx_p]

#%%Plot the domain

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
terrain = plt.get_cmap('terrain')
terrain_truncated = LinearSegmentedColormap.from_list(
    'terrain_truncated', terrain(np.linspace(0.25, 1, 100)))

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.pcolormesh(x*zi,y*zi,sfc.T,cmap=terrain_truncated)

for i in range(0,Ntwr):
    axs.plot(x[sel_coord_f[i][0]]*zi,y[sel_coord_f[i][1]]*zi,'ok')
    axs.plot(x[sel_coord_p[i][0]]*zi,y[sel_coord_p[i][1]]*zi,'vr')

axs.tick_params(axis='x', which='major', labelsize=12)
axs.set_xlabel(f'x [m]', fontsize=15)
axs.set_xlim(0,x[-1]*zi)
axs.set_ylim(0,y[-1]*zi)
axs.tick_params(axis='y', which='major', labelsize=12)
axs.set_ylabel(f'y [m]', fontsize=15)
plt.show()

#%% Bicheng Functions:
    
def get_dphidx(phi, wn):
  phi_c = np.fft.rfft(phi, axis=0, norm="ortho")
  dphidx_c = complex(0, 1) * wn[:, np.newaxis, np.newaxis] * phi_c
  dphidx_c[-1, :, :] = 0
  return np.fft.irfft(dphidx_c, axis=0, norm="ortho")

def get_dphidy(phi, wn):
  phi_c = np.fft.rfft(phi, axis=1, norm="ortho")
  dphidy_c = complex(0, 1) * wn[np.newaxis, :, np.newaxis] * phi_c
  dphidy_c[:, -1, :] = 0
  return np.fft.irfft(dphidy_c, axis=1, norm="ortho")

def get_dphidz(phi, dz):
  dphidz = np.zeros(phi.shape)
  dphidz[:,:,:-1] = (phi[:,:,1:]-phi[:,:,:-1]) / (dz)
  dphidz[:,:,-1] = dphidz[:,:,-2]
  return dphidz

def uvpnode2wnode(phi_c):
  phi_h = np.zeros(phi_c.shape)
  phi_h[:, :, 0] = 0
  phi_h[:, :, 1:] = 0.5*(phi_c[:, :, :-1] + phi_c[:, :, 1:])
  return phi_h

def wnode2uvpnode(phi_h):
  phi_c = np.zeros(phi_h.shape)
  phi_c[:, :, :-1] = 0.5*(phi_h[:, :, :-1]+phi_h[:, :, 1:])
  phi_c[:, :, -1] = phi_h[:, :, -1]
  return phi_c

def compute_d_twr(data, coord, dist, height, dz, zi, u_scale, LAD):

    npoints = coord.shape[0]
    d_dim = np.zeros(npoints)

    for i in range(npoints):
        ix, iy = coord[i]
        z_nan = dist[ix, iy, :]

        # Find first valid vertical index
        valid_start = np.argmax(z_nan > 0) - 5
        
        # Extract u and v from data array
        u = data.data[ix, iy, valid_start:, 0]
        v = data.data[ix, iy, valid_start:, 1]

        U = np.sqrt(u**2 + v**2)
        
        Z = np.linspace(1, height - 5, 16) * dz * zi - 0.5 * dz * zi

        # Make sure we have enough points
        max_len = min(height - 5, len(U), len(LAD))
        VAR = U[:max_len] * u_scale
        Y = (VAR[:max_len]**2) * 0.4 * np.array(LAD[:max_len])
        Z_use = Z[:max_len]

        d_dim[i] = trapezoid(Y * Z_use, Z_use) / trapezoid(Y, Z_use)

    return d_dim

def find_coordinates(elevation_map, N_twrs, loc='max'):
    """
    Finds the coordinates of local peaks or valleys in a topography map.

    Parameters
    ----------
    elevation_map : 2D numpy array
        Topography height map.
    N_twrs : int
        Number of towers to select.
    loc : str, optional
        'max' for peaks, 'min' for valleys. Default is 'max'.

    Returns
    -------
    coords : ndarray of shape (N_twrs, 2)
        Selected (i, j) coordinates.
    """
    import numpy as np

    arr = np.array(elevation_map)
    rows, cols = arr.shape
    
    if loc == 'flat':
        # Uniform random sampling of N_twrs coordinates
        all_indices = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'), axis=-1).reshape(-1, 2)
        selected = all_indices[np.random.choice(all_indices.shape[0], N_twrs, replace=False)]
        return selected

    # Prepare slices for neighbors (4-connectivity)
    up    = arr[:-2, 1:-1]
    down  = arr[2:, 1:-1]
    left  = arr[1:-1, :-2]
    right = arr[1:-1, 2:]
    center = arr[1:-1, 1:-1]

    if loc == 'max':
        # Local peak condition (center > all neighbors)
        peak_mask = (center >= up) & (center >= down) & (center >= left) & (center >= right)
        # Threshold to keep only the top 10% elevation
        threshold = 0.9 * np.max(arr)
        high_mask = center > threshold
        valid_mask = peak_mask & high_mask
    elif loc == 'min':
        # Local valley condition (low elevation below a certain threshold)
        threshold = 2 * np.min(arr)
        valid_mask = center <= threshold        
    else:
        raise ValueError("loc must be 'max' or 'min'")

    # Get indices of valid points and shift to original indexing
    indices = np.argwhere(valid_mask) + 1  # shift due to 1:-1 slicing

    if len(indices) < N_twrs:
        raise ValueError(f"Not enough {'peaks' if loc == 'max' else 'valleys'} found to select {N_twrs} towers.")

    # Randomly sample N_twrs from candidates
    selected = indices[np.random.choice(len(indices), N_twrs, replace=False)]

    return selected

#%%Compute ustar at the top of the canopy on uvp node

# ustar = np.zeros((Nx,Ny),order='F')

# ustar = ((wnode2uvpnode(data.data[:,:,:,8])[:,:,10] - data.data[:,:,10,0]*wnode2uvpnode(data.data[:,:,:,2])[:,:,10] + wnode2uvpnode(data.data[:,:,:,23])[:,:,10])**2 \
#          + (wnode2uvpnode(data.data[:,:,:,9])[:,:,10] - data.data[:,:,10,1]*wnode2uvpnode(data.data[:,:,:,2])[:,:,10] + wnode2uvpnode(data.data[:,:,:,24])[:,:,10])**2)**(1/4)

T_13 = wnode2uvpnode(((data.data[:,:,:,8]) - uvpnode2wnode(data.data[:,:,:,0])*(data.data[:,:,:,2]) - (data.data[:,:,:,23])))
T_23 = wnode2uvpnode(((data.data[:,:,:,9]) - uvpnode2wnode(data.data[:,:,:,1])*(data.data[:,:,:,2]) - (data.data[:,:,:,24])))

# Remove planar mean at each height level
u_pp = data.data[:,:,:,0] - data.data[:,:,:,0].mean(axis=(0, 1), keepdims=True)
v_pp = data.data[:,:,:,1] - data.data[:,:,:,1].mean(axis=(0, 1), keepdims=True)
w_pp = data.data[:,:,:,2] - data.data[:,:,:,2].mean(axis=(0, 1), keepdims=True)

# Compute dispersive stresses
D13 = (u_pp) * wnode2uvpnode(w_pp)
D23 = (v_pp) * wnode2uvpnode(w_pp)

cov_turb = -np.sqrt((T_13 + D13)**2 + (T_23 + D23)**2)
ustar = np.sqrt(-cov_turb[:, :, 10])  # slice at hc_n height

del T_13,T_23,u_pp,v_pp,w_pp,D13,D23,cov_turb

#%%Define tower averaging function

def average_over_selected_coords(data, selected_coords):
    """
    Compute the average over the (nx, NY) dimensions for selected (i, j) coordinates.
    
    Parameters:
    -----------
    data : np.ndarray
        A 3D array of shape (nx, NY, nz)
    selected_coords : np.ndarray
        Array of shape (n_points, 2), where each row is an (i, j) coordinate
    
    Returns:
    --------
    avg_profile : np.ndarray
        1D array of length nz, the average over selected (nx, NY) points
    """
    # Extract the profiles at selected (i, j) locations
    profiles = np.array([data[i, j, :] for i, j in selected_coords])

    # Average over the selected points (axis 0)
    avg_profile = np.mean(profiles, axis=0)
    
    return avg_profile

#%%

ls = ['-','--']
colors = ['k','b','g','r','c','m','y','lime','violet']

fig,axs = plt.subplots(1,1,tight_layout=True)

for i in range(len(cases)):
    if i < 6:
        data = xr.open_dataarray(cases[i]+'/Data_Momentum_4TKE.nc')
        sfc = np.load(path+'input_txt_files\\'+cases[i]+'\\sfc.npy')

        coord_f = np.argwhere(sfc==1.6)
        coord_p = np.argwhere(sfc==0)

        idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
        idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)

        sel_coord_f = coord_f[idx_f]
        sel_coord_p = coord_p[idx_p]
        
        del coord_f,coord_p,idx_f,idx_p
        
        ustar = np.zeros((Nx,Ny),order='F')

        ustar = ((wnode2uvpnode(data.data[:,:,:,8])[:,:,10] - data.data[:,:,10,0]*wnode2uvpnode(data.data[:,:,:,2])[:,:,10] + wnode2uvpnode(data.data[:,:,:,23])[:,:,10])**2 \
                 + (wnode2uvpnode(data.data[:,:,:,9])[:,:,10] - data.data[:,:,10,1]*wnode2uvpnode(data.data[:,:,:,2])[:,:,10] + wnode2uvpnode(data.data[:,:,:,24])[:,:,10])**2)**(1/4)
        
        Nz_SLayer = 200
        TwrAvgProf_f = np.zeros((Nz_Slayer),order='F')
        TwrAvgProf_p = np.zeros((Nz_Slayer),order='F')

        u_mag = np.sqrt(data.data[:,:,:,0]**2 + data.data[:,:,:,1]**2 + wnode2uvpnode(data.data[:,:,:,2])**2)
        u_norm = u_mag/ustar[:,:,np.newaxis]
        TwrAvgProf_f = average_over_selected_coords(u_norm[:,:,:Nz_SLayer], sel_coord_f)
        TwrAvgProf_p = average_over_selected_coords(u_norm[:,:,:Nz_SLayer], sel_coord_p)
        
        del u_mag,u_norm,data,sfc,ustar

        axs.plot(TwrAvgProf_f,z_uvp[:Nz_SLayer]/Hcanopy,c=colors[i],ls=ls[0],label='Forested')
        axs.plot(TwrAvgProf_p,z_uvp[:Nz_SLayer]/Hcanopy,c=colors[i],ls=ls[1],label='Patches')
        
    else:
        
        import random
        os.chdir('C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\Anisotropy\\')
        from TowerAnalysis import Twr_Anis_Multi,plot_topo_twr,Twr_TKE_Multi#,find_coordinates
        os.chdir("C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\UCLAanalysis\\Zev_UCLA_Project\\")
        from functions import build_phi, build_intf
        
        directory = 'E:\\PhD\\TopographyData\\'
        path_sin = directory + cases[i] + '\\'
        
        Nx,Ny = 256; nzTot = 384; mpiProc = 32
        
        phi = build_phi(path_sin + 'phi_functions\\', Nx, Ny, nzTot, mpiProc)
        intf, iintf = build_intf(phi, dz)

        coord_p = find_coordinates(intf,Ntwr,'max')
        coord_v = find_coordinates(intf,Ntwr,'min')

        mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
        # Broadcast to shape (Nx, Ny, Nz_SLayer, NumVariables)
        mask4D = np.expand_dims(mask, axis=-1)
        # Assign NaN in one operation
        data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
        
        Umag_p = np.zeros((Ntwr, Nz_SLayer), dtype='float64', order='F')
        Umag_v = np.zeros((Ntwr, Nz_SLayer), dtype='float64', order='F')
        ustar = np.zeros(Ntwr)

        # Pre-extract needed variable indices
        idx_u, idx_v, idx_w = var_idx = [0, 1, 2]
        idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

        for k, (ix, iy) in enumerate(coord_p):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5

            # Extract Ruw and Rvw at z_start + 16
            z_idx_ustar = z_start + 16
            Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
                   data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
                   data.data[ix, iy, z_idx_ustar, idx_txz]) * u_scale**2

            Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
                   data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
                   data.data[ix, iy, z_idx_ustar, idx_tyz]) * u_scale**2

            ustar[k] = (Ruw**2 + Rvw**2)**0.25

            # Slice velocity components over Nz_SLayer starting from z_start
            z_end = z_start + Nz_SLayer
            u = data.data[ix, iy, z_start:z_end, idx_u]
            v = data.data[ix, iy, z_start:z_end, idx_v]
            w = data.data[ix, iy, z_start:z_end, idx_w]

            # Compute magnitude and normalize
            vel_mag = np.sqrt(u**2 + v**2 + w**2)
            Umag_p[k, :] = vel_mag * (u_scale / ustar[k])
            
        TwrAvgProf_p.data[:,0] = np.mean(Umag_p,axis=(0))
            
        for k, (ix, iy) in enumerate(coord_v):
            # Find first index where dist > 0
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5

            # Extract Ruw and Rvw at z_start + 16
            z_idx_ustar = z_start + 16
            Ruw = (data.data[ix, iy, z_idx_ustar, idx_uw] -
                   data.data[ix, iy, z_idx_ustar, idx_u] * data.data[ix, iy, z_idx_ustar, idx_w] -
                   data.data[ix, iy, z_idx_ustar, idx_txz]) * u_scale**2

            Rvw = (data.data[ix, iy, z_idx_ustar, idx_vw] -
                   data.data[ix, iy, z_idx_ustar, idx_v] * data.data[ix, iy, z_idx_ustar, idx_w] -
                   data.data[ix, iy, z_idx_ustar, idx_tyz]) * u_scale**2

            ustar[k] = (Ruw**2 + Rvw**2)**0.25

            # Slice velocity components over Nz_SLayer starting from z_start
            z_end = z_start + Nz_SLayer
            u = data.data[ix, iy, z_start:z_end, idx_u]
            v = data.data[ix, iy, z_start:z_end, idx_v]
            w = data.data[ix, iy, z_start:z_end, idx_w]

            # Compute magnitude and normalize
            vel_mag = np.sqrt(u**2 + v**2 + w**2)
            Umag_v[k, :] = vel_mag * (u_scale / ustar[k])

        TwrAvgProf_v.data[:,0] = np.mean(Umag_v,axis=(0))
    

axs.set_xlabel(r"$\overline{U}/u_*$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    # axs.legend()
    
axs.set_ylim(0,2)
axs.axhline(1,ls='--',c='k')

plt.show()    
    
        
    




































