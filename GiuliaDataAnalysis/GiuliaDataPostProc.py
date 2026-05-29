#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 03:14:29 2025

@author: u1450851
"""

#%%

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
path = '/uufs/chpc.utah.edu/common/home/calaf-group2/GiuliaData/TKE_BUDGET_AND_RAV/'
os.chdir(path)  

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','Hom_Amazon_9mps']
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
    
# data = xr.open_dataarray(cases[case]+'/Data_Momentum.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
# data_2D = xr.open_dataarray(cases[case]+'/Data_Momentum_2D.nc')  #This uploads the data from the NetCDF files, that keep the xarray form.
data = xr.open_dataarray(cases[case]+'/Data_Momentum_4TKE.nc')

#%%Mask the velocity field to 

u = (data.data[:, :, 4, 0] >= 1.5).astype(int)

# fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.pcolormesh(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,u.T,cmap='jet')

# plt.show()

#%%Randomly select coordinates

coord_f = np.argwhere(u==0)
coord_p = np.argwhere(u==1)

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

axs.pcolormesh(x*zi,y*zi,u.T,cmap=terrain_truncated)

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

#%%Compute ustar at the top of the canopy on uvp node

ustar = np.zeros((Nx,Ny),order='F')

ustar = ((wnode2uvpnode(data.data[:,:,:,8])[:,:,10] - data.data[:,:,10,0]*wnode2uvpnode(data.data[:,:,:,2])[:,:,10] + wnode2uvpnode(data.data[:,:,:,23])[:,:,10])**2 \
         + (wnode2uvpnode(data.data[:,:,:,9])[:,:,10] - data.data[:,:,10,1]*wnode2uvpnode(data.data[:,:,:,2])[:,:,10] + wnode2uvpnode(data.data[:,:,:,24])[:,:,10])**2)**(1/4)

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

Nz_SLayer = 200
TwrAvgProf_f = xr.DataArray(np.zeros(shape = (Nz_SLayer,5),order='F'),\
                       dims=('z','variable'), coords = {'variable':['Umag','Shear','phi','skewness','kurtosis']})
    
TwrAvgProf_p = xr.DataArray(np.zeros(shape = (Nz_SLayer,5),order='F'),\
                       dims=('z','variable'), coords = {'variable':['Umag','Shear','phi','skewness','kurtosis']})
    
#%%Plot tower averaged profile for the horizontal velocity magnitude

u_mag = np.sqrt(data.data[:,:,:,0]**2 + data.data[:,:,:,1]**2 + wnode2uvpnode(data.data[:,:,:,2])**2)
u_norm = u_mag/ustar[:,:,np.newaxis]
TwrAvgProf_f[:,0] = average_over_selected_coords(u_norm[:,:,:Nz_SLayer], sel_coord_f)
TwrAvgProf_p[:,0] = average_over_selected_coords(u_norm[:,:,:Nz_SLayer], sel_coord_p)

#%%Plot profiles of u_mag/ustar

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_f.data[:,0],z_uvp[:Nz_SLayer]/Hcanopy,'-k',label='Forested')
axs.plot(TwrAvgProf_p.data[:,0],z_uvp[:Nz_SLayer]/Hcanopy,'--k',label='Patches')

axs.set_xlabel(r"$\overline{U}/u_*$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.legend()
axs.set_ylim(0,12)
axs.axhline(1,ls='--',c='k')

plt.show()

#%%Compute the shear stress

shear = ((wnode2uvpnode(data.data[:,:,:,8]) - data.data[:,:,:,0]*wnode2uvpnode(data.data[:,:,:,2]) + wnode2uvpnode(data.data[:,:,:,23]))**2 \
         + (wnode2uvpnode(data.data[:,:,:,9]) - data.data[:,:,:,1]*wnode2uvpnode(data.data[:,:,:,2]) + wnode2uvpnode(data.data[:,:,:,24]))**2)**(1/2)

shear_norm = shear/(ustar[:,:,np.newaxis])**2
TwrAvgProf_f[:,1] = average_over_selected_coords(shear_norm[:,:,:Nz_SLayer], sel_coord_f)
TwrAvgProf_p[:,1] = average_over_selected_coords(shear_norm[:,:,:Nz_SLayer], sel_coord_p)

#%%Plot shear stress profile

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_f[:,1],z_uvp[:Nz_SLayer]/Hcanopy,'-k',label='Forested')
axs.plot(TwrAvgProf_p[:,1],z_uvp[:Nz_SLayer]/Hcanopy,'--k',label='Patches')

axs.set_xlabel(r"$\sqrt{\overline{u'w'}^2 + \overline{v'w'}^2}/u_*^2$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.legend(fontsize=15)
axs.set_xlim(-0.1,2)
axs.set_ylim(0,12)
axs.axhline(1,ls='--',c='k')
axs.grid()
axs.tick_params(axis='both',labelsize=12)

plt.show()

#%%Compute skewness

# def custom_skewness_profile(data, selected_coords):
#     """
#     Compute vertical profile of skewness using the provided formula.

#     Parameters:
#     -----------
#     data : np.ndarray
#         3D array of shape (nx, NY, nz)
#     selected_coords : np.ndarray
#         Array of shape (n_points, 2), each row is (i, j)

#     Returns:
#     --------
#     skew_profile : np.ndarray
#         1D array of skewness at each vertical level (length nz)
#     """
#     nx, NY, nz = data.shape
#     N = selected_coords.shape[0]
#     skew_profile = np.zeros(nz)

#     for k in range(nz):
#         # Extract w values at current level for selected (i, j) coords
#         w = np.array([data[i, j, k] for i, j in selected_coords])
#         w_mean = np.mean(w)
#         w_prime = w - w_mean

#         # Numerator: mean of w'^3
#         num = np.mean(w_prime ** 3)

#         # Denominator: (mean of w'^2)^(3/2)
#         denom = (np.mean(w_prime ** 2)) ** 1.5

#         # Avoid division by zero
#         skew_profile[k] = num / denom if denom != 0 else 0.0

#     return skew_profile

w3_t = wnode2uvpnode(data.data[:,:,:,33] - 3*data.data[:,:,:,2]*data.data[:,:,:,6] + 2*data.data[:,:,:,2]**3 + (data.data[:,:,:,42] - data.data[:,:,:,2]*uvpnode2wnode(data.data[:,:,:,21])))
w2_t = wnode2uvpnode(data.data[:,:,:,6] - data.data[:,:,:,2]**2 + uvpnode2wnode(data.data[:,:,:,21]))
skewness = w3_t/(w2_t)**1.5

#%%Compute skewness

TwrAvgProf_f[:,3] = average_over_selected_coords(skewness[:,:,:Nz_SLayer], sel_coord_f)
TwrAvgProf_p[:,3] = average_over_selected_coords(skewness[:,:,:Nz_SLayer], sel_coord_p)

#%%Plot skewness

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(TwrAvgProf_f[:,3],z_uvp[:Nz_SLayer]/Hcanopy,'-k',label='Forested')
axs.plot(TwrAvgProf_p[:,3],z_uvp[:Nz_SLayer]/Hcanopy,'--k',label='Patches')

axs.set_xlabel(r"$Sk_u$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
axs.legend(fontsize=15)
# axs.set_xlim(-0.1,2)
axs.set_ylim(0,12)
axs.axhline(1,ls='--',c='k')
axs.grid()
axs.tick_params(axis='both',labelsize=12)

plt.show()

#%%Kurtosis

def custom_kurtosis_profile(data, selected_coords):
    """
    Compute vertical profile of kurtosis over selected (i, j) coordinates.

    Parameters:
    -----------
    data : np.ndarray
        3D array of shape (nx, NY, nz)
    selected_coords : np.ndarray
        Array of shape (n_points, 2) where each row is (i, j)

    Returns:
    --------
    kurt_profile : np.ndarray
        1D array of kurtosis at each vertical level (length nz)
    """
    nx, NY, nz = data.shape
    kurt_profile = np.zeros(nz)

    for k in range(nz):
        # Extract data at level k for all selected (i, j)
        values = np.array([data[i, j, k] for i, j in selected_coords])
        mean = np.mean(values)
        fluct = values - mean

        # Compute numerator and denominator
        num = np.mean(fluct ** 4)
        denom = (np.mean(fluct ** 2)) ** 2

        kurt_profile[k] = num / denom if denom != 0 else 0.0

    return kurt_profile

#%%Compute kurtosis for the towers

# kurt_f = custom_kurtosis_profile(data.data[:,:,:,2], sel_coord_f)
# kurt_p = custom_kurtosis_profile(data.data[:,:,:,2], sel_coord_p)

#%%Plot kurtosis

# fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(kurt_f,z_w/Hcanopy,'-k',label='Forested')
# axs.plot(kurt_p,z_w/Hcanopy,'--k',label='Patches')

# axs.set_xlabel(r"$Ku_u$",fontsize=15)
# axs.set_ylabel(r"$z/h_C$",fontsize=15)
# axs.legend(fontsize=15)
# # axs.set_xlim(-0.1,2)
# axs.set_ylim(0,12)
# axs.axhline(1,ls='--',c='k')
# axs.grid()
# axs.tick_params(axis='both',labelsize=12)

# plt.show()

#%%Functions needed to compute the velocity gradient

def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar(Nz_SLayer, z_d, u, v, twr=False):    

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]

    # Levels to use for logarithmic fit
    fit_levels = [20, 30, 70, 90]

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]

    # Fit the log profile
    coefs, _ = curve_fit(log_fit, z_data, U_data, maxfev=10000)
    a, b = coefs

    # Fitted velocity profile across the entire surface layer
    u_fit = log_fit(z_d[:Nz_SLayer], a, b)

    # Compute z0hi and ustar
    z0hi = 1 / b
    ustar = U_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit

def phi_m_loc(Nx, Ny, Nz_SLayer, z_d, u, v, avgdUdz, avgdVdz, ustar):
    kappa = 0.4

    U_mag = np.sqrt(u**2 + v**2)                   # Magnitude of velocity
    mean_dUdz = (u * avgdUdz + v * avgdVdz) / U_mag  # Directional mean shear
    phi_m_1d = (kappa * z_d[:Nz_SLayer] / ustar) * mean_dUdz

    return phi_m_1d

def compute_d_twr(data, coord, height, dz, zi, u_scale, LAD):
    
    d_dim = np.zeros(coord.shape[0])
    
    nlevels = len(LAD)

    z = (np.arange(1, height + 1)[:nlevels]) * dz * zi - (dz * zi) / 2
    Z = z  # Final vertical coordinates for integration

    for idx in range(coord.shape[0]):
        i, j = coord[idx]
        
        # Extract u, v and compute |U| above that level
        u = data.data[i, j, :,0]
        v = data.data[i, j, :,1]
        U = np.sqrt(u**2 + v**2)

        # Select the first `nlevels` points for integration (truncate if too short)
        U_slice = U[:nlevels]
        Z_slice = Z[:len(U_slice)]

        VAR = U_slice * u_scale
        Y = (VAR**2) * 0.4 * LAD

        num = trapezoid(Y * Z_slice, Z_slice)
        den = trapezoid(Y, Z_slice)

        d_dim[idx] = num / den if den != 0 else 0.0

    return d_dim

#%%Compute displacement height

disp_f = compute_d_twr(data, sel_coord_f, height, dz, zi, uscale, LAD)
disp_p = compute_d_twr(data, sel_coord_p, height, dz, zi, uscale, LAD)
dudz_uvp = wnode2uvpnode(data.data[:,:,:,22])
dvdz_uvp = wnode2uvpnode(data.data[:,:,:,23])

phi_m_2D = np.zeros((sel_coord_f.shape[0],Nz_Slayer))
z2D = np.zeros((sel_coord_f.shape[0],Nz_Slayer))
z0hi = np.zeros((sel_coord_f.shape[0]),'d',order='F')
ustar = np.zeros((sel_coord_f.shape[0]),'d',order='F')
z_over_d = np.zeros((sel_coord_f.shape[0],Nz_Slayer))

for k in range(sel_coord_f.shape[0]):
    loc = sel_coord_f[k]

    z = z_uvp
    z_d = (z - ((disp_f[k])/zi))
    z2D[k,:] = z_d[0:Nz_Slayer]
    z_over_d[k,:] = ((z_uvp[0:Nz_Slayer]*zi)-disp_f[k])/39 
    
    [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_Slayer,z_d,data.data[loc[0],loc[1],:,0],\
                                                            data.data[loc[0],loc[1],:,1],True)
    
    phi_m_2D[k,:] = phi_m_loc(Nx,Ny,Nz_Slayer,z_d,data.data[loc[0],loc[1],:Nz_Slayer,0],data.data[loc[0],loc[1],:Nz_Slayer,1],\
                              dudz_uvp[loc[0],loc[1],:Nz_Slayer],dvdz_uvp[loc[0],loc[1],:Nz_Slayer],ustar[k])
        
    print(f'Done with coord: {k}')

#%%Plot vertical gradient

# fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(np.mean(phi_m_2D,axis=(0)),z_uvp[:Nz_Slayer]/Hcanopy,'-k',label='Forested')
# # axs.plot(kurt_p,z_w/Hcanopy,'--k',label='Patches')

# axs.set_xlabel(r"$\phi_M$",fontsize=15)
# axs.set_ylabel(r"$z/h_C$",fontsize=15)
# axs.legend(fontsize=15)
# axs.set_xlim(-0.5,2)
# axs.set_ylim(0,12)
# axs.axhline(1,ls='--',c='k')
# axs.grid()
# axs.tick_params(axis='both',labelsize=12)

# plt.show()

#%%Cluster Plot

# ls = ['-',':']
# marker = ['o','v','*','+','s','d']
# mrkevr = 10
# plt_cnt = 0
# msize = 4

# fig,axs = plt.subplots(5,5,constrained_layout=True,figsize=(16,9.5),sharey=True,sharex='col')

# for i in range(len(cases)):
#     if i <3:
#         plt_cnt = 3
#     else:
#         plt_cnt = 4
#     #Load data
#     data = xr.open_dataarray(cases[i]+'/Data_Momentum.nc')
#     #Mask the data into forested and bare soil
#     u = copy.deepcopy(data.data[:,:,4,0])
#     u[(u<1.5)] = 0 
#     u[(u>0)] = 1
#     #Select random coordates
#     coord_f = np.argwhere(u==0)
#     coord_p = np.argwhere(u==1)
#     idx_f = np.random.choice(coord_f.shape[0],size=Ntwr,replace=False)
#     idx_p = np.random.choice(coord_p.shape[0],size=Ntwr,replace=False)
#     sel_coord_f = coord_f[idx_f]
#     sel_coord_p = coord_p[idx_p]
#     #Compute ustar at the top of the canopy
#     u_w = uvpnode2wnode(data.data[:,:,:,0])
#     v_w = uvpnode2wnode(data.data[:,:,:,1])
#     ustar = np.zeros((Nx,Ny),order='F')
#     ustar = ((data.data[:,:,10,14] - u_w[:,:,10]*data.data[:,:,10,2] - data.data[:,:,10,20])**2 \
#              + (data.data[:,:,10,15] - v_w[:,:,10]*data.data[:,:,10,2] - data.data[:,:,10,21])**2)**(1/4)
#     #Compute the horizontal velocity magnitude and plot it
#     u_mag = np.sqrt(data.data[:,:,:,0]**2 + data.data[:,:,:,1]**2)
#     u_norm = u_mag/ustar[:,:,np.newaxis]
#     u_norm_twr_f = average_over_selected_coords(u_norm, sel_coord_f)
#     u_norm_twr_p = average_over_selected_coords(u_norm, sel_coord_p)
    
#     axs[plt_cnt,0].plot(u_norm_twr_f,z_uvp/Hcanopy,c='k',ls=ls[0],marker=marker[i],markevery=mrkevr,ms=msize,label='Forested')
#     axs[plt_cnt,0].plot(u_norm_twr_p,z_uvp/Hcanopy,c='k',ls=ls[1],marker=marker[i],markevery=mrkevr,ms=msize,label='Patches')
#     #Compute the shear stress and plot it
#     shear = ((data.data[:,:,:,14] - u_w*data.data[:,:,:,2] - data.data[:,:,:,20])**2 \
#              + (data.data[:,:,:,15] - v_w*data.data[:,:,:,2] - data.data[:,:,:,21])**2)**(1/2)

#     shear_norm = shear/(ustar[:,:,np.newaxis])**2
#     shear_norm_f = average_over_selected_coords(shear_norm, sel_coord_f)
#     shear_norm_p = average_over_selected_coords(shear_norm, sel_coord_p)
    
#     axs[plt_cnt,1].plot(shear_norm_f,z_w/Hcanopy,c='k',ls=ls[0],marker=marker[i],markevery=mrkevr,ms=msize,label='Forested')
#     axs[plt_cnt,1].plot(shear_norm_p,z_w/Hcanopy,c='k',ls=ls[1],marker=marker[i],markevery=mrkevr,ms=msize,label='Patches')
    
#     #Compute and plot the non-dimensional velocity gradient
#     disp_f = compute_d_twr(data, sel_coord_f, height, dz, zi, uscale, LAD)
#     dudz_uvp = wnode2uvpnode(data.data[:,:,:,22])
#     dvdz_uvp = wnode2uvpnode(data.data[:,:,:,23])

#     phi_m_2D = np.zeros((sel_coord_f.shape[0],Nz_Slayer))
#     z2D = np.zeros((sel_coord_f.shape[0],Nz_Slayer))
#     z0hi = np.zeros((sel_coord_f.shape[0]),'d',order='F')
#     ustar = np.zeros((sel_coord_f.shape[0]),'d',order='F')
#     z_over_d = np.zeros((sel_coord_f.shape[0],Nz_Slayer))

#     for k in range(sel_coord_f.shape[0]):
#         loc = sel_coord_f[k]

#         z = z_uvp
#         z_d = (z - ((disp_f[k])/zi))
#         z2D[k,:] = z_d[0:Nz_Slayer]
#         z_over_d[k,:] = ((z_uvp[0:Nz_Slayer]*zi)-disp_f[k])/39 
        
#         [z0hi[k],ustar[k],U_mean,U_data,z_data,u_fit] = compute_ustar(Nz_Slayer,z_d,data.data[loc[0],loc[1],:,0],\
#                                                                 data.data[loc[0],loc[1],:,1],True)
        
#         phi_m_2D[k,:] = phi_m_loc(Nx,Ny,Nz_Slayer,z_d,data.data[loc[0],loc[1],:Nz_Slayer,0],data.data[loc[0],loc[1],:Nz_Slayer,1],\
#                                   dudz_uvp[loc[0],loc[1],:Nz_Slayer],dvdz_uvp[loc[0],loc[1],:Nz_Slayer],ustar[k])
            
#         # print(f'Done with coord: {k}')
    
#     axs[plt_cnt,2].plot(np.mean(phi_m_2D,axis=(0)),z_uvp[:Nz_Slayer]/Hcanopy,c='k',ls=ls[0],marker=marker[i],markevery=mrkevr,ms=msize,label='Forested')
    
#     #Compute skewness and plot it
#     skewU_f = custom_skewness_profile(data.data[:,:,:,2], sel_coord_f)
#     skewU_p = custom_skewness_profile(data.data[:,:,:,2], sel_coord_p)
    
#     axs[plt_cnt,3].plot(skewU_f,z_w/Hcanopy,c='k',ls=ls[0],marker=marker[i],markevery=mrkevr,ms=msize,label='Forested')
#     axs[plt_cnt,3].plot(skewU_p,z_w/Hcanopy,c='k',ls=ls[1],marker=marker[i],markevery=mrkevr,ms=msize,label='Patches')
#     #Compute Kurtosis and plot it
#     kurt_f = custom_kurtosis_profile(data.data[:,:,:,2], sel_coord_f)
#     kurt_p = custom_kurtosis_profile(data.data[:,:,:,2], sel_coord_p)
    
#     axs[plt_cnt,4].plot(kurt_f,z_w/Hcanopy,c='k',ls=ls[0],marker=marker[i],markevery=mrkevr,ms=msize,label='Forested')
#     axs[plt_cnt,4].plot(kurt_p,z_w/Hcanopy,c='k',ls=ls[1],marker=marker[i],markevery=mrkevr,ms=msize,label='Patches')
    
#     print(f'Done with case {i}')

# axs[4,0].set_xlabel(r"$\overline{U}/u_*$",fontsize=12); axs[4,0].set_xlim(-0.1,20)
# axs[4,1].set_xlabel(r"$\sqrt{\overline{u'w'}^2 + \overline{v'w'}^2}/u_*^2$",fontsize=12); axs[4,1].set_xlim(-0.5,2)
# axs[4,2].set_xlabel(r"$\phi_M$",fontsize=12); axs[4,2].set_xlim(-0.5,2)
# axs[4,3].set_xlabel(r"$Sk_w$",fontsize=12); axs[4,3].set_xlim(-2,7)
# axs[4,4].set_xlabel(r"$Ku_w$",fontsize=12); axs[4,4].set_xlim(-2,20)

# for i in range(len(axs[0])):
#     axs[i,0].set_ylabel(r"$z/h_C$",fontsize=12)
#     axs[i,0].set_ylim(0,12)
    
# for i in range(len(axs)):
#     for j in range(len(axs[1])):
#         axs[i,j].axhline(1,ls='--',c='k')
#         axs[i,j].grid()
#         axs[i,j].tick_params(axis='both',labelsize=10)

# # plt.tight_layout(pad=0.8)
# plt.show()

#%%Compute Resolved REynodls stresses and TKE

# terms_ptb = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,8),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['u2_t','uv_t',\
#                        'uw_t','v2_t','vw_t','w2_t','tke','tke_SGS']})

# wn_x = 2*np.pi*np.fft.rfftfreq(Nx, dx)
# wn_y = 2*np.pi*np.fft.rfftfreq(Ny, dy)

# # # Interpolate u and v to w node
# u_h = uvpnode2wnode(data_TKE.data[:,:,:,0])
# v_h = uvpnode2wnode(data_TKE.data[:,:,:,1])

# terms_ptb.data[:,:,:,0] = data_TKE.data[:,:,:,4] - data_TKE.data[:,:,:,0]**2
# terms_ptb.data[:,:,:,1] = data_TKE.data[:,:,:,7] - data_TKE.data[:,:,:,0]*data_TKE.data[:,:,:,1]
# terms_ptb.data[:,:,:,2] = wnode2uvpnode(data_TKE.data[:,:,:,8] - u_h*data_TKE.data[:,:,:,2])
# # terms_ptb.data[:,:,:,2] = wnode2uvpnode(terms_ptb.data[:,:,:,2])

# terms_ptb.data[:,:,:,3] = data_TKE.data[:,:,:,5] - data_TKE.data[:,:,:,1]**2
# terms_ptb.data[:,:,:,4] = wnode2uvpnode(data_TKE.data[:,:,:,9] - v_h*data_TKE.data[:,:,:,2])
# # terms_ptb.data[:,:,:,4] = wnode2uvpnode(terms_ptb.data[:,:,:,4])

# terms_ptb.data[:,:,:,5] = wnode2uvpnode(data_TKE.data[:,:,:,6] - data_TKE.data[:,:,:,2]**2)
# # terms_ptb.data[:,:,:,5] = wnode2uvpnode(terms_ptb.data[:,:,:,5])

# terms_ptb.data[:,:,:,6] = (terms_ptb.data[:,:,:,0] + terms_ptb.data[:,:,:,3] + terms_ptb.data[:,:,:,5]) / 2
# terms_ptb.data[:,:,:,7] = (data_TKE.data[:,:,:,19] + data_TKE.data[:,:,:,20] + data_TKE.data[:,:,:,21]) / 2

# terms_ptb.to_netcdf(path+cases[case]+'\\terms_ptb.nc')

# del terms_ptb

terms_ptb = xr.open_dataarray(path+cases[case]+'/terms_ptb.nc')

#%%#%% Calculate the TKE budget

# terms_bdg = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,15),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['adv_h','adv_v',\
#                        'adv','uturb_h','uturb_v','ttrans','pturb_h','pturb_v','ptrans','dissip','canopy','totdis',\
#                            'prod_h','prod_v','prod']})

# # Calculate the advection term (all on uvp-nodes)
# print('Calculating the advection term')
# # ue = data_TKE.data[:,:,:,0]*terms_ptb['tke']
# duedx = get_dphidx(data_TKE.data[:,:,:,0]*terms_ptb.data[:,:,:,6], wn_x)
# # ve = data_TKE.data[:,:,:,1]*terms_ptb['tke']
# dvedy = get_dphidy(data_TKE.data[:,:,:,1]*terms_ptb.data[:,:,:,6], wn_y)
# tkez = uvpnode2wnode(terms_ptb.data[:,:,:,6])
# # we = data_TKE.data[:,:,:,2]*tkez
# dwedz = get_dphidz(data_TKE.data[:,:,:,2]*tkez, dz)
# # dwedz = wnode2uvpnode(dwedz)
# # terms_bdg.data[:,:,:,0] = -duedx - dvedy
# # terms_bdg.data[:,:,:,1] = -dwedz

# #SGS
# # ue_sgs = data_TKE.data[:,:,:,0]*terms_ptb.data[:,:,:,7]
# due_sgsdx = get_dphidx(data_TKE.data[:,:,:,0]*terms_ptb.data[:,:,:,7], wn_x)
# # ve_sgs = data_TKE.data[:,:,:,1]*terms_ptb.data[:,:,:,7]
# dve_sgsdy = get_dphidy(data_TKE.data[:,:,:,1]*terms_ptb.data[:,:,:,7], wn_y)
# tke_sgsz = uvpnode2wnode(terms_ptb.data[:,:,:,7])
# # we_sgs = data_TKE.data[:,:,:,2]*tke_sgsz
# dwe_sgsdz = get_dphidz(data_TKE.data[:,:,:,2]*tke_sgsz, dz)
# # dwe_sgsdz = wnode2uvpnode(dwe_sgsdz)

# terms_bdg.data[:,:,:,0] = - duedx - dvedy - due_sgsdx - dve_sgsdy
# terms_bdg.data[:,:,:,1] = - dwedz - dwe_sgsdz

# del duedx,dvedy,tkez,dwedz,due_sgsdx,dve_sgsdy,tke_sgsz,dwe_sgsdz

# terms_bdg.data[:,:,:,2] = terms_bdg.data[:,:,:,0] + terms_bdg.data[:,:,:,1]

# # Calculate the turbulent transport term (horizontal, all on uvp-nodes)
# print('Calculating the horizontal turbulent transport term')
# uw_c = wnode2uvpnode(data_TKE.data[:,:,:,8])
# vw_c = wnode2uvpnode(data_TKE.data[:,:,:,9])
# w_c = wnode2uvpnode(data_TKE.data[:,:,:,2])
# w2_c = wnode2uvpnode(data_TKE.data[:,:,:,6])

# u3_t = data_TKE.data[:,:,:,25] - 3*data_TKE.data[:,:,:,0]*data_TKE.data[:,:,:,4] + 2*(data_TKE.data[:,:,:,0]**3)
# uv2_t = data_TKE.data[:,:,:,26] - 2*data_TKE.data[:,:,:,1]*data_TKE.data[:,:,:,7]\
#   + 2*data_TKE.data[:,:,:,0]*(data_TKE.data[:,:,:,1]**2) - data_TKE.data[:,:,:,0]*data_TKE.data[:,:,:,5]
# uw2_t = wnode2uvpnode(data_TKE.data[:,:,:,27]) - 2*w_c*uw_c + 2*data_TKE.data[:,:,:,0]*(w_c**2)\
#   - data_TKE.data[:,:,:,0]*w2_c
# ue = 0.5*(u3_t+uv2_t+uw2_t)
# del u3_t,uv2_t,uw2_t
# duedx = get_dphidx(ue, wn_x)
# del ue
# utxx_t = (data_TKE.data[:,:,:,34]) - data_TKE.data[:,:,:,0]*(data_TKE.data[:,:,:,19])
# vtxy_t = (data_TKE.data[:,:,:,43]) - data_TKE.data[:,:,:,1]*(data_TKE.data[:,:,:,22])
# wtxz_t = wnode2uvpnode((data_TKE.data[:,:,:,44]) - data_TKE.data[:,:,:,2]*(data_TKE.data[:,:,:,23]))
# # wtxz_t = wnode2uvpnode(wtxz_t)
# dutaudx = get_dphidx(0.5*(utxx_t+vtxy_t+wtxz_t), wn_x)
# del utxx_t,vtxy_t,wtxz_t
# # terms_bdg['uturb_h'] = -duedx-dutaudx

# u2v_t = data_TKE.data[:,:,:,28] - 2*data_TKE.data[:,:,:,0]*data_TKE.data[:,:,:,7]\
#   + 2*data_TKE.data[:,:,:,1]*(data_TKE.data[:,:,:,0]**2) - data_TKE.data[:,:,:,1]*data_TKE.data[:,:,:,4]
# v3_t = data_TKE.data[:,:,:,29] - 3*data_TKE.data[:,:,:,1]*data_TKE.data[:,:,:,5] + 2*(data_TKE.data[:,:,:,1]**3)
# vw2_t = wnode2uvpnode(data_TKE.data[:,:,:,30]) - 2*w_c*vw_c + 2*data_TKE.data[:,:,:,1]*w_c**2\
#   - data_TKE.data[:,:,:,1]*w2_c
# ve = 0.5*(u2v_t+v3_t+vw2_t)
# del u2v_t,v3_t,vw2_t
# dvedy = get_dphidy(ve, wn_y)
# del ve
# utxy_t = (data_TKE.data[:,:,:,45]) - data_TKE.data[:,:,:,0]*(data_TKE.data[:,:,:,22])
# vtyy_t = (data_TKE.data[:,:,:,38]) - data_TKE.data[:,:,:,1]*(data_TKE.data[:,:,:,20])
# wtyz_t = wnode2uvpnode((data_TKE.data[:,:,:,46]) - data_TKE.data[:,:,:,2]*(data_TKE.data[:,:,:,24]))
# # wtyz_t = wnode2uvpnode(wtyz_t)
# dutaudy = get_dphidy(0.5*(utxy_t+vtyy_t+wtyz_t), wn_y)
# del utxy_t,vtyy_t,wtyz_t

# terms_bdg.data[:,:,:,3] = -duedx-dutaudx-dvedy-dutaudy

# del duedx,dutaudx,dvedy,dutaudy,uw_c,vw_c,w_c,w2_c

# # Calculate the turbulent transport term (vertical, all on uvp-nodes)
# print('Calculating the vertical turbulent transport term')
# u2_h = uvpnode2wnode(data_TKE.data[:,:,:,4])
# v2_h = uvpnode2wnode(data_TKE.data[:,:,:,5])
# wu2_t = data_TKE.data[:,:,:,31] - 2*u_h*data_TKE.data[:,:,:,8] + 2*data_TKE.data[:,:,:,2]*u_h**2\
#   - data_TKE.data[:,:,:,2]*u2_h
# wv2_t = data_TKE.data[:,:,:,32] - 2*v_h*data_TKE.data[:,:,:,9] + 2*data_TKE.data[:,:,:,2]*v_h**2\
#   - data_TKE.data[:,:,:,2]*v2_h
# w3_t = data_TKE.data[:,:,:,33] - 3*data_TKE.data[:,:,:,2]*data_TKE.data[:,:,:,6] + 2*data_TKE.data[:,:,:,2]**3
# dwedz = get_dphidz(0.5*(wu2_t+wv2_t+w3_t),dz)
# del wu2_t,wv2_t,w3_t
# # dwedz = get_dphidz(we, dz)
# # dwedz = wnode2uvpnode(dwedz)
# utxz_t = (data_TKE.data[:,:,:,47]) - u_h*(data_TKE.data[:,:,:,23])
# vtyz_t = (data_TKE.data[:,:,:,48]) - v_h*(data_TKE.data[:,:,:,24])
# wtzz_t = (data_TKE.data[:,:,:,42]) - data_TKE.data[:,:,:,2]*uvpnode2wnode(data_TKE.data[:,:,:,21])
# dutaudz = get_dphidz(0.5*(utxz_t+vtyz_t+wtzz_t), dz)
# del utxz_t,vtyz_t,wtzz_t
# # dutaudz = wnode2uvpnode(dutaudz)
# terms_bdg.data[:,:,:,4] = -dwedz-dutaudz
# del dwedz,dutaudz

# terms_bdg.data[:,:,:,5] = terms_bdg.data[:,:,:,3] + terms_bdg.data[:,:,:,4]

# # Calculate the pressure transport term (all on uvp-nodes)
# print('Calculating the pressure transport term')
# # dpudx = get_dphidx(data_tavg['pu'], wn_x) - data_tavg['u']*get_dphidx(data_tavg['p'], wn_x)
# # pu_t = data_TKE.data[:,:,:,49] - data_TKE.data[:,:,:,3]*uvpnode2wnode(data_TKE.data[:,:,:,0])
# pu_t = (data_TKE.data[:,:,:,49]-0.5*(uvpnode2wnode(data_TKE.data[:,:,:,25]+data_TKE.data[:,:,:,26])+data_TKE.data[:,:,:,27]))\
#     - (data_TKE.data[:,:,:,3]-0.5*(uvpnode2wnode(data_TKE.data[:,:,:,4]+data_TKE.data[:,:,:,5])+data_TKE.data[:,:,:,6]))*uvpnode2wnode(data_TKE.data[:,:,:,0])
# # terms_ptb['pu_t'] = (data_tavg['pu']\
# #     -(1/3)*(data_tavg['utxx']+data_tavg['utyy']+data_tavg['utzz']))\
# #     - (data_tavg['p']\
# #     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['u']
# dpudx = wnode2uvpnode(get_dphidx(pu_t, wn_x))
# del pu_t

# # dpvdy = get_dphidy(data_tavg['pv'], wn_y) - data_tavg['v']*get_dphidy(data_tavg['p'], wn_y)
# # pv_t = data_TKE.data[:,:,:,50] - data_TKE.data[:,:,:,3]*uvpnode2wnode(data_TKE.data[:,:,:,1])
# pv_t = (data_TKE.data[:,:,:,50]-0.5*(uvpnode2wnode(data_TKE.data[:,:,:,28]+data_TKE.data[:,:,:,29])+data_TKE.data[:,:,:,30]))\
#     - (data_TKE.data[:,:,:,3]-0.5*(uvpnode2wnode(data_TKE.data[:,:,:,4]+data_TKE.data[:,:,:,5])+data_TKE.data[:,:,:,6]))*uvpnode2wnode(data_TKE.data[:,:,:,1])
# # terms_ptb['pv_t'] = (data_tavg['pv']\
# #     -(1/3)*(data_tavg['vtxx']+data_tavg['vtyy']+data_tavg['vtzz']))\
# #     - (data_tavg['p']\
# #     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['v']
# dpvdy = wnode2uvpnode(get_dphidy(pv_t, wn_y))
# del pv_t

# # p_h = data_tavg['p'] #uvpnode2wnode(data_tavg['p'])
# # dpwdz = get_dphidz(data_tavg['pw'], dz) - uvpnode2wnode(data_tavg['w'])*get_dphidz(data_tavg['p'], dz)
# # pw_t = data_TKE.data[:,:,:,51] - data_TKE.data[:,:,:,3]*data_TKE.data[:,:,:,2]
# pw_t = (data_TKE.data[:,:,:,51]-0.5*(data_TKE.data[:,:,:,31]+data_TKE.data[:,:,:,32]+data_TKE.data[:,:,:,33]))\
#     - (data_TKE.data[:,:,:,3]-0.5*(uvpnode2wnode(data_TKE.data[:,:,:,4]+data_TKE.data[:,:,:,5])+data_TKE.data[:,:,:,6]))*data_TKE.data[:,:,:,2]
# # terms_ptb['pw_t'] = (data_tavg['pw']\
# #     -(1/3)*(data_tavg['wtxx']+data_tavg['wtyy']+data_tavg['wtzz']))\
# #     - (data_tavg['p']\
# #     -(1/3)*(data_tavg['txx']+data_tavg['tyy']+data_tavg['tzz']))*data_tavg['w']
# dpwdz = get_dphidz(pw_t, dz)
# del pw_t
# # dpwdz = wnode2uvpnode(dpwdz)

# terms_bdg.data[:,:,:,6] = -dpudx-dpvdy
# del dpudx,dpvdy
# terms_bdg.data[:,:,:,7] = -dpwdz
# del dpwdz

# terms_bdg.data[:,:,:,8] = terms_bdg.data[:,:,:,6] + terms_bdg.data[:,:,:,7]

# # Calculate the dissipation rate (all on uvp-nodes)
# print('Calculating the dissipation term')
# # terms_drv['S11'] = terms_drv['dudx']
# # terms_drv['S12'] = 0.5*(terms_drv['dudy'] + terms_drv['dvdx'])
# # terms_drv['S13'] = 0.5*(terms_drv['dudz'] + terms_drv['dwdx'])
# # terms_drv['S22'] = terms_drv['dvdy']
# # terms_drv['S23'] = 0.5*(terms_drv['dvdz'] + terms_drv['dwdy'])
# # terms_drv['S33'] = terms_drv['dwdz']

# dissip = data_TKE.data[:,:,:,52] + data_TKE.data[:,:,:,55] + wnode2uvpnode(data_TKE.data[:,:,:,56]) + data_TKE.data[:,:,:,53] +\
#     wnode2uvpnode(data_TKE.data[:,:,:,57]) + data_TKE.data[:,:,:,54]

# terms_bdg.data[:,:,:,9] = dissip\
#   - (data_TKE.data[:,:,:,19])*data_TKE.data[:,:,:,10] - (data_TKE.data[:,:,:,20])*data_TKE.data[:,:,:,14]\
#   - (data_TKE.data[:,:,:,21])*wnode2uvpnode(data_TKE.data[:,:,:,18])\
#   - 2*(data_TKE.data[:,:,:,22])*0.5*(data_TKE.data[:,:,:,11]+data_TKE.data[:,:,:,13])\
#   - wnode2uvpnode(2*(data_TKE.data[:,:,:,23])*0.5*(data_TKE.data[:,:,:,12]+data_TKE.data[:,:,:,16]))\
#   - wnode2uvpnode(2*(data_TKE.data[:,:,:,24])*0.5*(data_TKE.data[:,:,:,15]+data_TKE.data[:,:,:,17]))
  
# del dissip

# terms_bdg.data[:,:,:,10] = wnode2uvpnode(data_TKE.data[:,:,:,-1] - data_TKE.data[:,:,:,2]*data_TKE.data[:,:,:,-4])\
#                             + (data_TKE.data[:,:,:,-3]-data_TKE.data[:,:,:,0]*data_TKE.data[:,:,:,-6])\
#                                + (data_TKE.data[:,:,:,-2]-data_TKE.data[:,:,:,1]*data_TKE.data[:,:,:,-5])
# # terms_bdg['canopy'] = wnode2uvpnode(terms_bdg['canopy'])
# # terms_bdg['canopy'] += ((data_tavg['ufdx']-data_tavg['u']*data_tavg['fdx'])\
# #   +(data_tavg['vfdy']-data_tavg['v']*data_tavg['fdy']))
  
# terms_bdg.data[:,:,:,11] = terms_bdg.data[:,:,:,9] + terms_bdg.data[:,:,:,10]

# # # Calculate the production (all on uvp-nodes)
# print('Calculating the production term')
# terms_bdg.data[:,:,:,12] =\
#   -(terms_ptb.data[:,:,:,0]*data_TKE.data[:,:,:,10] + terms_ptb.data[:,:,:,1]*data_TKE.data[:,:,:,11]
#   + terms_ptb.data[:,:,:,1]*data_TKE.data[:,:,:,13] + terms_ptb.data[:,:,:,3]*data_TKE.data[:,:,:,14]
#   + terms_ptb.data[:,:,:,2]*wnode2uvpnode(data_TKE.data[:,:,:,16]) + terms_ptb.data[:,:,:,4]*wnode2uvpnode(data_TKE.data[:,:,:,17])
#   + (data_TKE.data[:,:,:,19])*data_TKE.data[:,:,:,10] + (data_TKE.data[:,:,:,20])*data_TKE.data[:,:,:,14]
#   + 2*(data_TKE.data[:,:,:,22])*0.5*(data_TKE.data[:,:,:,11]+data_TKE.data[:,:,:,13]) + wnode2uvpnode((data_TKE.data[:,:,:,23])*data_TKE.data[:,:,:,16])
#   + wnode2uvpnode((data_TKE.data[:,:,:,24])*data_TKE.data[:,:,:,17])
#   )
# terms_bdg.data[:,:,:,13] =\
#   -(terms_ptb.data[:,:,:,2]*wnode2uvpnode(data_TKE.data[:,:,:,12]) + terms_ptb.data[:,:,:,4]*wnode2uvpnode(data_TKE.data[:,:,:,15])
#   + terms_ptb.data[:,:,:,5]*wnode2uvpnode(data_TKE.data[:,:,:,18])
#   + wnode2uvpnode((data_TKE.data[:,:,:,23])*data_TKE.data[:,:,:,12] + (data_TKE.data[:,:,:,24])*data_TKE.data[:,:,:,15])
#   + (data_TKE.data[:,:,:,21])*wnode2uvpnode(data_TKE.data[:,:,:,18])
#   )
# # terms_bdg['prod_dudz'] = -(terms_ptb['uw_t']*terms_drv['dudz'] + data_tavg['txz']*terms_drv['dudz'])
# # terms_bdg['prod_dvdz'] = -(terms_ptb['vw_t']*terms_drv['dvdz'] + data_tavg['tyz']*terms_drv['dvdz'])
# # terms_bdg['prod_dwdz'] = -(terms_ptb['w2_t']*terms_drv['dwdz'] + data_tavg['tzz']*terms_drv['dwdz'])

# terms_bdg.data[:,:,:,14] = terms_bdg.data[:,:,:,12] + terms_bdg.data[:,:,:,13]

# # terms_bdg['res'] = terms_bdg['prod_v'] + terms_bdg['prod_h'] - terms_bdg['canopy'] - terms_bdg['dissip']\
# #                     + terms_bdg['uturb_v'] + terms_bdg['uturb_h']\
# #                     + terms_bdg['adv_h'] + terms_bdg['adv_v'] + terms_bdg['pturb_h'] + terms_bdg['pturb_v'] 

# print('*'*80)

# terms_bdg.to_netcdf(path+cases[case] + '\\TKE_terms.nc') 

# del u_h,v_h

terms_bdg = xr.open_dataarray(path+cases[case] + '/TKE_terms.nc')


#%%Plot TKE

# terms_bdg = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,15),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['adv_h','adv_v',\
#                        'adv','uturb_h','uturb_v','ttrans','pturb_h','pturb_v','ptrans','dissip','canopy','totdis',\
#                            'prod_h','prod_v','prod']})

plt.figure(figsize=(4,8))
# plt.plot(np.nanmean(terms_bdg.data[:,:,:,2]+terms_bdg.data[:,:,:,5]+terms_bdg.data[:,:,:,8]-terms_bdg.data[:,:,:,9]\
#                     -terms_bdg.data[:,:,:,10]+terms_bdg.data[:,:,:,14],axis=(0,1)),z_uvp,c='pink',marker='o',markevery=8,label='res')
# plt.plot(np.nanmean(terms_bdg.data[:,:,:,2],axis=(0,1)),z_uvp,c='red',marker='s',markevery=8,label='Adv')
# plt.plot(np.nanmean(terms_bdg.data[:,:,:,5],axis=(0,1)),z_uvp,c='purple',marker='^',markevery=8,label='T_trans')
# plt.plot(np.nanmean(terms_bdg.data[:,:,:,8],axis=(0,1)),z_uvp,c='brown',marker='D',markevery=8,label='P_trans')
plt.plot(np.nanmean(-terms_bdg.data[:,:,:,9],axis=(0,1)),z_uvp,c='green',marker='v',markevery=8,label='Diss')
plt.plot(np.nanmean(-terms_bdg.data[:,:,:,10],axis=(0,1)),z_uvp,c='orange',marker='o',markevery=8,label='Can')
# plt.plot(np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='black',marker='o',markevery=8,label='TotDis')
plt.plot(np.nanmean(terms_bdg.data[:,:,:,14],axis=(0,1)),z_uvp,c='blue',marker='D',markevery=8,label='Prod')
plt.plot(np.nanmean(terms_bdg.data[:,:,:,-1]-terms_bdg.data[:,:,:,11],axis=(0,1)),z_uvp,c='yellow',marker='*',markevery=8,label='Prod-Diss')
# plt.plot(np.nanmean(terms_bdg['adv']+terms_bdg['ttrans']+terms_bdg['ptrans'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=8,label='A+TT+PT')

# plt.plot(np.nanmean(terms_bdg['res'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='pink',marker='o',markevery=5,label='res')
# plt.plot(np.nanmean(terms_bdg['adv'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='red',marker='s',markevery=5,label='Adv')
# plt.plot(np.nanmean(terms_bdg['ttrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='purple',marker='^',markevery=5,label='T_trans')
# plt.plot(np.nanmean(terms_bdg['ptrans'],axis=(0,1))/np.nanmean(-terms_bdg['totdis'],axis=(0,1)),z_ax,c='brown',marker='D',markevery=5,label='P_trans')
# plt.plot(np.nanmean(-terms_bdg['dissip'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='green',marker='v',markevery=5,label='Diss')
# plt.plot(np.nanmean(-terms_bdg['canopy'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='orange',marker='o',markevery=5,label='Can')
# plt.plot(np.nanmean(terms_bdg['prod'],axis=(0,1))/np.nanmean(terms_bdg['totdis'],axis=(0,1)),z_ax,c='blue',marker='D',markevery=7,label='Prod')
# plt.hlines(Hcanopy,-80,80,linestyle='--',colors='k')
# plt.hlines(2*Hcanopy,-80,80,linestyle='--',colors='k')
plt.vlines(0,0,5,linestyle='--',colors='grey')
# plt.vlines(1,0,5,linestyle='--',colors='grey')
# plt.vlines(-1,0,5,linestyle='--',colors='grey')
plt.ylim(z_uvp[0],z_uvp[-1])
# plt.xlim(-30,0)
# plt.title(r'Real')
plt.xlabel(r'$\left \langle de/dt \right \rangle / \left \langle \epsilon_{t} \right \rangle$')
# plt.xlabel(r'$\left \langle de/dt \right \rangle$')
plt.ylabel(r'$z/z_i$')
plt.legend(loc='upper right')
plt.show()

#%%Pcolor plots of TKE terms

yslice = 123

fig,axs = plt.subplots(1,1,tight_layout=True)

p = axs.pcolormesh(x,z_uvp,(terms_bdg.data[:,yslice,:,-1]-terms_bdg.data[:,yslice,:,11]).T,cmap='bwr',vmin=-500,vmax=500)
# p = axs.pcolormesh(x,z_uvp,terms_bdg.data[:,:,20,8].T,cmap='bwr',vmin=-500,vmax=500)
# p = axs.pcolormesh(x,y,terms_bdg.data[:,:,20,8].T,cmap='bwr',vmin=-500,vmax=500)

# axs.axhline(Hcanopy,ls='--',c='k')
# axs.axhline(230*dy,ls='--')
axs.set_xlabel(r'$x/z_i$',fontsize=15)
axs.set_ylabel(r'$z/z_i$',fontsize=15)
axs.set_title(f'{cases[case]}',fontsize=15)

cbar = plt.colorbar(p)

plt.show()

#%%Compute anisotropy

# def Anisotropy(Nx, Ny, Nz, R11, R22, R33, R12, R13, R23):
#     import numpy as np

#     # Compute turbulent kinetic energy
#     e = R11 + R22 + R33  # shape: (Nx, Ny, Nz)

#     # Allocate output arrays (using order='F' only matters for Fortran compatibility)
#     xB = np.zeros((Nx, Ny, Nz), order='F')
#     yB = np.zeros((Nx, Ny, Nz), order='F')
#     lambda3 = np.zeros((Nx, Ny, Nz), order='F')

#     # Identity matrix for broadcasting
#     Id = np.eye(3)

#     # Vectorized loop over domain
#     for k in range(Nz):
#         print(f"iteration = {k}")
        
#         # Extract 2D slices for this layer
#         R11_k, R22_k, R33_k = R11[:, :, k], R22[:, :, k], R33[:, :, k]
#         R12_k, R13_k, R23_k = R12[:, :, k], R13[:, :, k], R23[:, :, k]
#         e_k = e[:, :, k]

#         # Flatten for vectorized processing
#         shape2d = (Nx, Ny)
#         flat_size = Nx * Ny
#         indices = np.indices(shape2d).reshape(2, -1)
#         i_idx, j_idx = indices

#         # Build Reynolds stress tensor array (flat for vectorization)
#         R_tensor = np.empty((flat_size, 3, 3))
#         R_tensor[:, 0, 0] = R11_k[i_idx, j_idx]
#         R_tensor[:, 0, 1] = R12_k[i_idx, j_idx]
#         R_tensor[:, 0, 2] = R13_k[i_idx, j_idx]
#         R_tensor[:, 1, 0] = R12_k[i_idx, j_idx]
#         R_tensor[:, 1, 1] = R22_k[i_idx, j_idx]
#         R_tensor[:, 1, 2] = R23_k[i_idx, j_idx]
#         R_tensor[:, 2, 0] = R13_k[i_idx, j_idx]
#         R_tensor[:, 2, 1] = R23_k[i_idx, j_idx]
#         R_tensor[:, 2, 2] = R33_k[i_idx, j_idx]

#         e_flat = e_k[i_idx, j_idx]
#         B_tensor = R_tensor / e_flat[:, None, None] - Id / 3

#         # Eigen-decomposition
#         eigenvalues = np.linalg.eigvalsh(B_tensor)  # faster for symmetric matrices

#         # Sort eigenvalues in descending order
#         sorted_evals = np.sort(eigenvalues, axis=1)[:, ::-1]

#         # Compute lambda3
#         lambda3[:, :, k] = sorted_evals[:, 2].reshape(Nx, Ny)

#         # Compute C-coefficients
#         C1c = sorted_evals[:, 0] - sorted_evals[:, 1]
#         C2c = 2 * (sorted_evals[:, 1] - sorted_evals[:, 2])
#         C3c = 3 * sorted_evals[:, 2] + 1

#         # Compute barycentric coordinates
#         xB[:, :, k] = (C1c + 0.5 * C3c).reshape(Nx, Ny)
#         yB[:, :, k] = (C3c * np.sqrt(3) / 2).reshape(Nx, Ny)

#     return xB, yB, lambda3

def Anisotropy(Nx, Ny, Nz, R11, R22, R33, R12, R13, R23):
    """
    Compute barycentric invariants and smallest eigenvalue lambda3
    from Reynolds stress tensor components stored as xarray.DataArrays.
    """

    import numpy as np
    import xarray as xr

    # Extract dimensions and coordinates from one of the input arrays
    dims = R11.dims
    coords = R11.coords

    # Convert DataArrays to NumPy arrays
    R11v = R11.values
    R22v = R22.values
    R33v = R33.values
    R12v = R12.values
    R13v = R13.values
    R23v = R23.values

    # Calculate TKE
    e = R11v + R22v + R33v
    Ntot = Nx * Ny * Nz

    # Flatten arrays for batch processing
    e_flat = e.ravel(order='F')
    R11_flat = R11v.ravel(order='F')
    R22_flat = R22v.ravel(order='F')
    R33_flat = R33v.ravel(order='F')
    R12_flat = R12v.ravel(order='F')
    R13_flat = R13v.ravel(order='F')
    R23_flat = R23v.ravel(order='F')

    # Build full Reynolds stress tensor (Ntot, 3, 3)
    R_all = np.zeros((Ntot, 3, 3))
    R_all[:, 0, 0] = R11_flat
    R_all[:, 1, 1] = R22_flat
    R_all[:, 2, 2] = R33_flat
    R_all[:, 0, 1] = R12_flat
    R_all[:, 1, 0] = R12_flat
    R_all[:, 0, 2] = R13_flat
    R_all[:, 2, 0] = R13_flat
    R_all[:, 1, 2] = R23_flat
    R_all[:, 2, 1] = R23_flat

    # Avoid division by zero
    e_safe = np.where(e_flat == 0.0, 1e-12, e_flat)

    # Identity matrix for subtraction
    Id = np.eye(3)

    # Compute anisotropy tensor
    B_all = R_all / e_safe[:, None, None] - (1.0 / 3.0) * Id

    # Eigenvalues (symmetric case)
    eigvals_all = np.linalg.eigvalsh(B_all)
    eigvals_sorted = np.sort(eigvals_all, axis=1)[:, ::-1]

    lambda3_flat = eigvals_sorted[:, 2]
    C1c = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    C2c = 2 * (eigvals_sorted[:, 1] - eigvals_sorted[:, 2])
    C3c = 3 * eigvals_sorted[:, 2] + 1

    xB_flat = C1c + 0.5 * C3c
    yB_flat = C3c * (np.sqrt(3) / 2)

    # Reshape back to (Nx, Ny, Nz)
    xB = xr.DataArray(xB_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)
    yB = xr.DataArray(yB_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)
    lambda3 = xr.DataArray(lambda3_flat.reshape((Nx, Ny, Nz), order='F'), dims=dims, coords=coords)

    return xB, yB, lambda3

#%%Compute anisotropy and save to file

# anisotropy = xr.DataArray(np.ones(shape = (Nx,Ny,Nz,3),order='F'),\
#                        dims=('x','y','z','variable'), coords = {'variable':['xB','yB','lambda3']})

# anisotropy[:,:,:,0],anisotropy[:,:,:,1],anisotropy[:,:,:,2] = Anisotropy(Nx, Ny, Nz, terms_ptb[:,:,:,0] + data_TKE[:,:,:,19],\
#                                                                          terms_ptb[:,:,:,3] + data_TKE[:,:,:,20],\
#                                                                              terms_ptb[:,:,:,5] + data_TKE[:,:,:,21],\
#                                                                                  terms_ptb[:,:,:,1] + data_TKE[:,:,:,22],\
#                                                                                      terms_ptb[:,:,:,2] + wnode2uvpnode(data_TKE[:,:,:,23]),\
#                                                                                          terms_ptb[:,:,:,4] + wnode2uvpnode(data_TKE[:,:,:,24]))

# anisotropy.to_netcdf(path+cases[case] + '\\anisotropy.nc')

anisotropy = xr.open_dataarray(path+cases[case] + '/anisotropy.nc')

#%%Plot Anisotropy

os.chdir('C:\\Users\\udina\\Desktop\\UNIVERSITA\\PhD\\Research\\Python_Codes\\functions\\')

from Anisotropy_Functions import ColorAnisotropy

cmap = ColorAnisotropy()

yslice = 123

fig, axs=plt.subplots(1,1,constrained_layout=True,figsize=(12,5))
levels=np.linspace(0,np.sqrt(3)/2,10)
plt1=axs.contourf(x,z_uvp/Hcanopy,anisotropy.data[:,yslice,:,1].T,cmap=cmap,levels=levels,vmin=0,vmax=np.sqrt(3)/2)
axs.contour(x,z_uvp/Hcanopy,anisotropy.data[:,yslice,:,1].T,levels=[0.38],colors='black')
# plt1=axs.contourf(x,z[0:Nz_SLayer],np.transpose(np.mean(yB,axis=1)),cmap=cmap,levels=8)
# axs.pcolormesh(x,z,np.transpose(mask[:,Ny_Slice,:]),cmap='gray')
# axs.plot(x,z_tpg/zi,color='black')
axs.axhline(1,ls='--',c='k')
# axs.plot(x_ax,np.mean(intf,axis=(1)),color='black')
# axs.set_ylim([0,0.1])
# axs.set_xlim([0,1])
# axs.set_title('pu correlation y avg')
axs.set_ylabel(r'$z/z_i$')
axs.set_xlabel(r'$x/z_i$')
fig.colorbar(plt1,ax=axs)
axs.set_title(f'{cases[case]}',fontsize=15)
# plt.savefig(figPath+'yB.png',dpi=300,facecolor='white', edgecolor='white')


plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
import copy

# Optional: remove if not needed
# from scipy.stats import gaussian_kde

l = 0
level = 5

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

# Deep copy the arrays
tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11])

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)

# Select y_B and TKE slices
x_yB = anisotropy[:, :, 10:60, 1]
y_TKE = tmpNorm[:, :, 10:60]

# Compute binned statistics
TKE_median = []
TKE_std = []
yB_mean = []

j = 0.1
for i in range(28):
    if i == 0:
        mask = x_yB < j
        yB_mean.append(0.05)
    else:
        mask = (x_yB > j) & (x_yB < j + 0.025)
        yB_mean.append(j + 0.0125)
    TKE_vals = y_TKE[mask]
    TKE_median.append(np.nanmedian(TKE_vals))
    TKE_std.append(np.nanstd(TKE_vals))
    j += 0.025

# Plot results
axs.plot(yB_mean, TKE_median, c='k', marker='o')
axs.fill_between(
    yB_mean,
    np.array(TKE_median) - np.array(TKE_std),
    np.array(TKE_median) + np.array(TKE_std),
    alpha=0.5,
)

axs.axhline(0, color='k', linestyle='-.')
axs.axvline(0.38, color='k', linestyle='-.')
axs.text(0.38 + 0.01, 200, 'yB = 0.38', rotation=90, va='center', ha='left', color='black')
axs.axvline(0.36, color='k', linestyle='-.')
axs.text(0.36 - 0.02, 200, 'yB = 0.36', rotation=90, va='center', ha='left', color='black')

axs.set_xlabel(r'$y_B$', fontsize=18)
axs.set_ylabel(r'$\frac{P-D}{|D|}$', fontsize=21)
axs.set_xlim(0.15, 0.6)
axs.set_ylim(-70, 250)
axs.tick_params(axis='x', labelsize=12)
axs.tick_params(axis='y', labelsize=12)
axs.set_title(f'{cases[case]}',fontsize=15)

# Correlation text
corr = np.corrcoef(x_yB.values.flatten(), y_TKE.flatten())[0, 1]
axs.text(0.1, 0.9, f'r = {round(corr, 2)}', transform=axs.transAxes, fontsize=18)

plt.tight_layout()
plt.show()


#%% Plot TKE residual normalized by Dissipation with yB contour line

# Deep copy the arrays
tmpDIS = copy.deepcopy(terms_bdg[:, :, :, 11])
tmpRES = copy.deepcopy(terms_bdg[:, :, :, -1] - terms_bdg[:, :, :, 11])
tmpRES = np.where(np.abs(tmpRES) < 10, 0, tmpRES)

# Avoid divide-by-zero
with np.errstate(divide='ignore', invalid='ignore'):
    tmpNorm = np.where(tmpDIS != 0, tmpRES / np.abs(tmpDIS) * 100, np.nan)
    
levels=[-100,-1,1,100]
levels_2 = [0.38]
colors=['blue','white','red']

yslice = 123

fig,axs = plt.subplots(1,1,figsize=(10,5),tight_layout=True)

p = axs.contourf(x,z_uvp,tmpNorm[:,yslice,:].T,colors=colors,alpha=0.5,levels=levels,extend='both')

p.cmap.set_under('blue')
p.cmap.set_over('red')

sc = axs.contour(x,z_uvp,anisotropy[:,yslice,:,1].T,levels=levels_2,colors=['black'])
axs.axhline(39/zi,ls='--',c='k')

plt.show()










