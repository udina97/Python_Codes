#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:40:59 2026

@author: u1450851
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
import scipy.io

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')
from functions import get_dphidx,get_dphidy,get_dphidz,uvpnode2wnode,wnode2uvpnode,compute_d_twr,find_coordinates,average_over_selected_coords

#%%Load data

cases = ['Gap_12_9mps','Gap_8_9mps','Gap_4_9mps','Patch_12_9mps','Patch_8_9mps','Patch_4_9mps','ATTO','Sinusoidal','Flat','simulation_G']

case = 6

if case < 6:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/Data_Momentum_4TKE.nc')
elif case >= 6 and case < 9:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
    data = xr.open_dataarray(path_to_data + cases[case] + '/dataTKE.nc')
else:
    path_to_data = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/'
    data = scipy.io.loadmat(path_to_data + cases[case] + '/profiles.mat')

#%%Simulation Parameters

if case < 6:
    lx = 2*np.pi; ly = 2*np.pi; lz = 1
    nx,ny,nz = np.shape(data[:,:,:,0])
    zi = 1000
    canopyH = 39/zi
    dx = lx/nx; dy = ly/ny; dz = lz/nz
    x = np.arange(0,nx)*dx
    y = np.arange(0,ny)*dy
    z_w = np.arange(0,nz)*dz
    z_uvp = np.arange(0,nz)*dz + dz/2
    LAD = [0.2349432, 0.2715461, 0.2606477, 0.2706234, 0.289694, 0.2125121, 0.139206, 0.063390629, 0.03817526, 0.0219344]
    uscale = 0.313
elif case >= 6 and case < 9:
    lx = 2.88; ly = 2.88; lz = 0.96
    mpiProc = 32
    nx,ny,nz = np.shape(data[:,:,:,0])
    nz = nz + 5
    zi = 1000
    canopyH = 39/zi
    dx = lx/nx; dy = ly/ny; dz = lz/nz
    x = np.arange(0,nx)*dx
    y = np.arange(0,ny)*dy
    z_w = np.arange(0,nz)*dz
    z_uvp = np.arange(0,nz)*dz + dz/2
    LAD = [0.15655190, 0.20633190, 0.24492203, 0.28024144, 0.33267326, 0.33145316, 0.32065714, 0.28729650, 0.25240169, 0.17358901, 0.11740349, 0.064294815, \
    0.041340224, 0.023756023, 0.013134912, 0.013107183]
    uscale = 0.4
else:
    lx = data['lx'][0][0]; ly = data['ly'][0][0]; lz = data['lz'][0][0]
    nx = data['nx'][0][0]; ny = data['ny'][0][0]; nz = data['nz'][0][0]
    dx = data['dx'][0][0]; dy = data['dy'][0][0]; dz = data['dz'][0][0]
    z_uvp = data['z'][0]
    z_w = data['zi'][0]
    uscale = 1.23
    canopyH = 15.3

#%%

ls = ['-','--']
colors = ['k','b','g','r','c','m','y','lime','violet']
Ntwr = 100
Nz_SLayer = 200

#%%Funsctioons to compute ustar from log fit

def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar_G(Nz_SLayer, z_d, u, v, twr=False):    

    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]
    kappa = 0.4

    # Levels to use for logarithmic fit
    fit_levels = [80, 90, 100]

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]

    # Fit the log profile
    # coefs, _ = curve_fit(log_fit, z_data, U_data, maxfev=10000)
    # a, b = coefs
    slope, intercept = np.polyfit(np.log(z_data),U_data,1)
    ustar = kappa * slope
    z0hi = np.exp(-intercept / slope)
    
    # print("slope:", slope)
    # print("intercept:", intercept)
    print("z0:", z0hi)
    # print("z range:", np.nanmin(z_d), np.nanmax(z_d))

    # Evaluate the fit throughout the surface layer.
    u_fit = np.full(Nz_SLayer, np.nan, dtype=float)
    fit_mask = np.isfinite(z_d[:Nz_SLayer]) & (z_d[Nz_SLayer] > 0)

    u_fit[fit_mask] = (ustar / kappa  * np.log(z_d[:Nz_SLayer][fit_mask] / z0hi))

    # # Fitted velocity profile across the entire surface layer
    # u_fit = log_fit(z_d[:Nz_SLayer], a, b)

    # # Compute z0hi and ustar
    # z0hi = 1 / b
    # ustar = U_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit

def compute_d_twr_G(data, coord, height, dz, zi, u_scale, LAD):
    
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

def compute_ustar(Nz_SLayer, z_d, dist, u, v):    
    kappa = 0.4
    z_start = np.argmax(dist > 0) - 5
    z_d = np.asarray(z_d, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Compute velocity magnitude and restrict to surface layer
    # U = np.sqrt(u**2 + v**2)
    # U_mean = U[z_start:z_start+Nz_SLayer]
    U_mean = np.hypot(u[z_start:z_start+Nz_SLayer], v[z_start:z_start+Nz_SLayer])

    # Levels to use for logarithmic fit
    fit_levels = np.array([100, 110, 120], dtype=int)

    z_data = z_d[fit_levels]
    U_data = U_mean[fit_levels]
    valid = (np.isfinite(z_data) & np.isfinite(U_data) & (z_data > 0))
    if valid.sum() < 2:
        raise ValueError("At least two finite wind speeds at positive heights are required.")
    
    # Fit U = slope*ln(z-d) + intercept.
    slope, intercept = np.polyfit(np.log(z_data[valid]),U_data[valid],1)

    if slope <= 0:
        raise ValueError("The fitted wind speed does not increase with logarithmic height.")
        
    # Recover physical log-law parameters.
    ustar = kappa * slope
    z0hi = np.exp(-intercept / slope)
    
    # print("slope:", slope)
    # print("intercept:", intercept)
    print("z0:", z0hi)
    # print("z range:", np.nanmin(z_d), np.nanmax(z_d))

    # Evaluate the fit throughout the surface layer.
    u_fit = np.full(Nz_SLayer, np.nan, dtype=float)
    fit_mask = np.isfinite(z_d[:Nz_SLayer]) & (z_d[Nz_SLayer] > 0)

    u_fit[fit_mask] = (ustar / kappa  * np.log(z_d[:Nz_SLayer][fit_mask] / z0hi))

    return z0hi, ustar, U_mean, U_data, z_data, u_fit

#%%Select coordinates

from functions import build_phi, build_intf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
terrain = plt.get_cmap('terrain')
terrain_truncated = LinearSegmentedColormap.from_list(
    'terrain_truncated', terrain(np.linspace(0.25, 1, 100)))

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

# coord_p = find_coordinates(intf,Ntwr,'max')
# coord_v = find_coordinates(intf,Ntwr,'min')
# coord_f = find_coordinates(intf,Ntwr,'flat')

fig,axs = plt.subplots(1,1,tight_layout=True)

contour1 = axs.contourf(x * zi, y * zi, (intf*zi - (4.5*dz*zi)).T, levels=np.arange(0,100,1),cmap=terrain_truncated)
for i in range(Ntwr):
    # axs.plot(x[coord_p[i][0]]*zi,y[coord_p[i][1]]*zi,'ok')
    # axs.plot(x[coord_v[i][0]]*zi,y[coord_v[i][1]]*zi,'ok')
    axs.plot(x[coord_f[i][0]]*zi,y[coord_f[i][1]]*zi,'ok')
    
plt.show()

#%%

def velocity_inflection_indices(coords):
    """Return the vertical-profile index of the main inflection point."""
    inflection_idx = np.full(len(coords), -1, dtype=int)

    for k, (ix, iy) in enumerate(coords):
        u = data.data[ix, iy, :, idx_u]
        v = data.data[ix, iy, :, idx_v]
        velocity = np.hypot(u, v)

        valid = np.isfinite(velocity)
        indices = np.flatnonzero(valid)

        if indices.size < 4:
            continue

        U = velocity[indices]
        d2U = np.gradient(np.gradient(U))

        # Candidates where the second derivative changes sign.
        crossings = np.flatnonzero(d2U[:-1] * d2U[1:] <= 0)

        if crossings.size == 0:
            continue

        # Select the strongest curvature transition.
        strengths = np.abs(d2U[crossings] - d2U[crossings + 1])
        local_idx = crossings[np.argmax(strengths)]

        inflection_idx[k] = indices[local_idx]

    return inflection_idx


inflection_idx_f = velocity_inflection_indices(coord_f)

# Inflection-point index for tower k:
# z_idx_inflection = inflection_idx_f[k]

#%%Plot the tower coordinate with inflection point next to it

fig, axs = plt.subplots(1, 1, tight_layout=True)

contour1 = axs.contourf(x * zi,y * zi,(intf * zi - 4.5 * dz * zi).T,levels=np.arange(0, 100, 1),cmap=terrain_truncated)

for k, (ix, iy) in enumerate(coord_f):
    x_pos = x[ix] * zi
    y_pos = y[iy] * zi

    axs.plot(x_pos, y_pos, "ok")

    label = (str(inflection_idx_f[k]) if inflection_idx_f[k] >= 0 else "N/A")

    axs.annotate(label,xy=(x_pos, y_pos),xytext=(5, 5),textcoords="offset points",fontsize=8,color="black")

fig.colorbar(contour1, ax=axs, label="Surface elevation")
plt.show()

#%%Compute the friction velocity by fitting the logarithmic profile to all coordinates and from the shear for TOPOGRAPHY CASES

phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
intf, iintf = build_intf(phi, dz)

coord_p = find_coordinates(intf,Ntwr,'max')
coord_v = find_coordinates(intf,Ntwr,'min')
# coord_f = find_coordinates(intf,Ntwr,'flat')

z_profile = np.arange(nz) * dz * zi
zeds = np.ones((nx, ny, 1)) * z_profile
dist = copy.deepcopy(zeds)
del zeds,z_profile
dist -= intf[:, :, np.newaxis] * zi
mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
mask4D = np.expand_dims(mask, axis=-1)
data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
del mask,mask4D

height = math.ceil(canopyH/dz)

d_dim_p = compute_d_twr(data,coord_p,dist,height+5,dz,zi,uscale,LAD)
d_dim_v = compute_d_twr(data,coord_v,dist,height+5,dz,zi,uscale,LAD)
# d_dim_f = compute_d_twr(data,coord_f,dist,height+5,dz,zi,uscale,LAD)

# Variable indices
idx_u, idx_v, idx_w = 0, 1, 2
idx_uw, idx_vw = 8, 9
idx_txz, idx_tyz = 23, 24

# Allocate output vectors.
# np.nan makes failed or invalid calculations easy to identify.
ustar_shear_p = np.full(len(coord_p), np.nan, dtype=np.float64)
ustar_logfit_p = np.full(len(coord_p), np.nan, dtype=np.float64)
ustar_shear_v = np.full(len(coord_v), np.nan, dtype=np.float64)
ustar_logfit_v = np.full(len(coord_v), np.nan, dtype=np.float64)

# ustar_shear_f = np.full(len(coord_f), np.nan, dtype=np.float64)
# ustar_logfit_f = np.full(len(coord_f), np.nan, dtype=np.float64)

def calculate_ustar_vectors(coords, d_dim):
    """Calculate shear-stress and log-fit friction velocities."""

    n_towers = len(coords)

    ustar_shear = np.full(n_towers, np.nan, dtype=np.float64)
    ustar_logfit = np.full(n_towers, np.nan, dtype=np.float64)

    for k, (ix, iy) in enumerate(coords):
        # First vertical index above the local surface.
        positive_indices = np.flatnonzero(dist[ix, iy, :] > 0)

        if positive_indices.size == 0:
            continue

        z_start = positive_indices[0] - 5

        # Location used for the shear-stress calculation.
        z_idx_ustar = z_start + 16
        # z_idx_ustar = z_start + inflection_idx_f[k]

        if not 0 <= z_idx_ustar < data.data.shape[2]:
            continue

        u = data.data[ix, iy, z_idx_ustar, idx_u]
        v = data.data[ix, iy, z_idx_ustar, idx_v]
        w = data.data[ix, iy, z_idx_ustar, idx_w]

        uw = data.data[ix, iy, z_idx_ustar, idx_uw]
        vw = data.data[ix, iy, z_idx_ustar, idx_vw]

        txz = data.data[ix, iy, z_idx_ustar, idx_txz]
        tyz = data.data[ix, iy, z_idx_ustar, idx_tyz]

        # Resolved plus modeled Reynolds shear stresses.
        Ruw = uw - u * w - txz
        Rvw = vw - v * w - tyz

        # u* = sqrt(sqrt(Ruw^2 + Rvw^2))
        ustar_shear[k] = np.sqrt(np.hypot(Ruw, Rvw))

        # Displacement-corrected height for the log-profile fit.
        z_d = z_uvp - d_dim[k] / zi

        (_z0hi,ustar_logfit[k],_U_mean,_U_data,_z_data,_u_fit) = compute_ustar(Nz_SLayer,z_d,dist[ix, iy, :],data.data[ix, iy, :, idx_u],data.data[ix, iy, :, idx_v])

    return ustar_shear, ustar_logfit


# Calculate the two u* estimates at peak and valley coordinates.
ustar_shear_p, ustar_logfit_p = calculate_ustar_vectors(coord_p,d_dim_p)
ustar_shear_v, ustar_logfit_v = calculate_ustar_vectors(coord_v,d_dim_v)
# ustar_shear_f, ustar_logfit_f = calculate_ustar_vectors(coord_f,d_dim_f)

diff_p = abs(ustar_logfit_p-ustar_shear_p)
diff_v = abs(ustar_logfit_v-ustar_shear_v)
# diff_f = abs(ustar_logfit_f-ustar_shear_f)

#%%Compute ustar from log fit and shear for the GAP and PATCH cases

if case < 6:
    CANOPY_TOP_LEVEL_GAP_PATCH = 10
    idx_u, idx_v, idx_w = 0, 1, 2
    idx_uw, idx_vw = 8, 9
    idx_txz, idx_tyz = 23, 24

    sfc = np.load(path_to_data+'../input_txt_files/'+cases[case]+'/sfc.npy')

    def sample_surface_towers(sfc, value, Ntwr):
        coords = np.argwhere(np.isclose(sfc, value))
        if coords.shape[0] < Ntwr:
            raise ValueError(f"Only {coords.shape[0]} points found for sfc={value}; need {Ntwr}.")
        return coords[np.random.choice(coords.shape[0], size=Ntwr, replace=False)]

    sel_coord_f = sample_surface_towers(sfc, 1.6, Ntwr)
    sel_coord_p = sample_surface_towers(sfc, 0.0, Ntwr)

    u_w = uvpnode2wnode(data.data[:,:,:,idx_u])
    v_w = uvpnode2wnode(data.data[:,:,:,idx_v])
    Ruw = wnode2uvpnode(data.data[:,:,:,idx_uw] - u_w*data.data[:,:,:,idx_w] - data.data[:,:,:,idx_txz])
    Rvw = wnode2uvpnode(data.data[:,:,:,idx_vw] - v_w*data.data[:,:,:,idx_w] - data.data[:,:,:,idx_tyz])
    shear_field = np.hypot(Ruw, Rvw)
    del u_w, v_w, Ruw, Rvw

    height = math.ceil(canopyH/dz)
    disp_f = compute_d_twr_G(data, sel_coord_f, height, dz, zi, uscale, LAD)
    disp_p = np.zeros(sel_coord_p.shape[0], dtype=float)

    def calculate_gap_patch_ustar_vectors(coords, d_dim):
        n_towers = len(coords)
        z0hi = np.full(n_towers, np.nan, dtype=np.float64)
        ustar_shear = np.full(n_towers, np.nan, dtype=np.float64)
        ustar_logfit = np.full(n_towers, np.nan, dtype=np.float64)
        u_mag_twr = np.full((n_towers, Nz_SLayer), np.nan, dtype=np.float64)

        for k, (ix, iy) in enumerate(coords):
            z_d = z_uvp - d_dim[k] / zi
            z0hi[k], ustar_logfit[k], _U_mean, _U_data, _z_data, _u_fit = compute_ustar_G(
                Nz_SLayer, z_d, data.data[ix, iy, :, idx_u], data.data[ix, iy, :, idx_v], True)

            if CANOPY_TOP_LEVEL_GAP_PATCH < shear_field.shape[2]:
                ustar_shear[k] = np.sqrt(shear_field[ix, iy, CANOPY_TOP_LEVEL_GAP_PATCH])

            if np.isfinite(ustar_logfit[k]) and ustar_logfit[k] != 0:
                u_mag_twr[k, :] = np.hypot(data.data[ix, iy, :Nz_SLayer, idx_u], data.data[ix, iy, :Nz_SLayer, idx_v]) / ustar_logfit[k]

        return z0hi, ustar_shear, ustar_logfit, u_mag_twr, np.nanmean(u_mag_twr, axis=0)

    z0hi_f, ustar_shear_f, ustar_logfit_f, u_mag_twr_f, TwrAvgProf_f = (
        calculate_gap_patch_ustar_vectors(sel_coord_f, disp_f)
    )
    z0hi_p, ustar_shear_p, ustar_logfit_p, u_mag_twr_p, TwrAvgProf_p = (
        calculate_gap_patch_ustar_vectors(sel_coord_p, disp_p)
    )

    diff_f = np.abs(ustar_logfit_f - ustar_shear_f)
    diff_p = np.abs(ustar_logfit_p - ustar_shear_p)
else:
    print(f"Skipping GAP/PATCH tower ustar cell for non-GAP/PATCH case: {cases[case]}")

#%%Compute domain-averaged canopy-top friction velocity from the shear field

CANOPY_TOP_LEVEL_GAP_PATCH = 10
CANOPY_TOP_LEVEL_TOPO = 16

idx_u, idx_v, idx_w = 0, 1, 2
idx_uw, idx_vw = 8, 9
idx_txz, idx_tyz = 23, 24


if case >= 9:
    raise NotImplementedError(
        "Domain-averaged canopy-top ustar from the 3D shear field is only "
        "implemented for the NetCDF cases."
    )

data_array = data.data

# Total resolved plus modeled shear-stress magnitude on uvp nodes.
if case < 6:
    u_w = uvpnode2wnode(data_array[:, :, :, idx_u])
    v_w = uvpnode2wnode(data_array[:, :, :, idx_v])
    Ruw = (
        data_array[:, :, :, idx_uw]
        - u_w * data_array[:, :, :, idx_w]
        - data_array[:, :, :, idx_txz]
    )
    Rvw = (
        data_array[:, :, :, idx_vw]
        - v_w * data_array[:, :, :, idx_w]
        - data_array[:, :, :, idx_tyz]
    )
    Ruw = wnode2uvpnode(Ruw)
    Rvw = wnode2uvpnode(Rvw)
    del u_w, v_w
else:
    Ruw = (
        data_array[:, :, :, idx_uw]
        - data_array[:, :, :, idx_u] * data_array[:, :, :, idx_w]
        - data_array[:, :, :, idx_txz]
    )
    Rvw = (
        data_array[:, :, :, idx_vw]
        - data_array[:, :, :, idx_v] * data_array[:, :, :, idx_w]
        - data_array[:, :, :, idx_tyz]
    )

shear_field = np.sqrt(Ruw**2 + Rvw**2)
del Ruw, Rvw

if case < 6:
    canopy_top_idx = np.full((nx, ny), CANOPY_TOP_LEVEL_GAP_PATCH, dtype=int)
    valid_canopy_top = np.full(
        (nx, ny),
        CANOPY_TOP_LEVEL_GAP_PATCH < data_array.shape[2],
        dtype=bool,
    )
else:
    phi = build_phi(path_to_data + cases[case] + '/phi_functions/', nx, ny, nz, mpiProc)
    intf, iintf = build_intf(phi, dz)
    z_profile = np.arange(nz) * dz * zi
    zeds = np.ones((nx, ny, 1)) * z_profile
    dist = copy.deepcopy(zeds)
    del zeds,z_profile
    dist -= intf[:, :, np.newaxis] * zi
    mask = dist[:,:,5:] < 0  # shape (Nx, Ny, Nz_SLayer)
    mask4D = np.expand_dims(mask, axis=-1)
    data.data[:, :, :, :] = np.where(mask4D, np.nan, data.data[:, :, :, :])
    del mask,mask4D
    dist_offset = dist.shape[2] - data_array.shape[2]
    first_air_idx = np.argmax(dist > 0, axis=2) - dist_offset
    canopy_top_idx = first_air_idx + CANOPY_TOP_LEVEL_TOPO
    valid_canopy_top = (
        np.any(dist > 0, axis=2)
        & (canopy_top_idx >= 0)
        & (canopy_top_idx < data_array.shape[2])
    )

ix, iy = np.indices((nx, ny))
ustar_canopy_top = np.full((nx, ny), np.nan, dtype=float)
ustar_canopy_top[valid_canopy_top] = np.sqrt(
    shear_field[
        ix[valid_canopy_top],
        iy[valid_canopy_top],
        canopy_top_idx[valid_canopy_top],
    ]
)

ustar_shear_domain_avg = np.nanmean(ustar_canopy_top)
ustar_shear_domain_avg_dimensional = ustar_shear_domain_avg * uscale

print(
    f"{cases[case]} domain-averaged canopy-top ustar from shear: "
    f"{ustar_shear_domain_avg:.6g}"
)
print(
    f"{cases[case]} dimensional domain-averaged canopy-top ustar from shear: "
    f"{ustar_shear_domain_avg_dimensional:.6g}"
)

#%%

fig,axs = plt.subplots(1,1,tight_layout=True)

# axs.plot(np.mean(Umag_p,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[0],label='Peak')
# axs.plot(np.mean(Umag_v,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Valley')
axs.plot(np.mean(Umag_f,axis=(0)),z_uvp[:Nz_SLayer]/canopyH,c=colors[case],ls=ls[1],label='Flat')

axs.set_xlabel(r"$\overline{U}/u_*$",fontsize=15)
axs.set_ylabel(r"$z/h_C$",fontsize=15)
    
axs.set_ylim(0,10)
axs.axhline(1,ls='--',c='k')

plt.show() 















































