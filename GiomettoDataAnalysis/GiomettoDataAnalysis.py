#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 10:10:58 2025

@author: u1450851
"""

import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np

#%%Import data

# Path to your .mat file
os.chdir('/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/')
sim = 'simulation_G'
mat_file_path = sim + '/profiles.mat'

# Load the .mat file
data = scipy.io.loadmat(mat_file_path)

# Inspect the keys in the dictionary
print("Keys in .mat file:", data.keys())

#%%Simulation parameters

nx = data['nx'][0][0]
ny = data['ny'][0][0]
nz = data['nz'][0][0]

lx = data['lx'][0][0]
ly = data['ly'][0][0]
lz = data['lz'][0][0]

dx = data['dx'][0][0]
dy = data['dy'][0][0]
dz = data['dz'][0][0]

z_uvp = data['z'][0]
z_w = data['zi'][0]
uscale = 1.23
zscale = 15.3

#%%Reynolds stresses

R11 = data['uu_xy'].squeeze()
R22 = data['vv_xy'].squeeze()
R33 = data['ww_xy'].squeeze()
R12 = data['uv_xy'].squeeze()
R13 = data['uw_xy'].squeeze()
R23 = data['vw_xy'].squeeze()

R11d = data['uu_xy'].squeeze() + data['uud_xy'].squeeze()
R22d = data['vv_xy'].squeeze() + data['vvd_xy'].squeeze()
R33d = data['ww_xy'].squeeze() + data['wwd_xy'].squeeze()
R12d = data['uv_xy'].squeeze() + data['uvd_xy'].squeeze()
R13d = data['uw_xy'].squeeze() + data['uwd_xy'].squeeze()
R23d = data['vw_xy'].squeeze() + data['vwd_xy'].squeeze()

#%%Compute ustar

ustar = ((R13 + data['txz_xy'].squeeze())**2 + (R23 + data['tyz_xy'].squeeze())**2)**(0.25)
ustar_d = ((R13d + data['txz_xy'].squeeze())**2 + (R23d + data['tyz_xy'].squeeze())**2)**(0.25)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(ustar,z_uvp,c='k',label='ustar')
axs.plot(ustar_d,z_uvp,c='k',ls='--',label='ustar_d')

axs.set_ylabel(r"z",fontsize=12)
axs.set_xlabel(r"$u_*$",fontsize=12)
# axs.set_xlim(-0.1,10)
axs.set_ylim(z_uvp[0],z_uvp[-1])
axs.set_title('Friction velocity',fontsize = 12)
axs.legend()

plt.show()

#%%Velocity magnitude profile

u = data['u_xy'].squeeze()
v = data['v_xy'].squeeze()
w = data['w_xy'].squeeze()

Umag = np.sqrt(u**2 + v**2 + w**2)/np.max(ustar)
Umag_d = np.sqrt(u**2 + v**2 + w**2)/np.max(ustar_d)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(Umag,z_uvp,c='k',label='Umag')
axs.plot(Umag_d,z_uvp,c='k',ls='--',label='Umag_d')

axs.set_ylabel(r"z",fontsize=12)
axs.set_xlabel(r"$\overline{U}$",fontsize=12)
# axs.set_xlim(-0.1,10)
axs.set_ylim(z_uvp[0],z_uvp[-1])
axs.set_title('Velocity Magnitude',fontsize = 12)
axs.legend()

plt.show()

#%%Plot the shear stress profile

shear = np.sqrt((R13 + data['txz_xy'].squeeze())**2 + ((R23 + data['tyz_xy'].squeeze()))**2)/np.max(ustar)**2
shear_d = np.sqrt((R13d + data['txz_xy'].squeeze())**2 + (R23d + data['tyz_xy'].squeeze())**2)/np.max(ustar_d)**2

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(shear,z_uvp,c='k',label='Shear')
axs.plot(shear_d,z_uvp,c='k',ls='--',label='Shear_d')

axs.set_ylabel(r"z",fontsize=12)
axs.set_xlabel(r"$\sqrt{\overline{u'w'}^2 + \overline{v'w'}^2}$",fontsize=12)
# axs.set_xlim(-0.1,10)
axs.set_ylim(z_uvp[0],z_uvp[-1])
axs.set_title('Shear Stress', fontsize=12)
axs.legend()

plt.show()

#%%Compute the velocity gradient
kappa = 0.4


def log_fit(z, a, b):
    return a * np.log(b * z)

def compute_ustar(Nz_SLayer, z_d, u, v, twr=False):    
    from scipy.optimize import curve_fit
    # Compute velocity magnitude and restrict to surface layer
    U = np.sqrt(u**2 + v**2)
    U_mean = U[:Nz_SLayer]

    # Levels to use for logarithmic fit
    fit_levels = [50, 60, 70, 80]

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

def phi_m_loc(Nz_SLayer, z_d, u, v, avgdUdz, avgdVdz, ustar):
    kappa = 0.4

    U_mag = np.sqrt(u**2 + v**2)                   # Magnitude of velocity
    mean_dUdz = (u * avgdUdz + v * avgdVdz) / U_mag  # Directional mean shear
    phi_m_1d = (kappa * z_d[:Nz_SLayer] / ustar) * mean_dUdz

    return phi_m_1d

def compute_d_twr(data, coord, height, dz, zi, u_scale, LAD):
    from scipy.integrate import trapezoid
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

z_d = z_uvp
z0hi,ustar_v2,U_mean,U_data,z_data,u_fit = compute_ustar(nz, z_d, u, v)
phi = phi_m_loc(nz, z_d, u, v, data['dudz_xy'].squeeze(), data['dvdz_xy'].squeeze(), ustar_v2)

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(phi,z_uvp,c='k')

axs.set_xlabel(r"$\phi_M$",fontsize=12)
axs.set_ylabel(r"$z$",fontsize=12)

plt.show()

#%%tke Budget
tke_nd = zscale/uscale**3

diss = data['ds_xy'].squeeze()*tke_nd
mtr = data['mtr_xy'].squeeze()*tke_nd
ptrans = data['pt_xy'].squeeze()*tke_nd
prod = data['sp_xy'].squeeze()*tke_nd
prod_disp = data['spd_xy'].squeeze()*tke_nd
spm1 = data['spm1_xy'].squeeze()*tke_nd
spm2 = data['spm2_xy'].squeeze()*tke_nd
ttrans = data['tt_xy'].squeeze()*tke_nd
ttrans_disp = data['ttd_xy'].squeeze()*tke_nd
# res = diss[8:] + ptrans[7:] + prod[8:] + ttrans[7:] + mtr[7:]
pmd = diss[8:] + prod[8:] + prod_disp[8:] - spm2[7:]

z_plot = 80

z = z_uvp[:z_plot]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(diss[8:z_plot+8],z,c='k',label='diss')
axs.plot(mtr[7:z_plot+7],z,c='lime',label='mtr')
axs.plot(ptrans[7:z_plot+7],z,c='cyan',label='ptrans')
axs.plot(prod[8:z_plot+8],z,c='r',label='prod')
axs.plot(prod_disp[8:z_plot+8],z,c='r',ls='--',label='prod_disp')
# axs.plot(spm1[8:68],z,c='brown',label='spm1')
axs.plot(-spm2[7:z_plot+7],z,c='r',ls='-.',label='spm2')
axs.plot(ttrans[7:z_plot+7],z,c='b',label='ttrans')
axs.plot(ttrans_disp[7:z_plot+7],z,c='b',ls='--',label='ttrans_disp')
axs.plot(pmd[0:z_plot],z,c='grey',label='res')
axs.axhline(zscale,c='grey',linestyle='--')
axs.axhline(1.28*zscale,c='grey',linestyle='--')
# axs.axhline(crossings[1]-8,c='k',linestyle='--')

axs.set_ylim(0,z_plot)
axs.legend()

plt.show()


#%%Compute anisotropy

def Anisotropy1D(R11, R22, R33, R12, R13, R23):
   
    import numpy as np

    # TKE
    e = R11 + R22 + R33
    N = len(R11)

    # Build Reynolds stress tensor (N, 3, 3)
    R_all = np.zeros((N, 3, 3))
    R_all[:, 0, 0] = R11
    R_all[:, 1, 1] = R22
    R_all[:, 2, 2] = R33
    R_all[:, 0, 1] = R_all[:, 1, 0] = R12
    R_all[:, 0, 2] = R_all[:, 2, 0] = R13
    R_all[:, 1, 2] = R_all[:, 2, 1] = R23

    # Avoid division by zero
    e_safe = np.where(e == 0.0, 1e-12, e)

    # Identity matrix
    Id = np.eye(3)

    # Anisotropy tensor
    B_all = R_all / e_safe[:, None, None] - (1.0 / 3.0) * Id

    # Eigenvalues (symmetric case)
    eigvals_all = np.linalg.eigvalsh(B_all)
    eigvals_sorted = np.sort(eigvals_all, axis=1)[:, ::-1]

    lambda3 = eigvals_sorted[:, 2]
    C1c = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    C2c = 2 * (eigvals_sorted[:, 1] - eigvals_sorted[:, 2])
    C3c = 3 * eigvals_sorted[:, 2] + 1

    xB = C1c + 0.5 * C3c
    yB = C3c * (np.sqrt(3) / 2)

    return xB, yB, lambda3

xB,yB,lambda3 = Anisotropy1D(R11d, R22d, R33d, R12d, R13d, R23d)
import xarray as xr 
aniso = xr.DataArray(np.ones(shape = (nz,2),order='F'),\
                       dims=('z','variable'), coords = {'variable':['xB','yB']})
aniso[:,0] = xB
aniso[:,1] = yB
# aniso.to_netcdf('simulation_G/anisotropy.nc')
crossings = np.where((yB[:-1] < 0.38) & (yB[1:] > 0.38))[0]

fig,axs = plt.subplots(1,1,tight_layout=True)

axs.plot(yB[8:z_plot+8],z,c='k',label='yB')

axs.set_xlabel(r"$y_B$",fontsize=12)
axs.set_ylabel(r"$z$",fontsize=12)
axs.axvline(0.38,c='k',linestyle='--')
axs.axhline(crossings[1]-8,c='k',linestyle='--')
axs.set_ylim(0,z_plot)

plt.show()











































