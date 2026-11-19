#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:24:22 2026

@author: u1450851
"""

#%%Import libraries
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr

os.chdir('/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/')

from functions import wnode2uvpnode, uvpnode2wnode
from read_checkpoint_SC import read_checkpoint,read_checkpoint_aniso, read_checkpoint_sfc, read_checkpoint_sfc_L, read_checkpoint_mom, read_checkpoint_tke
from Anisotropy_Functions import ColorAnisotropy, Anisotropy
from scipy.stats import gaussian_kde
cmap = ColorAnisotropy()

from loadData import load_momentum

#%%Simulation parameters

nx = 64
ny = 64
nz = 64

lx = 5120
ly = 5120
lz = 2048

dx = lx/nx
dy = ly/ny
dz = lz/nz

zi = 1024
# zbl = 1099
uscale = 1
# wstar = 2.05
Tscale = 300
qstar = 0.24
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z_uvp = np.arange(0,nz)*dz + dz/2
z_w = np.arange(0,nz)*dz

#%%Load RAV data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 'simB_Patton2011_A5'
step = 136000
time_ext = '10min'

aniso = xr.open_dataarray(path+sim+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
mom3D = xr.open_dataarray(path+sim+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
sc3D = xr.open_dataarray(path+sim+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
tke3D = xr.open_dataarray(path+sim+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
mom2D = xr.open_dataarray(path+sim+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
sc2D = xr.open_dataarray(path+sim+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

#%%Compute the Reynolds stresses and anisotropy from RAV data

uu = mom3D[:,:,:,4] - mom3D[:,:,:,0]**2 - mom3D[:,:,:,16] + (2/3)*tke3D[:,:,:,1]
vv = mom3D[:,:,:,5] - mom3D[:,:,:,1]**2 - mom3D[:,:,:,17] + (2/3)*tke3D[:,:,:,1]
ww = wnode2uvpnode(mom3D[:,:,:,6] - mom3D[:,:,:,2]**2) - mom3D[:,:,:,18] + (2/3)*tke3D[:,:,:,1]
uv = mom3D[:,:,:,13] - mom3D[:,:,:,0]*mom3D[:,:,:,1] - mom3D[:,:,:,19]
uw = wnode2uvpnode(mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2] - mom3D[:,:,:,20])
vw = wnode2uvpnode(mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2] - mom3D[:,:,:,21])

[xB,yB,l3] = Anisotropy(nx,ny,nz,uu,vv,ww,uv,uw,vw)

#%%Compute L

T = uvpnode2wnode(sc3D[:,:,:,0])
T[:,:,0] = sc2D[:,:,-2]
# uw = (mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2] - mom3D[:,:,:,20])
# vw = (mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2] - mom3D[:,:,:,21])
L3D = -(((uw**2 + vw**2)**0.25)**3*sc3D[:,:,:,0])/(0.4*(9.8*zi/uscale**2)*wnode2uvpnode(sc3D[:,:,:,4] - mom3D[:,:,:,2]*uvpnode2wnode(sc3D[:,:,:,0]) - sc3D[:,:,:,7]))

#%%Compute phi_M

dudz = wnode2uvpnode(mom3D[:,:,:,22])
dvdz = wnode2uvpnode(mom3D[:,:,:,23])
# dUdz = (mom3D[:,:,:,0]*dudz + mom3D[:,:,:,1]*dvdz)/np.sqrt(mom3D[:,:,:,0]**2 + mom3D[:,:,:,1]**2)
dUdz = np.sqrt(dudz**2 + dvdz**2)
z3d = np.broadcast_to(z_uvp[None, None, :], (nx,ny,nz))/zi
# uw = (mom3D[:,:,:,14] - uvpnode2wnode(mom3D[:,:,:,0])*mom3D[:,:,:,2] - mom3D[:,:,:,20])
# vw = (mom3D[:,:,:,15] - uvpnode2wnode(mom3D[:,:,:,1])*mom3D[:,:,:,2] - mom3D[:,:,:,21])
ustar = (uw**2 + vw**2)**0.25
phi_m = (0.4*z3d/ustar)*dudz


#%%

from matplotlib.colors import LogNorm

# Select the surface level.
surface_level = 0

# Use z/zi because your L3D calculation uses g*zi/uscale**2.
z_surface = z_uvp[surface_level] / zi
zeta_surface = z_surface / L3D[:, :, surface_level]
phi_surface = phi_m[:, :, surface_level]

# Negative z/L is displayed using the positive coordinate -z/L.
x_data = -zeta_surface

# Keep unstable, finite, positive values within the requested axes.
valid = (
    np.isfinite(x_data)
    & np.isfinite(phi_surface)
    & (x_data >= 1e-3)
    & (x_data <= 1e2)
    & (phi_surface >= 1e-2)
    & (phi_surface <= 1e1)
)

if not np.any(valid):
    raise ValueError("No surface points fall within the requested plot limits.")

# Two-dimensional histogram with logarithmic bin edges.
x_edges = np.geomspace(1e-3, 1e2, 51)
y_edges = np.geomspace(1e-2, 1e1, 51)
counts, _, _ = np.histogram2d(
    x_data[valid], phi_surface[valid],
    bins=(x_edges, y_edges)
)

fig, ax = plt.subplots(figsize=(7, 5))

mesh = ax.pcolormesh(
    x_edges, y_edges, np.ma.masked_equal(counts.T, 0),
    norm=LogNorm(vmin=1, vmax=max(2, counts.max())),
    cmap="Blues",
    shading="auto",
)

# Table 1 unstable reference functions.
x_curve = np.geomspace(1e-3, 1e2, 500)
zeta_curve = -x_curve

phi_ho96 = (1 - 19*zeta_curve)**(-1/4)
phi_gr00 = (1 - 10*zeta_curve)**(-1/3)
phi_ky90 = (
    (1 + 0.6*zeta_curve**2) / (1 - 7.5*zeta_curve)
)**(1/3)
phi_brutsaert = (
    (0.37 - 0.24*np.abs(zeta_curve)**0.72)
    / (0.37 + np.abs(zeta_curve)**0.72)
    - 0.5*np.cbrt(zeta_curve)
)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

yB_max = 0.8
yB_values = np.linspace(0.1, 0.8, 8)
yB_norm = Normalize(vmin=0.1, vmax=0.8)

for yB_value in yB_values:
    # Coefficients for unstable phi_M, from Eq. 8.
    a = 0.24 - 0.38*yB_value if yB_value < 0.6 else 0.012
    b = 0.061
    c = 0.45 - 0.53*yB_value
    n = -0.12 + 6.4*yB_value

    # x_curve = -z/L, so |z/L| = x_curve.
    phi_anisotropy = (
        (a + b*x_curve**n) / (a + x_curve**n)
        + c*x_curve**(1/3)
    )

    ax.plot(
        x_curve,
        np.where(phi_anisotropy > 0, phi_anisotropy, np.nan),
        color=cmap(yB_norm(yB_value)),
        linewidth=1.7,
        zorder=3,
    )

fig.colorbar(
    ScalarMappable(norm=yB_norm, cmap=cmap),
    ax=ax,
    orientation="horizontal",
    label=r"$y_B$",
    pad=0.17,
    fraction=0.05,
)

ax.plot(x_curve, phi_ho96, color="black", lw=2, label="HÖ96")
ax.plot(x_curve, phi_gr00, color="magenta", lw=2, label="GR00")
ax.plot(x_curve, phi_ky90, color="royalblue", lw=2, label="KY90")
ax.plot(
    x_curve, phi_brutsaert,
    color="darkorange", lw=1.8, ls="--",
    label="Brutsaert (KY90 alternative)"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e-3)
ax.set_ylim(1e-2, 1e1)

ax.set_xticks([1e2, 1e1, 1, 1e-1, 1e-2, 1e-3])
ax.set_xticklabels([
    r"$-10^{2}$", r"$-10^{1}$", r"$-10^{0}$",
    r"$-10^{-1}$", r"$-10^{-2}$", r"$-10^{-3}$"
])
ax.set_xlabel(r"$z/L$")
ax.set_ylabel(r"$\phi_M$")
ax.legend()
fig.colorbar(mesh, ax=ax, label="Surface points per bin")
fig.suptitle(sim,fontsize=12)

plt.show()


#%%

from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# 1. Select surface points and set plot ranges
# ------------------------------------------------------------
surface_level = 0
n_stability_bins = 20

x_min, x_max = 1e-3, 1e2       # x = -z/L
phi_min, phi_max = 1e-2, 1e1
yB_max = np.sqrt(3) / 2

z_surface = z_uvp[surface_level] / zi
x_data = -z_surface / L3D[:, :, surface_level]
phi_data = phi_m[:, :, surface_level]
yB_data = yB[:, :, surface_level]

# Use the same surface samples for the density and medians.
valid = (
    np.isfinite(x_data)
    & np.isfinite(phi_data)
    & np.isfinite(yB_data)
    & (x_data >= x_min) & (x_data <= x_max)
    & (phi_data >= phi_min) & (phi_data <= phi_max)
    & (yB_data >= 0) & (yB_data <= yB_max)
)

if not np.any(valid):
    raise ValueError("No valid surface points fall within the plot ranges.")

x_values = x_data[valid]
phi_values = phi_data[valid]
yB_values = yB_data[valid]


# ------------------------------------------------------------
# 2. Bin stability and anisotropy, then find phi_M medians
# ------------------------------------------------------------
# Equal-width bins on the logarithmic x axis.
x_bin_edges = np.geomspace(x_min, x_max, n_stability_bins + 1)
x_bin_centers = np.sqrt(
    x_bin_edges[:-1] * x_bin_edges[1:]
)

# Eight equal-width anisotropy bins.
yB_bin_edges = np.linspace(0, yB_max, 9)
yB_bin_centers = (
    yB_bin_edges[:-1] + yB_bin_edges[1:]
) / 2

# Shape: (n_stability_bins, 8).
phi_medians, _, _, _ = binned_statistic_2d(
    x_values,
    yB_values,
    phi_values,
    statistic="median",
    bins=[x_bin_edges, yB_bin_edges],
)


# ------------------------------------------------------------
# 3. Make the density background from the same samples
# ------------------------------------------------------------
x_hist_edges = np.geomspace(x_min, x_max, 51)
phi_hist_edges = np.geomspace(phi_min, phi_max, 51)

counts, _, _ = np.histogram2d(
    x_values,
    phi_values,
    bins=[x_hist_edges, phi_hist_edges],
)

fig, ax = plt.subplots(
    figsize=(8, 5.5), layout="constrained"
)

mesh = ax.pcolormesh(
    x_hist_edges,
    phi_hist_edges,
    np.ma.masked_equal(counts.T, 0),
    norm=LogNorm(vmin=1, vmax=max(2, counts.max())),
    cmap="Greys",
    alpha=0.45,
    shading="auto",
)


# ------------------------------------------------------------
# 4. Add the classical unstable scaling curves
# ------------------------------------------------------------
x_curve = np.geomspace(x_min, x_max, 500)
zeta_curve = -x_curve

ax.plot(
    x_curve, (1 - 19*zeta_curve)**(-1/4),
    color="black", lw=2, label="HÖ96", zorder=3
)
ax.plot(
    x_curve, (1 - 10*zeta_curve)**(-1/3),
    color="magenta", lw=1.6, label="GR00", zorder=3
)
ax.plot(
    x_curve,
    ((1 + 0.6*zeta_curve**2) /
     (1 - 7.5*zeta_curve))**(1/3),
    color="royalblue", lw=1.6, label="KY90", zorder=3
)
ax.plot(
    x_curve,
    (0.37 - 0.24*np.abs(zeta_curve)**0.72)
    / (0.37 + np.abs(zeta_curve)**0.72)
    - 0.5*np.cbrt(zeta_curve),
    color="darkorange", lw=1.6, ls="--",
    label="Brutsaert", zorder=3
)


# ------------------------------------------------------------
# 5. Add Eq. 8 curves and matching data medians
# ------------------------------------------------------------
yB_norm = Normalize(vmin=0, vmax=yB_max)

for j, yB_center in enumerate(yB_bin_centers):
    color = cmap(yB_norm(yB_center))

    # Eq. 8 coefficients at this anisotropy-bin center.
    a = (
        0.24 - 0.38*yB_center
        if yB_center < 0.6 else 0.012
    )
    b = 0.061
    c = 0.45 - 0.53*yB_center
    n = -0.12 + 6.4*yB_center

    phi_eq8 = (
        (a + b*x_curve**n) / (a + x_curve**n)
        + c*x_curve**(1/3)
    )

    # Colored reference curve.
    ax.plot(
        x_curve,
        np.where(phi_eq8 > 0, phi_eq8, np.nan),
        color=color, lw=1.7, zorder=2
    )

    # Median of phi_M in each stability bin for this yB bin.
    ax.plot(
        x_bin_centers,
        phi_medians[:, j],
        linestyle="none",
        marker="o",
        markersize=4,
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.3,
        zorder=4,
    )


# ------------------------------------------------------------
# 6. Format the logarithmic axes and colorbars
# ------------------------------------------------------------
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x_max, x_min)
ax.set_ylim(phi_min, phi_max)

ax.set_xticks([1e2, 1e1, 1, 1e-1, 1e-2, 1e-3])
ax.set_xticklabels([
    r"$-10^{2}$", r"$-10^{1}$", r"$-10^{0}$",
    r"$-10^{-1}$", r"$-10^{-2}$", r"$-10^{-3}$"
])
ax.set_xlabel(r"$z/L$")
ax.set_ylabel(r"$\phi_M$")

handles, labels = ax.get_legend_handles_labels()
handles += [
    Line2D([0], [0], color=cmap(0.5), lw=1.7,
           label="Eq. 8"),
    Line2D([0], [0], color=cmap(0.5), marker="o",
           linestyle="none", markersize=4,
           label="Bin median"),
]
ax.legend(handles=handles, fontsize=8, ncol=2)

fig.colorbar(
    mesh, ax=ax, label="Surface points per bin", pad=0.02
)
fig.colorbar(
    ScalarMappable(norm=yB_norm, cmap=cmap),
    ax=ax, orientation="horizontal",
    label=r"$y_B$", pad=0.16, fraction=0.05
)
fig.suptitle(sim,fontsize=12)
plt.show()












































































