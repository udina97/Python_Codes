#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:21:36 2026

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
sim = 'simB_Patton2011'
step = 136000
time_ext = '60min'

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

#%%Plot flux variance vs z

# Select vertical levels before plotting.
level_start = 0
level_stop = 2 #nz//2          # Exclusive; use nz for all levels

if not (0 <= level_start < level_stop <= nz):
    raise ValueError("Require 0 <= level_start < level_stop <= nz")

levels = slice(level_start, level_stop)
selected_shape = uu[:, :, levels].shape

from matplotlib.colors import LogNorm

# L3d = np.broadcast_to(sc2D[:, :, 1][:, :, None], selected_shape)
L3d = L3D[:,:,levels]
z3d = np.broadcast_to(z_uvp[levels][None, None, :], selected_shape)/zi
# z3d = np.full(selected_shape, z_uvp[0])/zi
zeta = z3d / L3d

ustar = (uw[:, :, levels]**2 + vw[:, :, levels]**2)**0.25

xedges = np.geomspace(1e-4, 1e3, 101)
yedges = np.linspace(0, 10, 101)

histograms = []
for variance in (uu[:, :, levels], vv[:, :, levels], ww[:, :, levels]):
    sigma = np.sqrt(np.where(variance >= 0, variance, np.nan))
    ratio = sigma / np.where(ustar > 0, ustar, np.nan)

    valid = (
        np.isfinite(zeta) & np.isfinite(ratio)
        & (zeta < 0)
        & (-zeta >= 1e-4) & (-zeta <= 1e3)
        & (ratio >= 0) & (ratio <= 10)
    )
    histograms.append(
        np.histogram2d(-zeta[valid], ratio[valid],
                       bins=(xedges, yedges))[0]
    )

maxcount = max(h.max() for h in histograms)
if maxcount == 0:
    raise ValueError("No selected unstable samples fall within the axis limits.")

from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable

yB_max = np.sqrt(3) / 2
yB_values = (np.arange(8) + 0.5) * yB_max / 8
yB_norm = Normalize(vmin=0, vmax=yB_max)

# Table 1 coefficients for unstable stratification.
a_by_component = {
    "u": lambda yB: 0.784 - 2.582*np.log10(yB),
    "v": lambda yB: 0.725 - 2.702*np.log10(yB),
    "w": lambda yB: 1.119 - 0.019*yB - 0.065*yB**2 + 0.028*yB**3,
}

xcurve = np.geomspace(1e-4, 1e2, 500)  # xcurve = -z/L
unstable_shape = (1 + 3*xcurve)**(1/3)

fig, axes = plt.subplots(
    1, 3, figsize=(15, 4.5), sharey=True, layout="constrained"
)

for ax, hist, traditional_coefficient, component in zip(
    axes, histograms, (2.55, 2.05, 1.35), ("u", "v", "w")
):
    mesh = ax.pcolormesh(
        xedges, yedges, np.ma.masked_equal(hist.T, 0),
        norm=LogNorm(vmin=1, vmax=max(2, maxcount)),
        cmap="Greys", shading="auto"
    )

    # Traditional Figure 1 relation.
    ax.plot(
        xcurve, traditional_coefficient * unstable_shape,
        color="black", linewidth=2.2
    )

    # Anisotropy-dependent Table 1 relations.
    for yBidx in yB_values:
        ax.plot(
            xcurve,
            a_by_component[component](yBidx) * unstable_shape,
            color=cmap(yB_norm(yBidx)),
            linewidth=1.7
        )

    ax.set_xscale("log")
    ax.set_xlim(1e2, 1e-4)
    ax.set_ylim(0, 10)
    ax.set_xticks([1e2, 1, 1e-2, 1e-4])
    ax.set_xticklabels(
        [r"$-10^{2}$", r"$-10^{0}$", r"$-10^{-2}$", r"$-10^{-4}$"]
    )
    ax.set_title(rf"$\sigma_{{{component}}}/u_*$")
    ax.set_xlabel(r"$z/L$")

axes[0].set_ylabel(r"$\sigma_{u,v,w}/u_*$")
fig.colorbar(mesh, ax=axes, label="Samples per bin", pad=0.02)
fig.colorbar(
    ScalarMappable(norm=yB_norm, cmap=cmap),
    ax=axes, label=r"$y_B$", pad=0.02
)
plt.show()


#%%

from scipy.stats import binned_statistic_2d
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

# Plot options
level_start = 0
level_stop = 2                 # Exclusive
n_x_bins = 20
x_bin_spacing = "log"           # "log" or "linear"

if not (0 <= level_start < level_stop <= nz):
    raise ValueError("Require 0 <= level_start < level_stop <= nz")

levels = slice(level_start, level_stop)
shape = uu[:, :, levels].shape

# L3D uses g*zi/uscale**2, so use z/zi with this L3D.
z_nd = np.broadcast_to((z_uvp[levels] / zi)[None, None, :], shape)

# L3D = np.broadcast_to(sc2D[:, :, 1][:, :, None], shape)
L3D = -(((uw**2 + vw**2)**0.25)**3*sc3D[:,:,:,0])/(0.4*(9.8*zi/uscale**2)*wnode2uvpnode(sc3D[:,:,:,4] - mom3D[:,:,:,2]*uvpnode2wnode(sc3D[:,:,:,0]) - sc3D[:,:,:,7]))

with np.errstate(divide="ignore", invalid="ignore"):
    x_data = -z_nd / L3D[:, :, levels]  # Positive plotting coordinate: -z/L

yB_data = yB[:, :, levels]
ustar = (uw[:, :, levels]**2 + vw[:, :, levels]**2)**0.25
ustar = np.where(ustar > 0, ustar, np.nan)

variances = (uu[:, :, levels], vv[:, :, levels], ww[:, :, levels])
ratios = [np.sqrt(np.where(var >= 0, var, np.nan)) / ustar for var in variances]

# Equal-width bins in the chosen x coordinate.
if x_bin_spacing == "log":
    x_edges = np.geomspace(1e-4, 1e2, n_x_bins + 1)
    x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
elif x_bin_spacing == "linear":
    x_edges = np.linspace(1e-4, 1e2, n_x_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
else:
    raise ValueError("x_bin_spacing must be 'log' or 'linear'")

yB_max = np.sqrt(3) / 2
yB_edges = np.linspace(0, yB_max, 9)
yB_centers = (yB_edges[:-1] + yB_edges[1:]) / 2

# Median for each (x bin, yB bin), separately for u, v, and w.
common_valid = (np.isfinite(x_data) & np.isfinite(yB_data) & (x_data >= x_edges[0]) & (x_data <= x_edges[-1]) & (yB_data >= 0) & (yB_data <= yB_max))

medians = []
for ratio in ratios:
    valid = common_valid & np.isfinite(ratio)
    median, _, _, _ = binned_statistic_2d(x_data[valid], yB_data[valid], ratio[valid], statistic="median", bins=[x_edges, yB_edges])
    medians.append(median)       # Shape: (n_x_bins, 8)

# Unstable Table 1 coefficients.
a_by_component = {
    "u": lambda y: 0.784 - 2.582*np.log10(y),
    "v": lambda y: 0.725 - 2.702*np.log10(y),
    "w": lambda y: 1.119 - 0.019*y - 0.065*y**2 + 0.028*y**3,
}

x_curve = np.geomspace(1e-4, 1e2, 500)
shape_curve = (1 + 3*x_curve)**(1/3)
yB_norm = Normalize(vmin=0, vmax=yB_max)

fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True, layout="constrained")

for ax, median, ratio, classical_a, component in zip(
    axes, medians, ratios, (2.55, 2.05, 1.35), ("u", "v", "w")
):
    # Use the same x and yB sample selection as the medians.
    scatter_valid = (
        common_valid
        & np.isfinite(ratio)
        & (ratio >= 0)
        & (ratio <= 10)
    )

    ax.scatter(
        x_data[scatter_valid],
        ratio[scatter_valid],
        s=5,
        color="0.3",
        alpha=0.1,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )

    ax.plot(
        x_curve, classical_a * shape_curve,
        color="black", linewidth=2.2
    )

    for j, yB_center in enumerate(yB_centers):
        color = cmap(yB_norm(yB_center))

        # Colored anisotropy-dependent relation.
        ax.plot(x_curve, a_by_component[component](yB_center) * shape_curve, color=color, linewidth=1.6)

        # Data median at each x-bin center.
        ax.plot(x_centers, median[:, j], linestyle="none", marker="o", markersize=4, color=color, markeredgecolor="black", markeredgewidth=0.3)

    ax.set_xscale("log")
    ax.set_xlim(1e2, 1e-4)
    ax.set_ylim(0, 10)
    ax.set_xticks([1e2, 1, 1e-2, 1e-4])
    ax.set_xticklabels([r"$-10^{2}$", r"$-10^{0}$", r"$-10^{-2}$", r"$-10^{-4}$"])
    ax.set_title(rf"$\sigma_{{{component}}}/u_*$")
    ax.set_xlabel(r"$z/L$")

axes[0].set_ylabel(r"$\sigma_{u,v,w}/u_*$")
axes[0].legend(
    handles=[
        Line2D([0], [0], color="black", lw=2, label="Classical"),
        Line2D([0], [0], color=cmap(0.5), lw=1.6, label="Table 1"),
        Line2D([0], [0], color=cmap(0.5), marker="o",
               linestyle="none", markersize=4, label="Bin median"),
    ],
    fontsize=8,
)

fig.colorbar(
    ScalarMappable(norm=yB_norm, cmap=cmap),
    ax=axes, label=r"$y_B$"
)
fig.suptitle(sim,fontsize=12)
plt.show()
































































