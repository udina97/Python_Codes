#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:05:59 2026

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

from pathlib import Path
import re

_SNAPSHOT_NAME = re.compile(r"^snapshots_(\d+)\.dat$")

def load_output4_snapshots(
    path, nx, ny, nz, num_snapshots, start_iteration, num_scalars=1
):
    """Load velocity and scalar snapshots written by Fortran ``output4``.

    ``output4`` writes the fields consecutively as ``u``, ``v``, ``w``, then
    ``sc(:, :, :, 1)`` through ``sc(:, :, :, num_scalars)``.  All fields use
    Fortran array order.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory containing ``snapshots_#########.dat`` files.
    nx, ny, nz : int
        Global grid dimensions of each snapshot.
    num_snapshots : int
        Number of snapshot files to load.
    start_iteration : int
        Files at this iteration and later are selected in increasing iteration
        order.
    num_scalars : int, default=1
        Value of the Fortran ``nb_sc`` setting used when the snapshots were
        written.

    Returns
    -------
    u, v, w, sc : numpy.ndarray
        ``u``, ``v``, and ``w`` have shape ``(nx, ny, nz, num_snapshots)``.
        ``sc`` has shape ``(nx, ny, nz, num_scalars, num_snapshots)``.
    """
    dimensions = (nx, ny, nz)
    if any(not isinstance(size, (int, np.integer)) or size <= 0 for size in dimensions):
        raise ValueError("nx, ny, and nz must be positive integers")
    if not isinstance(num_snapshots, (int, np.integer)) or num_snapshots <= 0:
        raise ValueError("num_snapshots must be a positive integer")
    if not isinstance(num_scalars, (int, np.integer)) or num_scalars <= 0:
        raise ValueError("num_scalars must be a positive integer")

    snapshot_dir = Path(path)
    if not snapshot_dir.is_dir():
        raise NotADirectoryError(f"Snapshot directory does not exist: {snapshot_dir}")

    available = []
    for file_path in snapshot_dir.iterdir():
        match = _SNAPSHOT_NAME.match(file_path.name)
        if match and int(match.group(1)) >= start_iteration:
            available.append((int(match.group(1)), file_path))
    available.sort(key=lambda item: item[0])
    if len(available) < num_snapshots:
        raise FileNotFoundError(
            f"Requested {num_snapshots} snapshots from iteration {start_iteration}, "
            f"but found only {len(available)} in {snapshot_dir}"
        )

    count_per_field = nx * ny * nz
    fields_per_snapshot = 3 + num_scalars
    expected_bytes = fields_per_snapshot * count_per_field * np.dtype("<f8").itemsize
    u = np.empty((*dimensions, num_snapshots), dtype=np.float64, order="F")
    v = np.empty_like(u, order="F")
    w = np.empty_like(u, order="F")
    sc = np.empty((*dimensions, num_scalars, num_snapshots), dtype=np.float64, order="F")

    for time_index, (_, file_path) in enumerate(available[:num_snapshots]):
        if file_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Unexpected size for {file_path}: expected {expected_bytes} bytes "
                f"for {fields_per_snapshot} ({nx}, {ny}, {nz}) float64 fields"
            )

        values = np.fromfile(str(file_path), dtype="<f8")
        u[..., time_index] = values[:count_per_field].reshape(dimensions, order="F")
        v[..., time_index] = values[count_per_field : 2 * count_per_field].reshape(
            dimensions, order="F"
        )
        w[..., time_index] = values[2 * count_per_field : 3 * count_per_field].reshape(
            dimensions, order="F"
        )
        sc[..., time_index] = values[3 * count_per_field :].reshape(
            (*dimensions, num_scalars), order="F"
        )

    return u, v, w, sc

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

#%%Load data

path = '/scratch/general/nfs1/u1450851/LES_Sims/'
sim = 'simB_Patton2011'
step = 136000
time_ext = '30min'

aniso = xr.open_dataarray(path+sim+'/data/Anisotropy/Data_Anisotropy_'+str(step)+'.nc').data
mom3D = xr.open_dataarray(path+sim+'/data/Momentum3D/Data_Momentum_'+str(time_ext)+'.nc').data
sc3D = xr.open_dataarray(path+sim+'/data/Scalar3D/Data_Scalar_'+str(time_ext)+'.nc').data
# tke3D = xr.open_dataarray(path+sim+'/data/TKE_SGS_3D/Data_tke_sgs_'+str(time_ext)+'.nc').data
# mom2D = xr.open_dataarray(path+sim+'/data/Momentum2D/Data_Momentum_2D_'+str(time_ext)+'.nc').data
# sc2D = xr.open_dataarray(path+sim+'/data/Scalar2D/Data_Scalar_2D_'+str(time_ext)+'.nc').data

num_snap = 1
u, v, w, sc = load_output4_snapshots(path+sim+'/output_snap/', nx, ny, nz, num_snap, step, num_scalars=1)

#%%Plot istantaneous flow field and anisotropy

fig,axs = plt.subplots(1,3,tight_layout=True,figsize=(14,4))

axs[0].pcolormesh(x,z_uvp,u[:,ny//2,:,0].T,cmap='viridis')
axs[1].pcolormesh(x,z_uvp,v[:,ny//2,:,0].T,cmap='viridis')
axs[2].pcolormesh(x,z_w,w[:,ny//2,:,0].T,cmap='viridis')

axs[0].set_title(r'u',fontsize=12)
axs[1].set_title(r'v',fontsize=12)
axs[2].set_title(r'w',fontsize=12)

axs[0].set_ylabel(r'z [m]',fontsize=12)

for i in range(len(axs)):
    axs[i].set_xlabel(r'x [m]',fontsize=12)

plt.show()

fig,axs = plt.subplots(1,1,tight_layout=True,figsize=(7,4))

# axs.pcolormesh(x,z_uvp,aniso[:,ny//2,:,1].T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)
axs.pcolormesh(x,z_uvp,np.mean(aniso[:,:,:,1],axis=(1)).T,cmap=cmap,vmin=0,vmax=np.sqrt(3)/2)

axs.set_title(r'$y_B$',fontsize=12)

axs.set_ylabel(r'z [m]',fontsize=12)
axs.set_xlabel(r'x [m]',fontsize=12)

plt.show()

#%%Compute Reynolds stresses

u_prime = u[:,:,:,0] - mom3D[:,:,:,0]
v_prime = v[:,:,:,0] - mom3D[:,:,:,1]
w_prime = wnode2uvpnode(w[:,:,:,0] - mom3D[:,:,:,2])
T_prime= sc[:,:,:,0,0] - sc3D[:,:,:,0]

uu = u_prime*u_prime
vv = v_prime*v_prime
ww = w_prime*w_prime
uv = u_prime*v_prime
uw = u_prime*w_prime
vw = v_prime*w_prime

yB = aniso[:,:,:,1]

ustar = (uw**2 + vw**2)**(1/4)
L3D = -(ustar**3*sc[:,:,:,0,0])/((0.4*9.8*zi/uscale**2)*(w_prime*T_prime))

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
# L3D = -(((uw**2 + vw**2)**0.25)**3*sc3D[:,:,:,0])/(0.4*(9.8*zi/uscale**2)*wnode2uvpnode(sc3D[:,:,:,4] - mom3D[:,:,:,2]*uvpnode2wnode(sc3D[:,:,:,0]) - sc3D[:,:,:,7]))

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



















































