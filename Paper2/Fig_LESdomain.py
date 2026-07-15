#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:44:42 2026

@author: u1450851
"""
#%%
# Libraries and Functions
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %% Set up cases, paths, names

HOME = Path("/uufs/chpc.utah.edu/common/home/u1450851")
B_PATH = Path("/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims")
G_PATH = Path("/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData")

FUNCTIONS_PATH = HOME / "Python_Codes" / "UCLAanalysis" / "Zev_UCLA_Project"
if str(FUNCTIONS_PATH) not in sys.path:
    sys.path.insert(0, str(FUNCTIONS_PATH))

from functions import build_phi


B_CASES = {
    "Flat": "simflat_256x256x384_out5hr_v2",
    "Sinusoidal": "simbicheng_hill_256x256x384_out5hr",
    "ATTO": "simATTO_256x256x384_full_out5hr",
}
G_CASES = {
    "Gap": "Gap_8_9mps",
    "Patch": "Patch_8_9mps",
}


# %% Load grid information

zi = 1000.0
parameter_file = B_PATH / B_CASES["Flat"] / "output" / "ta1_field" / "parameters.txt"
with parameter_file.open("r") as param_file:
    param = [float(line.strip()) for line in param_file.readlines()]

Nx = int(param[0])
Ny = int(param[1])
Nz = int(param[2])
Lx = param[3]
Ly = param[4]
dx = param[6]
dy = param[7]
dz = param[8]
mpiProc = int(param[11])
z_shift = 4.5 * dz

x_b = np.arange(0, Nx) * dx * zi
y_b = np.arange(0, Ny) * dy * zi

Nx_g = 256
Ny_g = 256
Lx_g = 2 * np.pi
Ly_g = 2 * np.pi
dx_g = Lx_g / Nx_g
dy_g = Ly_g / Ny_g
x_g = np.arange(0, Nx_g) * dx_g * zi
y_g = np.arange(0, Ny_g) * dy_g * zi


def load_b_topography(case_directory):
    """Load the terrain interface and convert it to meters above the shifted base."""
    phi_path = B_PATH / case_directory / "output" / "phi_functions"
    phi = build_phi(str(phi_path) + "/", Nx, Ny, Nz, mpiProc)

    # Vectorized equivalent of build_intf from Paper1_FigureTopo.py.
    sign_change = phi[:, :, 3:-1] * phi[:, :, 4:] <= 0
    first_crossing = np.argmax(sign_change, axis=2) + 3
    has_crossing = np.any(sign_change, axis=2)
    interface_index = np.where(has_crossing, first_crossing - 1, 0)
    ix, iy = np.indices((Nx, Ny))
    phi_at_interface = phi[ix, iy, interface_index]
    intf = np.where(has_crossing, interface_index * dz - phi_at_interface, 0.0)

    return (intf - z_shift) * zi


def load_g_canopy(case_directory):
    """Load Giulia canopy footprints as a binary areal mask."""
    sfc_path = G_PATH / "input_txt_files" / case_directory / "sfc.npy"
    sfc = np.load(sfc_path)
    return np.where(sfc > 0, 1.0, 0.0)


# %% Plot LES domains

terrain = plt.get_cmap("terrain")
terrain_truncated = LinearSegmentedColormap.from_list(
    "terrain_truncated", terrain(np.linspace(0.25, 1, 100))
)
green_cmap = LinearSegmentedColormap.from_list("green_cmap", ["#ffffff", "#006d2c"])

fig = plt.figure(figsize=(13.5, 8.0))
gs = gridspec.GridSpec(2, 6, figure=fig)

axes_b = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:6]),
]
axes_g = [
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
]

labels = ["a)", "b)", "c)", "d)", "e)"]
topo_levels = np.arange(0, 101, 1)
topo_contour = None

for i, (case_name, case_directory) in enumerate(B_CASES.items()):
    topo = load_b_topography(case_directory)
    topo_contour = axes_b[i].contourf(
        x_b,
        y_b,
        topo.T,
        levels=topo_levels,
        cmap=terrain_truncated,
        extend="max",
    )
    axes_b[i].set_aspect("equal")
    axes_b[i].tick_params(axis="both", which="major", labelsize=12)
    axes_b[i].set_xlabel(r"$x$ [m]", fontsize=16)

for i, (case_name, case_directory) in enumerate(G_CASES.items()):
    canopy = load_g_canopy(case_directory)
    axes_g[i].pcolormesh(
        x_g,
        y_g,
        canopy.T,
        cmap=green_cmap,
        shading="auto",
        vmin=0,
        vmax=1,
    )
    axes_g[i].set_aspect("equal")
    axes_g[i].tick_params(axis="both", which="major", labelsize=12)
    axes_g[i].set_xlabel(r"$x$ [m]", fontsize=16)

axes_b[0].set_ylabel(r"$y$ [m]", fontsize=16)
axes_g[0].set_ylabel(r"$y$ [m]", fontsize=16)

for ax in axes_b[1:] + axes_g[1:]:
    ax.tick_params(labelleft=False)

for i, ax in enumerate(axes_b + axes_g):
    ax.text(
        0.0,
        1.0,
        labels[i],
        transform=ax.transAxes + ScaledTranslation(-4 / 72, +7 / 72, fig.dpi_scale_trans),
        fontsize=15,
        va="bottom",
        fontfamily="serif",
    )

divider = make_axes_locatable(axes_b[-1])
cbar_ax = divider.append_axes("right", size="4%", pad=0.06)
cbar = fig.colorbar(topo_contour, cax=cbar_ax)
cbar.set_label(r"$z$ [m]", fontsize=16, labelpad=8)
cbar.ax.tick_params(labelsize=12)

plt.subplots_adjust(
    left=0.07,
    bottom=0.08,
    right=0.93,
    top=0.93,
    wspace=0.16,
    hspace=0.32,
)

plt.savefig(
    HOME / "Pictures" / "Paper2" / "LESdomains.png",
    dpi=300,
    edgecolor="white",
    facecolor="white",
)
plt.show()

# %%
