#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite residual/yB pcolor figure with residual normalized by Ug.

This is based on Paper1_Figure3x3_ResYbComb.py. The figure layout and plotting
settings are preserved, but the residual normalization uses the horizontally
averaged velocity magnitude at the top of the domain:

    Ug = <sqrt(u^2 + v^2)>_{x,y, top}
"""

import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.transforms import ScaledTranslation

FUNCTIONS_DIR = '/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/'
sys.path.insert(0, FUNCTIONS_DIR)
from Anisotropy_Functions import ColorAnisotropy
from functions import build_intf, build_phi


PATH_FIG = '/uufs/chpc.utah.edu/common/home/u1450851/Pictures/Paper1/'
SIMS_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Sims/'
ANISO_OUT_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/'
NETCDF_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
GIULIA_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'

SIM_NAMES = {
    'flat': 'simflat_256x256x384_out5hr_v2',
    'sin': 'simbicheng_hill_256x256x384_out5hr',
    'atto': 'simATTO_256x256x384_full_out5hr',
}

TOPO_CASES = ['flat', 'sin', 'atto']
NETCDF_CASES = {'flat': 'Flat', 'sin': 'Sinusoidal', 'atto': 'ATTO'}
YSLICES_TOPO = {'flat': 100, 'sin': 150, 'atto': 63}
GIULIA_CASES = ['Gap_8_9mps', 'Patch_8_9mps']


def read_parameters(param_path):
    with open(param_path, 'r') as param_file:
        param = [float(line.strip()) for line in param_file.readlines()]

    return {
        'Nx': int(param[0]),
        'Ny': int(param[1]),
        'Nz': int(param[2]),
        'Lx': param[3],
        'Ly': param[4],
        'Lz': param[5],
        'dx': param[6],
        'dy': param[7],
        'dz': param[8],
        'mpiProc': int(param[11]),
    }


def velocity_scale_from_netcdf(case_name):
    data = xr.open_dataarray(os.path.join(NETCDF_DIR, case_name, 'dataTKE.nc'))
    vel_top = np.sqrt(data.data[:, :, -1, 0]**2 + data.data[:, :, -1, 1]**2)
    ug = np.nanmean(vel_top)
    if not np.isfinite(ug) or ug == 0:
        raise ValueError(f'Invalid Ug for {case_name}: {ug}')
    return ug


def velocity_scale_from_giulia(case_name, data):
    vel_top = np.sqrt(data.data[:, :, -1, 0]**2 + data.data[:, :, -1, 1]**2)
    ug = np.nanmean(vel_top)
    if not np.isfinite(ug) or ug == 0:
        raise ValueError(f'Invalid Ug for {case_name}: {ug}')
    return ug


def build_interface_and_distance(phi_dir, nx, ny, nz, dz, zi, mpi_proc):
    phi = build_phi(phi_dir, nx, ny, nz, mpi_proc)
    intf, iintf = build_intf(phi, dz)
    z_m = np.arange(nz) * dz * zi
    dist = z_m[np.newaxis, np.newaxis, :] - intf[:, :, np.newaxis] * zi
    return intf, iintf, dist


def load_topography_context(params, zi):
    nx = params['Nx']
    ny = params['Ny']
    nz = params['Nz']
    dz = params['dz']
    mpi_proc = params['mpiProc']

    intf = {}
    dist = {}
    for case_key, sim_name in SIM_NAMES.items():
        phi_dir = os.path.join(SIMS_DIR, sim_name, 'output', 'phi_functions') + '/'
        intf[case_key], _, dist[case_key] = build_interface_and_distance(
            phi_dir, nx, ny, nz, dz, zi, mpi_proc
        )

    return intf, dist


def load_anisotropy(params, dist):
    nx = params['Nx']
    ny = params['Ny']
    nz = params['Nz']
    aniso = {}
    aniso_var = ['xB', 'yB', 'type']

    for case_key, sim_name in SIM_NAMES.items():
        path = os.path.join(ANISO_OUT_DIR, sim_name, f'Anisotropy_clustering_{sim_name}_v2.nc')
        aniso_data = xr.open_dataarray(path)
        case_aniso = {
            var: np.reshape(np.copy(aniso_data.data[:, idx]), (nx, ny, nz))
            for idx, var in enumerate(aniso_var)
        }
        for var in aniso_var:
            case_aniso[var][dist[case_key] < 0] = np.nan
        aniso[case_key] = case_aniso

    return aniso


def load_tke_terms():
    return {
        case_key: np.load(
            os.path.join(ANISO_OUT_DIR, sim_name, 'TKE_terms_v2.npy'),
            allow_pickle=True,
        ).item()
        for case_key, sim_name in SIM_NAMES.items()
    }


def plot_topography_rows(axs, params, intf, dist, aniso, tke, ug, cmap, pcolor_settings):
    nx = params['Nx']
    nz = params['Nz']
    dx = params['dx']
    dz = params['dz']
    zi = 1000.0
    z_shift = 4.5 * dz
    h_canopy = 39 / zi
    x = np.arange(0, nx) * dx
    z = np.arange(0, nz - 5) * dz / h_canopy

    levels = pcolor_settings['levels']
    levels_2 = pcolor_settings['levels_2']
    levels_3 = pcolor_settings['levels_3']
    levels_yb = pcolor_settings['levels_yb']
    colors = pcolor_settings['colors']

    p1 = None
    p2 = None
    sc2 = None

    for row, case_key in enumerate(TOPO_CASES):
        yslice = YSLICES_TOPO[case_key]
        canopy_top = (intf[case_key][:, yslice] - z_shift + h_canopy) / h_canopy
        ground = (intf[case_key][:, yslice] - z_shift) / h_canopy

        tmp_res = (tke[case_key]['prod'] - tke[case_key]['totdis'])[:, :, 5:] * h_canopy / ug[case_key]**3
        tmp_res = tmp_res.copy()
        tmp_res[dist[case_key][:, :, 5:] < 0] = np.nan

        p1 = axs[row][0].contourf(
            x, z, tmp_res[:, yslice, :].T, cmap='bwr', levels=levels_3, extend='both'
        )
        p1.cmap.set_under('blue')
        p1.cmap.set_over('red')
        axs[row][0].plot(x, canopy_top, ls='--', c='k')
        axs[row][0].plot(x, ground, ls='-', c='k')

        axs[row][1].plot(x, canopy_top, ls='--', c='k')
        axs[row][1].plot(x, ground, ls='-', c='k')
        axs[row][1].contour(
            x, z, aniso[case_key]['yB'][:, yslice, 5:].T, levels=levels_2, colors=['black']
        )
        sc2 = axs[row][1].contourf(
            x,
            z,
            aniso[case_key]['yB'][:, yslice, 5:].T,
            levels=levels_yb,
            cmap=cmap,
            vmin=0,
            vmax=np.sqrt(3) / 2,
        )

        raw_res = (tke[case_key]['prod'] - tke[case_key]['totdis'])[:, :, 5:].copy()
        tmp_dis = tke[case_key]['totdis'][:, :, 5:].copy()
        raw_res[np.abs(raw_res) < 1] = 0
        tmp_norm = raw_res / np.abs(tmp_dis)

        p2 = axs[row][2].contourf(
            x,
            z,
            (tmp_norm[:, yslice, :] * 100).T,
            colors=colors,
            alpha=0.5,
            levels=levels,
            extend='both',
        )
        p2.cmap.set_under('blue')
        p2.cmap.set_over('red')
        axs[row][2].contour(
            x, z, aniso[case_key]['yB'][:, yslice, 5:].T, levels=levels_2, colors=['black']
        )
        axs[row][2].plot(x, canopy_top, ls='--', c='k')
        axs[row][2].plot(x, ground, ls='-', c='k')

    return p1, sc2, p2


def plot_giulia_rows(axs, cmap, pcolor_settings):
    nx = 256
    nz = 256
    lx = 2 * np.pi
    lz = 1
    dx = lx / nx
    dz = lz / nz
    x = np.arange(0, nx) * dx
    z_uvp = np.arange(0, nz) * dz + dz / 2
    zi = 1000
    h_canopy = 39 / zi
    yslice = 190

    levels = pcolor_settings['levels']
    levels_2 = pcolor_settings['levels_2']
    levels_3 = pcolor_settings['levels_3']
    levels_yb = pcolor_settings['levels_yb']
    colors = pcolor_settings['colors']

    p1 = None
    p2 = None
    sc2 = None

    for idx, case_name in enumerate(GIULIA_CASES):
        row = idx + 4
        data = xr.open_dataarray(os.path.join(GIULIA_DIR, case_name, 'Data_Momentum_4TKE.nc'))
        terms_bdg = xr.open_dataarray(os.path.join(GIULIA_DIR, case_name, 'TKE_terms.nc'))
        anisotropy = xr.open_dataarray(os.path.join(GIULIA_DIR, case_name, 'anisotropy.nc'))

        ug = velocity_scale_from_giulia(case_name, data)
        res = terms_bdg.data[:, :, :, -1] + terms_bdg.data[:, :, :, 11]
        res_norm = res * h_canopy / ug**3

        p1 = axs[row][0].contourf(
            x, z_uvp / h_canopy, res_norm[:, yslice, :].T, cmap='bwr', levels=levels_3, extend='both'
        )
        p1.cmap.set_under('blue')
        p1.cmap.set_over('red')
        axs[row][0].axhline(0, ls='-', color='k')
        axs[row][0].axhline(1, ls='--', color='k')

        axs[row][1].axhline(0, ls='-', color='k')
        axs[row][1].axhline(1, ls='--', color='k')
        axs[row][1].contour(
            x, z_uvp / h_canopy, anisotropy[:, yslice, :, 1].T, levels=levels_2, colors=['black']
        )
        sc2 = axs[row][1].contourf(
            x,
            z_uvp / h_canopy,
            anisotropy[:, yslice, :, 1].T,
            cmap=cmap,
            levels=levels_yb,
            vmin=0,
            vmax=np.sqrt(3) / 2,
        )

        tmp_dis = terms_bdg.data[:, :, :, 11].copy()
        tmp_res = (terms_bdg.data[:, :, :, -1] + terms_bdg.data[:, :, :, 11]).copy()
        tmp_res = np.where(np.abs(res) < 20, 0, tmp_res)

        with np.errstate(divide='ignore', invalid='ignore'):
            tmp_norm = np.where(tmp_dis != 0, tmp_res / np.abs(tmp_dis) * 100, np.nan)

        p2 = axs[row][2].contourf(
            x,
            z_uvp / h_canopy,
            tmp_norm[:, yslice, :].T,
            colors=colors,
            alpha=0.5,
            levels=levels,
            extend='both',
        )
        p2.cmap.set_under('blue')
        p2.cmap.set_over('red')
        axs[row][2].contour(
            x, z_uvp / h_canopy, anisotropy[:, yslice, :, 1].T, levels=levels_2, colors=['black']
        )
        axs[row][2].axhline(0, ls='-', color='k')
        axs[row][2].axhline(1, ls='--', color='k')

    return p1, sc2, p2


def make_axes():
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 1, 0.1, 1, 1], figure=fig)
    axs = [[None] * 3 for _ in range(6)]

    for row in [0, 1, 2, 4, 5]:
        for col in range(3):
            axs[row][col] = fig.add_subplot(gs[row, col], sharey=axs[row][0] if col > 0 else None)

    return fig, axs


def format_figure(fig, axs, p1, sc2, p2):
    labels = [
        'a1)', 'a2)', 'a3)', 'b1)', 'b2)', 'b3)', 'c1)', 'c2)', 'c3)',
        'd1)', 'd2)', 'd3)', 'e1)', 'e2)', 'e3)'
    ]

    cbar_ax = fig.add_axes([0.083, 0.07, 0.28, 0.01])
    cbar1 = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
    cbar1.set_label(r'$R\cdot\frac{h_C}{U_g^{3}}$ ', fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')
    cbar1.ax.tick_params(labelsize=9)
    cbar1.set_ticks([-0.003, -0.002, -0.001, 0.001, 0.002, 0.003])
    cbar1.set_ticklabels(['-0.003', '-0.002', '-0.001', '0.001', '0.002', '0.003'])

    cbar_ax = fig.add_axes([0.39, 0.07, 0.28, 0.01])
    cbar1 = fig.colorbar(sc2, cax=cbar_ax, orientation='horizontal')
    cbar1.set_label(r'$yB$', fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')
    cbar1.ax.tick_params(labelsize=9)

    cbar_ax = fig.add_axes([0.70, 0.07, 0.28, 0.01])
    cbar1 = fig.colorbar(p2, cax=cbar_ax, orientation='horizontal')
    cbar1.set_label(r'$\frac{P-\varepsilon}{|\varepsilon|}$', fontsize=14, labelpad=5, rotation=360, rotation_mode='anchor')
    cbar1.ax.tick_params(labelsize=9)

    for row in range(3):
        for col in range(1, 3):
            axs[row][col].tick_params(labelleft=False)
            axs[row][col].set_ylabel('')

    for row in range(4, 6):
        for col in range(1, 3):
            axs[row][col].tick_params(labelleft=False)
            axs[row][col].set_ylabel('')

    for col in range(3):
        axs[0][col].tick_params(labelbottom=False)
        axs[1][col].tick_params(labelbottom=False)
        axs[4][col].tick_params(labelbottom=False)
        axs[0][col].set_ylim(0, 20)
        axs[1][col].set_ylim(0, 20)
        axs[2][col].set_ylim(0, 20)
        axs[4][col].set_ylim(0, 20)
        axs[5][col].set_ylim(0, 20)

    for row in range(3):
        axs[row][0].set_ylabel(r'$z/h_C$', fontsize=14)
    for row in range(4, 6):
        axs[row][0].set_ylabel(r'$z/h_C$', fontsize=14)

    for col in range(3):
        axs[2][col].set_xlabel(r'$x/z_i$', fontsize=14)
        axs[5][col].set_xlabel(r'$x/z_i$', fontsize=14)

    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.96, wspace=0.08, hspace=0.4)

    label_idx = 0
    for row in range(3):
        for col in range(3):
            axs[row][col].text(
                0.0,
                1.0,
                labels[label_idx],
                transform=axs[row][col].transAxes + ScaledTranslation(-5 / 72, +5 / 72, fig.dpi_scale_trans),
                fontsize=12,
                va='bottom',
                ha='left',
                fontfamily='serif',
            )
            label_idx += 1

    for row in range(4, 6):
        for col in range(3):
            axs[row][col].text(
                0.0,
                1.0,
                labels[label_idx],
                transform=axs[row][col].transAxes + ScaledTranslation(-5 / 72, +5 / 72, fig.dpi_scale_trans),
                fontsize=12,
                va='bottom',
                ha='left',
                fontfamily='serif',
            )
            label_idx += 1


def main():
    zi = 1000.0
    flat_param_path = os.path.join(SIMS_DIR, SIM_NAMES['flat'], 'output', 'ta1_field', 'parameters.txt')
    params = read_parameters(flat_param_path)

    cmap = ColorAnisotropy()
    pcolor_settings = {
        'levels': [-100, -1, 1, 100],
        'levels_2': [0.38],
        'levels_3': [-0.001, -0.0005, -0.0001, 0.0001, 0.0005, 0.001],
        'levels_yb': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'colors': ['blue', 'white', 'red'],
    }

    intf, dist = load_topography_context(params, zi)
    aniso = load_anisotropy(params, dist)
    tke = load_tke_terms()
    ug = {
        case_key: velocity_scale_from_netcdf(netcdf_case)
        for case_key, netcdf_case in NETCDF_CASES.items()
    }

    fig, axs = make_axes()
    p1, sc2, p2 = plot_topography_rows(axs, params, intf, dist, aniso, tke, ug, cmap, pcolor_settings)
    p1, sc2, p2 = plot_giulia_rows(axs, cmap, pcolor_settings)

    format_figure(fig, axs, p1, sc2, p2)

    # plt.savefig(PATH_FIG + 'ResYB_Combo_All_Ug.png', dpi=300, facecolor='None', edgecolor='None')
    plt.show()


if __name__ == '__main__':
    main()
