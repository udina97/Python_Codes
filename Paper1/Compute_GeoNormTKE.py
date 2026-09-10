#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute reduced TKE budget profiles normalized by Ug.

This follows the profile extraction in Compute_TKEprof.py, but replaces the
u_* normalization with Ug, where Ug is the velocity magnitude at the top of the
domain. TKE budget terms are normalized with canopyH / Ug**3.
"""

import copy
import os
import sys

import numpy as np
import scipy.io
import xarray as xr

FUNCTIONS_DIR = '/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/'
sys.path.insert(0, FUNCTIONS_DIR)
from functions import build_intf, build_phi, find_coordinates


CASES = [
    'Gap_12_9mps',
    'Gap_8_9mps',
    'Gap_4_9mps',
    'Patch_12_9mps',
    'Patch_8_9mps',
    'Patch_4_9mps',
    'ATTO',
    'Sinusoidal',
    'Flat',
    'simulation_G',
]

GIULIA_DATA_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/'
BEN_DATA_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/'
GIOMETTO_DATA_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/'
PROFILE_DIR = '/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/'

NTWR = 100
NZ_SLAYER = 200
RNG = np.random.default_rng()


def save_profile(case_name, suffix, profile):
    suffix_part = f'_{suffix}' if suffix else ''
    out_path = os.path.join(PROFILE_DIR, f'RedTKE_{case_name}{suffix_part}_Ug.npy')
    np.save(out_path, profile)
    print(f'Saved {out_path}')


def require_valid_ug(ug, case_name, label):
    if not np.isfinite(ug) or ug == 0:
        raise ValueError(f'Invalid Ug for {case_name} ({label}): {ug}')
    return ug


def sample_coordinates(coords, n_towers, case_name, label):
    if coords.shape[0] < n_towers:
        raise ValueError(
            f'Not enough {label} coordinates for {case_name}: '
            f'found {coords.shape[0]}, need {n_towers}'
        )
    idx = RNG.choice(coords.shape[0], size=n_towers, replace=False)
    return coords[idx]


def average_selected_profiles(profile_4d, coords):
    profiles = np.array([profile_4d[i, j, :, :] for i, j in coords])
    return np.nanmean(profiles, axis=0).T


def load_case(case_idx):
    case_name = CASES[case_idx]

    if case_idx < 6:
        path_to_data = GIULIA_DATA_DIR
        data = xr.open_dataarray(path_to_data + case_name + '/Data_Momentum_4TKE.nc')
        terms_bdg = xr.open_dataarray(path_to_data + case_name + '/TKE_terms.nc')
    elif case_idx < 9:
        path_to_data = BEN_DATA_DIR
        data = xr.open_dataarray(path_to_data + case_name + '/dataTKE.nc')
        terms_bdg = xr.open_dataarray(path_to_data + case_name + '/TKE_terms.nc')
    else:
        path_to_data = GIOMETTO_DATA_DIR
        data = scipy.io.loadmat(path_to_data + case_name + '/profiles.mat')
        terms_bdg = None

    return path_to_data, data, terms_bdg


def get_simulation_parameters(case_idx, data):
    if case_idx < 6:
        lx = 2 * np.pi
        ly = 2 * np.pi
        lz = 1
        nx, ny, nz = np.shape(data[:, :, :, 0])
        zi = 1000
        canopy_h = 39 / zi
        dx = lx / nx
        dy = ly / ny
        dz = lz / nz
        mpi_proc = None
    elif case_idx < 9:
        lx = 2.88
        ly = 2.88
        lz = 0.96
        mpi_proc = 32
        nx, ny, nz = np.shape(data[:, :, :, 0])
        nz = nz + 5
        zi = 1000
        canopy_h = 39 / zi
        dx = lx / nx
        dy = ly / ny
        dz = lz / nz
    else:
        nx = data['nx'][0][0]
        ny = data['ny'][0][0]
        nz = data['nz'][0][0]
        dx = data['dx'][0][0]
        dy = data['dy'][0][0]
        dz = data['dz'][0][0]
        zi = None
        canopy_h = 15.3
        mpi_proc = None

    return nx, ny, nz, dx, dy, dz, zi, canopy_h, mpi_proc


def compute_giulia_profiles(case_idx, path_to_data, data, terms_bdg):
    case_name = CASES[case_idx]
    _, _, _, _, _, _, _, canopy_h, _ = get_simulation_parameters(case_idx, data)

    sfc = np.load(path_to_data + '../input_txt_files/' + case_name + '/sfc.npy')
    coord_f = sample_coordinates(np.argwhere(sfc == 1.6), NTWR, case_name, 'forested')
    coord_p = sample_coordinates(np.argwhere(sfc == 0), NTWR, case_name, 'patch')

    vel_mag = np.sqrt(data.data[:, :, :, 0]**2 + data.data[:, :, :, 1]**2)
    ug = require_valid_ug(np.nanmean(vel_mag[:, :, -1]), case_name, 'domain top')
    tke_nd = canopy_h / ug**3

    prod = terms_bdg.data[:, :, :NZ_SLAYER, 14] * tke_nd
    diss = terms_bdg.data[:, :, :NZ_SLAYER, 11] * tke_nd
    res = (terms_bdg.data[:, :, :NZ_SLAYER, 14] + terms_bdg.data[:, :, :NZ_SLAYER, 11]) * tke_nd
    tke_terms = np.stack((prod, diss, res), axis=-1)

    save_profile(case_name, 'f', average_selected_profiles(tke_terms, coord_f))
    save_profile(case_name, 'p', average_selected_profiles(tke_terms, coord_p))


def make_ben_dist(path_to_data, case_name, data, terms_bdg, nx, ny, nz, dz, zi, mpi_proc):
    phi = build_phi(path_to_data + case_name + '/phi_functions/', nx, ny, nz, mpi_proc)
    intf, _ = build_intf(phi, dz)

    z_profile = np.arange(nz) * dz * zi
    zeds = np.ones((nx, ny, 1)) * z_profile
    dist = copy.deepcopy(zeds)
    dist -= intf[:, :, np.newaxis] * zi

    mask = dist[:, :, 5:] < 0
    mask_4d = np.expand_dims(mask, axis=-1)
    data.data[:, :, :, :] = np.where(mask_4d, np.nan, data.data[:, :, :, :])
    terms_bdg.data[:, :, :, :] = np.where(mask_4d, np.nan, terms_bdg.data[:, :, :, :])

    return intf, dist


def extract_ben_tke_profiles(terms_bdg, dist, coords, canopy_h, ug):
    idx_p, idx_d = 14, 11
    tke_profiles = np.full((3, coords.shape[0], NZ_SLAYER), np.nan, dtype='float64', order='F')
    tke_nd = canopy_h / ug**3

    for k, (ix, iy) in enumerate(coords):
        z_start = max(np.argmax(dist[ix, iy, :] > 0) - 5, 0)
        z_end = z_start + NZ_SLAYER

        prod = terms_bdg.data[ix, iy, z_start:z_end, idx_p]
        diss = -terms_bdg.data[ix, iy, z_start:z_end, idx_d]
        res = terms_bdg.data[ix, iy, z_start:z_end, idx_p] - terms_bdg.data[ix, iy, z_start:z_end, idx_d]

        profile_len = min(NZ_SLAYER, prod.shape[0])
        tke_profiles[0, k, :profile_len] = prod[:profile_len] * tke_nd
        tke_profiles[1, k, :profile_len] = diss[:profile_len] * tke_nd
        tke_profiles[2, k, :profile_len] = res[:profile_len] * tke_nd

    return np.nanmean(tke_profiles, axis=1)


def compute_ben_profiles(case_idx, path_to_data, data, terms_bdg):
    case_name = CASES[case_idx]
    nx, ny, nz, _, _, dz, zi, canopy_h, mpi_proc = get_simulation_parameters(case_idx, data)
    intf, dist = make_ben_dist(path_to_data, case_name, data, terms_bdg, nx, ny, nz, dz, zi, mpi_proc)

    top_speed = np.sqrt(
        data.data[:, :, -1, 0]**2
        + data.data[:, :, -1, 1]**2
    )
    ug = require_valid_ug(np.nanmean(top_speed), case_name, 'domain top')

    if case_name == 'Flat':
        coord_f = find_coordinates(intf, NTWR, 'flat')
        save_profile(case_name, '', extract_ben_tke_profiles(terms_bdg, dist, coord_f, canopy_h, ug))
        return

    coord_p = find_coordinates(intf, NTWR, 'max')
    coord_v = find_coordinates(intf, NTWR, 'min')
    save_profile(case_name, 'p', extract_ben_tke_profiles(terms_bdg, dist, coord_p, canopy_h, ug))
    save_profile(case_name, 'v', extract_ben_tke_profiles(terms_bdg, dist, coord_v, canopy_h, ug))


def compute_giometto_profiles(case_idx, data):
    case_name = CASES[case_idx]
    _, _, _, _, _, _, _, canopy_h, _ = get_simulation_parameters(case_idx, data)

    u_xy = data['u_xy'].squeeze()
    v_xy = data['v_xy'].squeeze()
    w_xy = data['w_xy'].squeeze()
    u_tw = data['u_tw'].squeeze()
    v_tw = data['v_tw'].squeeze()
    w_tw = data['w_tw'].squeeze()

    vel_xy = np.sqrt(u_xy**2 + v_xy**2)
    vel_tw = np.sqrt(u_tw**2 + v_tw**2)
    ug_xy = require_valid_ug(vel_xy[-1], case_name, 'xy top')
    ug_tw = require_valid_ug(vel_tw[-1], case_name, 'tw top')

    tke_nd_xy = canopy_h / ug_xy**3
    tke_nd_tw = canopy_h / ug_tw**3

    diss_xy = data['ds_xy'].squeeze() * tke_nd_xy
    diss_tw = data['ds_tw'].squeeze() * tke_nd_tw

    prod_xy = (
        data['sp_xy'].squeeze()[8:] * tke_nd_xy
        + data['spd_xy'].squeeze()[8:] * tke_nd_xy
        - data['spm2_xy'].squeeze()[7:] * tke_nd_xy
    )
    prod_tw = data['sp_tw'].squeeze() * tke_nd_tw

    res_xy = diss_xy[8:] + prod_xy
    res_tw = diss_tw[8:] + prod_tw[8:]

    tke_prof_xy = np.zeros((3, 152), order='F')
    tke_prof_tw = np.zeros((3, 152), order='F')

    tke_prof_xy[0, :] = prod_xy
    tke_prof_xy[1, :] = diss_xy[8:]
    tke_prof_xy[2, :] = res_xy

    tke_prof_tw[0, :] = prod_tw[8:]
    tke_prof_tw[1, :] = diss_tw[8:]
    tke_prof_tw[2, :] = res_tw

    save_profile(case_name, 'xy', tke_prof_xy)
    save_profile(case_name, 'tw', tke_prof_tw)


def main():
    for case_idx, case_name in enumerate(CASES):
        print(f'Processing {case_name}')
        path_to_data, data, terms_bdg = load_case(case_idx)

        if case_idx < 6:
            compute_giulia_profiles(case_idx, path_to_data, data, terms_bdg)
        elif case_idx < 9:
            compute_ben_profiles(case_idx, path_to_data, data, terms_bdg)
        else:
            compute_giometto_profiles(case_idx, data)


if __name__ == '__main__':
    main()
