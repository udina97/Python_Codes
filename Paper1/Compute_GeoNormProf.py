#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute velocity-magnitude and shear profiles normalized by Ug.

Ug is taken as the velocity magnitude at the top of the domain for each case.
Velocity profiles are normalized by Ug and shear-stress-magnitude profiles are
normalized by Ug**2.
"""

import copy
import os
import sys

import numpy as np
import scipy.io
import xarray as xr

FUNCTIONS_DIR = '/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/'
sys.path.insert(0, FUNCTIONS_DIR)
from functions import build_intf, build_phi, find_coordinates, uvpnode2wnode, wnode2uvpnode


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


def save_profile(prefix, case_name, suffix, profile):
    suffix_part = f'_{suffix}' if suffix else ''
    out_path = os.path.join(PROFILE_DIR, f'{prefix}_{case_name}{suffix_part}_Ug.npy')
    np.save(out_path, profile)
    print(f'Saved {out_path}')


def require_valid_ug(ug, case_name, label):
    if not np.isfinite(ug) or ug == 0:
        raise ValueError(f'Invalid Ug for {case_name} ({label}): {ug}')
    return ug


def average_selected_profiles(profile_3d, coords):
    profiles = np.array([profile_3d[i, j, :] for i, j in coords])
    return np.nanmean(profiles, axis=0)


def sample_coordinates(coords, n_towers, case_name, label):
    if coords.shape[0] < n_towers:
        raise ValueError(
            f'Not enough {label} coordinates for {case_name}: '
            f'found {coords.shape[0]}, need {n_towers}'
        )
    idx = RNG.choice(coords.shape[0], size=n_towers, replace=False)
    return coords[idx]


def load_case(case_idx):
    case_name = CASES[case_idx]

    if case_idx < 6:
        path_to_data = GIULIA_DATA_DIR
        data = xr.open_dataarray(path_to_data + case_name + '/Data_Momentum_4TKE.nc')
    elif case_idx < 9:
        path_to_data = BEN_DATA_DIR
        data = xr.open_dataarray(path_to_data + case_name + '/dataTKE.nc')
    else:
        path_to_data = GIOMETTO_DATA_DIR
        data = scipy.io.loadmat(path_to_data + case_name + '/profiles.mat')

    return path_to_data, data


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
        z_uvp = np.arange(0, nz) * dz + dz / 2
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
        z_uvp = np.arange(0, nz) * dz + dz / 2
        return nx, ny, nz, dx, dy, dz, zi, canopy_h, z_uvp, mpi_proc
    else:
        lx = data['lx'][0][0]
        ly = data['ly'][0][0]
        lz = data['lz'][0][0]
        nx = data['nx'][0][0]
        ny = data['ny'][0][0]
        nz = data['nz'][0][0]
        dx = data['dx'][0][0]
        dy = data['dy'][0][0]
        dz = data['dz'][0][0]
        z_uvp = data['z'][0]
        zi = None
        canopy_h = 15.3

    return nx, ny, nz, dx, dy, dz, zi, canopy_h, z_uvp, None


def compute_giulia_profiles(case_idx, path_to_data, data):
    case_name = CASES[case_idx]
    sfc = np.load(path_to_data + '../input_txt_files/' + case_name + '/sfc.npy')

    coord_f = sample_coordinates(np.argwhere(sfc == 1.6), NTWR, case_name, 'forested')
    coord_p = sample_coordinates(np.argwhere(sfc == 0), NTWR, case_name, 'patch')

    u = data.data[:, :, :, 0]
    v = data.data[:, :, :, 1]
    vel_mag = np.sqrt(u**2 + v**2)
    ug = require_valid_ug(np.nanmean(vel_mag[:, :, -1]), case_name, 'domain top')

    t_13 = wnode2uvpnode(
        data.data[:, :, :, 8]
        - uvpnode2wnode(data.data[:, :, :, 0]) * data.data[:, :, :, 2]
        - data.data[:, :, :, 23]
    )
    t_23 = wnode2uvpnode(
        data.data[:, :, :, 9]
        - uvpnode2wnode(data.data[:, :, :, 1]) * data.data[:, :, :, 2]
        - data.data[:, :, :, 24]
    )
    shear = np.sqrt(t_13**2 + t_23**2)

    vel_norm = vel_mag[:, :, :NZ_SLAYER] / ug
    shear_norm = shear[:, :, :NZ_SLAYER] / ug**2

    save_profile('U', case_name, 'f', average_selected_profiles(vel_norm, coord_f))
    save_profile('U', case_name, 'p', average_selected_profiles(vel_norm, coord_p))
    save_profile('Shear', case_name, 'f', average_selected_profiles(shear_norm, coord_f))
    save_profile('Shear', case_name, 'p', average_selected_profiles(shear_norm, coord_p))


def make_ben_dist(path_to_data, case_name, data, nx, ny, nz, dz, zi, mpi_proc):
    phi = build_phi(path_to_data + case_name + '/phi_functions/', nx, ny, nz, mpi_proc)
    intf, _ = build_intf(phi, dz)

    z_profile = np.arange(nz) * dz * zi
    zeds = np.ones((nx, ny, 1)) * z_profile
    dist = copy.deepcopy(zeds)
    dist -= intf[:, :, np.newaxis] * zi

    mask = dist[:, :, 5:] < 0
    mask_4d = np.expand_dims(mask, axis=-1)
    data.data[:, :, :, :] = np.where(mask_4d, np.nan, data.data[:, :, :, :])

    return intf, dist


def extract_ben_profiles(data, dist, coords, ug):
    idx_u, idx_v, idx_w = 0, 1, 2
    idx_uw, idx_vw, idx_txz, idx_tyz = 8, 9, 23, 24

    vel_profiles = np.full((coords.shape[0], NZ_SLAYER), np.nan, dtype='float64', order='F')
    shear_profiles = np.full((coords.shape[0], NZ_SLAYER), np.nan, dtype='float64', order='F')

    for k, (ix, iy) in enumerate(coords):
        z_start = max(np.argmax(dist[ix, iy, :] > 0) - 5, 0)
        z_end = z_start + NZ_SLAYER

        u = data.data[ix, iy, z_start:z_end, idx_u]
        v = data.data[ix, iy, z_start:z_end, idx_v]
        w = data.data[ix, iy, z_start:z_end, idx_w]

        uw = (
            data.data[ix, iy, z_start:z_end, idx_uw]
            - data.data[ix, iy, z_start:z_end, idx_u] * data.data[ix, iy, z_start:z_end, idx_w]
            - data.data[ix, iy, z_start:z_end, idx_txz]
        )
        vw = (
            data.data[ix, iy, z_start:z_end, idx_vw]
            - data.data[ix, iy, z_start:z_end, idx_v] * data.data[ix, iy, z_start:z_end, idx_w]
            - data.data[ix, iy, z_start:z_end, idx_tyz]
        )

        profile_len = min(NZ_SLAYER, u.shape[0])
        vel_profiles[k, :profile_len] = np.sqrt(u[:profile_len]**2 + v[:profile_len]**2) / ug
        shear_profiles[k, :profile_len] = np.sqrt(uw[:profile_len]**2 + vw[:profile_len]**2) / ug**2

    return np.nanmean(vel_profiles, axis=0), np.nanmean(shear_profiles, axis=0)


def compute_ben_profiles(case_idx, path_to_data, data):
    case_name = CASES[case_idx]
    nx, ny, nz, _, _, dz, zi, _, _, mpi_proc = get_simulation_parameters(case_idx, data)
    intf, dist = make_ben_dist(path_to_data, case_name, data, nx, ny, nz, dz, zi, mpi_proc)

    top_speed = np.sqrt(
        data.data[:, :, -1, 0]**2
        + data.data[:, :, -1, 1]**2
        # + data.data[:, :, -1, 2]**2
    )
    ug = require_valid_ug(np.nanmean(top_speed), case_name, 'domain top')

    if case_name == 'Flat':
        coord_f = find_coordinates(intf, NTWR, 'flat')
        vel_f, shear_f = extract_ben_profiles(data, dist, coord_f, ug)
        save_profile('U', case_name, '', vel_f)
        save_profile('Shear', case_name, '', shear_f)
        return

    coord_p = find_coordinates(intf, NTWR, 'max')
    coord_v = find_coordinates(intf, NTWR, 'min')
    vel_p, shear_p = extract_ben_profiles(data, dist, coord_p, ug)
    vel_v, shear_v = extract_ben_profiles(data, dist, coord_v, ug)

    save_profile('U', case_name, 'p', vel_p)
    save_profile('U', case_name, 'v', vel_v)
    save_profile('Shear', case_name, 'p', shear_p)
    save_profile('Shear', case_name, 'v', shear_v)


def compute_giometto_profiles(case_idx, data):
    case_name = CASES[case_idx]

    u_xy = data['u_xy'].squeeze()
    v_xy = data['v_xy'].squeeze()
    # w_xy = data['w_xy'].squeeze()
    u_tw = data['u_tw'].squeeze()
    v_tw = data['v_tw'].squeeze()
    # w_tw = data['w_tw'].squeeze()

    vel_xy = np.sqrt(u_xy**2 + v_xy**2)
    vel_tw = np.sqrt(u_tw**2 + v_tw**2)
    ug_xy = require_valid_ug(vel_xy[-1], case_name, 'xy top')
    ug_tw = require_valid_ug(vel_tw[-1], case_name, 'tw top')

    r13_xy = data['uw_xy'].squeeze()
    r13_tw = data['uw_tw'].squeeze()
    r23_xy = data['vw_xy'].squeeze()
    r23_tw = data['vw_tw'].squeeze()
    r13d = r13_xy + data['uwd_xy'].squeeze()
    r23d = r23_xy + data['vwd_xy'].squeeze()

    shear_xy = np.sqrt(
        (r13d + data['txz_xy'].squeeze())**2
        + (r23d + data['tyz_xy'].squeeze())**2
    )
    shear_tw = np.sqrt(
        (r13_tw + data['txz_tw'].squeeze())**2
        + (r23_tw + data['tyz_tw'].squeeze())**2
    )

    save_profile('U', case_name, 'xy', vel_xy / ug_xy)
    save_profile('U', case_name, 'tw', vel_tw / ug_tw)
    save_profile('Shear', case_name, 'xy', shear_xy / ug_xy**2)
    save_profile('Shear', case_name, 'tw', shear_tw / ug_tw**2)


def main():
    for case_idx, case_name in enumerate(CASES):
        print(f'Processing {case_name}')
        path_to_data, data = load_case(case_idx)

        if case_idx < 6:
            compute_giulia_profiles(case_idx, path_to_data, data)
        elif case_idx < 9:
            compute_ben_profiles(case_idx, path_to_data, data)
        else:
            compute_giometto_profiles(case_idx, data)


if __name__ == '__main__':
    main()
