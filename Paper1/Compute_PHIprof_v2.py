#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and save phi_M profiles for all Paper 1 cases.

This v2 script uses the same phi_M and log-fit ustar calculation currently used
in phiM_vs_yB_RSL.py, then saves profiles with the original names plus _v2.
"""

import math
import os
import sys

import numpy as np
import scipy.io
import xarray as xr
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit


FUNCTIONS_DIR = "/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions/"
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

from functions import build_intf, build_phi, compute_d_twr, find_coordinates, wnode2uvpnode


GIULIA_DATA_DIR = "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/TKE_BUDGET_AND_RAV/"
BEN_DATA_DIR = "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/NetCDF_data/"
URBAN_DATA_DIR = "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/urban-profiles-for-Marc-Calaf-08-14-2025/urban-profiles-for-Marc-Calaf-08-14-2025/"
OUTPUT_DIR = "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles/"

GIULIA_CASES = [
    "Gap_12_9mps",
    "Gap_8_9mps",
    "Gap_4_9mps",
    "Patch_12_9mps",
    "Patch_8_9mps",
    "Patch_4_9mps",
]
BEN_CASES = ["ATTO", "Sinusoidal", "Flat"]
URBAN_CASES = ["simulation_G"]

NTWR = 100
NZ_SLAYER = 200

GIULIA_FIT_LEVELS = [80, 90, 100]
BEN_FIT_LEVELS = {
    "ATTO": [160, 170, 180],
    "Sinusoidal": [160, 170, 180],
    "Flat": [100, 110, 120],
}
URBAN_FIT_LEVELS = [100, 110, 120]


def log_fit(z, a, b):
    return a * np.log(b * z)


def compute_ustar_from_profile(nz_slayer, z_d, u, v, fit_levels):
    kappa = 0.4
    fit_levels = np.asarray(fit_levels, dtype=int)

    u_mag = np.sqrt(u**2 + v**2)
    u_mean = u_mag[:nz_slayer]

    z_data = z_d[fit_levels]
    u_data = u_mean[fit_levels]

    coefs, _ = curve_fit(log_fit, z_data, u_data, maxfev=10000)
    _, b = coefs

    z0hi = 1 / b
    ustar = u_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar


def compute_ustar_from_terrain_profile(nz_slayer, z_d, dist, u, v, fit_levels):
    kappa = 0.4
    z_start = np.argmax(dist > 0) - 5
    fit_levels = np.asarray(fit_levels, dtype=int)

    u_mag = np.sqrt(u**2 + v**2)
    u_mean = u_mag[z_start : z_start + nz_slayer]

    z_data = z_d[fit_levels]
    u_data = u_mean[fit_levels]

    coefs, _ = curve_fit(log_fit, z_data, u_data, maxfev=10000)
    _, b = coefs

    z0hi = 1 / b
    ustar = u_mean[fit_levels[0]] / ((1 / kappa) * np.log(z_d[fit_levels[0]] / z0hi))

    return z0hi, ustar


def phi_m_loc(nz_slayer, z_d, u, v, avgdudz, avgdvdz, ustar):
    kappa = 0.4
    u_mag = np.sqrt(u**2 + v**2)
    mean_dudz = (u * avgdudz + v * avgdvdz) / u_mag
    return (kappa * z_d[:nz_slayer] / ustar) * mean_dudz


def compute_d_twr_giulia(data, coord, height, dz, zi, u_scale, lad):
    d_dim = np.zeros(coord.shape[0])
    nlevels = len(lad)

    z = (np.arange(1, height + 1)[:nlevels]) * dz * zi - (dz * zi) / 2
    lad = np.asarray(lad)

    for idx, (i, j) in enumerate(coord):
        u = data.data[i, j, :, 0]
        v = data.data[i, j, :, 1]
        u_mag = np.sqrt(u**2 + v**2)

        u_slice = u_mag[:nlevels]
        z_slice = z[: len(u_slice)]

        var = u_slice * u_scale
        y = (var**2) * 0.4 * lad[: len(u_slice)]

        den = trapezoid(y, z_slice)
        d_dim[idx] = trapezoid(y * z_slice, z_slice) / den if den != 0 else 0.0

    return d_dim


def save_profile(name, profile):
    path = os.path.join(OUTPUT_DIR, f"{name}_v2.npy")
    np.save(path, profile)
    print(f"Saved {path}")


def compute_giulia_case(case_name):
    print(f"Computing {case_name}")

    data = xr.open_dataarray(os.path.join(GIULIA_DATA_DIR, case_name, "Data_Momentum_4TKE.nc"))

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
    uscale = 0.313
    lad = [
        0.2349432,
        0.2715461,
        0.2606477,
        0.2706234,
        0.289694,
        0.2125121,
        0.139206,
        0.063390629,
        0.03817526,
        0.0219344,
    ]

    del dx, dy, canopy_h

    sfc = np.load(os.path.join(GIULIA_DATA_DIR, "..", "input_txt_files", case_name, "sfc.npy"))
    coord_f = np.argwhere(sfc == 1.6)
    coord_p = np.argwhere(sfc == 0)
    sel_coord_f = coord_f[np.random.choice(coord_f.shape[0], size=NTWR, replace=False)]
    sel_coord_p = coord_p[np.random.choice(coord_p.shape[0], size=NTWR, replace=False)]

    height = math.ceil((39 / zi) / dz)
    disp_f = compute_d_twr_giulia(data, sel_coord_f, height, dz, zi, uscale, lad)
    disp_p = np.zeros(NTWR, order="F")

    dudz_uvp = wnode2uvpnode(data.data[:, :, :, 12])
    dvdz_uvp = wnode2uvpnode(data.data[:, :, :, 15])

    def average_phi(coords, disp):
        phi_2d = np.zeros((coords.shape[0], NZ_SLAYER))
        ustar = np.zeros(coords.shape[0], dtype="d", order="F")

        for k, loc in enumerate(coords):
            z_d = z_uvp - disp[k] / zi
            _, ustar[k] = compute_ustar_from_profile(
                NZ_SLAYER,
                z_d,
                data.data[loc[0], loc[1], :, 0],
                data.data[loc[0], loc[1], :, 1],
                GIULIA_FIT_LEVELS,
            )
            phi_2d[k, :] = phi_m_loc(
                NZ_SLAYER,
                z_d,
                data.data[loc[0], loc[1], :NZ_SLAYER, 0],
                data.data[loc[0], loc[1], :NZ_SLAYER, 1],
                dudz_uvp[loc[0], loc[1], :NZ_SLAYER],
                dvdz_uvp[loc[0], loc[1], :NZ_SLAYER],
                ustar[k],
            )

        return np.mean(phi_2d, axis=0)

    save_profile(f"PHI_{case_name}_f", average_phi(sel_coord_f, disp_f))
    save_profile(f"PHI_{case_name}_p", average_phi(sel_coord_p, disp_p))


def make_terrain_distance(intf, nz, dz, zi):
    nx, ny = intf.shape
    z_profile = np.arange(nz) * dz * zi
    dist = np.ones((nx, ny, 1)) * z_profile
    dist -= intf[:, :, np.newaxis] * zi
    return dist


def mask_terrain_data(data, dist):
    mask = dist[:, :, 5:] < 0
    mask4d = np.expand_dims(mask, axis=-1)
    data.data[:, :, :, :] = np.where(mask4d, np.nan, data.data[:, :, :, :])


def compute_ben_case(case_name):
    print(f"Computing {case_name}")

    data = xr.open_dataarray(os.path.join(BEN_DATA_DIR, case_name, "dataTKE.nc"))

    lx = 2.88
    ly = 2.88
    lz = 0.96
    mpi_proc = 32
    nx, ny, nz_data = np.shape(data[:, :, :, 0])
    nz = nz_data + 5
    zi = 1000
    canopy_h = 39 / zi
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    z_uvp = np.arange(0, nz) * dz + dz / 2
    uscale = 0.4
    lad = [
        0.15655190,
        0.20633190,
        0.24492203,
        0.28024144,
        0.33267326,
        0.33145316,
        0.32065714,
        0.28729650,
        0.25240169,
        0.17358901,
        0.11740349,
        0.064294815,
        0.041340224,
        0.023756023,
        0.013134912,
        0.013107183,
    ]

    del dx, dy

    phi = build_phi(os.path.join(BEN_DATA_DIR, case_name, "phi_functions") + "/", nx, ny, nz, mpi_proc)
    intf, _ = build_intf(phi, dz)
    dist = make_terrain_distance(intf, nz, dz, zi)
    mask_terrain_data(data, dist)

    height = math.ceil(canopy_h / dz)
    fit_levels = BEN_FIT_LEVELS[case_name]

    def average_phi(coords, disp):
        phi_2d = np.zeros((coords.shape[0], NZ_SLAYER), dtype="float64", order="F")
        ustar = np.zeros(coords.shape[0], dtype="d", order="F")

        for k, loc in enumerate(coords):
            ix, iy = loc
            z_start = np.argmax(dist[ix, iy, :] > 0) - 5
            z_d = z_uvp - disp[k] / zi
            _, ustar[k] = compute_ustar_from_terrain_profile(
                NZ_SLAYER,
                z_d,
                dist[ix, iy, :],
                data.data[ix, iy, :, 0],
                data.data[ix, iy, :, 1],
                fit_levels,
            )
            phi_2d[k, :] = phi_m_loc(
                NZ_SLAYER,
                z_d,
                data.data[ix, iy, z_start : z_start + NZ_SLAYER, 0],
                data.data[ix, iy, z_start : z_start + NZ_SLAYER, 1],
                data.data[ix, iy, z_start : z_start + NZ_SLAYER, 12],
                data.data[ix, iy, z_start : z_start + NZ_SLAYER, 15],
                ustar[k],
            )

        return np.mean(phi_2d, axis=0)

    if case_name == "Flat":
        coord_f = find_coordinates(intf, NTWR, "flat")
        disp_f = compute_d_twr(data, coord_f, dist, height + 5, dz, zi, uscale, lad)
        save_profile(f"PHI_{case_name}", average_phi(coord_f, disp_f))
        return

    coord_p = find_coordinates(intf, NTWR, "max")
    coord_v = find_coordinates(intf, NTWR, "min")
    disp_p = compute_d_twr(data, coord_p, dist, height + 5, dz, zi, uscale, lad)
    disp_v = compute_d_twr(data, coord_v, dist, height + 5, dz, zi, uscale, lad)

    save_profile(f"PHI_{case_name}_p", average_phi(coord_p, disp_p))
    save_profile(f"PHI_{case_name}_v", average_phi(coord_v, disp_v))


def compute_urban_case(case_name):
    print(f"Computing {case_name}")

    data = scipy.io.loadmat(os.path.join(URBAN_DATA_DIR, case_name, "profiles.mat"))

    nz = data["nz"][0][0]
    z_uvp = data["z"][0]
    canopy_h = 15.3
    nz_slayer = nz - 8
    z_d = z_uvp[:-8] - (2 / 3) * canopy_h

    u_xy = data["u_xy"].squeeze()[8:]
    v_xy = data["v_xy"].squeeze()[8:]
    u_tw = data["u_tw"].squeeze()[8:]
    v_tw = data["v_tw"].squeeze()[8:]

    _, ustar_xy = compute_ustar_from_profile(nz_slayer, z_d, u_xy, v_xy, URBAN_FIT_LEVELS)
    _, ustar_tw = compute_ustar_from_profile(nz_slayer, z_d, u_tw, v_tw, URBAN_FIT_LEVELS)

    phi_xy = phi_m_loc(
        nz_slayer,
        z_d,
        u_xy,
        v_xy,
        data["dudz_xy"].squeeze()[8:],
        data["dvdz_xy"].squeeze()[8:],
        ustar_xy,
    )
    phi_tw = phi_m_loc(
        nz_slayer,
        z_d,
        u_tw,
        v_tw,
        data["dudz_tw"].squeeze()[8:],
        data["dvdz_tw"].squeeze()[8:],
        ustar_tw,
    )

    save_profile(f"PHI_{case_name}_xy", phi_xy)
    save_profile(f"PHI_{case_name}_tw", phi_tw)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for case_name in GIULIA_CASES:
        compute_giulia_case(case_name)

    for case_name in BEN_CASES:
        compute_ben_case(case_name)

    for case_name in URBAN_CASES:
        compute_urban_case(case_name)


if __name__ == "__main__":
    main()
