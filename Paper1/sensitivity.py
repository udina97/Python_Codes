#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity of tower-averaged vertical profiles to the number of virtual towers.

The heterogeneous canopy cases are the Giulia gap/patch simulations used in
Paper1:

    g1200 -> Gap_12_9mps      i1200 -> Patch_12_9mps
    g800  -> Gap_8_9mps       i800  -> Patch_8_9mps
    g400  -> Gap_4_9mps       i400  -> Patch_4_9mps

For each case and canopy class, the script computes profiles with 1, 10, 100,
and 1000 virtual towers. The 1000-tower mean is used as the convergence
reference, and the convergence curve is the RMS profile error relative to that
reference.
"""

from __future__ import print_function

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np


PAPER1_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/"
    "GiuliaData/TKE_BUDGET_AND_RAV"
)
DEFAULT_INPUT_ROOT = Path(
    "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/"
    "GiuliaData/input_txt_files"
)
DEFAULT_OUTPUT_DIR = PAPER1_DIR / "sensitivity_outputs"

CASE_ALIASES = OrderedDict(
    [
        ("g1200", "Gap_12_9mps"),
        ("g800", "Gap_8_9mps"),
        ("g400", "Gap_4_9mps"),
        ("i1200", "Patch_12_9mps"),
        ("i800", "Patch_8_9mps"),
        ("i400", "Patch_4_9mps"),
    ]
)
CASE_LABELS = {value: key for key, value in CASE_ALIASES.items()}

PROFILE_ALIASES = OrderedDict(
    [
        ("u", "U"),
        ("shear", "Shear"),
        ("phi", "PHI"),
        ("skew", "Skew"),
    ]
)

SURFACE_VALUES = OrderedDict([("forested", 1.6), ("patches", 0.0)])

ZI = 1000.0
CANOPY_H = 39.0 / ZI
U_SCALE = 0.313
LAD = np.array(
    [
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
    ],
    dtype=float,
)
KAPPA = 0.4
NZ_SLAYER = 200


def wnode_to_uvp_column(phi_h):
    phi_h = np.asarray(phi_h, dtype=float)
    phi_c = np.zeros(phi_h.shape, dtype=float)
    phi_c[:-1] = 0.5 * (phi_h[:-1] + phi_h[1:])
    phi_c[-1] = phi_h[-1]
    return phi_c


def uvp_to_wnode_column(phi_c):
    phi_c = np.asarray(phi_c, dtype=float)
    phi_h = np.zeros(phi_c.shape, dtype=float)
    phi_h[0] = 0.0
    phi_h[1:] = 0.5 * (phi_c[:-1] + phi_c[1:])
    return phi_h


def log_fit(z, a, b):
    return a * np.log(b * z)


def compute_ustar(nz_slayer, z_d, u, v):
    from scipy.optimize import curve_fit

    fit_levels = np.array([50, 60, 70, 90], dtype=int)
    if np.max(fit_levels) >= nz_slayer:
        return np.nan

    u_mag = np.sqrt(u ** 2 + v ** 2)
    z_data = np.asarray(z_d[fit_levels], dtype=float)
    u_data = np.asarray(u_mag[:nz_slayer][fit_levels], dtype=float)
    valid = np.isfinite(z_data) & np.isfinite(u_data) & (z_data > 0.0)
    if np.count_nonzero(valid) < 3:
        return np.nan

    try:
        coefs, _ = curve_fit(log_fit, z_data[valid], u_data[valid], maxfev=10000)
    except Exception:
        return np.nan

    b = coefs[1]
    if not np.isfinite(b) or b <= 0.0:
        return np.nan

    z0hi = 1.0 / b
    denom = (1.0 / KAPPA) * np.log(z_d[fit_levels[0]] / z0hi)
    if not np.isfinite(denom) or denom == 0.0:
        return np.nan

    return u_mag[fit_levels[0]] / denom


def column(data, ix, iy, var_idx):
    return np.asarray(data[ix, iy, :, var_idx], dtype=float)


def compute_displacement_heights(data, coords, dz, zi, u_scale, lad):
    height = int(math.ceil(CANOPY_H / dz))
    nlevels = min(height, len(lad))
    z = np.arange(1, nlevels + 1, dtype=float) * dz * zi - 0.5 * dz * zi
    lad_use = np.asarray(lad[:nlevels], dtype=float)
    d_dim = np.zeros(coords.shape[0], dtype=float)

    for idx, (ix, iy) in enumerate(coords):
        u = column(data, ix, iy, 0)[:nlevels]
        v = column(data, ix, iy, 1)[:nlevels]
        u_mag = np.sqrt(u ** 2 + v ** 2)
        y = (u_mag * u_scale) ** 2 * KAPPA * lad_use[: len(u_mag)]
        z_use = z[: len(u_mag)]
        den = np.trapz(y, z_use)
        d_dim[idx] = np.trapz(y * z_use, z_use) / den if den != 0.0 else 0.0

    return d_dim


def compute_tower_profile_stack(data, coords, surface_name, profiles, z_uvp, dz):
    nz_slayer = min(NZ_SLAYER, len(z_uvp))
    stacks = {}
    for profile in profiles:
        stacks[profile] = np.full((coords.shape[0], nz_slayer), np.nan, dtype=float)

    if surface_name == "forested":
        displacement = compute_displacement_heights(data, coords, dz, ZI, U_SCALE, LAD)
    else:
        displacement = np.zeros(coords.shape[0], dtype=float)

    for idx, (ix, iy) in enumerate(coords):
        u = column(data, ix, iy, 0)
        v = column(data, ix, iy, 1)
        w = column(data, ix, iy, 2)
        z_d = z_uvp - displacement[idx] / ZI
        ustar = compute_ustar(nz_slayer, z_d, u, v)

        if not np.isfinite(ustar) or ustar == 0.0:
            continue

        if "U" in stacks:
            w_uvp = wnode_to_uvp_column(w)
            u_mag = np.sqrt(u ** 2 + v ** 2 + w_uvp ** 2)
            stacks["U"][idx, :] = u_mag[:nz_slayer] / ustar

        if "Shear" in stacks:
            uw = column(data, ix, iy, 8) - uvp_to_wnode_column(u) * w - column(data, ix, iy, 23)
            vw = column(data, ix, iy, 9) - uvp_to_wnode_column(v) * w - column(data, ix, iy, 24)
            t13 = wnode_to_uvp_column(uw)
            t23 = wnode_to_uvp_column(vw)
            stacks["Shear"][idx, :] = np.sqrt(t13[:nz_slayer] ** 2 + t23[:nz_slayer] ** 2) / (
                ustar ** 2
            )

        if "PHI" in stacks:
            dudz = wnode_to_uvp_column(column(data, ix, iy, 12))
            dvdz = wnode_to_uvp_column(column(data, ix, iy, 15))
            uv_mag = np.sqrt(u ** 2 + v ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_dudz = (u * dudz + v * dvdz) / uv_mag
                stacks["PHI"][idx, :] = (KAPPA * z_d[:nz_slayer] / ustar) * mean_dudz[:nz_slayer]

        if "Skew" in stacks:
            w_second_moment = column(data, ix, iy, 6)
            w2 = wnode_to_uvp_column(w_second_moment - w ** 2)
            w3 = wnode_to_uvp_column(column(data, ix, iy, 33) - 3.0 * w * w_second_moment + 2.0 * w ** 3)
            with np.errstate(divide="ignore", invalid="ignore"):
                stacks["Skew"][idx, :] = w3[:nz_slayer] / (w2[:nz_slayer] ** 1.5)

    return stacks


def rms_error(profile, reference):
    finite = np.isfinite(profile) & np.isfinite(reference)
    if not np.any(finite):
        return np.nan, np.nan, np.nan
    diff = profile[finite] - reference[finite]
    rms = np.sqrt(np.mean(diff ** 2))
    denom = np.sqrt(np.mean(reference[finite] ** 2))
    rel = rms / denom if denom != 0.0 else np.nan
    max_abs = np.max(np.abs(diff))
    return rms, rel, max_abs


def normalize_cases(cases):
    normalized = []
    for case in cases:
        key = case.lower()
        if key in CASE_ALIASES:
            normalized.append(CASE_ALIASES[key])
        elif case in CASE_LABELS:
            normalized.append(case)
        else:
            valid = ", ".join(list(CASE_ALIASES.keys()) + list(CASE_LABELS.keys()))
            raise ValueError("Unknown case '{}'. Valid cases: {}".format(case, valid))
    return normalized


def normalize_profiles(profile_args):
    lowered = [p.lower() for p in profile_args]
    if "all" in lowered:
        return list(PROFILE_ALIASES.values())
    profiles = []
    for profile in lowered:
        if profile not in PROFILE_ALIASES:
            raise ValueError("Unknown profile '{}'. Valid profiles: all, {}".format(profile, ", ".join(PROFILE_ALIASES)))
        profiles.append(PROFILE_ALIASES[profile])
    return list(OrderedDict((p, None) for p in profiles).keys())


def load_case_data(case_name, data_root, load_data, engine):
    try:
        import xarray as xr
    except ImportError:
        raise SystemExit(
            "xarray is required to read the raw NetCDF files. Activate the "
            "same Python environment used for the other Paper1 scripts."
        )

    data_path = data_root / case_name / "Data_Momentum_4TKE.nc"
    if not data_path.exists():
        raise FileNotFoundError("Missing data file: {}".format(data_path))

    open_kwargs = {}
    if engine:
        open_kwargs["engine"] = engine

    try:
        data_array = xr.open_dataarray(str(data_path), **open_kwargs)
    except ValueError as exc:
        raise SystemExit(
            "Could not open '{}'. xarray found the file, but the needed NetCDF "
            "backend is not available in this Python environment. Activate an "
            "environment with netCDF4 or h5netcdf installed, or pass --engine "
            "with the backend used by that environment.\n\n{}".format(data_path, exc)
        )

    if load_data:
        data_array = data_array.load()

    return data_array.data, data_array.shape


def load_surface_coords(case_name, input_root):
    sfc_path = input_root / case_name / "sfc.npy"
    if not sfc_path.exists():
        raise FileNotFoundError("Missing surface mask: {}".format(sfc_path))

    sfc = np.load(str(sfc_path))
    coords = OrderedDict()
    for surface, value in SURFACE_VALUES.items():
        coords[surface] = np.argwhere(np.isclose(sfc, value))
    return coords


def profile_statistics(profile_stack, tower_counts, reference_towers, n_repeats, random_state):
    if not np.any(np.isfinite(profile_stack)):
        raise ValueError("No finite tower profiles were computed for this profile/case/surface.")

    reference_profile = np.nanmean(profile_stack, axis=0)
    stats = OrderedDict()

    for ntwr in tower_counts:
        if ntwr == reference_towers:
            subset_profiles = np.array([reference_profile])
        else:
            subset_profiles = np.zeros((n_repeats, profile_stack.shape[1]), dtype=float)
            for repeat in range(n_repeats):
                subset = random_state.choice(reference_towers, size=ntwr, replace=False)
                subset_profiles[repeat, :] = np.nanmean(profile_stack[subset, :], axis=0)

        rms_values = []
        rel_values = []
        max_values = []
        for profile in subset_profiles:
            rms, rel, max_abs = rms_error(profile, reference_profile)
            rms_values.append(rms)
            rel_values.append(rel)
            max_values.append(max_abs)

        stats[ntwr] = {
            "mean_profile": np.nanmean(subset_profiles, axis=0),
            "std_profile": np.nanstd(subset_profiles, axis=0),
            "rms_error_mean": float(np.nanmean(rms_values)),
            "rms_error_std": float(np.nanstd(rms_values)),
            "rel_rms_error_mean": float(np.nanmean(rel_values)),
            "rel_rms_error_std": float(np.nanstd(rel_values)),
            "max_abs_error_mean": float(np.nanmean(max_values)),
            "max_abs_error_std": float(np.nanstd(max_values)),
        }

    return reference_profile, stats


def write_summary_csv(rows, output_path):
    fieldnames = [
        "case_label",
        "case_name",
        "surface",
        "profile",
        "ntwr",
        "reference_towers",
        "n_repeats",
        "rms_error_mean",
        "rms_error_std",
        "rel_rms_error_mean",
        "rel_rms_error_std",
        "max_abs_error_mean",
        "max_abs_error_std",
    ]
    with output_path.open("w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_convergence(rows, profiles, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    for profile in profiles:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        matching = [row for row in rows if row["profile"] == profile]
        line_keys = sorted(set((row["case_label"], row["surface"]) for row in matching))

        for case_label, surface in line_keys:
            line_rows = [
                row
                for row in matching
                if row["case_label"] == case_label and row["surface"] == surface
            ]
            line_rows = sorted(line_rows, key=lambda row: int(row["ntwr"]))
            x = np.array([int(row["ntwr"]) for row in line_rows], dtype=float)
            y = np.array([float(row["rel_rms_error_mean"]) for row in line_rows], dtype=float)
            y = np.maximum(y, 1.0e-12)
            label = "{} {}".format(case_label, surface[0])
            ax.plot(x, y, marker="o", linewidth=1.4, label=label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of virtual towers")
        ax.set_ylabel("Relative RMS profile error")
        ax.set_title("{} convergence to 1000-tower profile".format(profile))
        ax.grid(True, which="both", alpha=0.35)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(str(output_dir / "sensitivity_convergence_{}.png".format(profile.lower())), dpi=300)
        plt.close(fig)


def run_analysis(args):
    cases = normalize_cases(args.cases)
    profiles = normalize_profiles(args.profiles)
    tower_counts = sorted(set(int(count) for count in args.tower_counts))
    if min(tower_counts) <= 0:
        raise ValueError("Tower counts must be positive.")
    if args.n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")

    reference_towers = int(args.reference_towers or max(tower_counts))
    if reference_towers < max(tower_counts):
        raise ValueError("reference_towers must be >= the largest requested tower count.")

    data_root = Path(args.data_root).expanduser()
    input_root = Path(args.input_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = np.random.RandomState(args.seed)
    summary_rows = []
    npz_payload = OrderedDict()

    for case_name in cases:
        case_label = CASE_LABELS[case_name]
        print("Loading {} ({})".format(case_label, case_name))
        data, shape = load_case_data(case_name, data_root, args.load_data, args.engine)
        nz = shape[2]
        dz = 1.0 / nz
        z_uvp = np.arange(0, nz, dtype=float) * dz + dz / 2.0
        z_over_h = z_uvp[: min(NZ_SLAYER, nz)] / CANOPY_H
        npz_payload["z_over_h"] = z_over_h

        coords_by_surface = load_surface_coords(case_name, input_root)
        for surface_name, all_coords in coords_by_surface.items():
            if all_coords.shape[0] < reference_towers:
                raise ValueError(
                    "{} {} has only {} candidate towers; requested {} reference towers.".format(
                        case_label, surface_name, all_coords.shape[0], reference_towers
                    )
                )

            pool_idx = random_state.choice(all_coords.shape[0], size=reference_towers, replace=False)
            pool_coords = all_coords[pool_idx, :]
            print(
                "  {}: computing {} tower profiles from {} candidates".format(
                    surface_name, reference_towers, all_coords.shape[0]
                )
            )

            stacks = compute_tower_profile_stack(data, pool_coords, surface_name, profiles, z_uvp, dz)
            for profile_name, stack in stacks.items():
                reference_profile, stats = profile_statistics(
                    stack, tower_counts, reference_towers, args.n_repeats, random_state
                )
                safe_base = "{}__{}__{}".format(case_label, surface_name, profile_name)
                npz_payload["reference__" + safe_base] = reference_profile

                for ntwr, stat in stats.items():
                    npz_payload["mean__{}__Ntwr{}".format(safe_base, ntwr)] = stat["mean_profile"]
                    npz_payload["std__{}__Ntwr{}".format(safe_base, ntwr)] = stat["std_profile"]
                    row = {
                        "case_label": case_label,
                        "case_name": case_name,
                        "surface": surface_name,
                        "profile": profile_name,
                        "ntwr": ntwr,
                        "reference_towers": reference_towers,
                        "n_repeats": 1 if ntwr == reference_towers else args.n_repeats,
                        "rms_error_mean": stat["rms_error_mean"],
                        "rms_error_std": stat["rms_error_std"],
                        "rel_rms_error_mean": stat["rel_rms_error_mean"],
                        "rel_rms_error_std": stat["rel_rms_error_std"],
                        "max_abs_error_mean": stat["max_abs_error_mean"],
                        "max_abs_error_std": stat["max_abs_error_std"],
                    }
                    summary_rows.append(row)

                first_ntwr = tower_counts[0]
                first_rel = stats[first_ntwr]["rel_rms_error_mean"]
                print("    {}: Ntwr={} relative RMS error = {:.3e}".format(profile_name, first_ntwr, first_rel))

    csv_path = output_dir / "sensitivity_convergence.csv"
    npz_path = output_dir / "sensitivity_profiles.npz"
    write_summary_csv(summary_rows, csv_path)
    np.savez(str(npz_path), **npz_payload)

    if not args.no_plots:
        plot_convergence(summary_rows, profiles, output_dir)

    print("Saved summary: {}".format(csv_path))
    print("Saved profiles: {}".format(npz_path))
    if not args.no_plots:
        print("Saved convergence plots in: {}".format(output_dir))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute virtual tower-count sensitivity curves for Paper1 heterogeneous canopy profiles."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASE_ALIASES.keys()),
        help="Case aliases or raw case names. Defaults to all gXXXX/iXXXX cases.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["all"],
        help="Profiles to analyze: all, u, shear, phi, skew. Default: all.",
    )
    parser.add_argument(
        "--tower-counts",
        nargs="+",
        type=int,
        default=[1, 10, 100, 1000],
        help="Virtual tower counts for the convergence curve.",
    )
    parser.add_argument(
        "--reference-towers",
        type=int,
        default=None,
        help="Reference tower count. Default: largest value in --tower-counts.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=100,
        help="Number of random subsamples for each non-reference tower count.",
    )
    parser.add_argument("--seed", type=int, default=314159, help="Random seed for reproducible tower sampling.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Directory containing case NetCDF folders.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Directory containing case sfc.npy folders.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV, NPZ, and plot outputs.")
    parser.add_argument(
        "--engine",
        default=None,
        help="Optional xarray backend engine, for example netcdf4 or h5netcdf.",
    )
    parser.add_argument(
        "--load-data",
        action="store_true",
        help="Load each NetCDF array into memory before tower extraction. Faster if enough RAM is available.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/NPZ only.")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
