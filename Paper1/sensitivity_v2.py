#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random-tower sensitivity test for the heterogeneous canopy cases.

The default cases are:

    g1200 -> Gap_12_9mps      i1200 -> Patch_12_9mps
    g800  -> Gap_8_9mps       i800  -> Patch_8_9mps
    g400  -> Gap_4_9mps       i400  -> Patch_4_9mps

For each requested tower count and surface pool, the script draws 100 random
virtual-tower samples, computes one tower-averaged raw shear-stress profile
per sample, then summarizes the spread among those 100 profiles with a
height-integrated relative RMS uncertainty. The shear-stress profile is not
normalized by ustar. The default surface pools are forested towers
(sfc == 1.6) and empty-area towers (sfc == 0.0):

    metric = sqrt( int sigma(z)^2 dz / int mean(z)^2 dz )

where sigma(z) is the standard deviation among the sampled averaged profiles.
"""

from __future__ import print_function

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sensitivity import (
    CANOPY_H,
    CASE_ALIASES,
    load_case_data,
    uvp_to_wnode_column,
    wnode_to_uvp_column,
)


PAPER1_DIR = Path(__file__).resolve().parent
DEFAULT_GIULIA_ROOT = Path(
    "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData"
)
DEFAULT_OUTPUT_DIR = PAPER1_DIR / "sensitivity_v2_outputs"

DEFAULT_CASES = ["g1200", "g800", "g400", "i1200", "i800", "i400"]
CASE_GRID = [["g1200", "g800", "g400"], ["i1200", "i800", "i400"]]
SURFACE_VALUES = OrderedDict([("forested", 1.6), ("empty", 0.0)])
SURFACE_ALIASES = {
    "forested": "forested",
    "empty": "empty",
    "patches": "empty",
    "all": "all",
}
DEFAULT_TOWER_COUNTS = [10, 50, 100, 500]
DEFAULT_N_REPEATS = 100
DEFAULT_NZ_SLAYER = 200


def normalize_case(case):
    key = case.lower()
    if key in CASE_ALIASES:
        return CASE_ALIASES[key], key
    for alias, case_name in CASE_ALIASES.items():
        if case == case_name:
            return case_name, alias
    valid = ", ".join(list(CASE_ALIASES.keys()) + list(CASE_ALIASES.values()))
    raise ValueError("Unknown case '{}'. Valid cases: {}".format(case, valid))


def normalize_cases(cases):
    normalized = []
    for case in cases:
        case_name, case_alias = normalize_case(case)
        normalized.append((case_name, case_alias))
    return list(
        OrderedDict(
            (case_alias, (case_name, case_alias)) for case_name, case_alias in normalized
        ).values()
    )


def load_surface_mask(case_name, input_root):
    sfc_path = input_root / case_name / "sfc.npy"
    if not sfc_path.exists():
        raise FileNotFoundError("Missing surface mask: {}".format(sfc_path))
    return np.load(str(sfc_path))


def normalize_surfaces(surfaces):
    normalized = []
    for surface in surfaces:
        key = surface.lower()
        if key not in SURFACE_ALIASES:
            raise ValueError(
                "Unknown surface '{}'. Use all, forested, empty, or patches.".format(surface)
            )
        normalized.append(SURFACE_ALIASES[key])
    return list(OrderedDict((surface, None) for surface in normalized).keys())


def candidate_coords_from_surface(sfc, surface):
    if surface == "all":
        mask = np.zeros(sfc.shape, dtype=bool)
        for value in SURFACE_VALUES.values():
            mask |= np.isclose(sfc, value)
        return np.argwhere(mask)

    if surface not in SURFACE_VALUES:
        raise ValueError(
            "Unknown surface '{}'. Use all, forested, empty, or patches.".format(surface)
        )

    return np.argwhere(np.isclose(sfc, SURFACE_VALUES[surface]))


def column(data, ix, iy, var_idx):
    return np.asarray(data[ix, iy, :, var_idx], dtype=float)


def compute_shear_profile_stack(data, coords, nz_slayer, progress_every=1000):
    """Return raw shear-stress profiles for every coordinate in coords."""
    n_towers = coords.shape[0]
    nz_use = min(nz_slayer, data.shape[2])
    profiles = np.full((n_towers, nz_use), np.nan, dtype=float)

    for idx, (ix, iy) in enumerate(coords):
        if progress_every and idx and idx % progress_every == 0:
            print("    computed {:d}/{:d} unique tower profiles".format(idx, n_towers))

        u = column(data, ix, iy, 0)
        v = column(data, ix, iy, 1)
        w = column(data, ix, iy, 2)
        uw = column(data, ix, iy, 8) - uvp_to_wnode_column(u) * w - column(data, ix, iy, 23)
        vw = column(data, ix, iy, 9) - uvp_to_wnode_column(v) * w - column(data, ix, iy, 24)
        t13 = wnode_to_uvp_column(uw)
        t23 = wnode_to_uvp_column(vw)
        profiles[idx, :] = np.sqrt(t13[:nz_use] ** 2 + t23[:nz_use] ** 2)

    return profiles


def build_random_samples(n_candidates, tower_counts, n_repeats, random_state):
    samples = OrderedDict()
    for ntwr in tower_counts:
        samples[ntwr] = [
            random_state.choice(n_candidates, size=ntwr, replace=False) for _ in range(n_repeats)
        ]
    return samples


def unique_sample_indices(samples):
    all_indices = []
    for sample_list in samples.values():
        all_indices.extend(sample_list)
    return np.unique(np.concatenate(all_indices))


def average_sample_profiles(unique_profiles, unique_indices, sample_indices):
    index_to_row = {int(candidate_idx): row for row, candidate_idx in enumerate(unique_indices)}
    averaged = np.full((len(sample_indices), unique_profiles.shape[1]), np.nan, dtype=float)

    for repeat, sample in enumerate(sample_indices):
        rows = [index_to_row[int(candidate_idx)] for candidate_idx in sample]
        with np.errstate(invalid="ignore"):
            averaged[repeat, :] = np.nanmean(unique_profiles[rows, :], axis=0)

    return averaged


def metric_levels(z_over_h, max_z_over_h):
    finite = np.isfinite(z_over_h)
    if max_z_over_h is not None:
        finite &= z_over_h <= max_z_over_h
    return finite


def integrated_uncertainty_metrics(averaged_profiles, z_over_h, use_levels):
    mean_profile = np.nanmean(averaged_profiles, axis=0)
    std_profile = np.nanstd(averaged_profiles, axis=0, ddof=1)
    valid = use_levels & np.isfinite(z_over_h) & np.isfinite(mean_profile) & np.isfinite(std_profile)

    if np.count_nonzero(valid) < 2:
        raise ValueError("Not enough finite vertical levels to compute the integrated metric.")

    z = z_over_h[valid]
    mean_use = mean_profile[valid]
    std_use = std_profile[valid]

    std_sq_int = np.trapz(std_use ** 2, z)
    mean_sq_int = np.trapz(mean_use ** 2, z)
    z_span = z[-1] - z[0]

    absolute_rms_spread = math.sqrt(std_sq_int / z_span) if z_span > 0.0 else np.nan
    relative_rms_spread = (
        math.sqrt(std_sq_int / mean_sq_int) if mean_sq_int > 0.0 else np.nan
    )
    mean_integrated_std = np.trapz(std_use, z) / z_span if z_span > 0.0 else np.nan

    return {
        "mean_profile": mean_profile,
        "std_profile": std_profile,
        "absolute_rms_spread": float(absolute_rms_spread),
        "relative_rms_spread": float(relative_rms_spread),
        "mean_integrated_std": float(mean_integrated_std),
        "n_finite_levels": int(np.count_nonzero(valid)),
        "z_over_h_min": float(z[0]),
        "z_over_h_max": float(z[-1]),
    }


def write_summary_csv(rows, output_path):
    fieldnames = [
        "case_alias",
        "case_name",
        "surface",
        "ntwr",
        "n_repeats",
        "n_unique_towers",
        "n_candidate_towers",
        "relative_rms_spread",
        "absolute_rms_spread",
        "mean_integrated_std",
        "n_finite_levels",
        "z_over_h_min",
        "z_over_h_max",
    ]
    with output_path.open("w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    row_surfaces = set(row["surface"] for row in rows)
    surfaces = [surface for surface in ["forested", "empty", "all"] if surface in row_surfaces]

    for row_idx, case_row in enumerate(CASE_GRID):
        for col_idx, case_alias in enumerate(case_row):
            ax = axes[row_idx, col_idx]
            case_rows = [row for row in rows if row["case_alias"] == case_alias]

            for surface in surfaces:
                surface_rows = sorted(
                    [row for row in case_rows if row["surface"] == surface],
                    key=lambda row: int(row["ntwr"]),
                )
                if not surface_rows:
                    continue
                x = np.array([int(row["ntwr"]) for row in surface_rows], dtype=float)
                y = np.array(
                    [float(row["relative_rms_spread"]) for row in surface_rows],
                    dtype=float,
                )
                ax.plot(x, y, marker="o", linewidth=1.8, label=surface)

            ax.set_xscale("log")
            ax.set_title(case_alias)
            ax.grid(True, which="both", alpha=0.35)
            ax.xaxis.set_major_formatter(ScalarFormatter())

    for ax in axes[-1, :]:
        ax.set_xlabel("Number of virtual towers")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative height-integrated RMS spread")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle("Raw shear-stress tower-sampling uncertainty", y=0.99)
    fig.tight_layout()
    if handles:
        fig.subplots_adjust(top=0.88)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def run_analysis(args):
    cases = normalize_cases(args.cases)
    tower_counts = sorted(set(int(count) for count in args.tower_counts))
    if min(tower_counts) <= 0:
        raise ValueError("Tower counts must be positive.")
    if args.n_repeats <= 1:
        raise ValueError("Use at least two repeats to compute profile spread.")
    surfaces = normalize_surfaces(args.surfaces)

    giulia_root = Path(args.giulia_root).expanduser()
    data_root = giulia_root / "TKE_BUDGET_AND_RAV"
    input_root = giulia_root / "input_txt_files"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = np.random.RandomState(args.seed)
    summary_rows = []
    npz_payload = OrderedDict()

    for case_name, case_alias in cases:
        print("Loading {} ({})".format(case_alias, case_name))
        data, shape = load_case_data(case_name, data_root, args.load_data, args.engine)
        nz = shape[2]
        dz = 1.0 / nz
        z_uvp = np.arange(0, nz, dtype=float) * dz + dz / 2.0
        nz_use = min(args.nz_slayer, nz)
        z_over_h = z_uvp[:nz_use] / CANOPY_H
        use_levels = metric_levels(z_over_h, args.max_z_over_h)

        npz_payload["z_over_h__{}".format(case_alias)] = z_over_h
        npz_payload["metric_levels__{}".format(case_alias)] = use_levels

        sfc = load_surface_mask(case_name, input_root)

        for surface in surfaces:
            candidates = candidate_coords_from_surface(sfc, surface)
            if candidates.shape[0] < max(tower_counts):
                raise ValueError(
                    "{} {} has only {} candidate towers; requested {}.".format(
                        case_alias, surface, candidates.shape[0], max(tower_counts)
                    )
                )

            print(
                "{} {}: drawing {} repeats for tower counts {}".format(
                    case_alias,
                    surface,
                    args.n_repeats,
                    ", ".join(str(count) for count in tower_counts),
                )
            )
            samples = build_random_samples(
                candidates.shape[0], tower_counts, args.n_repeats, random_state
            )
            unique_indices = unique_sample_indices(samples)
            unique_coords = candidates[unique_indices, :]

            print(
                "  computing {} unique tower profiles from {} candidates".format(
                    unique_coords.shape[0], candidates.shape[0]
                )
            )
            unique_profiles = compute_shear_profile_stack(
                data,
                unique_coords,
                nz_use,
                progress_every=args.progress_every,
            )

            for ntwr, sample_indices in samples.items():
                averaged_profiles = average_sample_profiles(
                    unique_profiles, unique_indices, sample_indices
                )
                stats = integrated_uncertainty_metrics(averaged_profiles, z_over_h, use_levels)

                safe_base = "{}__{}__Ntwr{}".format(case_alias, surface, ntwr)
                npz_payload["profiles__" + safe_base] = averaged_profiles
                npz_payload["mean__" + safe_base] = stats["mean_profile"]
                npz_payload["std__" + safe_base] = stats["std_profile"]

                row = {
                    "case_alias": case_alias,
                    "case_name": case_name,
                    "surface": surface,
                    "ntwr": ntwr,
                    "n_repeats": args.n_repeats,
                    "n_unique_towers": unique_coords.shape[0],
                    "n_candidate_towers": candidates.shape[0],
                    "relative_rms_spread": stats["relative_rms_spread"],
                    "absolute_rms_spread": stats["absolute_rms_spread"],
                    "mean_integrated_std": stats["mean_integrated_std"],
                    "n_finite_levels": stats["n_finite_levels"],
                    "z_over_h_min": stats["z_over_h_min"],
                    "z_over_h_max": stats["z_over_h_max"],
                }
                summary_rows.append(row)

                print(
                    "  Ntwr={:d}: relative integrated RMS spread = {:.4e}".format(
                        ntwr, stats["relative_rms_spread"]
                    )
                )

    csv_path = output_dir / "heterogeneous_raw_shear_uncertainty.csv"
    npz_path = output_dir / "heterogeneous_raw_shear_sampled_profiles.npz"
    plot_path = output_dir / "heterogeneous_raw_shear_uncertainty_vs_towers.png"

    write_summary_csv(summary_rows, csv_path)
    np.savez(str(npz_path), **npz_payload)
    if not args.no_plot:
        plot_metric(summary_rows, plot_path)

    print("Saved summary: {}".format(csv_path))
    print("Saved sampled profiles: {}".format(npz_path))
    if not args.no_plot:
        print("Saved metric plot: {}".format(plot_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute raw shear-stress tower-sampling uncertainty."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        help=(
            "Case aliases or raw case names. Default: "
            "{}.".format(" ".join(DEFAULT_CASES))
        ),
    )
    parser.add_argument(
        "--giulia-root",
        default=str(DEFAULT_GIULIA_ROOT),
        help="Root containing TKE_BUDGET_AND_RAV and input_txt_files.",
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        default=["forested", "empty"],
        choices=["all", "forested", "empty", "patches"],
        help="Tower candidate pools to analyze. Default: forested empty.",
    )
    parser.add_argument(
        "--tower-counts",
        nargs="+",
        type=int,
        default=DEFAULT_TOWER_COUNTS,
        help="Number of virtual towers to average. Default: 10 50 100 500.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Random samples per tower count. Default: 100.",
    )
    parser.add_argument(
        "--nz-slayer",
        type=int,
        default=DEFAULT_NZ_SLAYER,
        help="Number of vertical levels kept in each shear profile. Default: 200.",
    )
    parser.add_argument(
        "--max-z-over-h",
        type=float,
        default=10.0,
        help="Maximum z/h_C included in the integrated metric. Default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314159,
        help="Random seed for reproducible tower sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV, NPZ, and plot outputs.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="Optional xarray backend engine, for example netcdf4 or h5netcdf.",
    )
    parser.add_argument(
        "--load-data",
        action="store_true",
        help="Load the full NetCDF array into memory before extracting towers.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many unique tower profiles. Use 0 to disable.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Write CSV/NPZ only.")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
