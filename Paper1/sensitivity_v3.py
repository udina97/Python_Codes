#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tower-count sensitivity for normalized shear in the heterogeneous canopy cases.

The script loads the gap and patch heterogeneous canopy cases, computes the
raw 3D shear-stress magnitude from the full momentum field using the same
T_13/T_23 definition used in sensitivity_v2.py, normalizes each virtual tower's
shear profile by its local ustar**2, and then compares random N-tower mean
profiles against a 1000-tower reference profile.

For each region, the normalized deviation profile is

    D_N(z) = abs(S_N(z) - S_ref(z)) / S_ref(z)

where S_ref is the 1000-tower mean profile for that region. The plotted metric
is the mean D_N(z) profile across the 100 random resampling realizations. The
shaded bands show the interquartile range of D_N(z) across those realizations.

A second figure shows the normalized 1000-tower reference shear profile and
one example resampling realization for each requested N.
"""

from __future__ import print_function

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sensitivity import CANOPY_H, CASE_ALIASES, KAPPA, LAD, U_SCALE, ZI, load_case_data


PAPER1_DIR = Path(__file__).resolve().parent
DEFAULT_GIULIA_ROOT = Path(
    "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData"
)
DEFAULT_OUTPUT_DIR = PAPER1_DIR / "sensitivity_v3_outputs"

DEFAULT_CASES = ["g1200", "g800", "g400", "i1200", "i800", "i400"]
CASE_GRID = [["g1200", "g800", "g400"], ["i1200", "i800", "i400"]]
PANEL_LABELS = [
    ["a)", "b)", "c)"],
    ["d)", "e)", "f)"],
]
DEFAULT_TOWER_COUNTS = [1, 10, 50, 100, 500]
DEFAULT_REFERENCE_TOWERS = 1000
DEFAULT_N_REPEATS = 100
DEFAULT_NZ_SLAYER = 200
PROFILE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

SURFACE_VALUES = OrderedDict([("forested", 1.6), ("open_gap", 0.0)])
SURFACE_LABELS = {"forested": "Forested", "open_gap": "Empty/open gap"}


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


def candidate_coords_from_surface(sfc, surface):
    if surface not in SURFACE_VALUES:
        raise ValueError(
            "Unknown surface '{}'. Valid surfaces: {}.".format(
                surface, ", ".join(SURFACE_VALUES)
            )
        )
    return np.argwhere(np.isclose(sfc, SURFACE_VALUES[surface]))


def uvp_to_wnode_field(phi_c):
    """Interpolate a 3D uvp-node field to w nodes along the vertical axis."""
    phi_c = np.asarray(phi_c, dtype=float)
    phi_h = np.zeros(phi_c.shape, dtype=float)
    phi_h[:, :, 0] = 0.0
    phi_h[:, :, 1:] = 0.5 * (phi_c[:, :, :-1] + phi_c[:, :, 1:])
    return phi_h


def wnode_to_uvp_field(phi_h):
    """Interpolate a 3D w-node field to uvp nodes along the vertical axis."""
    phi_h = np.asarray(phi_h, dtype=float)
    phi_c = np.empty(phi_h.shape, dtype=float)
    phi_c[:, :, :-1] = 0.5 * (phi_h[:, :, :-1] + phi_h[:, :, 1:])
    phi_c[:, :, -1] = phi_h[:, :, -1]
    return phi_c


def data_field(data, var_idx):
    return np.array(data[:, :, :, var_idx], dtype=float, copy=True)


def column(data, ix, iy, var_idx):
    return np.asarray(data[ix, iy, :, var_idx], dtype=float)


def compute_raw_shear_field(data):
    """Return sqrt(T_13**2 + T_23**2) for the full 3D domain."""
    print("Computing full 3D raw shear field")

    w = data_field(data, 2)

    u_w = uvp_to_wnode_field(data_field(data, 0))
    uw = data_field(data, 8)
    uw -= u_w * w
    del u_w
    uw -= data_field(data, 23)
    t13 = wnode_to_uvp_field(uw)
    del uw

    v_w = uvp_to_wnode_field(data_field(data, 1))
    vw = data_field(data, 9)
    vw -= v_w * w
    del v_w, w
    vw -= data_field(data, 24)
    t23 = wnode_to_uvp_field(vw)
    del vw

    shear = np.sqrt(t13 ** 2 + t23 ** 2)
    del t13, t23
    return shear


def log_fit(z, a, b):
    return a * np.log(b * z)


def compute_ustar_g(nz_slayer, z_d, u, v):
    """Compute local ustar using the g1200 log-fit method from Compute_SHEARprof.py."""
    from scipy.optimize import curve_fit

    fit_levels = np.array([50, 60, 70, 90], dtype=int)
    if np.max(fit_levels) >= nz_slayer:
        return np.nan

    u_mag = np.sqrt(u ** 2 + v ** 2)
    u_mean = u_mag[:nz_slayer]
    z_data = np.asarray(z_d[fit_levels], dtype=float)
    u_data = np.asarray(u_mean[fit_levels], dtype=float)
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

    ustar = u_mean[fit_levels[0]] / denom
    return float(ustar) if np.isfinite(ustar) and ustar != 0.0 else np.nan


def compute_displacement_heights_g(data, coords, dz):
    """Compute local forest displacement heights as in Compute_SHEARprof.py."""
    height = int(math.ceil(CANOPY_H / dz))
    nlevels = min(height, len(LAD))
    z_dim = np.arange(1, height + 1, dtype=float)[:nlevels] * dz * ZI
    z_dim -= 0.5 * dz * ZI
    lad_use = np.asarray(LAD[:nlevels], dtype=float)
    displacement = np.zeros(coords.shape[0], dtype=float)

    for idx, (ix, iy) in enumerate(coords):
        u = column(data, ix, iy, 0)[:nlevels]
        v = column(data, ix, iy, 1)[:nlevels]
        u_mag = np.sqrt(u ** 2 + v ** 2)
        z_use = z_dim[: len(u_mag)]
        y = (u_mag * U_SCALE) ** 2 * KAPPA * lad_use[: len(u_mag)]
        den = np.trapz(y, z_use)
        displacement[idx] = np.trapz(y * z_use, z_use) / den if den != 0.0 else 0.0

    return displacement


def normalized_shear_profile_stack(
    data, shear_field, coords, surface, z_uvp, dz, nz_use, progress_every=1000
):
    """Return shear/ustar**2 profiles for every coordinate in coords."""
    n_towers = coords.shape[0]
    profiles = np.full((n_towers, nz_use), np.nan, dtype=float)

    if surface == "forested":
        print("  computing local forest displacement heights")
        displacement = compute_displacement_heights_g(data, coords, dz)
    else:
        displacement = np.zeros(n_towers, dtype=float)

    ustars = np.full(n_towers, np.nan, dtype=float)
    for idx, (ix, iy) in enumerate(coords):
        if progress_every and idx and idx % progress_every == 0:
            print(
                "    normalized {:d}/{:d} unique tower profiles".format(
                    idx, n_towers
                )
            )

        z_d = z_uvp - displacement[idx] / ZI
        ustar = compute_ustar_g(
            nz_use,
            z_d,
            column(data, ix, iy, 0),
            column(data, ix, iy, 1),
        )
        ustars[idx] = ustar
        if np.isfinite(ustar) and ustar != 0.0:
            profiles[idx, :] = shear_field[ix, iy, :nz_use] / (ustar ** 2)

    return profiles, ustars, displacement


def draw_samples(n_candidates, tower_count, n_repeats, random_state):
    return [
        random_state.choice(n_candidates, size=tower_count, replace=False)
        for _ in range(n_repeats)
    ]


def unique_sample_indices(reference_indices, samples_by_count):
    all_indices = [reference_indices]
    for sample_list in samples_by_count.values():
        all_indices.extend(sample_list)
    return np.unique(np.concatenate(all_indices))


def profile_index_lookup(unique_indices):
    return {int(candidate_idx): row for row, candidate_idx in enumerate(unique_indices)}


def tower_mean_profile(profile_stack, lookup, sample_indices):
    rows = [lookup[int(candidate_idx)] for candidate_idx in sample_indices]
    with np.errstate(invalid="ignore"):
        return np.nanmean(profile_stack[rows, :], axis=0)


def mean_profiles_for_samples(profile_stack, lookup, sample_indices):
    profiles = np.full((len(sample_indices), profile_stack.shape[1]), np.nan, dtype=float)
    for repeat, indices in enumerate(sample_indices):
        profiles[repeat, :] = tower_mean_profile(profile_stack, lookup, indices)
    return profiles


def profile_levels(z_over_h, reference_profile, max_z_over_h):
    valid = np.isfinite(z_over_h) & np.isfinite(reference_profile)
    valid &= np.abs(reference_profile) != 0.0
    if max_z_over_h is not None:
        valid &= z_over_h <= max_z_over_h
    return valid


def deviation_statistics(sample_profiles, reference_profile, z_over_h, use_levels):
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation_profiles = np.abs(sample_profiles - reference_profile) / np.abs(
            reference_profile
        )

    mean_deviation_profile = np.nanmean(deviation_profiles, axis=0)
    q25_deviation_profile = np.nanpercentile(deviation_profiles, 25.0, axis=0)
    q75_deviation_profile = np.nanpercentile(deviation_profiles, 75.0, axis=0)

    finite_levels = use_levels & np.isfinite(mean_deviation_profile)
    if np.count_nonzero(finite_levels) < 2:
        raise ValueError("Not enough finite levels to compute deviation statistics.")

    return {
        "deviation_profiles": deviation_profiles,
        "mean_deviation_profile": mean_deviation_profile,
        "q25_deviation_profile": q25_deviation_profile,
        "q75_deviation_profile": q75_deviation_profile,
        "use_levels": use_levels,
        "n_finite_levels": int(np.count_nonzero(finite_levels)),
        "z_over_h_min": float(z_over_h[finite_levels][0]),
        "z_over_h_max": float(z_over_h[finite_levels][-1]),
    }


def write_profile_csv(records, output_path):
    fieldnames = [
        "case_alias",
        "case_name",
        "surface",
        "ntwr",
        "reference_towers",
        "n_repeats",
        "n_candidate_towers",
        "z_over_h",
        "mean_deviation",
        "q25_deviation",
        "q75_deviation",
    ]
    with output_path.open("w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            valid = record["use_levels"]
            z_over_h = record["z_over_h"]
            mean_profile = record["mean_deviation_profile"]
            q25_profile = record["q25_deviation_profile"]
            q75_profile = record["q75_deviation_profile"]
            for level in np.where(valid)[0]:
                writer.writerow(
                    {
                        "case_alias": record["case_alias"],
                        "case_name": record["case_name"],
                        "surface": record["surface"],
                        "ntwr": record["ntwr"],
                        "reference_towers": record["reference_towers"],
                        "n_repeats": record["n_repeats"],
                        "n_candidate_towers": record["n_candidate_towers"],
                        "z_over_h": z_over_h[level],
                        "mean_deviation": mean_profile[level],
                        "q25_deviation": q25_profile[level],
                        "q75_deviation": q75_profile[level],
                    }
                )


def surface_linestyle(surface):
    return "-" if surface == "forested" else "--"


def plot_deviation_profiles(records, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    tower_counts = sorted(set(int(record["ntwr"]) for record in records))
    color_by_count = {
        ntwr: PROFILE_COLORS[idx % len(PROFILE_COLORS)]
        for idx, ntwr in enumerate(tower_counts)
    }

    for row_idx, case_row in enumerate(CASE_GRID):
        for col_idx, case_alias in enumerate(case_row):
            ax = axes[row_idx, col_idx]
            case_records = sorted(
                [record for record in records if record["case_alias"] == case_alias],
                key=lambda record: (int(record["ntwr"]), record["surface"]),
            )

            for record in case_records:
                valid = (
                    record["use_levels"]
                    & np.isfinite(record["mean_deviation_profile"])
                    & np.isfinite(record["q25_deviation_profile"])
                    & np.isfinite(record["q75_deviation_profile"])
                )
                color = color_by_count[int(record["ntwr"])]
                linestyle = surface_linestyle(record["surface"])
                z = record["z_over_h"][valid]
                mean_profile = record["mean_deviation_profile"][valid]
                q25_profile = record["q25_deviation_profile"][valid]
                q75_profile = record["q75_deviation_profile"][valid]

                ax.plot(
                    mean_profile,
                    z,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                )
                ax.fill_betweenx(
                    z,
                    q25_profile,
                    q75_profile,
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                )

            ax.axhline(1.0, color="0.25", linestyle=":", linewidth=1.0)
            ax.text(
                0.03,
                0.96,
                PANEL_LABELS[row_idx][col_idx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
            )
            ax.set_xlim(left=0.0)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(True, alpha=0.35)

            if row_idx == len(CASE_GRID) - 1:
                ax.set_xlabel(r"$\overline{r}_N$", fontsize=16)
            if col_idx == 0:
                ax.set_ylabel(r"$z/h_C$", fontsize=16)
            if not case_records:
                ax.set_visible(False)

    z_max = max(
        record["z_over_h"][record["use_levels"]][-1]
        for record in records
        if np.any(record["use_levels"])
    )
    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_ylim(bottom=0.0, top=z_max)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def plot_example_shear_profiles(records, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    tower_counts = sorted(
        set(int(record["ntwr"]) for record in records if not record["is_reference"])
    )
    color_by_count = {
        ntwr: PROFILE_COLORS[idx % len(PROFILE_COLORS)]
        for idx, ntwr in enumerate(tower_counts)
    }

    for row_idx, case_row in enumerate(CASE_GRID):
        for col_idx, case_alias in enumerate(case_row):
            ax = axes[row_idx, col_idx]
            case_records = sorted(
                [record for record in records if record["case_alias"] == case_alias],
                key=lambda record: (int(record["ntwr"]), record["surface"]),
            )

            for record in case_records:
                valid = record["use_levels"] & np.isfinite(record["shear_profile"])
                color = color_by_count.get(int(record["ntwr"]), "k")
                linestyle = surface_linestyle(record["surface"])
                linewidth = 2.4 if record["is_reference"] else 1.6
                ax.plot(
                    record["shear_profile"][valid],
                    record["z_over_h"][valid],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )

            ax.axhline(1.0, color="0.25", linestyle=":", linewidth=1.0)
            ax.text(
                0.03,
                0.96,
                PANEL_LABELS[row_idx][col_idx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
            )
            ax.set_xlim(left=0.0)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(True, alpha=0.35)

            if row_idx == len(CASE_GRID) - 1:
                ax.set_xlabel(r"Normalized shear, $\tau/u_*^2$", fontsize=16)
            if col_idx == 0:
                ax.set_ylabel(r"$z/h_C$", fontsize=16)
            if not case_records:
                ax.set_visible(False)

    z_max = max(
        record["z_over_h"][record["use_levels"]][-1]
        for record in records
        if np.any(record["use_levels"])
    )
    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_ylim(bottom=0.0, top=z_max)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def run_analysis(args):
    cases = normalize_cases(args.cases)
    tower_counts = sorted(set(int(count) for count in args.tower_counts))
    if min(tower_counts) <= 0:
        raise ValueError("Tower counts must be positive.")
    if max(tower_counts) > args.reference_towers:
        raise ValueError("Tower counts must be <= reference_towers.")
    if args.n_repeats <= 1:
        raise ValueError("Use at least two repeats to compute uncertainty.")
    if args.example_repeat < 0 or args.example_repeat >= args.n_repeats:
        raise ValueError("example_repeat must be between 0 and n_repeats - 1.")

    giulia_root = Path(args.giulia_root).expanduser()
    data_root = giulia_root / "TKE_BUDGET_AND_RAV"
    input_root = giulia_root / "input_txt_files"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = np.random.RandomState(args.seed)

    profile_records = []
    example_shear_records = []
    npz_payload = OrderedDict()

    for case_name, case_alias in cases:
        print("Loading {} ({})".format(case_alias, case_name))
        data, shape = load_case_data(case_name, data_root, args.load_data, args.engine)
        nz = shape[2]
        dz = 1.0 / nz
        z_uvp = np.arange(0, nz, dtype=float) * dz + dz / 2.0
        nz_use = min(args.nz_slayer, nz)
        z_over_h = z_uvp[:nz_use] / CANOPY_H

        sfc = load_surface_mask(case_name, input_root)
        shear_field = compute_raw_shear_field(data)
        npz_payload["z_over_h__" + case_alias] = z_over_h

        for surface in SURFACE_VALUES:
            candidates = candidate_coords_from_surface(sfc, surface)
            if candidates.shape[0] < args.reference_towers:
                raise ValueError(
                    "{} {} has only {} candidate towers; requested {} reference towers.".format(
                        case_alias,
                        surface,
                        candidates.shape[0],
                        args.reference_towers,
                    )
                )

            reference_indices = random_state.choice(
                candidates.shape[0], size=args.reference_towers, replace=False
            )
            samples_by_count = OrderedDict()
            for ntwr in tower_counts:
                samples_by_count[ntwr] = draw_samples(
                    candidates.shape[0], ntwr, args.n_repeats, random_state
                )

            unique_indices = unique_sample_indices(reference_indices, samples_by_count)
            unique_coords = candidates[unique_indices, :]
            print(
                "{} {}: normalizing {} unique tower profiles from {} candidates".format(
                    case_alias,
                    SURFACE_LABELS[surface],
                    unique_coords.shape[0],
                    candidates.shape[0],
                )
            )
            (
                unique_profiles,
                unique_ustars,
                unique_displacement,
            ) = normalized_shear_profile_stack(
                data,
                shear_field,
                unique_coords,
                surface,
                z_uvp,
                dz,
                nz_use,
                progress_every=args.progress_every,
            )
            lookup = profile_index_lookup(unique_indices)
            reference_coords = candidates[reference_indices, :]
            reference_profile = tower_mean_profile(
                unique_profiles, lookup, reference_indices
            )
            use_levels = profile_levels(z_over_h, reference_profile, args.max_z_over_h)

            safe_surface = "{}__{}".format(case_alias, surface)
            npz_payload["reference_profile__" + safe_surface] = reference_profile
            npz_payload["reference_coords__" + safe_surface] = reference_coords
            npz_payload["unique_coords__" + safe_surface] = unique_coords
            npz_payload["unique_ustar__" + safe_surface] = unique_ustars
            npz_payload["unique_displacement__" + safe_surface] = unique_displacement
            example_shear_records.append(
                {
                    "case_alias": case_alias,
                    "case_name": case_name,
                    "surface": surface,
                    "ntwr": args.reference_towers,
                    "is_reference": True,
                    "z_over_h": z_over_h,
                    "use_levels": use_levels,
                    "shear_profile": reference_profile,
                }
            )

            print(
                "{} {}: reference from {} towers; {} finite unique ustar values".format(
                    case_alias,
                    SURFACE_LABELS[surface],
                    args.reference_towers,
                    np.count_nonzero(np.isfinite(unique_ustars)),
                )
            )

            for ntwr in tower_counts:
                print(
                    "  N={}: averaging {} normalized random samples".format(
                        ntwr, args.n_repeats
                    )
                )
                sample_profiles = mean_profiles_for_samples(
                    unique_profiles, lookup, samples_by_count[ntwr]
                )
                example_profile = sample_profiles[args.example_repeat, :]
                stats = deviation_statistics(
                    sample_profiles, reference_profile, z_over_h, use_levels
                )

                safe_base = "{}__{}__Ntwr{}".format(case_alias, surface, ntwr)
                npz_payload["sample_mean_profiles__" + safe_base] = sample_profiles
                npz_payload["example_shear_profile__" + safe_base] = example_profile
                npz_payload["deviation_profiles__" + safe_base] = stats[
                    "deviation_profiles"
                ]
                npz_payload["mean_deviation_profile__" + safe_base] = stats[
                    "mean_deviation_profile"
                ]
                npz_payload["q25_deviation_profile__" + safe_base] = stats[
                    "q25_deviation_profile"
                ]
                npz_payload["q75_deviation_profile__" + safe_base] = stats[
                    "q75_deviation_profile"
                ]
                npz_payload["use_levels__" + safe_base] = stats["use_levels"]

                record = {
                    "case_alias": case_alias,
                    "case_name": case_name,
                    "surface": surface,
                    "ntwr": ntwr,
                    "reference_towers": args.reference_towers,
                    "n_repeats": args.n_repeats,
                    "n_candidate_towers": candidates.shape[0],
                    "z_over_h": z_over_h,
                    "use_levels": stats["use_levels"],
                    "mean_deviation_profile": stats["mean_deviation_profile"],
                    "q25_deviation_profile": stats["q25_deviation_profile"],
                    "q75_deviation_profile": stats["q75_deviation_profile"],
                    "n_finite_levels": stats["n_finite_levels"],
                    "z_over_h_min": stats["z_over_h_min"],
                    "z_over_h_max": stats["z_over_h_max"],
                }
                profile_records.append(record)
                example_shear_records.append(
                    {
                        "case_alias": case_alias,
                        "case_name": case_name,
                        "surface": surface,
                        "ntwr": ntwr,
                        "is_reference": False,
                        "z_over_h": z_over_h,
                        "use_levels": stats["use_levels"],
                        "shear_profile": example_profile,
                    }
                )

                print(
                    "    mean profile deviation over plotted levels = {:.4e}".format(
                        np.nanmean(
                            stats["mean_deviation_profile"][stats["use_levels"]]
                        )
                    )
                )

        del data, shear_field, sfc

    csv_path = output_dir / "heterogeneous_normalized_shear_deviation_profiles.csv"
    npz_path = output_dir / "heterogeneous_normalized_shear_deviation_profiles.npz"
    plot_path = output_dir / "heterogeneous_normalized_shear_deviation_profiles.png"
    shear_plot_path = output_dir / "heterogeneous_normalized_shear_example_profiles.png"

    write_profile_csv(profile_records, csv_path)
    np.savez(str(npz_path), **npz_payload)
    if not args.no_plot:
        plot_deviation_profiles(profile_records, plot_path)
        plot_example_shear_profiles(example_shear_records, shear_plot_path)

    print("Saved profile CSV: {}".format(csv_path))
    print("Saved profiles: {}".format(npz_path))
    if not args.no_plot:
        print("Saved plot: {}".format(plot_path))
        print("Saved example shear plot: {}".format(shear_plot_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot heterogeneous-case normalized-shear deviation profiles using "
            "a 1000-tower reference profile for forested and open-gap regions."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        help=(
            "Case aliases or raw case names. Default: {}.".format(
                " ".join(DEFAULT_CASES)
            )
        ),
    )
    parser.add_argument(
        "--giulia-root",
        default=str(DEFAULT_GIULIA_ROOT),
        help="Root containing TKE_BUDGET_AND_RAV and input_txt_files.",
    )
    parser.add_argument(
        "--tower-counts",
        nargs="+",
        type=int,
        default=DEFAULT_TOWER_COUNTS,
        help="Numbers of virtual towers to test. Default: 1 10 50 100 500.",
    )
    parser.add_argument(
        "--reference-towers",
        type=int,
        default=DEFAULT_REFERENCE_TOWERS,
        help="Number of towers used for each regional reference profile. Default: 1000.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Random resampling realizations per tower count. Default: 100.",
    )
    parser.add_argument(
        "--example-repeat",
        type=int,
        default=0,
        help=(
            "Zero-based resampling realization used in the normalized-shear "
            "example profile figure. Default: 0."
        ),
    )
    parser.add_argument(
        "--nz-slayer",
        type=int,
        default=DEFAULT_NZ_SLAYER,
        help="Number of vertical levels kept in each profile. Default: 200.",
    )
    parser.add_argument(
        "--max-z-over-h",
        type=float,
        default=10.0,
        help="Maximum z/h_C shown and written in the profile outputs. Default: 10.",
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
        help="Load the NetCDF array into memory before computing shear.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many unique normalized tower profiles. Use 0 to disable.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Write CSV/NPZ only.")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
