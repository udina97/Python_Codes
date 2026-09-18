#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pooled-profile bootstrap regression for yB-vs-normalized-TKE-residual.

Workflow:
1. For each case, load yB and normalized TKE residual profiles from canopy top
   to 5 canopy heights.
2. For each bootstrap, sample Np profiles with replacement from every case.
3. Pool all sampled points from all cases.
4. Fit one equally case-weighted linear regression line using points in the
   selected yB interval.
5. In the same yB interval, compute 25th and 75th residual quantiles in bins
   of width 0.025.
6. Repeat Nb times.
7. Plot the median regression line and the median lower/upper quantile curves.
"""

from argparse import ArgumentParser
from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


FUNCTIONS_DIR = Path("/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions")
if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCTIONS_DIR))

from functions import build_intf, build_phi  # noqa: E402


CASES = [
    "Gap_12_9mps",
    "Gap_8_9mps",
    "Gap_4_9mps",
    "Patch_12_9mps",
    "Patch_8_9mps",
    "Patch_4_9mps",
    "ATTO",
    "Sinusoidal",
    "Flat",
]

GAP_PATCH_DATA_ROOT = Path(
    "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/GiuliaData/"
    "TKE_BUDGET_AND_RAV"
)
TOPO_DATA_ROOT = Path(
    "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/AnisotropyData/"
    "NetCDF_data"
)
OUTPUT_DIR = Path(
    "/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/Paper1/"
    "bootstrapIva_outputs"
)
PROFILE_DIR = Path(
    "/uufs/chpc.utah.edu/common/home/calaf-group2/Ben_research/Paper1/Profiles"
)

ZI = 1000.0
CANOPY_H = 39.0 / ZI
CANOPY_H_METERS = CANOPY_H * ZI
HEIGHT_MAX_MULTIPLIER = 6.0

N_BOOTSTRAP = 100
N_SAMPLE_PROFILES = 10000
RANDOM_SEED = 20260911

YB_FIT_LIMITS = (0.25, 0.45)
YB_BIN_WIDTH = 0.025
QUANTILE_LIMITS = (25.0, 75.0)

YB_PLOT_LIMITS = (0.15, 0.60)
TKE_PLOT_LIMITS = (-70, 250)

PROFILE_COLORS = [
    "green",
    "limegreen",
    "lightgreen",
    "beige",
    "khaki",
    "gold",
    "peachpuff",
    "sandybrown",
    "saddlebrown",
]
PROFILE_LABELS = [
    "g1200",
    "g800",
    "g400",
    "i1200",
    "i800",
    "i400",
    "ATTO",
    "Sinusoidal",
    "Flat",
]


def parse_args():
    parser = ArgumentParser(
        description="Bootstrap pooled profile clouds and fit one median regression line."
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-sample-profiles", type=int, default=N_SAMPLE_PROFILES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--yb-min", type=float, default=YB_FIT_LIMITS[0])
    parser.add_argument("--yb-max", type=float, default=YB_FIT_LIMITS[1])
    parser.add_argument("--yb-bin-width", type=float, default=YB_BIN_WIDTH)
    parser.add_argument("--iq-low", type=float, default=QUANTILE_LIMITS[0])
    parser.add_argument("--iq-high", type=float, default=QUANTILE_LIMITS[1])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    if args.n_sample_profiles <= 0:
        parser.error("--n-sample-profiles must be positive")
    if args.yb_min >= args.yb_max:
        parser.error("--yb-min must be smaller than --yb-max")
    if args.yb_bin_width <= 0.0:
        parser.error("--yb-bin-width must be positive")
    if not 0.0 <= args.iq_low < args.iq_high <= 100.0:
        parser.error("--iq-low and --iq-high must satisfy 0 <= low < high <= 100")
    return args


def load_case(case_name, case_index):
    """Load TKE terms and anisotropy arrays using the Paper1 data paths."""
    import xarray as xr

    if case_index < 6:
        case_dir = GAP_PATCH_DATA_ROOT / case_name
        terms_bdg = xr.open_dataarray(case_dir / "TKE_terms.nc").data
        anisotropy = xr.open_dataarray(case_dir / "anisotropy.nc").data
        lz = 1.0
        topo_case = False
        mpi_proc = None
    else:
        case_dir = TOPO_DATA_ROOT / case_name
        terms_bdg = xr.open_dataarray(case_dir / "TKE_terms.nc").data
        anisotropy = xr.open_dataarray(case_dir / "anisotropy.nc").data
        lz = 0.96
        topo_case = True
        mpi_proc = 32

    nx, ny, nz_data = terms_bdg[:, :, :, 0].shape
    nz_full = nz_data + 5 if topo_case else nz_data
    dz = lz / nz_full

    return {
        "case_dir": case_dir,
        "terms_bdg": terms_bdg,
        "anisotropy": anisotropy,
        "nx": nx,
        "ny": ny,
        "nz_data": nz_data,
        "nz_full": nz_full,
        "dz": dz,
        "topo_case": topo_case,
        "mpi_proc": mpi_proc,
    }


def compute_normalized_tke_residual(terms_bdg, topo_case, dist_terms=None):
    """Compute normalized residual using the existing Paper1 convention."""
    tmp_dis = np.array(terms_bdg[:, :, :, 11], copy=True)

    if topo_case:
        tmp_res = np.array(terms_bdg[:, :, :, -1] - terms_bdg[:, :, :, 11], copy=True)
        if dist_terms is not None:
            near_canopy = (dist_terms > 38.0) & (dist_terms < 42.0)
            residual_ref = np.nanmedian(tmp_res[near_canopy])
        else:
            residual_ref = np.nan
        if not np.isfinite(residual_ref):
            residual_ref = np.nanmax(np.abs(np.nanmedian(tmp_res, axis=(0, 1))))
    else:
        tmp_res = np.array(terms_bdg[:, :, :, -1] + terms_bdg[:, :, :, 11], copy=True)
        residual_ref = np.nanmax(np.abs(np.nanmedian(tmp_res, axis=(0, 1))))

    if np.isfinite(residual_ref) and residual_ref != 0:
        tmp_res[np.abs(tmp_res) < 0.01 * abs(residual_ref)] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tmp_dis != 0, tmp_res / np.abs(tmp_dis) * 100.0, np.nan)


def build_topography_distance(case_dir, nx, ny, nz_full, dz, mpi_proc):
    """Distance above local immersed-boundary interface, in meters."""
    phi = build_phi(str(case_dir / "phi_functions") + "/", nx, ny, nz_full, mpi_proc)
    intf, _ = build_intf(phi, dz)
    z_profile = np.arange(nz_full) * dz * ZI
    dist_full = z_profile[np.newaxis, np.newaxis, :] - intf[:, :, np.newaxis] * ZI
    return dist_full[:, :, 5:]


def prepare_case(case_name, case_index):
    """Return yB and residual profiles restricted to H through 5H."""
    meta = load_case(case_name, case_index)
    yb = meta["anisotropy"][:, :, :, 1]

    if meta["topo_case"]:
        dist_terms = build_topography_distance(
            meta["case_dir"],
            meta["nx"],
            meta["ny"],
            meta["nz_full"],
            meta["dz"],
            meta["mpi_proc"],
        )
        residual = compute_normalized_tke_residual(
            meta["terms_bdg"],
            meta["topo_case"],
            dist_terms=dist_terms,
        )
        height_mask = (dist_terms >= CANOPY_H_METERS) & (
            dist_terms <= HEIGHT_MAX_MULTIPLIER * CANOPY_H_METERS
        )
        yb_profiles = np.where(height_mask, yb, np.nan).reshape(-1, yb.shape[2])
        residual_profiles = np.where(height_mask, residual, np.nan).reshape(-1, residual.shape[2])
    else:
        residual = compute_normalized_tke_residual(meta["terms_bdg"], meta["topo_case"])
        z_uvp = np.arange(meta["nz_data"]) * meta["dz"] + 0.5 * meta["dz"]
        z_indices = np.where(
            (z_uvp >= CANOPY_H) & (z_uvp <= HEIGHT_MAX_MULTIPLIER * CANOPY_H)
        )[0]
        if z_indices.size == 0:
            raise ValueError(f"No vertical indices found between H and 5H for {case_name}.")

        yb_profiles = yb[:, :, z_indices].reshape(-1, z_indices.size)
        residual_profiles = residual[:, :, z_indices].reshape(-1, z_indices.size)

    return {
        "case": case_name,
        "yb_profiles": yb_profiles.astype(np.float32, copy=False),
        "residual_profiles": residual_profiles.astype(np.float32, copy=False),
    }


def make_yb_bins(yb_min, yb_max, bin_width):
    """Build yB bin edges and centers."""
    n_full_bins = int(np.floor((yb_max - yb_min) / bin_width + 1e-12))
    edges = yb_min + np.arange(n_full_bins + 1) * bin_width
    if edges.size == 0 or not np.isclose(edges[-1], yb_max):
        edges = np.append(edges, yb_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def sample_case_points(case_data, n_sample_profiles, rng, yb_min, yb_max):
    """Sample profiles from one case and return valid points inside the yB interval."""
    n_profiles = case_data["yb_profiles"].shape[0]
    sample_idx = rng.randint(0, n_profiles, size=n_sample_profiles)

    yb = case_data["yb_profiles"][sample_idx].ravel()
    residual = case_data["residual_profiles"][sample_idx].ravel()
    valid = (
        np.isfinite(yb)
        & np.isfinite(residual)
        & (yb >= yb_min)
        & (yb <= yb_max)
    )
    return yb[valid], residual[valid]


def fit_regression(yb, residual, weights):
    """Fit residual = slope * yB + intercept by weighted linear regression."""
    finite = np.isfinite(yb) & np.isfinite(residual) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(finite) < 2:
        return np.nan, np.nan

    x = yb[finite]
    y = residual[finite]
    w = weights[finite]
    sum_w = np.sum(w)
    if sum_w <= 0.0:
        return np.nan, np.nan

    x_mean = np.sum(w * x) / sum_w
    y_mean = np.sum(w * y) / sum_w
    x_centered = x - x_mean
    denominator = np.sum(w * x_centered * x_centered)
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return np.nan, np.nan

    slope = np.sum(w * x_centered * (y - y_mean)) / denominator
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def zero_crossing(slope, intercept):
    """Return yB where the fitted line crosses residual=0."""
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope == 0.0:
        return np.nan
    return float(-intercept / slope)


def weighted_percentile(values, weights, quantiles):
    """Compute weighted percentiles for one finite sample."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(valid) == 0:
        return np.full(len(quantiles), np.nan)

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative_weight = np.cumsum(weights)
    targets = np.asarray(quantiles) / 100.0 * cumulative_weight[-1]
    return np.interp(targets, cumulative_weight, values)


def binned_quantiles(yb, residual, weights, bin_edges, quantiles):
    """Compute equally case-weighted residual quantiles in yB bins."""
    q_low = np.full(bin_edges.size - 1, np.nan)
    q_high = np.full(bin_edges.size - 1, np.nan)

    for i in range(bin_edges.size - 1):
        if i == bin_edges.size - 2:
            in_bin = (yb >= bin_edges[i]) & (yb <= bin_edges[i + 1])
        else:
            in_bin = (yb >= bin_edges[i]) & (yb < bin_edges[i + 1])

        if np.count_nonzero(in_bin) > 0:
            q_low[i], q_high[i] = weighted_percentile(
                residual[in_bin],
                weights[in_bin],
                quantiles,
            )

    return q_low, q_high


def nanmedian_columns(values):
    """Column-wise nanmedian without warnings for columns that are all NaN."""
    med = np.full(values.shape[1], np.nan)
    finite_cols = np.any(np.isfinite(values), axis=0)
    if np.any(finite_cols):
        med[finite_cols] = np.nanmedian(values[:, finite_cols], axis=0)
    return med


def finite_median(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def run_bootstrap(case_profiles, args, rng, bin_edges, bin_centers):
    slopes = np.full(args.n_bootstrap, np.nan)
    intercepts = np.full(args.n_bootstrap, np.nan)
    crossings = np.full(args.n_bootstrap, np.nan)
    n_points = np.zeros(args.n_bootstrap, dtype=int)
    n_active_cases = np.zeros(args.n_bootstrap, dtype=int)
    q_low_all = np.full((args.n_bootstrap, bin_centers.size), np.nan)
    q_high_all = np.full((args.n_bootstrap, bin_centers.size), np.nan)

    for boot in range(args.n_bootstrap):
        yb_parts = []
        residual_parts = []
        weight_parts = []

        for case_data in case_profiles:
            yb, residual = sample_case_points(
                case_data,
                args.n_sample_profiles,
                rng,
                args.yb_min,
                args.yb_max,
            )
            if yb.size == 0:
                continue

            yb_parts.append(yb)
            residual_parts.append(residual)
            weight_parts.append(np.full(yb.size, 1.0 / yb.size))

        if not yb_parts:
            continue
        yb_pool = np.concatenate(yb_parts)
        residual_pool = np.concatenate(residual_parts)
        weights_pool = np.concatenate(weight_parts)
        n_points[boot] = yb_pool.size
        n_active_cases[boot] = len(yb_parts)

        slopes[boot], intercepts[boot] = fit_regression(yb_pool, residual_pool, weights_pool)
        crossings[boot] = zero_crossing(slopes[boot], intercepts[boot])
        q_low_all[boot], q_high_all[boot] = binned_quantiles(
            yb_pool,
            residual_pool,
            weights_pool,
            bin_edges,
            (args.iq_low, args.iq_high),
        )

        if (boot + 1) % 10 == 0 or boot == args.n_bootstrap - 1:
            print(f"  completed {boot + 1}/{args.n_bootstrap} bootstraps")

    median_slope = finite_median(slopes)
    median_intercept = finite_median(intercepts)
    line_yb = np.linspace(args.yb_min, args.yb_max, 200)
    median_line = median_slope * line_yb + median_intercept
    median_line_crossing = zero_crossing(median_slope, median_intercept)

    return {
        "slopes": slopes,
        "intercepts": intercepts,
        "crossings": crossings,
        "n_points": n_points,
        "n_active_cases": n_active_cases,
        "line_yb": line_yb,
        "median_line": median_line,
        "median_slope": median_slope,
        "median_intercept": median_intercept,
        "median_line_crossing": median_line_crossing,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "q_low_all": q_low_all,
        "q_high_all": q_high_all,
        "q_low_median": nanmedian_columns(q_low_all),
        "q_high_median": nanmedian_columns(q_high_all),
    }


def save_crossovers(result, output_dir):
    path = output_dir / "bootstrapIva_crossovers.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["bootstrap", "n_points", "n_active_cases", "slope", "intercept", "zero_crossing"]
        )
        for i in range(result["crossings"].size):
            writer.writerow(
                [
                    i + 1,
                    result["n_points"][i],
                    result["n_active_cases"][i],
                    f"{result['slopes'][i]:.10g}",
                    f"{result['intercepts'][i]:.10g}",
                    f"{result['crossings'][i]:.10g}",
                ]
            )
    return path


def save_plot_data(result, output_dir):
    path = output_dir / "bootstrapIva_plot_data.npz"
    np.savez(
        path,
        line_yb=result["line_yb"],
        median_line=result["median_line"],
        median_slope=result["median_slope"],
        median_intercept=result["median_intercept"],
        median_line_crossing=result["median_line_crossing"],
        bin_edges=result["bin_edges"],
        bin_centers=result["bin_centers"],
        q_low_median=result["q_low_median"],
        q_high_median=result["q_high_median"],
        slopes=result["slopes"],
        intercepts=result["intercepts"],
        crossings=result["crossings"],
        n_points=result["n_points"],
        n_active_cases=result["n_active_cases"],
        q_low_all=result["q_low_all"],
        q_high_all=result["q_high_all"],
    )
    return path


def save_plot_data_csv(result, output_dir):
    path = output_dir / "bootstrapIva_plot_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "yB_bin_left",
                "yB_bin_right",
                "yB_bin_center",
                "median_regression_at_bin_center",
                "q_low_median",
                "q_high_median",
            ]
        )
        for i, center in enumerate(result["bin_centers"]):
            writer.writerow(
                [
                    f"{result['bin_edges'][i]:.10g}",
                    f"{result['bin_edges'][i + 1]:.10g}",
                    f"{center:.10g}",
                    f"{result['median_slope'] * center + result['median_intercept']:.10g}",
                    f"{result['q_low_median'][i]:.10g}",
                    f"{result['q_high_median'][i]:.10g}",
                ]
            )
    return path


def save_summary(result, args, output_dir):
    path = output_dir / "bootstrapIva_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_bootstrap", args.n_bootstrap])
        writer.writerow(["n_sample_profiles_per_case", args.n_sample_profiles])
        writer.writerow(["height_range", "H_to_5H"])
        writer.writerow(["regression_fit", "equally_case_weighted_linear_regression"])
        writer.writerow(["quantile_weighting", "equally_case_weighted"])
        writer.writerow(["yb_min", f"{args.yb_min:.10g}"])
        writer.writerow(["yb_max", f"{args.yb_max:.10g}"])
        writer.writerow(["yb_bin_width", f"{args.yb_bin_width:.10g}"])
        writer.writerow(["quantile_low_percent", f"{args.iq_low:.10g}"])
        writer.writerow(["quantile_high_percent", f"{args.iq_high:.10g}"])
        writer.writerow(["median_slope", f"{result['median_slope']:.10g}"])
        writer.writerow(["median_intercept", f"{result['median_intercept']:.10g}"])
        writer.writerow(["median_line_zero_crossing", f"{result['median_line_crossing']:.10g}"])
        writer.writerow(["bootstrap_zero_crossing_median", f"{finite_median(result['crossings']):.10g}"])
    return path


def plot_regressionline_case_medians(ax):
    """Overlay the saved per-case median profiles from RegressionLine.py."""
    for case_name, color, label in zip(CASES, PROFILE_COLORS, PROFILE_LABELS):
        prof = np.load(PROFILE_DIR / f"ResTKEvsYB_{case_name}_Q.npy")
        ax.plot(prof[3, :], prof[0, :], c=color, label=label)
        ax.fill_between(
            prof[3, :],
            np.array(prof[1, :]),
            np.array(prof[2, :]),
            alpha=0.1,
            color=color,
        )


def extend_center_values_to_edges(bin_edges, bin_centers, values):
    """Extend center-sampled bin values smoothly to the first and last bin edges."""
    return (
        np.concatenate(([bin_edges[0]], bin_centers, [bin_edges[-1]])),
        np.concatenate(([values[0]], values, [values[-1]])),
    )


def plot_result(result, output_dir, show=False):
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 5))

    plot_regressionline_case_medians(ax)

    q_x, q_low = extend_center_values_to_edges(
        result["bin_edges"],
        result["bin_centers"],
        result["q_low_median"],
    )
    _, q_high = extend_center_values_to_edges(
        result["bin_edges"],
        result["bin_centers"],
        result["q_high_median"],
    )
    ax.fill_between(
        q_x,
        q_low,
        q_high,
        color="0.65",
        alpha=0.35,
        label="_nolegend_",
    )
    ax.plot(
        result["line_yb"],
        result["median_line"],
        color="black",
        linewidth=2.0,
        label="_nolegend_",
    )
    ax.axhline(0, color="k", linestyle=":")
    ax.axvline(result["median_line_crossing"], color="k", linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"$y_B$", fontsize=18)
    ax.set_ylabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=21)
    ax.set_xlim(*YB_PLOT_LIMITS)
    ax.set_ylim(*TKE_PLOT_LIMITS)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

    path = output_dir / "bootstrapIva_regression_quantiles.png"
    fig.savefig(path, dpi=300, edgecolor="white", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    bin_edges, bin_centers = make_yb_bins(args.yb_min, args.yb_max, args.yb_bin_width)

    case_profiles = []
    for case_index, case_name in enumerate(CASES):
        print(f"Loading {case_name} ({case_index + 1}/{len(CASES)})")
        case_data = prepare_case(case_name, case_index)
        case_profiles.append(case_data)
        print(f"  profiles: {case_data['yb_profiles'].shape[0]}")

    print(
        f"\nRunning {args.n_bootstrap} bootstraps with "
        f"{args.n_sample_profiles} profiles per case."
    )
    print(
        f"Height range: H to {HEIGHT_MAX_MULTIPLIER:g}H; "
        f"yB fit/bin range: {args.yb_min:g} to {args.yb_max:g}; "
        f"bin width: {args.yb_bin_width:g}"
    )
    print("Regression: equally case-weighted linear regression")

    result = run_bootstrap(case_profiles, args, rng, bin_edges, bin_centers)

    crossovers_path = save_crossovers(result, args.output_dir)
    plot_npz_path = save_plot_data(result, args.output_dir)
    plot_csv_path = save_plot_data_csv(result, args.output_dir)
    summary_path = save_summary(result, args, args.output_dir)
    figure_path = plot_result(result, args.output_dir, show=args.show)

    print("\nMedian regression:")
    print(f"  slope: {result['median_slope']:.6g}")
    print(f"  intercept: {result['median_intercept']:.6g}")
    print(f"  zero crossing: {result['median_line_crossing']:.6g}")
    print(f"\nSaved crossovers to {crossovers_path}")
    print(f"Saved reusable plot arrays to {plot_npz_path}")
    print(f"Saved reusable plot CSV to {plot_csv_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
