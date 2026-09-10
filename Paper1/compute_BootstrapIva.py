#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pooled-profile bootstrap regression for yB-vs-normalized-TKE-residual.

Each bootstrap realization samples 10,000 x-y profiles with replacement from
each study case, extracts each sampled profile from canopy top to 10 canopy
heights, pools all valid points from all cases, and fits one regression line

    TKE_residual = m * yB + q

The final plotted line is the pointwise median of the 1,000 fitted lines. The
shading is the interquartile range of those lines. Crossover values are saved
for all bootstrap realizations along with their median and interquartile range.
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
LABELS = ["g1200", "g800", "g400", "i1200", "i800", "i400", "ATTO", "Sinusoidal", "Flat"]

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

ZI = 1000.0
CANOPY_H = 39.0 / ZI
CANOPY_H_METERS = CANOPY_H * ZI

N_BOOTSTRAP = 1000
N_SAMPLE_PROFILES = 10000
RANDOM_SEED = 20260909
MIN_VALID_POINTS = 10
REQUIRE_POSITIVE_TO_NEGATIVE_CROSSING = True
YB_FIT_LIMITS = (0.25, 0.45)

YB_LINE = np.linspace(0.0, np.sqrt(3.0) / 2.0, 300)
YB_PLOT_LIMITS = (0.15, 0.60)
TKE_PLOT_LIMITS = (-70, 250)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Bootstrap one pooled yB-vs-normalized-TKE-residual regression line "
            "from profile resampling across all Paper1 cases."
        )
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-sample-profiles", type=int, default=N_SAMPLE_PROFILES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--yb-min",
        type=float,
        default=YB_FIT_LIMITS[0],
        help="Minimum yB value included in each regression fit. Default includes no lower cutoff.",
    )
    parser.add_argument(
        "--yb-max",
        type=float,
        default=YB_FIT_LIMITS[1],
        help="Maximum yB value included in each regression fit. Default includes no upper cutoff.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving it.")
    args = parser.parse_args()
    if args.yb_min is not None and args.yb_max is not None and args.yb_min >= args.yb_max:
        parser.error("--yb-min must be smaller than --yb-max")
    return args


def load_case(case_name, case_index):
    """Load TKE terms and anisotropy arrays using the Paper1 data paths."""
    import xarray as xr

    if case_index < 6:
        case_dir = GAP_PATCH_DATA_ROOT / case_name
        terms_bdg = xr.open_dataarray(case_dir / "TKE_terms.nc").data
        anisotropy = xr.open_dataarray(case_dir / "anisotropy.nc").data
        lx = ly = 2.0 * np.pi
        lz = 1.0
        topo_case = False
        mpi_proc = None
    else:
        case_dir = TOPO_DATA_ROOT / case_name
        terms_bdg = xr.open_dataarray(case_dir / "TKE_terms.nc").data
        anisotropy = xr.open_dataarray(case_dir / "anisotropy.nc").data
        lx = ly = 2.88
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
        "dx": lx / nx,
        "dy": ly / ny,
        "dz": dz,
        "topo_case": topo_case,
        "mpi_proc": mpi_proc,
    }


def compute_normalized_tke_residual(terms_bdg, topo_case, dist_terms=None):
    """Compute normalized residual using the same conventions as the existing scripts."""
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
    """Distance above the local immersed-boundary interface, in meters."""
    phi = build_phi(str(case_dir / "phi_functions") + "/", nx, ny, nz_full, mpi_proc)
    intf, _ = build_intf(phi, dz)
    z_profile = np.arange(nz_full) * dz * ZI
    dist_full = z_profile[np.newaxis, np.newaxis, :] - intf[:, :, np.newaxis] * ZI
    return dist_full[:, :, 5:]


def empty_stats():
    """Regression sufficient statistics for pooled valid points."""
    return {
        "n": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_xx": 0.0,
        "sum_xy": 0.0,
        "sum_yy": 0.0,
    }


def compute_column_stats(x_vals, y_vals, height_mask, yb_min=None, yb_max=None):
    """Compute per-x-y-profile regression statistics over valid height levels."""
    valid = height_mask & np.isfinite(x_vals) & np.isfinite(y_vals)
    if yb_min is not None:
        valid &= x_vals >= yb_min
    if yb_max is not None:
        valid &= x_vals <= yb_max

    x_clean = np.where(valid, x_vals, 0.0)
    y_clean = np.where(valid, y_vals, 0.0)

    return {
        "n": np.sum(valid, axis=2).astype(np.int64).ravel(),
        "sum_x": np.sum(x_clean, axis=2).ravel(),
        "sum_y": np.sum(y_clean, axis=2).ravel(),
        "sum_xx": np.sum(x_clean * x_clean, axis=2).ravel(),
        "sum_xy": np.sum(x_clean * y_clean, axis=2).ravel(),
        "sum_yy": np.sum(y_clean * y_clean, axis=2).ravel(),
    }


def add_sampled_columns_to_stats(stats, column_stats, sampled_columns):
    """Add sampled profile statistics to a pooled realization."""
    weights = np.bincount(sampled_columns, minlength=column_stats["n"].size)
    stats["n"] += int(np.dot(weights, column_stats["n"]))
    stats["sum_x"] += float(np.dot(weights, column_stats["sum_x"]))
    stats["sum_y"] += float(np.dot(weights, column_stats["sum_y"]))
    stats["sum_xx"] += float(np.dot(weights, column_stats["sum_xx"]))
    stats["sum_xy"] += float(np.dot(weights, column_stats["sum_xy"]))
    stats["sum_yy"] += float(np.dot(weights, column_stats["sum_yy"]))
    return stats


def fit_from_stats(stats):
    """
    Fit y = m*x + q from sufficient statistics.

    This matches the standardized total-least-squares fit used in the existing
    bootstrap script, but avoids rebuilding the full sampled point cloud.
    """
    if stats["n"] < MIN_VALID_POINTS:
        return np.nan, np.nan

    n = float(stats["n"])
    x_mean = stats["sum_x"] / n
    y_mean = stats["sum_y"] / n

    sxx = stats["sum_xx"] - stats["sum_x"] ** 2 / n
    syy = stats["sum_yy"] - stats["sum_y"] ** 2 / n
    sxy = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n

    if (
        sxx <= 0.0
        or syy <= 0.0
        or not np.isfinite(sxx)
        or not np.isfinite(syy)
        or not np.isfinite(sxy)
        or np.isclose(sxy, 0.0)
    ):
        return np.nan, np.nan

    slope = np.sign(sxy) * np.sqrt(syy / n) / np.sqrt(sxx / n)
    intercept = y_mean - slope * x_mean

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan, np.nan

    return slope, intercept


def crossing_from_fit(slope, intercept):
    """Return yB crossing for the fitted line."""
    valid_direction = (
        slope < 0.0 if REQUIRE_POSITIVE_TO_NEGATIVE_CROSSING else slope != 0.0
    )
    if not np.isfinite(slope) or not np.isfinite(intercept) or not valid_direction:
        return np.nan
    return -intercept / slope


def prepare_case(case_name, case_index, yb_min=None, yb_max=None):
    """Load one case and reduce it to per-profile statistics."""
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
        tke_norm = compute_normalized_tke_residual(
            meta["terms_bdg"], meta["topo_case"], dist_terms=dist_terms
        )
        height_mask = (dist_terms >= CANOPY_H_METERS) & (
            dist_terms <= 10.0 * CANOPY_H_METERS
        )
    else:
        tke_norm = compute_normalized_tke_residual(meta["terms_bdg"], meta["topo_case"])
        z_uvp = np.arange(meta["nz_data"]) * meta["dz"] + 0.5 * meta["dz"]
        z_indices = np.where((z_uvp >= CANOPY_H) & (z_uvp <= 10.0 * CANOPY_H))[0]
        if z_indices.size == 0:
            raise ValueError(f"No vertical indices found between H and 10H for {case_name}.")

        yb = yb[:, :, z_indices]
        tke_norm = tke_norm[:, :, z_indices]
        height_mask = np.ones_like(yb, dtype=bool)

    column_stats = compute_column_stats(
        yb,
        tke_norm,
        height_mask,
        yb_min=yb_min,
        yb_max=yb_max,
    )
    n_valid_columns = int(np.count_nonzero(column_stats["n"] > 0))
    n_valid_points = int(np.sum(column_stats["n"]))

    return {
        "case": case_name,
        "label": LABELS[case_index],
        "n_columns": meta["nx"] * meta["ny"],
        "n_valid_columns": n_valid_columns,
        "n_valid_points": n_valid_points,
        "column_stats": column_stats,
    }


def run_bootstrap(case_data, n_bootstrap, n_sample_profiles, rng):
    """Run pooled-profile bootstrap fits across all cases."""
    slopes = np.full(n_bootstrap, np.nan)
    intercepts = np.full(n_bootstrap, np.nan)
    crossovers = np.full(n_bootstrap, np.nan)
    n_points = np.zeros(n_bootstrap, dtype=np.int64)

    for boot in range(n_bootstrap):
        stats = empty_stats()
        for case in case_data:
            sampled_columns = rng.integers(
                0,
                case["n_columns"],
                size=n_sample_profiles,
            )
            add_sampled_columns_to_stats(stats, case["column_stats"], sampled_columns)

        slope, intercept = fit_from_stats(stats)
        slopes[boot] = slope
        intercepts[boot] = intercept
        crossovers[boot] = crossing_from_fit(slope, intercept)
        n_points[boot] = stats["n"]

        if (boot + 1) % 100 == 0 or boot == n_bootstrap - 1:
            print(f"  completed {boot + 1}/{n_bootstrap} bootstrap fits")

    line_ensemble = slopes[:, np.newaxis] * YB_LINE[np.newaxis, :] + intercepts[:, np.newaxis]
    line_median = np.nanmedian(line_ensemble, axis=0)
    line_q25 = np.nanpercentile(line_ensemble, 25, axis=0)
    line_q75 = np.nanpercentile(line_ensemble, 75, axis=0)

    return {
        "slopes": slopes,
        "intercepts": intercepts,
        "crossovers": crossovers,
        "n_points": n_points,
        "line_yb": YB_LINE,
        "line_median": line_median,
        "line_q25": line_q25,
        "line_q75": line_q75,
        "n_valid_bootstrap": int(np.count_nonzero(np.isfinite(crossovers))),
        "crossover_median": float(np.nanmedian(crossovers)),
        "crossover_q25": float(np.nanpercentile(crossovers, 25)),
        "crossover_q75": float(np.nanpercentile(crossovers, 75)),
        "slope_median": float(np.nanmedian(slopes)),
        "intercept_median": float(np.nanmedian(intercepts)),
    }


def save_crossovers_csv(result, output_dir):
    """Save one row per bootstrap fit, with crossover summary columns included."""
    path = output_dir / "bootstrapIva_crossovers.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bootstrap",
                "n_points",
                "slope",
                "intercept",
                "crossover",
                "crossover_median",
                "crossover_q25",
                "crossover_q75",
            ]
        )
        for boot, (n_points, slope, intercept, crossover) in enumerate(
            zip(
                result["n_points"],
                result["slopes"],
                result["intercepts"],
                result["crossovers"],
            ),
            start=1,
        ):
            writer.writerow(
                [
                    boot,
                    n_points,
                    f"{slope:.10g}",
                    f"{intercept:.10g}",
                    f"{crossover:.10g}",
                    f"{result['crossover_median']:.10g}",
                    f"{result['crossover_q25']:.10g}",
                    f"{result['crossover_q75']:.10g}",
                ]
            )
    return path


def save_summary_csv(
    case_data,
    result,
    output_dir,
    n_bootstrap,
    n_sample_profiles,
    seed,
    yb_min,
    yb_max,
):
    """Save concise run metadata and crossover summary."""
    path = output_dir / "bootstrapIva_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_bootstrap", n_bootstrap])
        writer.writerow(["n_sample_profiles_per_case", n_sample_profiles])
        writer.writerow(["random_seed", seed])
        writer.writerow(["height_range", "H_to_10H"])
        writer.writerow(["yb_fit_min", "" if yb_min is None else f"{yb_min:.10g}"])
        writer.writerow(["yb_fit_max", "" if yb_max is None else f"{yb_max:.10g}"])
        writer.writerow(["n_cases", len(case_data)])
        writer.writerow(["n_valid_bootstrap_crossovers", result["n_valid_bootstrap"]])
        writer.writerow(["crossover_median", f"{result['crossover_median']:.10g}"])
        writer.writerow(["crossover_q25", f"{result['crossover_q25']:.10g}"])
        writer.writerow(["crossover_q75", f"{result['crossover_q75']:.10g}"])
        writer.writerow(["slope_median", f"{result['slope_median']:.10g}"])
        writer.writerow(["intercept_median", f"{result['intercept_median']:.10g}"])
        for case in case_data:
            writer.writerow([f"{case['label']}_valid_profiles", case["n_valid_columns"]])
            writer.writerow([f"{case['label']}_valid_points", case["n_valid_points"]])
    return path


def save_line_csv(result, output_dir):
    """Save the median regression line and its interquartile band."""
    path = output_dir / "bootstrapIva_regression_line.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["yB", "line_median", "line_q25", "line_q75"])
        for row in zip(
            result["line_yb"],
            result["line_median"],
            result["line_q25"],
            result["line_q75"],
        ):
            writer.writerow([f"{value:.10g}" for value in row])
    return path


def save_npz(result, output_dir):
    """Save detailed arrays for reuse without reparsing CSV files."""
    path = output_dir / "bootstrapIva_results.npz"
    np.savez(
        path,
        slopes=result["slopes"],
        intercepts=result["intercepts"],
        crossovers=result["crossovers"],
        n_points=result["n_points"],
        line_yb=result["line_yb"],
        line_median=result["line_median"],
        line_q25=result["line_q25"],
        line_q75=result["line_q75"],
    )
    return path


def plot_result(result, output_dir, show=False):
    """Plot the median pooled bootstrap regression line and interquartile shading."""
    fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(8, 5))

    axs.fill_between(
        result["line_yb"],
        result["line_q25"],
        result["line_q75"],
        color="0.65",
        alpha=0.35,
        label="IQR",
    )
    axs.plot(
        result["line_yb"],
        result["line_median"],
        color="black",
        linewidth=2.0,
        label="Median bootstrap line",
    )
    axs.axhline(0, color="k", linestyle=":")
    axs.axvline(result["crossover_median"], color="k", linestyle="--", linewidth=1.0)

    axs.set_xlabel(r"$y_B$", fontsize=18)
    axs.set_ylabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=21)
    axs.set_xlim(*YB_PLOT_LIMITS)
    axs.set_ylim(*TKE_PLOT_LIMITS)
    axs.tick_params(axis="x", labelsize=12)
    axs.tick_params(axis="y", labelsize=12)
    axs.legend(loc="best", fontsize=12)

    path = output_dir / "bootstrapIva_regression_line.png"
    fig.savefig(path, dpi=300, edgecolor="white", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    case_data = []
    for case_index, case_name in enumerate(CASES):
        print(f"Preparing {case_name} ({case_index + 1}/{len(CASES)})")
        prepared = prepare_case(
            case_name,
            case_index,
            yb_min=args.yb_min,
            yb_max=args.yb_max,
        )
        case_data.append(prepared)
        print(
            f"  valid profiles={prepared['n_valid_columns']}, "
            f"valid points={prepared['n_valid_points']}"
        )

    print(
        f"\nRunning {args.n_bootstrap} pooled bootstrap fits with "
        f"{args.n_sample_profiles} profiles per case."
    )
    result = run_bootstrap(
        case_data,
        args.n_bootstrap,
        args.n_sample_profiles,
        rng,
    )

    crossovers_path = save_crossovers_csv(result, args.output_dir)
    summary_path = save_summary_csv(
        case_data,
        result,
        args.output_dir,
        args.n_bootstrap,
        args.n_sample_profiles,
        args.seed,
        args.yb_min,
        args.yb_max,
    )
    line_csv_path = save_line_csv(result, args.output_dir)
    npz_path = save_npz(result, args.output_dir)
    figure_path = plot_result(result, args.output_dir, show=args.show)

    print("\nPooled bootstrap crossover yB median/IQR:")
    print(
        f"  {result['crossover_median']:.4f} "
        f"[{result['crossover_q25']:.4f}, {result['crossover_q75']:.4f}]"
    )
    print(f"\nSaved crossover CSV to {crossovers_path}")
    print(f"Saved summary CSV to {summary_path}")
    print(f"Saved line CSV to {line_csv_path}")
    print(f"Saved detailed arrays to {npz_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
