#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-one-out cross validation for yB-vs-normalized-TKE-residual crossings.

For each held-out case, the script fits

    TKE_residual = m * yB + q

using all valid points from the other eight cases, then compares that predicted
crossing yB = -q / m with the crossing obtained from the held-out case alone.
"""

from pathlib import Path
import sys

import numpy as np
import xarray as xr


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
    "leave1out_outputs"
)

ZI = 1000.0
CANOPY_H = 39.0 / ZI
CANOPY_H_METERS = CANOPY_H * ZI
MIN_VALID_POINTS = 10
REQUIRE_POSITIVE_TO_NEGATIVE_CROSSING = True
USE_CASE_BALANCED_WEIGHTS = True


def load_case(case_name, case_index):
    """Load TKE terms and anisotropy arrays using the Paper1 data paths."""
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
    """Compute normalized residual using the same conventions as the bootstrap script."""
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
    """Regression sufficient statistics for pooled data."""
    return {
        "n": 0,
        "sum_w": 0.0,
        "sum_wx": 0.0,
        "sum_wy": 0.0,
        "sum_wxx": 0.0,
        "sum_wxy": 0.0,
        "sum_wyy": 0.0,
    }


def add_values_to_stats(stats, x_vals, y_vals, point_weight=1.0):
    """Accumulate valid x/y values into regression sufficient statistics."""
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return stats

    x = x_vals[valid]
    y = y_vals[valid]
    stats["n"] += n_valid
    stats["sum_w"] += point_weight * n_valid
    stats["sum_wx"] += point_weight * float(np.sum(x))
    stats["sum_wy"] += point_weight * float(np.sum(y))
    stats["sum_wxx"] += point_weight * float(np.sum(x * x))
    stats["sum_wxy"] += point_weight * float(np.sum(x * y))
    stats["sum_wyy"] += point_weight * float(np.sum(y * y))
    return stats


def combine_stats(stats_list, case_balanced_weights=False):
    """Combine case-level sufficient statistics into one pooled-data fit."""
    combined = empty_stats()
    for stats in stats_list:
        if case_balanced_weights:
            if stats["n"] == 0:
                continue
            scale = 1.0 / stats["n"]
        else:
            scale = 1.0

        combined["n"] += stats["n"]
        combined["sum_w"] += scale * stats["sum_w"]
        combined["sum_wx"] += scale * stats["sum_wx"]
        combined["sum_wy"] += scale * stats["sum_wy"]
        combined["sum_wxx"] += scale * stats["sum_wxx"]
        combined["sum_wxy"] += scale * stats["sum_wxy"]
        combined["sum_wyy"] += scale * stats["sum_wyy"]
    return combined


def fit_from_stats(stats):
    """Fit y = m*x + q from sufficient statistics."""
    # if stats["n"] < MIN_VALID_POINTS or stats["sum_w"] <= 0:
    #     return np.nan, np.nan

    # sum_w = stats["sum_w"]
    # denom = stats["sum_wxx"] - stats["sum_wx"] ** 2 / sum_w
    # if denom == 0 or not np.isfinite(denom):
    #     return np.nan, np.nan

    # slope = (stats["sum_wxy"] - stats["sum_wx"] * stats["sum_wy"] / sum_w) / denom
    # intercept = stats["sum_wy"] / sum_w - slope * stats["sum_wx"] / sum_w
    # return slope, intercept
    
    if stats["n"] < MIN_VALID_POINTS or stats["sum_w"] <= 0:
        return np.nan, np.nan

    sum_w = stats["sum_w"]

    x_mean = stats["sum_wx"] / sum_w
    y_mean = stats["sum_wy"] / sum_w

    sxx = stats["sum_wxx"] - stats["sum_wx"] ** 2 / sum_w
    syy = stats["sum_wyy"] - stats["sum_wy"] ** 2 / sum_w
    sxy = stats["sum_wxy"] - stats["sum_wx"] * stats["sum_wy"] / sum_w

    if (
        sxx <= 0.0
        or syy <= 0.0
        or not np.isfinite(sxx)
        or not np.isfinite(syy)
        or not np.isfinite(sxy)
    ):
        return np.nan, np.nan

    x_scale = np.sqrt(sxx / sum_w)
    y_scale = np.sqrt(syy / sum_w)

    # Standardized TLS is the principal-axis fit after x/y are scaled
    # to unit variance. In standardized space, the slope is +/-1.
    if np.isclose(sxy, 0.0):
        return np.nan, np.nan

    slope_norm = np.sign(sxy)
    slope = slope_norm * y_scale / x_scale
    intercept = y_mean - slope * x_mean

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan, np.nan

    return slope, intercept


def crossing_from_fit(slope, intercept):
    """Return yB crossing for positive-to-negative regressions."""
    valid_direction = (
        slope < 0.0 if REQUIRE_POSITIVE_TO_NEGATIVE_CROSSING else slope != 0.0
    )
    if not np.isfinite(slope) or not np.isfinite(intercept) or not valid_direction:
        return np.nan
    return -intercept / slope


def compute_case_stats(case_name, case_index):
    """Build regression stats for all valid points in one case."""
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
        height_mask = (dist_terms >= CANOPY_H_METERS) & (dist_terms <= 10.0 * CANOPY_H_METERS)
        stats = add_values_to_stats(empty_stats(), yb[height_mask], tke_norm[height_mask])
    else:
        tke_norm = compute_normalized_tke_residual(meta["terms_bdg"], meta["topo_case"])
        z_uvp = np.arange(meta["nz_data"]) * meta["dz"] + 0.5 * meta["dz"]
        z_indices = np.where((z_uvp >= CANOPY_H) & (z_uvp <= 10.0 * CANOPY_H))[0]
        if z_indices.size == 0:
            raise ValueError(f"No vertical indices found between H and 5H for {case_name}.")
        stats = add_values_to_stats(empty_stats(), yb[:, :, z_indices], tke_norm[:, :, z_indices])

    slope, intercept = fit_from_stats(stats)
    crossing = crossing_from_fit(slope, intercept)

    return {
        "case": case_name,
        "label": LABELS[case_index],
        "stats": stats,
        "case_slope": slope,
        "case_intercept": intercept,
        "case_crossing": crossing,
    }


def run_leave_one_out(case_results):
    """Fit all leave-one-out training clouds and compare against held-out cases."""
    loo_results = []
    for heldout_index, heldout in enumerate(case_results):
        train_stats_list = (
            result["stats"]
            for index, result in enumerate(case_results)
            if index != heldout_index
        )
        train_stats = combine_stats(
            train_stats_list,
            case_balanced_weights=USE_CASE_BALANCED_WEIGHTS,
        )
        train_slope, train_intercept = fit_from_stats(train_stats)
        train_crossing = crossing_from_fit(train_slope, train_intercept)
        error = train_crossing - heldout["case_crossing"]

        loo_results.append(
            {
                "heldout_case": heldout["case"],
                "heldout_label": heldout["label"],
                "train_n": train_stats["n"],
                "heldout_n": heldout["stats"]["n"],
                "train_slope": train_slope,
                "train_intercept": train_intercept,
                "predicted_crossing": train_crossing,
                "heldout_slope": heldout["case_slope"],
                "heldout_intercept": heldout["case_intercept"],
                "heldout_crossing": heldout["case_crossing"],
                "crossing_error": error,
                "abs_crossing_error": abs(error) if np.isfinite(error) else np.nan,
            }
        )
    return loo_results


def save_case_crossings(case_results):
    """Save each individual case fit and crossing."""
    path = OUTPUT_DIR / "case_crossings_10_w.csv"
    rows = [
        "case,label,n_points,slope,intercept,crossover\n",
    ]
    for result in case_results:
        rows.append(
            f"{result['case']},{result['label']},{result['stats']['n']},"
            f"{result['case_slope']:.10g},{result['case_intercept']:.10g},"
            f"{result['case_crossing']:.10g}\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def save_leave_one_out_results(loo_results):
    """Save the nine leave-one-out estimates."""
    path = OUTPUT_DIR / "leave1out_cross_validation_10_w.csv"
    rows = [
        "case_balanced_weights,heldout_case,heldout_label,train_n,heldout_n,train_slope,train_intercept,"
        "predicted_crossing,heldout_slope,heldout_intercept,heldout_crossing,"
        "crossing_error,abs_crossing_error\n",
    ]
    for result in loo_results:
        rows.append(
            f"{USE_CASE_BALANCED_WEIGHTS},{result['heldout_case']},{result['heldout_label']},"
            f"{result['train_n']},{result['heldout_n']},"
            f"{result['train_slope']:.10g},{result['train_intercept']:.10g},"
            f"{result['predicted_crossing']:.10g},"
            f"{result['heldout_slope']:.10g},{result['heldout_intercept']:.10g},"
            f"{result['heldout_crossing']:.10g},"
            f"{result['crossing_error']:.10g},{result['abs_crossing_error']:.10g}\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def save_summary(loo_results):
    """Save robustness summary for the nine leave-one-out estimates."""
    predicted = np.array([result["predicted_crossing"] for result in loo_results])
    heldout = np.array([result["heldout_crossing"] for result in loo_results])
    abs_error = np.array([result["abs_crossing_error"] for result in loo_results])

    path = OUTPUT_DIR / "leave1out_summary_10_w.csv"
    rows = [
        "metric,value\n",
        f"case_balanced_weights,{USE_CASE_BALANCED_WEIGHTS}\n",
        f"n_leave1out_estimates,{len(loo_results)}\n",
        f"predicted_crossing_median,{np.nanmedian(predicted):.10g}\n",
        f"predicted_crossing_q25,{np.nanpercentile(predicted, 25):.10g}\n",
        f"predicted_crossing_q75,{np.nanpercentile(predicted, 75):.10g}\n",
        f"heldout_crossing_median,{np.nanmedian(heldout):.10g}\n",
        f"heldout_crossing_q25,{np.nanpercentile(heldout, 25):.10g}\n",
        f"heldout_crossing_q75,{np.nanpercentile(heldout, 75):.10g}\n",
        f"median_abs_error,{np.nanmedian(abs_error):.10g}\n",
        f"max_abs_error,{np.nanmax(abs_error):.10g}\n",
    ]
    path.write_text("".join(rows), encoding="utf-8")


def print_summary(loo_results):
    """Print the main leave-one-out diagnostics."""
    predicted = np.array([result["predicted_crossing"] for result in loo_results])
    heldout = np.array([result["heldout_crossing"] for result in loo_results])
    abs_error = np.array([result["abs_crossing_error"] for result in loo_results])

    print("\nLeave-one-out results:")
    for result in loo_results:
        print(
            f"  {result['heldout_label']:>10s}: predicted={result['predicted_crossing']:.4f}, "
            f"heldout={result['heldout_crossing']:.4f}, "
            f"error={result['crossing_error']:.4f}"
        )

    print("\nRobustness summary:")
    print(
        "  predicted crossing median/IQR: "
        f"{np.nanmedian(predicted):.4f} "
        f"[{np.nanpercentile(predicted, 25):.4f}, "
        f"{np.nanpercentile(predicted, 75):.4f}]"
    )
    print(
        "  held-out crossing median/IQR: "
        f"{np.nanmedian(heldout):.4f} "
        f"[{np.nanpercentile(heldout, 25):.4f}, "
        f"{np.nanpercentile(heldout, 75):.4f}]"
    )
    print(f"  median absolute error: {np.nanmedian(abs_error):.4f}")
    print(f"  max absolute error: {np.nanmax(abs_error):.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_results = []
    for case_index, case_name in enumerate(CASES):
        print(f"Computing case fit for {case_name} ({case_index + 1}/{len(CASES)})")
        case_result = compute_case_stats(case_name, case_index)
        case_results.append(case_result)
        print(
            f"  crossing yB: {case_result['case_crossing']:.4f} "
            f"(N={case_result['stats']['n']})"
        )

    loo_results = run_leave_one_out(case_results)
    save_case_crossings(case_results)
    save_leave_one_out_results(loo_results)
    save_summary(loo_results)
    print_summary(loo_results)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
