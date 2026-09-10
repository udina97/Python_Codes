#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap yB-vs-normalized-TKE-residual regression analysis.

For each case, this script repeatedly samples x-y columns with replacement,
extracts the vertical profile from canopy top to 5 canopy heights, fits

    TKE_residual = m * yB + q

to the full sampled point cloud, and stores the zero-crossing yB = -q / m.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
    "bootstrap_regression_outputs"
)

ZI = 1000.0
CANOPY_H = 39.0 / ZI
CANOPY_H_METERS = CANOPY_H * ZI

N_BOOTSTRAP = 1000 #1000
N_SAMPLE_COLUMNS = 10000 #10000
RANDOM_SEED = 20260902
MIN_VALID_POINTS = 10
REQUIRE_POSITIVE_TO_NEGATIVE_CROSSING = True

YB_LINE = np.linspace(0.0, np.sqrt(3.0) / 2.0, 300)
YB_PLOT_LIMITS = (0.15, 0.60)
TKE_PLOT_LIMITS = (-70, 250)


def load_case(case_name, case_index):
    """Load TKE terms and anisotropy arrays using the original script paths."""
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
    """
    Reproduce the normalized TKE residual conventions used by
    Compute_TKEres_vs_YB.py.
    """
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


def linear_fit(x_vals, y_vals):
    """Fit y = m*x + q while ignoring invalid points."""
    # valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    # if np.count_nonzero(valid) < MIN_VALID_POINTS:
    #     return np.nan, np.nan

    # x = x_vals[valid]
    # y = y_vals[valid]
    # x_mean = np.mean(x)
    # y_mean = np.mean(y)
    # denom = np.sum((x - x_mean) ** 2)
    # if denom == 0 or not np.isfinite(denom):
    #     return np.nan, np.nan

    # slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    # intercept = y_mean - slope * x_mean
    # return slope, intercept
    
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    if np.count_nonzero(valid) < MIN_VALID_POINTS:
        return np.nan, np.nan

    x = x_vals[valid]
    y = y_vals[valid]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_scale = np.std(x)
    y_scale = np.std(y)

    if (
        x_scale == 0
        or y_scale == 0
        or not np.isfinite(x_scale)
        or not np.isfinite(y_scale)
    ):
        return np.nan, np.nan

    x_norm = (x - x_mean) / x_scale
    y_norm = (y - y_mean) / y_scale

    centered = np.column_stack((x_norm, y_norm))

    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    if singular_values[0] == 0 or not np.isfinite(singular_values[0]):
        return np.nan, np.nan

    # Dominant principal direction in standardized coordinates.
    direction = vh[0]

    if np.isclose(direction[0], 0.0):
        return np.nan, np.nan

    slope_norm = direction[1] / direction[0]

    # Transform y_norm = slope_norm * x_norm back to physical units.
    slope = slope_norm * y_scale / x_scale
    intercept = y_mean - slope * x_mean

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan, np.nan

    return slope, intercept


def sampled_cloud_for_flat_case(yb, tke_norm, sample_x, sample_y, z_indices):
    """Return flattened yB and TKE residual values for sampled flat-case columns."""
    yb_cloud = yb[sample_x[:, np.newaxis], sample_y[:, np.newaxis], z_indices]
    tke_cloud = tke_norm[sample_x[:, np.newaxis], sample_y[:, np.newaxis], z_indices]
    return yb_cloud.ravel(), tke_cloud.ravel()


def sampled_cloud_for_topography_case(yb, tke_norm, dist_terms, sample_x, sample_y):
    """Return flattened yB and TKE residual values for sampled terrain-case columns."""
    yb_cloud = yb[sample_x, sample_y, :]
    tke_cloud = tke_norm[sample_x, sample_y, :]
    dist_cloud = dist_terms[sample_x, sample_y, :]
    height_mask = (dist_cloud >= CANOPY_H_METERS) & (dist_cloud <= 10.0 * CANOPY_H_METERS)
    valid = height_mask & np.isfinite(yb_cloud) & np.isfinite(tke_cloud)
    return yb_cloud[valid], tke_cloud[valid]


def run_case(case_name, case_index, rng):
    """Run all bootstrap fits for one case."""
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
        z_indices = None
    else:
        dist_terms = None
        tke_norm = compute_normalized_tke_residual(meta["terms_bdg"], meta["topo_case"])
        z_uvp = np.arange(meta["nz_data"]) * meta["dz"] + 0.5 * meta["dz"]
        z_indices = np.where((z_uvp >= CANOPY_H) & (z_uvp <= 10.0 * CANOPY_H))[0]

    if z_indices is not None and z_indices.size == 0:
        raise ValueError(f"No vertical indices found between H and 5H for {case_name}.")

    slopes = np.full(N_BOOTSTRAP, np.nan)
    intercepts = np.full(N_BOOTSTRAP, np.nan)
    crossovers = np.full(N_BOOTSTRAP, np.nan)

    for boot in range(N_BOOTSTRAP):
        sample_x = rng.integers(0, meta["nx"], size=N_SAMPLE_COLUMNS)
        sample_y = rng.integers(0, meta["ny"], size=N_SAMPLE_COLUMNS)

        if meta["topo_case"]:
            x_vals, y_vals = sampled_cloud_for_topography_case(
                yb, tke_norm, dist_terms, sample_x, sample_y
            )
        else:
            x_vals, y_vals = sampled_cloud_for_flat_case(
                yb, tke_norm, sample_x, sample_y, z_indices
            )

        slope, intercept = linear_fit(x_vals, y_vals)
        slopes[boot] = slope
        intercepts[boot] = intercept
        valid_crossing_direction = (
            slope < 0.0 if REQUIRE_POSITIVE_TO_NEGATIVE_CROSSING else slope != 0.0
        )
        if np.isfinite(slope) and valid_crossing_direction and np.isfinite(intercept):
            crossovers[boot] = -intercept / slope

    line_ensemble = slopes[:, np.newaxis] * YB_LINE[np.newaxis, :] + intercepts[:, np.newaxis]
    line_median = np.nanmedian(line_ensemble, axis=0)
    line_q25 = np.nanpercentile(line_ensemble, 25, axis=0)
    line_q75 = np.nanpercentile(line_ensemble, 75, axis=0)

    return {
        "case": case_name,
        "slopes": slopes,
        "intercepts": intercepts,
        "crossovers": crossovers,
        "line_yb": YB_LINE,
        "line_median": line_median,
        "line_q25": line_q25,
        "line_q75": line_q75,
        "n_valid_bootstrap": int(np.count_nonzero(np.isfinite(crossovers))),
        "crossover_median": np.nanmedian(crossovers),
        "crossover_q25": np.nanpercentile(crossovers, 25),
        "crossover_q75": np.nanpercentile(crossovers, 75),
        "slope_median": np.nanmedian(slopes),
        "intercept_median": np.nanmedian(intercepts),
    }


def save_case_result(result):
    """Save the detailed bootstrap arrays for one case."""
    np.savez(
        OUTPUT_DIR / f"bootstrap_regression_{result['case']}.npz",
        slopes=result["slopes"],
        intercepts=result["intercepts"],
        crossovers=result["crossovers"],
        line_yb=result["line_yb"],
        line_median=result["line_median"],
        line_q25=result["line_q25"],
        line_q75=result["line_q75"],
    )


def save_summary(results):
    """Save one CSV row per case with crossover median and IQR."""
    summary_path = OUTPUT_DIR / "bootstrap_regression_summary.csv"
    header = (
        "case,n_valid_bootstrap,crossover_median,crossover_q25,crossover_q75,"
        "slope_median,intercept_median\n"
    )
    rows = [header]
    for result in results:
        rows.append(
            f"{result['case']},{result['n_valid_bootstrap']},"
            f"{result['crossover_median']:.10g},"
            f"{result['crossover_q25']:.10g},"
            f"{result['crossover_q75']:.10g},"
            f"{result['slope_median']:.10g},"
            f"{result['intercept_median']:.10g}\n"
        )
    summary_path.write_text("".join(rows), encoding="utf-8")


def plot_results(results):
    """Plot all bootstrap regression lines on one RegressionLine-style axis."""
    fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(8, 5))

    colors = ['green','limegreen','lightgreen','beige','khaki','gold','peachpuff','sandybrown','saddlebrown']
    labels = ['g1200','g800','g400','i1200','i800','i400','ATTO','Sinusoidal','Flat']

    for result, color, label in zip(results, colors, labels):
        axs.plot(result["line_yb"], result["line_median"], color=color, label=label)
        axs.fill_between(
            result["line_yb"],
            result["line_q25"],
            result["line_q75"],
            alpha=0.1,
            color=color,
        )

    axs.axhline(0, color="k", linestyle=":")
    # axs.axvline(0.2875, color="k", linestyle=":")
    # axs.axvline(0.4375, color="k", linestyle=":")

    axs.set_xlabel(r"$y_B$", fontsize=18)
    axs.set_ylabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=21)
    axs.set_xlim(*YB_PLOT_LIMITS)
    axs.set_ylim(*TKE_PLOT_LIMITS)
    axs.tick_params(axis="x", labelsize=12)
    axs.tick_params(axis="y", labelsize=12)
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "bootstrap_regression_lines.png",
        dpi=300,
        edgecolor="white",
        facecolor="white",
    )
    plt.show()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    results = []
    for case_index, case_name in enumerate(CASES):
        print(f"Processing {case_name} ({case_index + 1}/{len(CASES)})")
        result = run_case(case_name, case_index, rng)
        save_case_result(result)
        results.append(result)
        print(
            f"  crossover yB median/IQR: {result['crossover_median']:.2f} "
            f"[{result['crossover_q25']:.2f}, {result['crossover_q75']:.2f}]"
        )

    save_summary(results)
    plot_results(results)
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
