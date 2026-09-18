#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample yB-vs-normalized-TKE-residual points once across all cases and plot a
pooled 2D histogram.

Workflow:
1. For each case, load yB and normalized TKE residual profiles from canopy top
   to a user-selected multiple of canopy height.
2. For each case, sample N x,y profile locations with replacement.
3. Pool all finite sampled profile points from all cases.
4. Compute and plot a 2D histogram with yB on the x axis and residual on the
   y axis.
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
    "resyb_2dhist_outputs"
)

ZI = 1000.0
CANOPY_H = 39.0 / ZI
CANOPY_H_METERS = CANOPY_H * ZI
HEIGHT_MAX_MULTIPLIER = 10.0

N_SAMPLE_PROFILES = 10000
RANDOM_SEED = 20260916

YB_LIMITS = (0.0, 0.8)
RESIDUAL_LIMITS = (-150.0, 300.0)
YB_BINS = 160
RESIDUAL_BINS = 180


def parse_args():
    parser = ArgumentParser(
        description="Compute one pooled 2D histogram of sampled residual-vs-yB points."
    )
    parser.add_argument("--n-sample-profiles", type=int, default=N_SAMPLE_PROFILES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--height-max-multiplier", type=float, default=HEIGHT_MAX_MULTIPLIER)
    parser.add_argument("--yb-min", type=float, default=YB_LIMITS[0])
    parser.add_argument("--yb-max", type=float, default=YB_LIMITS[1])
    parser.add_argument("--residual-min", type=float, default=RESIDUAL_LIMITS[0])
    parser.add_argument("--residual-max", type=float, default=RESIDUAL_LIMITS[1])
    parser.add_argument("--yb-bins", type=int, default=YB_BINS)
    parser.add_argument("--residual-bins", type=int, default=RESIDUAL_BINS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--save-points",
        action="store_true",
        help="Also save pooled sampled yB/residual arrays to an NPZ file.",
    )

    args = parser.parse_args()
    if args.n_sample_profiles <= 0:
        parser.error("--n-sample-profiles must be positive")
    if args.height_max_multiplier <= 1.0:
        parser.error("--height-max-multiplier must be greater than 1")
    if args.yb_min >= args.yb_max:
        parser.error("--yb-min must be smaller than --yb-max")
    if args.residual_min >= args.residual_max:
        parser.error("--residual-min must be smaller than --residual-max")
    if args.yb_bins <= 0 or args.residual_bins <= 0:
        parser.error("--yb-bins and --residual-bins must be positive")
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


def prepare_case(case_name, case_index, height_max_multiplier):
    """Return yB and residual profiles restricted to H through height_max_multiplier * H."""
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
            dist_terms <= height_max_multiplier * CANOPY_H_METERS
        )
        yb_profiles = np.where(height_mask, yb, np.nan).reshape(-1, yb.shape[2])
        residual_profiles = np.where(height_mask, residual, np.nan).reshape(
            -1,
            residual.shape[2],
        )
    else:
        residual = compute_normalized_tke_residual(meta["terms_bdg"], meta["topo_case"])
        z_uvp = np.arange(meta["nz_data"]) * meta["dz"] + 0.5 * meta["dz"]
        z_indices = np.where(
            (z_uvp >= CANOPY_H) & (z_uvp <= height_max_multiplier * CANOPY_H)
        )[0]
        if z_indices.size == 0:
            raise ValueError(
                f"No vertical indices found between H and {height_max_multiplier:g}H "
                f"for {case_name}."
            )

        yb_profiles = yb[:, :, z_indices].reshape(-1, z_indices.size)
        residual_profiles = residual[:, :, z_indices].reshape(-1, z_indices.size)

    return {
        "case": case_name,
        "yb_profiles": yb_profiles.astype(np.float32, copy=False),
        "residual_profiles": residual_profiles.astype(np.float32, copy=False),
    }


def sample_case_points(case_data, n_sample_profiles, rng):
    """Sample x,y profiles from one case and return all finite points."""
    n_profiles = case_data["yb_profiles"].shape[0]
    sample_idx = rng.randint(0, n_profiles, size=n_sample_profiles)

    yb = case_data["yb_profiles"][sample_idx].ravel()
    residual = case_data["residual_profiles"][sample_idx].ravel()
    valid = np.isfinite(yb) & np.isfinite(residual)
    return yb[valid], residual[valid]


def sample_all_cases(case_profiles, n_sample_profiles, rng):
    yb_parts = []
    residual_parts = []
    case_id_parts = []
    counts = []

    for case_id, case_data in enumerate(case_profiles):
        yb, residual = sample_case_points(case_data, n_sample_profiles, rng)
        yb_parts.append(yb)
        residual_parts.append(residual)
        case_id_parts.append(np.full(yb.size, case_id, dtype=np.int16))
        counts.append(
            {
                "case": case_data["case"],
                "n_profiles_available": case_data["yb_profiles"].shape[0],
                "n_profiles_sampled": n_sample_profiles,
                "n_valid_points": yb.size,
            }
        )

    return (
        np.concatenate(yb_parts),
        np.concatenate(residual_parts),
        np.concatenate(case_id_parts),
        counts,
    )


def compute_histogram(yb, residual, args):
    hist, yb_edges, residual_edges = np.histogram2d(
        yb,
        residual,
        bins=(args.yb_bins, args.residual_bins),
        range=((args.yb_min, args.yb_max), (args.residual_min, args.residual_max)),
    )
    return hist, yb_edges, residual_edges


def save_histogram(hist, yb_edges, residual_edges, counts, args):
    npz_path = args.output_dir / "resyb_2dhist.npz"
    np.savez(
        npz_path,
        hist=hist,
        yb_edges=yb_edges,
        residual_edges=residual_edges,
        cases=np.array(CASES),
        n_sample_profiles=args.n_sample_profiles,
        height_max_multiplier=args.height_max_multiplier,
        yb_limits=np.array([args.yb_min, args.yb_max]),
        residual_limits=np.array([args.residual_min, args.residual_max]),
    )

    csv_path = args.output_dir / "resyb_2dhist_counts.csv"
    yb_centers = 0.5 * (yb_edges[:-1] + yb_edges[1:])
    residual_centers = 0.5 * (residual_edges[:-1] + residual_edges[1:])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["yB_center", "residual_center", "count"])
        for i, yb_center in enumerate(yb_centers):
            for j, residual_center in enumerate(residual_centers):
                if hist[i, j] > 0:
                    writer.writerow(
                        [
                            f"{yb_center:.10g}",
                            f"{residual_center:.10g}",
                            f"{hist[i, j]:.10g}",
                        ]
                    )

    summary_path = args.output_dir / "resyb_2dhist_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "n_profiles_available", "n_profiles_sampled", "n_valid_points"])
        for row in counts:
            writer.writerow(
                [
                    row["case"],
                    row["n_profiles_available"],
                    row["n_profiles_sampled"],
                    row["n_valid_points"],
                ]
            )

    return npz_path, csv_path, summary_path


def save_points(yb, residual, case_ids, args):
    path = args.output_dir / "resyb_sampled_points.npz"
    np.savez(
        path,
        yb=yb.astype(np.float32, copy=False),
        residual=residual.astype(np.float32, copy=False),
        case_ids=case_ids,
        cases=np.array(CASES),
    )
    return path


def plot_histogram(hist, yb_edges, residual_edges, args):
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 5.5))

    mesh = ax.pcolormesh(
        yb_edges,
        residual_edges,
        hist.T,
        cmap="hot_r",
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Sampled point count", fontsize=13)

    ax.axhline(0, color="k", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$y_B$", fontsize=18)
    ax.set_ylabel(r"$\frac{P-\varepsilon}{|\varepsilon|}$", fontsize=21)
    ax.set_xlim(args.yb_min, args.yb_max)
    ax.set_ylim(args.residual_min, args.residual_max)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    path = args.output_dir / "resyb_2dhist_heatmap.png"
    fig.savefig(path, dpi=300, edgecolor="white", facecolor="white")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    case_profiles = []
    for case_index, case_name in enumerate(CASES):
        print(f"Loading {case_name} ({case_index + 1}/{len(CASES)})")
        case_data = prepare_case(case_name, case_index, args.height_max_multiplier)
        case_profiles.append(case_data)
        print(f"  profiles available: {case_data['yb_profiles'].shape[0]}")

    print(
        f"\nSampling {args.n_sample_profiles} x,y profiles per case with replacement "
        f"from H to {args.height_max_multiplier:g}H."
    )
    yb, residual, case_ids, counts = sample_all_cases(
        case_profiles,
        args.n_sample_profiles,
        rng,
    )

    hist, yb_edges, residual_edges = compute_histogram(yb, residual, args)

    npz_path, csv_path, summary_path = save_histogram(hist, yb_edges, residual_edges, counts, args)
    figure_path = plot_histogram(hist, yb_edges, residual_edges, args)

    points_path = None
    if args.save_points:
        points_path = save_points(yb, residual, case_ids, args)

    print(f"\nPooled valid sampled points: {yb.size}")
    print(f"Points inside plotted histogram range: {int(hist.sum())}")
    print(f"Saved histogram arrays to {npz_path}")
    print(f"Saved nonzero histogram counts to {csv_path}")
    print(f"Saved sample summary to {summary_path}")
    if points_path is not None:
        print(f"Saved sampled point arrays to {points_path}")
    print(f"Saved heatmap to {figure_path}")


if __name__ == "__main__":
    main()
