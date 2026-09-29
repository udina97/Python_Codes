#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare checkpoint shear stress and TKE profiles across LES resolutions.

The default setup loads:
    test_tkesgs_tii_64
    test_tkesgs_tii_128
    test_tkesgs_tii_256
at step 110000.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


FUNCTIONS_DIR = "/uufs/chpc.utah.edu/common/home/u1450851/Python_Codes/functions"
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

from read_checkpoint_SC import read_checkpoint_aniso


DEFAULT_BASE_PATH = "/scratch/general/nfs1/u1450851/LES_Sims/"
DEFAULT_SIM_PREFIX = "test_tkesgs_tii_lowshear_"
DEFAULT_STEP = 110000
DEFAULT_RESOLUTIONS = (64, 128, 256)
LZ = 1.0

RESOLUTION_COLORS = {
    64: "green",
    128: "red",
    256: "blue",
}

COMPONENT_LINESTYLES = {
    "resolved": "-",
    "sgs": "--",
    "total": ":",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot horizontal-median UW shear stress and TKE checkpoint profiles."
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Base directory containing LES simulation folders.",
    )
    parser.add_argument(
        "--sim-prefix",
        default=DEFAULT_SIM_PREFIX,
        help=(
            "Simulation name prefix. Each resolution is appended unless "
            "--sim-template is supplied."
        ),
    )
    parser.add_argument(
        "--sim-template",
        default=None,
        help=(
            "Optional simulation name template using {resolution}, e.g. "
            "'test_tkesgs_tii_{resolution}'."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help="Checkpoint step to load.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_RESOLUTIONS),
        help="List of resolutions to load.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving the figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save and close without opening an interactive plot window.",
    )
    return parser.parse_args()


def simulation_name(resolution, sim_prefix, sim_template):
    if sim_template is not None:
        return sim_template.format(resolution=resolution)
    return f"{sim_prefix}{resolution}"


def load_checkpoint(base_path, sim_name, step, resolution):
    checkpoint_path = os.path.join(base_path, sim_name, "output_checkpoint")
    return read_checkpoint_aniso(
        checkpoint_path,
        [step],
        resolution,
        resolution,
        resolution,
    )


def compute_reynolds_and_tke(checkpoint, nz):
    u = checkpoint["u_new"][:, :, :nz]
    v = checkpoint["v_new"][:, :, :nz]
    w = checkpoint["w_new"][:, :, :nz]

    rxx_res = checkpoint["uu_new"][:, :, :nz] - u * u
    ryy_res = checkpoint["vv_new"][:, :, :nz] - v * v
    rzz_res = checkpoint["ww_new"][:, :, :nz] - w * w

    rxy_res = checkpoint["uv_new"][:, :, :nz] - u * v
    rxz_res = checkpoint["uw_new"][:, :, :nz] - u * w
    ryz_res = checkpoint["vw_new"][:, :, :nz] - v * w

    tke_sgs = checkpoint["tke_sgs_new"][:, :, :nz]
    rxx_sgs = -checkpoint["txx_new"][:, :, :nz] + (2.0 / 3.0) * tke_sgs
    ryy_sgs = -checkpoint["tyy_new"][:, :, :nz] + (2.0 / 3.0) * tke_sgs
    rzz_sgs = -checkpoint["tzz_new"][:, :, :nz] + (2.0 / 3.0) * tke_sgs

    rxy_sgs = -checkpoint["txy_new"][:, :, :nz]
    rxz_sgs = -checkpoint["txz_new"][:, :, :nz]
    ryz_sgs = -checkpoint["tyz_new"][:, :, :nz]

    stresses = {
        "resolved": {
            "xx": rxx_res,
            "yy": ryy_res,
            "zz": rzz_res,
            "xy": rxy_res,
            "xz": rxz_res,
            "yz": ryz_res,
        },
        "sgs": {
            "xx": rxx_sgs,
            "yy": ryy_sgs,
            "zz": rzz_sgs,
            "xy": rxy_sgs,
            "xz": rxz_sgs,
            "yz": ryz_sgs,
        },
    }
    stresses["total"] = {
        key: stresses["resolved"][key] + stresses["sgs"][key]
        for key in stresses["resolved"]
    }

    tke = {}
    for component in ("resolved", "sgs", "total"):
        tke[component] = 0.5 * (
            stresses[component]["xx"]
            + stresses[component]["yy"]
            + stresses[component]["zz"]
        )

    return stresses, tke


def horizontal_median(field):
    return np.median(field, axis=(0, 1))


def plot_profiles(results, step, sim_label, output=None, show=True):
    nrows = len(results)
    fig, axs = plt.subplots(
        nrows,
        2,
        figsize=(10, 3.2 * nrows),
        sharey=False,
        tight_layout=True,
    )

    if nrows == 1:
        axs = np.array([axs])

    for row, (resolution, data) in enumerate(results.items()):
        color = RESOLUTION_COLORS.get(resolution, "black")
        z = data["z"]

        for component, linestyle in COMPONENT_LINESTYLES.items():
            axs[row, 0].plot(
                horizontal_median(data["stresses"][component]["xz"]),
                z,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=component,
            )
            axs[row, 1].plot(
                horizontal_median(data["tke"][component]),
                z,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=component,
            )

        axs[row, 0].set_ylabel(r"$z/L_z$")
        axs[row, 0].set_title(f"{resolution}: UW shear stress")
        axs[row, 1].set_title(f"{resolution}: TKE")
        axs[row, 0].grid(True, alpha=0.3)
        axs[row, 1].grid(True, alpha=0.3)

    axs[-1, 0].set_xlabel(r"$\widetilde{u'w'}$")
    axs[-1, 1].set_xlabel("TKE")

    component_handles = [
        Line2D([0], [0], color="black", linestyle=linestyle, lw=2)
        for linestyle in COMPONENT_LINESTYLES.values()
    ]

    axs[0, 0].legend(
        component_handles,
        list(COMPONENT_LINESTYLES.keys()),
        title="Component",
        loc="best",
    )

    # fig.suptitle(f"{sim_label}, step {step}", y=1.03)

    if output is not None:
        fig.savefig(output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    results = {}

    for resolution in args.resolutions:
        sim_name = simulation_name(resolution, args.sim_prefix, args.sim_template)
        print(f"Loading {sim_name}, step {args.step}")
        checkpoint = load_checkpoint(args.base_path, sim_name, args.step, resolution)
        stresses, tke = compute_reynolds_and_tke(
            checkpoint,
            resolution,
        )
        dz = LZ / resolution
        results[resolution] = {
            "stresses": stresses,
            "tke": tke,
            "z": np.arange(resolution) * dz + 0.5 * dz,
        }

    sim_label = (
        args.sim_template
        if args.sim_template is not None
        else f"{args.sim_prefix}<resolution>"
    )
    plot_profiles(
        results,
        args.step,
        sim_label,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
