#!/usr/bin/env python3
"""Plot the anisotropy-dependent surface phi and psi functions from the LES.

The unstable anisotropy curves reproduce the current surface_module.f90:
psi_m and psi_h are limited separately by ln(z1/z0m)-0.1 and
ln(z1/z0h)-0.1, without a stability cutoff. Momentum phi becomes 1 when
psi_m reaches its ceiling; heat phi plateaus at its crossing value while
psi_h remains capped. Stable anisotropy and black reference curves follow
the existing LES formulations. The defaults z1=16 m and z0m=z0h=0.1 m
match the simB_Patton2011 parameter file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "functions"))
from Anisotropy_Functions import ColorAnisotropy


NPNT = 100
BRUTSAERT_LIMIT = 0.41 ** -3


def phi_m_unstable(zeta: np.ndarray, yb: float) -> np.ndarray:
    yb_min = max(yb, 0.1)
    a = 0.012 if yb > 0.6 else 0.24 - 0.38 * yb_min
    b = 0.061
    c = 0.45 - 0.53 * yb_min
    exponent = -0.12 + 6.4 * yb_min
    magnitude = np.abs(zeta)
    power = magnitude ** exponent
    return np.minimum((a + b * power) / (a + power)
                      + c * magnitude ** (1.0 / 3.0), 1.0)


def phi_h_unstable(zeta: np.ndarray, yb: float) -> np.ndarray:
    d = 0.48 + 1.8 * max(yb, 0.1)
    return np.minimum(d * ((3.0 - 2.5 * zeta) /
                           (1.0 - 10.0 * zeta + 50.0 * zeta**2)) ** (1.0 / 3.0),
                      1.0)


def phi_m_stable(zeta: np.ndarray, yb: float) -> np.ndarray:
    yb_min = max(yb, 0.1)
    a = 0.76 + 1.5 * yb_min
    b = 6.3 - 4.3 * yb_min
    return np.maximum(a + b * zeta, 1.0)


def phi_h_stable(zeta: np.ndarray, yb: float) -> np.ndarray:
    yb_min = max(yb, 0.1)
    c = 0.34 if yb_min > 0.6 else 1.9 - 2.6 * yb_min
    d = 6.7 - 10.0 * yb_min
    return np.maximum(c + d * zeta, 1.0)


def integrate_phi(zeta_end: np.ndarray, phi_function, yb: float) -> tuple[np.ndarray, np.ndarray]:
    """Return LES psi and endpoint phi on the fine_mesh grid, starting at zero."""
    zeta_vec = zeta_end[:, None] * np.linspace(0.0, 1.0, NPNT)[None, :]
    phi_vec = phi_function(zeta_vec, yb)
    # integrate_phi treats the first interval as 0.5*(1-phi(2)), not as
    # a conventional trapezoid with a value evaluated at zeta=0.
    psi = 0.5 * (1.0 - phi_vec[:, 1])
    psi += np.sum(0.5 * (zeta_vec[:, 2:] - zeta_vec[:, 1:-1]) *
                  ((1.0 - phi_vec[:, 1:-1]) / zeta_vec[:, 1:-1] +
                   (1.0 - phi_vec[:, 2:]) / zeta_vec[:, 2:]), axis=1)
    return psi, phi_vec[:, -1]


def bruts_psi_m(y: np.ndarray) -> np.ndarray:
    """Analytic Brutsaert unstable momentum psi from compute_PSI."""
    a, b = 0.33, 0.41
    y = np.minimum(y, BRUTSAERT_LIMIT)
    x = (y / a) ** (1.0 / 3.0)
    root3 = np.sqrt(3.0)
    return (np.log(a + y) - 3.0 * b * y ** (1.0 / 3.0)
            + b * a ** (1.0 / 3.0) / 2.0 * np.log((1.0 + x)**2 /
                                                     (1.0 - x + x**2))
            + root3 * b * a ** (1.0 / 3.0) * np.arctan((2.0 * x - 1.0) / root3)
            - np.log(a) + root3 * b * a ** (1.0 / 3.0) * np.pi / 6.0)


def unstable_phi_psi_capped(zeta: np.ndarray, yb: float, phi_function,
                            psi_limit: float, *, heat: bool
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the Fortran zero-start bin and fixed log(1+|zeta|) mesh."""
    log_spacing = 0.05
    seed = np.expm1(log_spacing)
    psi_out = np.empty_like(zeta)
    phi_out = np.empty_like(zeta)

    def linear_value(y: float) -> tuple[float, float]:
        if y <= 0.0:
            return 0.0, float(phi_function(np.array([0.0]), yb)[0])
        psi, phi = integrate_phi(np.array([-y]), phi_function, yb)
        return float(psi[0]), float(phi[0])

    for index, endpoint in enumerate(-zeta):
        y = float(endpoint)
        psi, phi = linear_value(min(y, seed))
        if psi >= psi_limit:
            if heat:
                lower, upper = 0.0, min(y, seed)
                phi = float(phi_function(np.array([0.0]), yb)[0])
                for _ in range(12):
                    midpoint = 0.5 * (lower + upper)
                    trial_psi, trial_phi = linear_value(midpoint)
                    if trial_psi < psi_limit:
                        lower, phi = midpoint, trial_phi
                    else:
                        upper = midpoint
            else:
                phi = 1.0
            psi_out[index], phi_out[index] = psi_limit, phi
            continue

        if y > seed:
            y_prev = seed
            t_prev = log_spacing
            t_end = np.log1p(y)
            while t_prev < t_end:
                t_next = min(t_end, t_prev + log_spacing)
                y_next = y if t_next == t_end else np.expm1(t_next)
                phi_next = float(phi_function(np.array([-y_next]), yb)[0])
                increment = 0.5 * (y_next - y_prev) * (
                    (1.0 - phi) / y_prev + (1.0 - phi_next) / y_next)
                if psi + increment >= psi_limit:
                    if heat:
                        fraction = np.clip((psi_limit - psi) / increment, 0.0, 1.0)
                        y_cross = y_prev + fraction * (y_next - y_prev)
                        phi = float(phi_function(np.array([-y_cross]), yb)[0])
                    else:
                        phi = 1.0
                    psi = psi_limit
                    break
                psi += increment
                phi = phi_next
                y_prev, t_prev = y_next, t_next
        psi_out[index], phi_out[index] = psi, phi
    return psi_out, phi_out


def reference_functions(zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Brutsaert unstable or Högström hog96 stable phi_m, phi_h, psi_m, psi_h."""
    if np.all(zeta < 0.0):
        y = -zeta
        a, b, c, d, exponent = 0.33, 0.41, 0.33, 0.057, 0.78
        phi_m = np.where(y > BRUTSAERT_LIMIT, 1.0,
                         (a + b * y ** (4.0 / 3.0)) / (a + y))
        psi_m = bruts_psi_m(y)
        y_h = np.minimum(y, 30.0)
        phi_h = (c + d * y_h**exponent) / (c + y_h**exponent)
        psi_h = (1.0 - d) / exponent * np.log((c + y_h**exponent) / c)
    elif np.all(zeta > 0.0):
        limited = np.minimum(zeta, 10.0)
        phi_m = 1.0 + 5.3 * limited
        phi_h = 1.0 + 8.0 * limited
        psi_m = -5.3 * limited
        psi_h = -8.0 * limited
    else:
        raise ValueError("Pass either entirely negative or entirely positive stability values")
    return phi_m, phi_h, psi_m, psi_h


def wall_functions(zeta: np.ndarray, yb: float,
                   z_over_z0m: float, z_over_z0h: float,
                   psi_margin: float = 0.1
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate anisotropic functions with roughness-dependent unstable ceilings."""
    if np.all(zeta < 0.0):
        momentum_limit = np.log(z_over_z0m) - psi_margin
        heat_limit = np.log(z_over_z0h) - psi_margin
        psi_m, phi_m = unstable_phi_psi_capped(
            zeta, yb, phi_m_unstable, momentum_limit, heat=False)
        psi_h, phi_h = unstable_phi_psi_capped(
            zeta, yb, phi_h_unstable, heat_limit, heat=True)
    elif np.all(zeta > 0.0):
        zeta_grid = np.minimum(zeta, 10.0)
        psi_m, phi_m = integrate_phi(zeta_grid, phi_m_stable, yb)
        phi_m = np.where(psi_m > 0.0, 1.0, phi_m)
        psi_m = np.minimum(psi_m, 0.0)
        psi_h, phi_h = integrate_phi(zeta_grid, phi_h_stable, yb)
        phi_h = np.where(psi_h > 0.0, 1.0, phi_h)
        psi_h = np.minimum(psi_h, 0.0)
    else:
        raise ValueError("Pass either entirely negative or entirely positive stability values")
    return phi_m, phi_h, psi_m, psi_h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("AnisoFluxGradFunc_v6.png"),
                        help="Figure path (default: AnisoFluxGradFunc_v6.png beside this script)")
    parser.add_argument("--z1", type=float, default=16.0,
                        help="First grid-cell center height dz/2 in meters (default: 16)")
    parser.add_argument("--z0m", type=float, default=0.1,
                        help="Momentum roughness length in meters (default: 0.1)")
    parser.add_argument("--z0h", type=float, default=0.1,
                        help="Heat roughness length in meters (default: 0.1)")
    parser.add_argument("--psi-margin", type=float, default=0.1,
                        help="Margin subtracted from each local wall-law logarithm")
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0.0
           for value in (args.z1, args.z0m, args.z0h)):
        parser.error("--z1, --z0m, and --z0h must be finite and positive")
    if not np.isfinite(args.psi_margin) or args.psi_margin < 0.0:
        parser.error("--psi-margin must be finite and nonnegative")
    z_over_z0m = args.z1 / args.z0m
    z_over_z0h = args.z1 / args.z0h
    psi_m_limit = np.log(z_over_z0m) - args.psi_margin
    psi_h_limit = np.log(z_over_z0h) - args.psi_margin
    if min(psi_m_limit, psi_h_limit) <= 0.0:
        parser.error("both ln(z1/z0) values must exceed --psi-margin")

    unstable = -np.logspace(2, -4, 600)
    stable = np.logspace(-4, 2, 600)
    yb_values = np.linspace(0.1, 0.6, 6)
    cmap = ColorAnisotropy()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex="col")
    titles = (r"Unstable $\phi_m$", r"Stable $\phi_m$",
              r"Unstable $\phi_h$", r"Stable $\phi_h$")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    for yb in yb_values:
        color = cmap(yb)
        phi_m_u, phi_h_u, psi_m_u, psi_h_u = wall_functions(
            unstable, yb, z_over_z0m, z_over_z0h, args.psi_margin)
        phi_m_s, phi_h_s, psi_m_s, psi_h_s = wall_functions(
            stable, yb, z_over_z0m, z_over_z0h, args.psi_margin)
        label = rf"$y_B={yb:.1f}$"
        for ax, x, values in (
            (axes[0, 0], unstable, phi_m_u), (axes[0, 1], stable, phi_m_s),
            (axes[0, 2], unstable, phi_h_u), (axes[0, 3], stable, phi_h_s),
            (axes[1, 0], unstable, psi_m_u), (axes[1, 1], stable, psi_m_s),
            (axes[1, 2], unstable, psi_h_u), (axes[1, 3], stable, psi_h_s),
        ):
            ax.plot(x, values, color=color, linewidth=1.8, label=label)

    phi_m_u_ref, phi_h_u_ref, psi_m_u_ref, psi_h_u_ref = reference_functions(unstable)
    phi_m_s_ref, phi_h_s_ref, psi_m_s_ref, psi_h_s_ref = reference_functions(stable)
    for ax, x, values, linestyle in (
        (axes[0, 0], unstable, phi_m_u_ref, "-"),
        (axes[0, 1], stable, phi_m_s_ref, "--"),
        (axes[0, 2], unstable, phi_h_u_ref, "-"),
        (axes[0, 3], stable, phi_h_s_ref, "--"),
        (axes[1, 0], unstable, psi_m_u_ref, "-"),
        (axes[1, 1], stable, psi_m_s_ref, "--"),
        (axes[1, 2], unstable, psi_h_u_ref, "-"),
        (axes[1, 3], stable, psi_h_s_ref, "--"),
    ):
        ax.plot(x, values, color="black", linestyle=linestyle,
                linewidth=2.4, zorder=10)

    for col in range(4):
        for row in range(2):
            ax = axes[row, col]
            if col % 2 == 0:
                ax.set_xscale("symlog", linthresh=1e-5)
                ax.set_xlim(-1e2, -1e-4)
            else:
                ax.set_xscale("log")
                ax.set_xlim(1e-4, 1e2)
            ax.grid(True, which="major", linestyle=":", linewidth=0.6)
        axes[1, col].set_xlabel(r"Stability $\zeta=z_1/L$")
    for col in (0, 2):
        axes[0, col].set_ylim(0, 1)
    for col in (1, 3):
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylim(1, 1e2)
    for col, label in enumerate((r"$\phi_m$", r"$\phi_m$", r"$\phi_h$", r"$\phi_h$")):
        axes[0, col].set_ylabel(label)
    for col, label in enumerate((r"$\psi_m$", r"$\psi_m$", r"$\psi_h$", r"$\psi_h$")):
        axes[1, col].set_ylabel(label)
        axes[1, col].axhline(0, color="0.4", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(psi_m_limit, color="0.35", linestyle=":", linewidth=1)
    axes[1, 2].axhline(psi_h_limit, color="0.35", linestyle=":", linewidth=1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.extend((Line2D([], [], color="black", linewidth=2.4),
                    Line2D([], [], color="black", linestyle="--", linewidth=2.4)))
    labels.extend(("Brutsaert (unstable)", "Högström (stable)"))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=8, frameon=False, fontsize=9, columnspacing=1.0)
    fig.suptitle(rf"Anisotropy-dependent surface scaling functions "
                 rf"(v6; $z_1={args.z1:g}$ m, $z_{{0m}}={args.z0m:g}$ m, "
                 rf"$z_{{0h}}={args.z0h:g}$ m; "
                 rf"$\psi_{{m,\max}}={psi_m_limit:.2f}$, $\psi_{{h,\max}}={psi_h_limit:.2f}$)",
                 y=0.995)
    fig.text(0.5, 0.018,
             r"Colored unstable curves have no stability cutoff; the black Brutsaert heat "
             r"reference retains its original $|\zeta|=30$ convention.",
             ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.11,
                        wspace=0.34, hspace=0.20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
