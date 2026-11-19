#!/usr/bin/env python3
"""Compare reference and anisotropy-dependent LES wall-law friction velocity.

With --snapshot-step, replay one sm_wallstress_flux call using its snapshot,
surface checkpoint, and anisotropy checkpoint. Without that option, retain the
multi-snapshot averaging analysis. The default unstable anisotropy scalings
match the current surface_module.f90: momentum and heat psi have separate
roughness-dependent ceilings, with no stability-based cutoff. The standard
Brutsaert law is unchanged. The optional clipped momentum mode is retained
only to reproduce the earlier comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


DEFAULT_SIM = Path("/scratch/general/nfs1/u1450851/LES_Sims/simB_Patton2011")
KAPPA = 0.4
G = 9.81
NPNT = 100  # compute_phi_anisotropy in surface_module.f90
PSI_MARGIN = 0.1
UNSTABLE_LOG_SPACING = 0.05


def read_namelist(path: Path) -> dict[str, str]:
    """Read scalar values from the run's Fortran namelist (not a general parser)."""
    result = {}
    for line in path.read_text().splitlines():
        line = line.split("!", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip().lower()] = value.strip().rstrip(",")
    return result


def number(config: dict[str, str], key: str) -> float:
    return float(config[key].split()[0].replace("d", "e").replace("D", "E"))


def flag(config: dict[str, str], key: str) -> bool:
    return config[key].split()[0].lower() == ".true."


def read_fields(path: Path, shape: tuple[int, int, int], nfields: int) -> np.ndarray:
    expected = np.prod(shape) * nfields * 8
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {path.stat().st_size}")
    values = np.fromfile(str(path), dtype="<f8")
    n = int(np.prod(shape))
    return np.stack([values[i*n:(i+1)*n].reshape(shape, order="F")
                     for i in range(nfields)])


def read_surface(path: Path, nx: int, ny: int) -> dict[str, np.ndarray]:
    names = ("flux", "value", "ustar", "phi_m", "psi_m", "phi_h", "psi_h", "L")
    if path.stat().st_size != len(names) * nx * ny * 8:
        raise ValueError(f"Unexpected surface checkpoint size: {path}")
    values = np.fromfile(str(path), dtype="<f8")
    n = nx * ny
    return {key: values[i*n:(i+1)*n].reshape((nx, ny), order="F")
            for i, key in enumerate(names)}


def read_yb(path: Path, nx: int, ny: int, nz: int, level: int) -> np.ndarray:
    # IO_checkpoint.f90 writes xB then yB, each (nx,ny,nz) float64, Fortran order.
    n = nx * ny * nz
    if path.stat().st_size < 2 * n * 8:
        raise ValueError(f"Anisotropy checkpoint is too short: {path}")
    with path.open("rb") as stream:
        stream.seek(n * 8)
        # One horizontal plane in Fortran order, level is 1-based in the LES.
        stream.seek((level - 1) * nx * ny * 8, 1)
        return np.fromfile(stream, dtype="<f8", count=nx * ny).reshape((nx, ny), order="F")


def layer_filter(field: np.ndarray, lx_nd: float, ly_nd: float) -> np.ndarray:
    """The G_test spectral cutoff with alpha=2 and zeroed Nyquist modes."""
    nx, ny = field.shape
    dx, dy = lx_nd / nx, ly_nd / ny
    delta = 2.0 * np.sqrt(dx * dy)
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    k2 = kx[:, None]**2 + ky[None, :]**2
    keep = k2 < (np.pi / delta)**2
    keep[-1, :] = False
    keep[:, ny // 2] = False
    # Fortran FFTW forward/backward normalization is supplied by G=1/(nx*ny).
    # NumPy's inverse transform already includes the same normalization.
    spectrum = np.fft.fft(np.fft.rfft(field, axis=0), axis=1)
    return np.fft.irfft(np.fft.ifft(spectrum * keep, axis=1), n=nx, axis=0).real


def bruts_psi_m(y: np.ndarray) -> np.ndarray:
    """Brutsaert unstable momentum psi, including the LES upper-y cap."""
    a, b = 0.33, 0.41
    y = np.minimum(y, b**-3)
    x = (y / a)**(1 / 3)
    psi_zero = -np.log(a) + np.sqrt(3) * b * a**(1 / 3) * np.pi / 6
    return (np.log(a + y) - 3*b*y**(1/3)
            + b*a**(1/3)/2 * np.log((1+x)**2 / (1-x+x*x))
            + np.sqrt(3)*b*a**(1/3)*np.arctan((2*x-1)/np.sqrt(3))
            + psi_zero)


def standard_psi(L: np.ndarray, flux: np.ndarray, dz_nd: float) -> tuple[np.ndarray, np.ndarray]:
    psim = np.zeros_like(L)
    psih = np.zeros_like(L)
    active = np.isfinite(L) & (flux != 0)
    unstable = active & (L < 0)
    stable = active & (L > 0)
    if np.any(unstable):
        y = -0.5 * dz_nd / L[unstable]
        psim[unstable] = bruts_psi_m(y)
        c, d, power = 0.33, 0.057, 0.78
        yy = np.minimum(y, 30.0)
        psih[unstable] = ((1-d)/power) * np.log((c+yy**power)/c)
    if np.any(stable):
        zeta = np.minimum(0.5 * dz_nd / L[stable], 10.0)
        psim[stable] = -5.3 * zeta  # stable_function='hog96'
        psih[stable] = -8.0 * zeta
    return psim, psih


def integrate_phi(z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """The 100-point trapezoid loop in integrate_phi, including its first bin."""
    integral = 0.5 * (1 - phi[:, 1])
    integral += np.sum(0.5 * (z[:, 2:] - z[:, 1:-1]) *
                       ((1-phi[:, 1:-1])/z[:, 1:-1] +
                        (1-phi[:, 2:])/z[:, 2:]), axis=1)
    return integral


def dynamic_unstable_phi_psi(zeta: np.ndarray, yb: np.ndarray,
                             psi_limit: float, *, heat: bool
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Integrate unstable phi as in surface_module.f90, returning psi and phi.

    The first 100-point interval preserves integrate_phi's zero-start rule.
    Subsequent bins are fixed in log(1+|zeta|); only psi, not stability,
    determines whether momentum phi becomes 1 or heat phi plateaus.
    """
    if zeta.ndim != 1 or yb.shape != zeta.shape or np.any(zeta >= 0):
        raise ValueError("Expected matching one-dimensional unstable zeta and yB")
    if not zeta.size:
        empty = np.empty(0, dtype=float)
        return empty, empty.copy()
    if not np.isfinite(psi_limit) or psi_limit <= 0:
        raise ValueError("The unstable wall-law psi ceiling must be positive")

    def phi_value(y: np.ndarray, yb_local: np.ndarray) -> np.ndarray:
        yb_min = np.maximum(yb_local, 0.1)
        if heat:
            d = 0.48 + 1.8 * yb_min
            return np.minimum(d * ((3.0 + 2.5*y) /
                                   (1.0 + 10.0*y + 50.0*y*y))**(1.0/3.0), 1.0)
        a = np.where(yb_local > 0.6, 0.012, 0.24 - 0.38*yb_min)
        c = 0.45 - 0.53*yb_min
        exponent = -0.12 + 6.4*yb_min
        power = y**exponent
        return np.minimum((a + 0.061*power)/(a + power) + c*y**(1.0/3.0), 1.0)

    def linear_value(y: np.ndarray, yb_local: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
        z = -y[:, None] * np.linspace(0.0, 1.0, NPNT)[None, :]
        phi_grid = phi_value(-z, yb_local[:, None])
        return integrate_phi(z, phi_grid), phi_grid[:, -1]

    y = -zeta
    seed = np.expm1(UNSTABLE_LOG_SPACING)
    base = np.minimum(y, seed)
    psi, phi = linear_value(base, yb)
    capped = psi >= psi_limit
    if np.any(capped):
        if heat:
            indices = np.flatnonzero(capped)
            lower = np.zeros(indices.size)
            upper = base[indices].copy()
            phi_lower = phi_value(lower, yb[indices])
            for _ in range(12):
                midpoint = 0.5 * (lower + upper)
                trial_psi, trial_phi = linear_value(midpoint, yb[indices])
                below = trial_psi < psi_limit
                lower = np.where(below, midpoint, lower)
                upper = np.where(below, upper, midpoint)
                phi_lower = np.where(below, trial_phi, phi_lower)
            phi[indices] = phi_lower
        else:
            phi[capped] = 1.0
        psi[capped] = psi_limit

    active = (y > seed) & ~capped
    y_previous = np.full(zeta.size, seed)
    t_previous = np.full(zeta.size, UNSTABLE_LOG_SPACING)
    t_endpoint = np.log1p(y)
    while np.any(active):
        indices = np.flatnonzero(active)
        t_next = np.minimum(t_endpoint[indices],
                            t_previous[indices] + UNSTABLE_LOG_SPACING)
        y_next = np.where(t_next == t_endpoint[indices], y[indices], np.expm1(t_next))
        phi_next = phi_value(y_next, yb[indices])
        increment = 0.5 * (y_next - y_previous[indices]) * (
            (1.0 - phi[indices])/y_previous[indices] + (1.0 - phi_next)/y_next)
        reached = psi[indices] + increment >= psi_limit
        if np.any(reached):
            hit = indices[reached]
            if heat:
                fraction = np.clip((psi_limit - psi[hit])/increment[reached], 0.0, 1.0)
                crossing = y_previous[hit] + fraction * (y_next[reached] - y_previous[hit])
                phi[hit] = phi_value(crossing, yb[hit])
            else:
                phi[hit] = 1.0
            psi[hit] = psi_limit
            active[hit] = False
        remaining = ~reached
        if np.any(remaining):
            not_hit = indices[remaining]
            psi[not_hit] += increment[remaining]
            phi[not_hit] = phi_next[remaining]
            y_previous[not_hit] = y_next[remaining]
            t_previous[not_hit] = t_next[remaining]
            active[not_hit[t_next[remaining] >= t_endpoint[not_hit]]] = False
    return psi, phi


def clipped_unstable_psi_m(zeta: np.ndarray, yb: np.ndarray) -> np.ndarray:
    """Original surface_module.f90 unstable anisotropic momentum branch.

    Above the Brutsaert threshold the LES substitutes the Brutsaert psi_m
    value. Below it, phi_m is integrated on the LES 100-point mesh, whose
    stability endpoint is clipped at -30, and psi_m is bounded to [0, cap].
    """
    y = -zeta
    threshold = 0.41 ** -3
    psi_cap = float(bruts_psi_m(np.array([threshold]))[0])
    psi = np.full_like(zeta, psi_cap)
    below = y <= threshold
    if np.any(below):
        z_end = np.maximum(zeta[below], -30.0)
        z = z_end[:, None] * np.linspace(0.0, 1.0, NPNT)[None, :]
        magnitude = np.abs(z)
        yb0 = yb[below]
        ybmin = np.maximum(yb0, 0.1)
        a = np.where(yb0 > 0.6, 0.012, 0.24 - 0.38 * ybmin)[:, None]
        c = (0.45 - 0.53 * ybmin)[:, None]
        exponent = (-0.12 + 6.4 * ybmin)[:, None]
        power = magnitude ** exponent
        phi = np.minimum((a + 0.061 * power) / (a + power)
                         + c * magnitude ** (1.0 / 3.0), 1.0)
        psi[below] = np.clip(integrate_phi(z, phi), 0.0, psi_cap)
    return psi


def aniso_psi(L: np.ndarray, flux: np.ndarray, yb: np.ndarray,
              dz_nd: float, z0m_nd: float, z0h_nd: float,
              unstable_phi_m_version: str = "extended") -> tuple[np.ndarray, np.ndarray]:
    psim = np.zeros_like(L)
    psih = np.zeros_like(L)
    active = np.isfinite(L) & (flux != 0)
    for unstable in (True, False):
        mask = active & ((L < 0) if unstable else (L > 0))
        if not np.any(mask):
            continue
        l = L[mask]
        yb0 = yb[mask]
        ybmin = np.maximum(yb0, 0.1)
        if unstable:
            zeta = 0.5 * dz_nd / l
            if unstable_phi_m_version == "extended":
                log_m = np.log((0.5*dz_nd)/z0m_nd)
                log_h = np.log((0.5*dz_nd)/z0h_nd)
                if min(log_m, log_h) <= PSI_MARGIN:
                    raise ValueError("Unstable wall-law logarithms must exceed the 0.1 psi margin")
                pm, _phi_m = dynamic_unstable_phi_psi(
                    zeta, yb0, log_m - PSI_MARGIN, heat=False)
                ph, _phi_h = dynamic_unstable_phi_psi(
                    zeta, yb0, log_h - PSI_MARGIN, heat=True)
            elif unstable_phi_m_version == "clipped":
                # Retained only to reproduce the older stability-clipped plot.
                pm = clipped_unstable_psi_m(zeta, yb0)
                z_end = np.maximum(zeta, -30.0)
                z = z_end[:, None] * np.linspace(0, 1, NPNT)[None, :]
                d = 0.48 + 1.8*ybmin
                phih = np.minimum(d[:, None] *
                                  ((3 - 2.5*z)/(1 - 10*z + 50*z*z))**(1/3), 1.0)
                ph = np.maximum(integrate_phi(z, phih), 0.0)
                logh = np.log((0.5*dz_nd)/z0h_nd)
                ph = np.where(ph > logh, logh - PSI_MARGIN, ph)
            else:
                raise ValueError(f"Unknown unstable phi_m version: {unstable_phi_m_version}")
        else:
            z_end = np.minimum(0.5 * dz_nd / l, 10.0)
            z = z_end[:, None] * np.linspace(0, 1, NPNT)[None, :]
            a = 0.76 + 1.5*ybmin
            b = 6.3 - 4.3*ybmin
            phim = np.maximum(a[:, None] + b[:, None]*z, 1.0)
            pm = np.minimum(integrate_phi(z, phim), 0.0)
            c = np.where(ybmin > 0.6, 0.34, 1.9 - 2.6*ybmin)
            d = 6.7 - 10*ybmin
            phih = np.maximum(c[:, None] + d[:, None]*z, 1.0)
            ph = np.minimum(integrate_phi(z, phih), 0.0)
        psim[mask] = pm
        psih[mask] = ph
    return psim, psih


def calculate(u1: np.ndarray, v1: np.ndarray, temp1: np.ndarray,
              surface: dict[str, np.ndarray], yb: np.ndarray, config: dict[str, str],
              anisotropic: bool,
              unstable_phi_m_version: str = "extended") -> dict[str, np.ndarray]:
    nx, ny = u1.shape
    lx_nd = number(config, "lx") / number(config, "zi")
    ly_nd = number(config, "ly") / number(config, "zi")
    dz_nd = number(config, "lz") / number(config, "nz") / number(config, "zi")
    z0_nd = number(config, "lbc_dz0") / number(config, "zi")
    z0h_nd = number(config, "lbc_dz0_sc") / number(config, "zi")
    if not (0 < z0_nd < dz_nd/2 and 0 < z0h_nd < dz_nd/2):
        raise ValueError("Roughness lengths must lie below the first velocity/scalar level")
    wall_model = config["wall_model"].strip(" '\"").lower()
    if wall_model == "wall_filter":
        u1 = layer_filter(u1, lx_nd, ly_nd)
        v1 = layer_filter(v1, lx_nd, ly_nd)
        temp1 = layer_filter(temp1, lx_nd, ly_nd)
    elif wall_model == "wall_homog":
        u1 = np.full((nx, ny), u1.mean())
        v1 = np.full((nx, ny), v1.mean())
        temp1 = np.full((nx, ny), temp1.mean())
    elif wall_model != "wall_local":
        raise ValueError(f"Unsupported wall_model: {wall_model}")
    speed = np.hypot(u1, v1)
    flux = surface["flux"].copy()
    if wall_model == "wall_homog":
        flux = np.full((nx, ny), flux.mean())
    # sm_wallstress_flux starts with the previous psi_m. It does not reset psi.
    ustar0 = KAPPA * speed / (np.log(0.5*dz_nd/z0_nd) - surface["psi_m"])
    if number(config, "lbc_sc") == 1:
        flux = (surface["value"]-temp1)*ustar0*KAPPA / (
            np.log(0.5*dz_nd/z0h_nd)-surface["psi_h"])
    g_hat = G * number(config, "zi") / number(config, "u_scale")**2
    with np.errstate(divide="ignore", invalid="ignore"):
        L0 = -ustar0**3 * temp1 / (KAPPA*g_hat*flux)
    if anisotropic:
        psim, psih = aniso_psi(L0, flux, yb, dz_nd, z0_nd, z0h_nd,
                              unstable_phi_m_version)
    else:
        psim, psih = standard_psi(L0, flux, dz_nd)
    ustar = KAPPA * speed / (np.log(0.5*dz_nd/z0_nd)-psim)
    if number(config, "lbc_sc") == 1:
        flux = (surface["value"]-temp1)*ustar*KAPPA / (
            np.log(0.5*dz_nd/z0h_nd)-psih)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = -ustar**3 * temp1 / (KAPPA*g_hat*flux)
    return {"ustar_nd": ustar, "heat_flux_nd": flux, "L_nd": L,
            "ustar0_nd": ustar0, "L0_nd": L0, "filtered_speed": speed,
            "psi_m": psim, "psi_h": psih, "filtered_u": u1,
            "filtered_v": v1, "filtered_temp": temp1}


def plot_fields(results: dict[str, dict[str, np.ndarray]], lx: float, ly: float,
                output: Path, step: int, level: int) -> None:
    x = np.linspace(0, lx, next(iter(results.values()))["ustar"].shape[0]+1)
    y = np.linspace(0, ly, next(iter(results.values()))["ustar"].shape[1]+1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    scales = {}
    for name in ("ustar", "heat_flux"):
        values = np.concatenate([fields[name].ravel() for fields in results.values()])
        scales[name] = (np.nanmin(values), np.nanmax(values))
    for row, (label, fields) in enumerate(results.items()):
        for col, (name, units) in enumerate((("ustar", "m s$^{-1}$"),
                                              ("heat_flux", "K m s$^{-1}$"))):
            lo, hi = scales[name]
            if np.isclose(lo, hi):
                lo, hi = lo-0.01*max(abs(lo), 1), hi+0.01*max(abs(hi), 1)
            mesh = axes[row, col].pcolormesh(x, y, fields[name].T, shading="flat",
                                              vmin=lo, vmax=hi)
            fig.colorbar(mesh, ax=axes[row, col], label=units)
            axes[row, col].set(title=f"{label}: {name.replace('_', ' ')}",
                               xlabel="x [m]", ylabel="y [m]", aspect="equal")
    fig.suptitle(f"Surface fields after snapshot {step}; anisotropic $y_B$ level {level}")
    fig.savefig(output / "surface_fields.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for col, (name, units) in enumerate((("ustar", "m s$^{-1}$"),
                                          ("heat_flux", "K m s$^{-1}$"))):
        all_values = np.concatenate([r[name].ravel() for r in results.values()])
        all_values = all_values[np.isfinite(all_values)]
        lo, hi = np.min(all_values), np.max(all_values)
        if np.isclose(lo, hi):
            axes[col].axvline(lo, linewidth=2, color="C0",
                              label="both branches: prescribed flux")
            axes[col].set(xlim=(lo-0.1*max(abs(lo), 1), hi+0.1*max(abs(hi), 1)),
                          xlabel=f"{name.replace('_', ' ')} [{units}]",
                          ylabel="PDF (point mass)")
            axes[col].legend()
            continue
        bins = np.linspace(lo, hi, 51)
        for label, fields in results.items():
            values = fields[name].ravel()
            values = values[np.isfinite(values)]
            axes[col].hist(values, bins=bins, density=True, histtype="step",
                           linewidth=1.8, label=label)
        axes[col].set(xlabel=f"{name.replace('_', ' ')} [{units}]", ylabel="PDF")
        axes[col].legend()
    fig.savefig(output / "surface_pdfs.png", dpi=170)
    plt.close(fig)


def error_metrics(python_field: np.ndarray, les_field: np.ndarray) -> dict[str, float]:
    difference = python_field - les_field
    return {"mean_absolute_error": float(np.mean(np.abs(difference))),
            "root_mean_square_error": float(np.sqrt(np.mean(difference*difference))),
            "maximum_absolute_error": float(np.max(np.abs(difference)))}


def anisotropy_diagnostics(results: dict[str, dict[str, np.ndarray]],
                           yb: np.ndarray, cfg: dict[str, str],
                           output: Path, step: int, level: int) -> dict[str, float]:
    """Show the signed response and the wall-law terms that produce it."""
    standard = results["standard"]
    aniso = results["anisotropic"]
    delta = aniso["ustar"] - standard["ustar"]
    delta_psi = aniso["psi_m"] - standard["psi_m"]
    standard_over_aniso = np.full(delta.shape, np.nan)
    np.divide(standard["ustar"], aniso["ustar"], out=standard_over_aniso,
              where=np.abs(aniso["ustar"]) > 1e-12)
    minimum_reference = 0.01  # avoid dividing by nearly zero ustar
    valid_relative = standard["ustar"] >= minimum_reference
    relative_pct = np.full(delta.shape, np.nan)
    relative_pct[valid_relative] = 100 * delta[valid_relative] / standard["ustar"][valid_relative]
    z1_nd = 0.5 * number(cfg, "lz") / number(cfg, "nz") / number(cfg, "zi")
    with np.errstate(divide="ignore", invalid="ignore"):
        instability = -z1_nd / standard["L0_nd"]
    cap_limit = 0.41**-3
    capped = (standard["L0_nd"] < 0) & (instability > cap_limit)
    x = np.linspace(0, number(cfg, "lx"), delta.shape[0]+1)
    y = np.linspace(0, number(cfg, "ly"), delta.shape[1]+1)

    def map_plot(field: np.ndarray, filename: str, title: str, units: str) -> None:
        finite = field[np.isfinite(field)]
        limit = max(float(np.max(np.abs(finite))), 1e-14) if finite.size else 1e-14
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        mesh = ax.pcolormesh(x, y, field.T, shading="flat", cmap="RdBu_r",
                             norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        fig.colorbar(mesh, ax=ax, label=units)
        ax.set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    map_plot(delta, "anisotropic_minus_standard_ustar.png",
             f"Anisotropic minus standard $u_*$, $y_B$ level {level}",
             r"$\Delta u_*$ [m s$^{-1}$]")
    map_plot(relative_pct, "anisotropic_relative_ustar.png",
             f"Relative change in $u_*$ (reference $u_* \geq$ {minimum_reference} m/s)",
             r"$100\Delta u_*/u_{*,\mathrm{standard}}$ [%]")
    map_plot(delta_psi, "anisotropic_minus_standard_psi_m.png",
             f"Anisotropic minus standard $\psi_m$, $y_B$ level {level}",
             r"$\Delta\psi_m$ [dimensionless]")

    ratio_values = standard_over_aniso[np.isfinite(standard_over_aniso)]
    if ratio_values.size:
        ratio_span = max(float(np.max(np.abs(ratio_values - 1))), 1e-12)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        mesh = ax.pcolormesh(x, y, standard_over_aniso.T, shading="flat",
                             cmap="RdBu_r", norm=TwoSlopeNorm(
                                 vmin=1-ratio_span, vcenter=1, vmax=1+ratio_span))
        fig.colorbar(mesh, ax=ax, label=r"$u_{*,\mathrm{standard}}/u_{*,\mathrm{anisotropic}}$ [dimensionless]")
        ax.set(title=f"Standard/aniso $u_*$ ratio, $y_B$ level {level}",
               xlabel="x [m]", ylabel="y [m]", aspect="equal")
        fig.savefig(output / "standard_over_anisotropic_ustar.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        if np.min(ratio_values) == np.max(ratio_values):
            ax.axvline(float(ratio_values[0]), color="C0", linewidth=2,
                       label="point mass")
            ax.legend()
        else:
            ax.hist(ratio_values, bins=60, density=True, color="C0")
        ax.axvline(1, color="k", linestyle="--", linewidth=1, label="equal $u_*$")
        ax.set(title=f"Spatial PDF of standard/aniso $u_*$ ratio, $y_B$ level {level}",
               xlabel=r"$u_{*,\mathrm{standard}}/u_{*,\mathrm{anisotropic}}$",
               ylabel="Spatial PDF")
        fig.savefig(output / "standard_over_anisotropic_ustar_pdf.png", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    if np.min(delta) == np.max(delta):
        axes[0, 0].axvline(float(delta.flat[0]), color="C0", linewidth=2,
                           label="point mass")
        axes[0, 0].legend()
    else:
        bins = np.linspace(float(np.min(delta)), float(np.max(delta)), 61)
        axes[0, 0].hist(delta.ravel(), bins=bins, density=True, color="C0")
    axes[0, 0].axvline(0, color="k", linewidth=1)
    axes[0, 0].set(xlabel=r"$\Delta u_*$ [m s$^{-1}$]", ylabel="Spatial PDF",
                   title="Signed local changes")

    axes[0, 1].scatter(standard["ustar"], aniso["ustar"], s=5, alpha=0.25,
                       color="C0", rasterized=True)
    common_max = max(float(np.max(standard["ustar"])), float(np.max(aniso["ustar"])))
    axes[0, 1].plot([0, common_max], [0, common_max], "k--", linewidth=1)
    axes[0, 1].set(xlabel=r"Standard $u_*$ [m s$^{-1}$]",
                   ylabel=r"Anisotropic $u_*$ [m s$^{-1}$]",
                   title="Cell-by-cell comparison", xlim=(0, common_max),
                   ylim=(0, common_max), aspect="equal")

    axes[1, 0].scatter(yb[~capped], delta[~capped], s=5, alpha=0.25,
                       color="C0", label="anisotropic law active", rasterized=True)
    axes[1, 0].scatter(yb[capped], delta[capped], s=5, alpha=0.25,
                       color="C1", label="shared high-instability cap", rasterized=True)
    axes[1, 0].axhline(0, color="k", linewidth=1)
    axes[1, 0].set(xlabel=r"$y_B$ at wall-model level",
                   ylabel=r"$\Delta u_*$ [m s$^{-1}$]",
                   title="Response versus anisotropy")
    axes[1, 0].legend(markerscale=2)

    unstable = np.isfinite(instability) & (instability > 0)
    axes[1, 1].scatter(instability[unstable & ~capped], delta[unstable & ~capped],
                       s=5, alpha=0.25, color="C0", rasterized=True)
    axes[1, 1].scatter(instability[capped], delta[capped],
                       s=5, alpha=0.25, color="C1", rasterized=True)
    axes[1, 1].axvline(cap_limit, color="k", linestyle="--", linewidth=1,
                       label=f"shared cap: {cap_limit:.1f}")
    axes[1, 1].axhline(0, color="k", linewidth=1)
    axes[1, 1].set(xscale="log", xlabel=r"$-z_1/L_0$ (first-pass instability)",
                   ylabel=r"$\Delta u_*$ [m s$^{-1}$]",
                   title="Response versus stability")
    axes[1, 1].legend()
    fig.suptitle(f"Anisotropic response at wall call {step+1}; $y_B$ level {level}")
    fig.savefig(output / "anisotropy_diagnostics.png", dpi=170)
    plt.close(fig)

    np.savez_compressed(output / "anisotropy_difference_fields.npz",
                        ustar_anisotropic_minus_standard=delta,
                        ustar_relative_change_percent=relative_pct,
                        ustar_standard_over_anisotropic=standard_over_aniso,
                        psi_m_anisotropic_minus_standard=delta_psi,
                        first_pass_instability=instability,
                        shared_high_instability_cap=capped, yB=yb)
    nonzero = np.abs(delta) > 1e-12
    return {"mean_signed_m_s": float(np.mean(delta)),
            "median_signed_m_s": float(np.median(delta)),
            "mean_absolute_m_s": float(np.mean(np.abs(delta))),
            "minimum_signed_m_s": float(np.min(delta)),
            "maximum_signed_m_s": float(np.max(delta)),
            "fraction_positive": float(np.mean(delta > 1e-12)),
            "fraction_negative": float(np.mean(delta < -1e-12)),
            "fraction_unchanged_within_1e_12_m_s": float(np.mean(~nonzero)),
            "fraction_at_shared_high_instability_cap": float(np.mean(capped)),
            "relative_change_reference_floor_m_s": minimum_reference,
            "fraction_masked_from_relative_change": float(np.mean(~valid_relative)),
            "fraction_with_absolute_relative_change_over_1_percent": float(
                np.mean(np.abs(relative_pct[valid_relative]) > 1)) if np.any(valid_relative) else None,
            "fraction_with_absolute_relative_change_over_5_percent": float(
                np.mean(np.abs(relative_pct[valid_relative]) > 5)) if np.any(valid_relative) else None,
            "standard_over_anisotropic_ratio_mean": float(np.mean(ratio_values)) if ratio_values.size else None,
            "standard_over_anisotropic_ratio_minimum": float(np.min(ratio_values)) if ratio_values.size else None,
            "standard_over_anisotropic_ratio_maximum": float(np.max(ratio_values)) if ratio_values.size else None,
            "mean_absolute_psi_m_difference": float(np.mean(np.abs(delta_psi)))}


def plot_comparison(python_fields: dict[str, np.ndarray],
                    les_fields: dict[str, np.ndarray], lx: float, ly: float,
                    output: Path, snapshot_step: int, branch: str) -> None:
    """Plot matched fields, their spatial PDFs, and absolute ustar error."""
    names = (("ustar", r"m s$^{-1}$"), ("heat_flux", r"K m s$^{-1}$"))
    x = np.linspace(0, lx, python_fields["ustar"].shape[0]+1)
    y = np.linspace(0, ly, python_fields["ustar"].shape[1]+1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for row, (name, units) in enumerate(names):
        combined = np.concatenate((python_fields[name].ravel(), les_fields[name].ravel()))
        lo, hi = np.min(combined), np.max(combined)
        if np.isclose(lo, hi):
            padding = 0.01 * max(abs(lo), 1)
            lo, hi = lo-padding, hi+padding
        for col, (source, fields) in enumerate((("Python", python_fields),
                                                ("LES checkpoint", les_fields))):
            mesh = axes[row, col].pcolormesh(x, y, fields[name].T, shading="flat",
                                              vmin=lo, vmax=hi)
            fig.colorbar(mesh, ax=axes[row, col], label=units)
            axes[row, col].set(title=f"{source}: {name.replace('_', ' ')}",
                               xlabel="x [m]", ylabel="y [m]", aspect="equal")
    fig.suptitle(f"{branch} wall law: snapshot {snapshot_step} to LES checkpoint {snapshot_step+1}")
    fig.savefig(output / "comparison_fields.png", dpi=170)
    plt.close(fig)


def available_steps(sim_dir: Path, start: int, end: int) -> list[int]:
    """Return saved snapshot steps in an inclusive interval."""
    steps = []
    for path in (sim_dir / "output_snap").glob("snapshots_*.dat"):
        try:
            step = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if start <= step <= end:
            steps.append(step)
    return sorted(steps)


def plot_snapshot_ustar(standard: np.ndarray, anisotropic: np.ndarray,
                        lx: float, ly: float, output: Path,
                        step: int, level: int,
                        reference_fields: tuple[np.ndarray, np.ndarray] | None = None,
                        version_label: str = "extended",
                        filename: str = "snapshot_ustar_comparison.png",
                        comparison_anisotropic: np.ndarray | None = None,
                        title_override: str | None = None,
                        save_snapshot_fields: bool = True) -> dict[str, float]:
    """Show the two fields, their response maps, and all requested spatial PDFs."""
    difference = anisotropic - standard
    absolute_difference = np.abs(difference)
    ratio = np.full_like(standard, np.nan)
    np.divide(standard, anisotropic, out=ratio,
              where=np.isfinite(anisotropic) & (np.abs(anisotropic) > 1e-12))
    scale_standard, scale_anisotropic = (reference_fields if reference_fields is not None
                                        else (standard, anisotropic))
    scale_difference = scale_anisotropic - scale_standard
    scale_ratio = np.full_like(scale_standard, np.nan)
    np.divide(scale_standard, scale_anisotropic, out=scale_ratio,
              where=np.isfinite(scale_anisotropic) & (np.abs(scale_anisotropic) > 1e-12))
    if comparison_anisotropic is not None:
        if reference_fields is not None or comparison_anisotropic.shape != standard.shape:
            raise ValueError("Shared scales require a matching comparison field and no reference fields")
        other_ratio = np.full_like(standard, np.nan)
        np.divide(standard, comparison_anisotropic, out=other_ratio,
                  where=np.isfinite(comparison_anisotropic) &
                  (np.abs(comparison_anisotropic) > 1e-12))
        scale_anisotropic = np.concatenate((anisotropic.ravel(), comparison_anisotropic.ravel()))
        scale_difference = np.concatenate((difference.ravel(),
                                           (comparison_anisotropic-standard).ravel()))
        scale_ratio = np.concatenate((ratio.ravel(), other_ratio.ravel()))
    x = np.linspace(0, lx, standard.shape[0] + 1)
    y = np.linspace(0, ly, standard.shape[1] + 1)
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)

    def map_panel(ax, field: np.ndarray, title: str, label: str,
                  cmap: str, **kwargs) -> None:
        mesh = ax.pcolormesh(x, y, field.T, shading="flat", cmap=cmap, **kwargs)
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")

    finite_ustar = np.concatenate((scale_standard[np.isfinite(scale_standard)],
                                   scale_anisotropic[np.isfinite(scale_anisotropic)]))
    lo, hi = float(np.min(finite_ustar)), float(np.max(finite_ustar))
    if np.isclose(lo, hi):
        padding = 0.01 * max(abs(lo), 1.0)
        lo, hi = lo - padding, hi + padding
    for ax, field, title in ((axes[0, 0], standard, "Standard $u_*$"),
                             (axes[0, 1], anisotropic, "Anisotropy-adjusted $u_*$")):
        map_panel(ax, field, title, r"$u_*$ [m s$^{-1}$]", "viridis",
                  vmin=lo, vmax=hi)
    signed_span = max(float(np.nanmax(np.abs(scale_difference))), 1e-14)
    map_panel(axes[0, 2], difference, "Signed difference: anisotropic − standard",
              r"$\Delta u_*$ [m s$^{-1}$]", "RdBu_r",
              norm=TwoSlopeNorm(vmin=-signed_span, vcenter=0, vmax=signed_span))
    map_panel(axes[0, 3], absolute_difference, "Absolute difference",
              r"$|\Delta u_*|$ [m s$^{-1}$]", "magma", vmin=0,
              vmax=signed_span)
    finite_ratio = ratio[np.isfinite(ratio)]
    finite_scale_ratio = scale_ratio[np.isfinite(scale_ratio)]
    if finite_scale_ratio.size:
        ratio_span = max(float(np.max(np.abs(finite_scale_ratio - 1.0))), 1e-12)
        map_panel(axes[1, 0], ratio, "Standard / anisotropic $u_*$",
                  r"$u_{*,\mathrm{standard}}/u_{*,\mathrm{anisotropic}}$",
                  "RdBu_r", norm=TwoSlopeNorm(vmin=1-ratio_span, vcenter=1,
                                                vmax=1+ratio_span))
    else:
        axes[1, 0].set_visible(False)

    def pdf(ax, values: np.ndarray, label: str, color: str,
            bins: np.ndarray | int = 60) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        if np.isclose(np.min(values), np.max(values)):
            ax.axvline(float(values[0]), color=color, linewidth=2, label=label + " (point mass)")
        else:
            ax.hist(values, bins=bins, density=True, histtype="step",
                    linewidth=2, color=color, label=label)

    shared_bins = np.linspace(lo, hi, 61)
    signed_bins = np.linspace(float(np.nanmin(scale_difference)),
                              float(np.nanmax(scale_difference)), 61)
    ratio_bins = np.linspace(float(np.min(finite_scale_ratio)),
                             float(np.max(finite_scale_ratio)), 61) if finite_scale_ratio.size else 60
    pdf(axes[1, 1], standard.ravel(), "Standard", "C0", shared_bins)
    pdf(axes[1, 1], anisotropic.ravel(), "Anisotropic", "C1", shared_bins)
    axes[1, 1].set(title="$u_*$ spatial PDFs", xlabel=r"$u_*$ [m s$^{-1}$]",
                   ylabel="Spatial PDF")
    axes[1, 1].legend()
    pdf(axes[1, 2], difference.ravel(), "Anisotropic − standard", "C0", signed_bins)
    axes[1, 2].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1, 2].set(title="Signed-difference PDF",
                   xlabel=r"$\Delta u_*$ [m s$^{-1}$]", ylabel="Spatial PDF")
    pdf(axes[1, 3], ratio.ravel(), "Standard / anisotropic", "C0", ratio_bins)
    axes[1, 3].axvline(1, color="k", linestyle="--", linewidth=1)
    axes[1, 3].set(title="Ratio PDF",
                   xlabel=r"$u_{*,\mathrm{standard}}/u_{*,\mathrm{anisotropic}}$",
                   ylabel="Spatial PDF")
    if reference_fields is not None:
        reference_fig, reference_axes = plt.subplots(1, 3)
        pdf(reference_axes[0], scale_standard.ravel(), "Standard", "C0", shared_bins)
        pdf(reference_axes[0], scale_anisotropic.ravel(), "Anisotropic", "C1", shared_bins)
        pdf(reference_axes[1], scale_difference.ravel(), "Anisotropic − standard",
            "C0", signed_bins)
        reference_axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
        pdf(reference_axes[2], scale_ratio.ravel(), "Standard / anisotropic",
            "C0", ratio_bins)
        reference_axes[2].axvline(1, color="k", linestyle="--", linewidth=1)
        for ax, reference_ax in zip(axes[1, 1:], reference_axes):
            ax.set_xlim(reference_ax.get_xlim())
            ax.set_ylim(reference_ax.get_ylim())
        for ax, values, bins in ((axes[1, 2], difference, signed_bins),
                                 (axes[1, 3], finite_ratio, ratio_bins)):
            if values.size:
                peak = float(np.max(np.histogram(values, bins=bins, density=True)[0]))
                if peak > ax.get_ylim()[1]:
                    ax.text(0.97, 0.97, f"PDF peak {peak:.1f} (off-scale)",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=8, bbox={"facecolor": "white", "alpha": 0.85,
                                              "edgecolor": "none"})
        limits = {"ustar_colorbar_m_s": [lo, hi],
                  "signed_difference_colorbar_m_s": [-signed_span, signed_span],
                  "absolute_difference_colorbar_m_s": [0.0, signed_span],
                  "standard_over_anisotropic_colorbar": [1-ratio_span, 1+ratio_span],
                  "pdf_axes": [{"x": list(ax.get_xlim()), "y": list(ax.get_ylim())}
                               for ax in reference_axes],
                  "map_axes_m": {"x": [0.0, lx], "y": [0.0, ly]}}
        (output / "reference_plot_limits.json").write_text(json.dumps(limits, indent=2) + "\n")
        plt.close(reference_fig)
    elif comparison_anisotropic is not None:
        def shared_pdf_limits(ax, bins: np.ndarray,
                              fields: tuple[np.ndarray, ...]) -> None:
            peaks = []
            for field in fields:
                finite = field[np.isfinite(field)]
                if finite.size:
                    peaks.append(float(np.max(np.histogram(finite, bins=bins,
                                                           density=True)[0])))
            ax.set(xlim=(float(bins[0]), float(bins[-1])),
                   ylim=(0.0, 1.05*max(peaks, default=1.0)))

        shared_pdf_limits(axes[1, 1], shared_bins,
                          (standard, anisotropic, comparison_anisotropic))
        shared_pdf_limits(axes[1, 2], signed_bins,
                          (difference, comparison_anisotropic-standard))
        shared_pdf_limits(axes[1, 3], ratio_bins,
                          (ratio, other_ratio))
    suffix = (" (clipped unstable $\\phi_m$)" if version_label == "clipped"
              else " (dynamic unstable $\\psi$ limits)")
    scale_note = (" (previous plot scales)" if reference_fields is not None else
                  " (shared scales)" if comparison_anisotropic is not None else "")
    fig.suptitle(title_override if title_override is not None else
                 f"Snapshot {step}: replayed wall call {step+1}; anisotropy $y_B$ level {level}"
                 f"{suffix}{scale_note}")
    fig.savefig(output / filename, dpi=170)
    plt.close(fig)

    if save_snapshot_fields:
        np.savez_compressed(output / "snapshot_ustar_fields.npz",
                            ustar_standard=standard,
                            ustar_anisotropic=anisotropic,
                            ustar_anisotropic_minus_standard=difference,
                            ustar_absolute_difference=absolute_difference,
                            ustar_standard_over_anisotropic=ratio)
    return {"standard_mean_m_s": float(np.mean(standard)),
            "anisotropic_mean_m_s": float(np.mean(anisotropic)),
            "signed_difference_mean_m_s": float(np.mean(difference)),
            "absolute_difference_mean_m_s": float(np.mean(absolute_difference)),
            "signed_difference_minimum_m_s": float(np.min(difference)),
            "signed_difference_maximum_m_s": float(np.max(difference)),
            "standard_over_anisotropic_ratio_mean": float(np.mean(finite_ratio)) if finite_ratio.size else None,
            "standard_over_anisotropic_ratio_minimum": float(np.min(finite_ratio)) if finite_ratio.size else None,
            "standard_over_anisotropic_ratio_maximum": float(np.max(finite_ratio)) if finite_ratio.size else None}


def plot_averaged_ustar(standard: np.ndarray, anisotropic: np.ndarray,
                        lx: float, ly: float, output: Path, level: int,
                        steps: list[int]) -> dict[str, float]:
    """Save maps and PDFs based on time-averaged ustar fields."""
    difference = anisotropic - standard
    ratio = np.full_like(standard, np.nan)
    np.divide(anisotropic, standard, out=ratio, where=np.abs(standard) > 1e-12)
    x = np.linspace(0, lx, standard.shape[0] + 1)
    y = np.linspace(0, ly, standard.shape[1] + 1)
    first, last = steps[0], steps[-1]

    def symmetric_map(field: np.ndarray, filename: str, title: str, label: str,
                      center: float = 0.0) -> None:
        finite = field[np.isfinite(field)]
        span = max(float(np.max(np.abs(finite - center))), 1e-14)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        mesh = ax.pcolormesh(x, y, field.T, shading="flat", cmap="RdBu_r",
                             norm=TwoSlopeNorm(vmin=center-span, vcenter=center,
                                               vmax=center+span))
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    title_suffix = f"mean of {len(steps)} snapshots ({first}--{last}); $y_B$ level {level}"
    symmetric_map(difference, "mean_anisotropic_minus_standard_ustar.png",
                  f"Mean anisotropic minus standard $u_*$\n{title_suffix}",
                  r"$\overline{u}_{*,\mathrm{anisotropic}}-\overline{u}_{*,\mathrm{standard}}$ [m s$^{-1}$]")
    symmetric_map(ratio, "mean_anisotropic_over_standard_ustar.png",
                  f"Mean anisotropic/standard $u_*$ ratio\n{title_suffix}",
                  r"$\overline{u}_{*,\mathrm{anisotropic}}/\overline{u}_{*,\mathrm{standard}}$",
                  center=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, values, center, xlabel, title in (
        (axes[0], difference[np.isfinite(difference)], 0.0,
         r"$\overline{u}_{*,\mathrm{anisotropic}}-\overline{u}_{*,\mathrm{standard}}$ [m s$^{-1}$]",
         "PDF of mean ustar difference"),
        (axes[1], ratio[np.isfinite(ratio)], 1.0,
         r"$\overline{u}_{*,\mathrm{anisotropic}}/\overline{u}_{*,\mathrm{standard}}$",
         "PDF of mean ustar ratio")):
        if np.isclose(values.min(), values.max()):
            ax.axvline(float(values[0]), color="C0", linewidth=2)
        else:
            ax.hist(values, bins=60, density=True, color="C0")
        ax.axvline(center, color="k", linestyle="--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel="Spatial PDF", title=title)
    fig.suptitle(title_suffix)
    fig.savefig(output / "mean_ustar_difference_and_ratio_pdfs.png", dpi=170)
    plt.close(fig)
    return {"difference_mean_m_s": float(np.mean(difference)),
            "difference_minimum_m_s": float(np.min(difference)),
            "difference_maximum_m_s": float(np.max(difference)),
            "anisotropic_over_standard_ratio_mean": float(np.nanmean(ratio)),
            "anisotropic_over_standard_ratio_minimum": float(np.nanmin(ratio)),
            "anisotropic_over_standard_ratio_maximum": float(np.nanmax(ratio))}

    ustar_absolute_difference = np.abs(python_fields["ustar"] - les_fields["ustar"])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, ustar_absolute_difference.T, shading="flat",
                         cmap="magma", vmin=0)
    fig.colorbar(mesh, ax=ax, label=r"$|u_{*,\mathrm{Python}}-u_{*,\mathrm{LES}}|$ [m s$^{-1}$]")
    ax.set(title=f"Absolute ustar difference: snapshot {snapshot_step} vs checkpoint {snapshot_step+1}",
           xlabel="x [m]", ylabel="y [m]", aspect="equal")
    fig.savefig(output / "ustar_absolute_difference.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for col, (name, units) in enumerate(names):
        python_values = python_fields[name].ravel()
        les_values = les_fields[name].ravel()
        all_values = np.concatenate((python_values, les_values))
        lo, hi = np.min(all_values), np.max(all_values)
        if np.isclose(lo, hi):
            # A prescribed constant flux has a point-mass PDF, not a density curve.
            axes[col].axvline(lo, color="C0", linewidth=2,
                              label="Python = LES (point mass)")
            padding = 0.1*max(abs(lo), 1)
            axes[col].set(xlim=(lo-padding, hi+padding), ylabel="PDF (point mass)")
        else:
            bins = np.linspace(lo, hi, 51)
            axes[col].hist(python_values, bins=bins, density=True,
                           color="C0", alpha=0.4, label="Python")
            axes[col].hist(les_values, bins=bins, density=True,
                           histtype="step", color="C1", linewidth=1.8,
                           label="LES checkpoint")
            axes[col].set(ylabel="PDF")
        axes[col].set(xlabel=f"{name.replace('_', ' ')} [{units}]")
        axes[col].legend()
    fig.suptitle(f"Spatial PDFs for wall call {snapshot_step+1}")
    fig.savefig(output / "comparison_pdfs.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--snapshot-step", type=int, default=None,
                        help="Replay one matched snapshot and make the eight-panel ustar comparison")
    parser.add_argument("--start-step", type=int, default=136110)
    parser.add_argument("--end-step", type=int, default=146010)
    parser.add_argument("--anisotropy-level", type=int, default=5,
                        help="1-based vertical grid point for yB (default: 5)")
    parser.add_argument("--unstable-phi-m-version", choices=("extended", "clipped", "both"),
                        default="extended",
                        help="Current dynamic-psi law, earlier stability-clipped law, or both for averaging")
    parser.add_argument("--reference-fields", type=Path, default=None,
                        help="Saved extended-law snapshot fields supplying fixed plot scales for clipped mode")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Destination for snapshot or averaged fields and plots")
    args = parser.parse_args()
    if args.start_step > args.end_step:
        parser.error("--start-step must not exceed --end-step")
    cfg = read_namelist(args.sim_dir / "parameters.nlst")
    nx, ny, nz = (int(number(cfg, k)) for k in ("nx", "ny", "nz"))
    run_level = int(number(cfg, "anisotropy_surface_level")) if "anisotropy_surface_level" in cfg else 1
    level = args.anisotropy_level
    if not 1 <= level <= nz:
        raise ValueError(f"Anisotropy level must be between 1 and nz={nz}; got {level}")
    if int(number(cfg, "nb_sc")) != 1:
        raise ValueError("This script currently expects one temperature scalar")
    if flag(cfg, "sfc_z0_files") or flag(cfg, "sfc_sc_z0_files"):
        raise ValueError("Spatial roughness files need to be read for this run")
    if flag(cfg, "sfc_lsm"):
        raise ValueError("Land surface model boundary condition is unsupported")
    if flag(cfg, "sfc_sc_timedep") or flag(cfg, "sfc_sc_files"):
        raise ValueError("Time-varying or spatial scalar boundary input is unsupported")
    if cfg["scalar_passive"].split()[0].lower() != ".false.":
        raise ValueError("Temperature must be an active scalar")
    if cfg["lbc_sc"].split()[0] not in ("0", "1"):
        raise ValueError("Expected flux (0) or value (1) scalar boundary condition")
    first_level_m = 0.5 * number(cfg, "lz") / number(cfg, "nz")
    dynamic_psi_m_limit = float(np.log(first_level_m / number(cfg, "lbc_dz0")) - PSI_MARGIN)
    dynamic_psi_h_limit = float(np.log(first_level_m / number(cfg, "lbc_dz0_sc")) - PSI_MARGIN)
    checkpoint = args.sim_dir / "output_checkpoint"
    if args.snapshot_step is not None:
        if args.unstable_phi_m_version == "both":
            parser.error("--unstable-phi-m-version both is only supported for averaging")
        step = args.snapshot_step
        snap = args.sim_dir / "output_snap" / f"snapshots_{step:09d}.dat"
        surface_path = checkpoint / f"checkpoint_wsc_{step:09d}_sfc"
        yb_path = checkpoint / f"checkpoint_aniso_{step:09d}"
        missing = [path for path in (snap, surface_path, yb_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Snapshot comparison needs matching saved state:\n" +
                                    "\n".join(str(path) for path in missing))
        if step + 1 < int(number(cfg, "scalar_init")):
            raise ValueError("Selected snapshot precedes active-scalar initialization")
        if args.output_dir is None:
            args.output_dir = (Path(__file__).resolve().parent / "LESustarheat_output" /
                               (f"snapshot_{step:09d}_yB_level_{level:02d}_dynamic_psi"
                                if args.unstable_phi_m_version == "extended" else
                                f"snapshot_{step:09d}_yB_level_{level:02d}_clipped_phi_m"))
        u, v, _w, sc = read_fields(snap, (nx, ny, nz), 4)
        surface = read_surface(surface_path, nx, ny)
        yb = read_yb(yb_path, nx, ny, nz, level)
        if not np.all(np.isfinite(yb)):
            raise ValueError(f"yB contains non-finite values at step {step}")
        standard = calculate(u[:, :, 0], v[:, :, 0], sc[:, :, 0],
                             surface, yb, cfg, False)
        anisotropic = calculate(u[:, :, 0], v[:, :, 0], sc[:, :, 0],
                                surface, yb, cfg, True, args.unstable_phi_m_version)
        u_scale = number(cfg, "u_scale")
        reference_fields = None
        reference_path = args.reference_fields
        if reference_path is None:
            reference_path = (Path(__file__).resolve().parent / "LESustarheat_output" /
                              f"snapshot_{step:09d}_yB_level_{level:02d}_extended_phi_m" /
                              "snapshot_ustar_fields.npz")
        if reference_path.is_file():
            with np.load(reference_path) as saved:
                reference_fields = (saved["ustar_standard"], saved["ustar_anisotropic"])
            if not np.allclose(reference_fields[0], standard["ustar_nd"] * u_scale,
                               rtol=0.0, atol=1e-12):
                raise ValueError("Reference fields do not match the standard ustar for this snapshot")
        elif args.unstable_phi_m_version == "clipped" or args.reference_fields is not None:
            raise FileNotFoundError(f"Reference fields missing: {reference_path}")
        else:
            reference_path = None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        standard_ustar = standard["ustar_nd"] * u_scale
        anisotropic_ustar = anisotropic["ustar_nd"] * u_scale
        diagnostics = plot_snapshot_ustar(
            standard_ustar, anisotropic_ustar, number(cfg, "lx"), number(cfg, "ly"),
            args.output_dir, step, level,
            reference_fields if args.unstable_phi_m_version == "clipped" else None,
            args.unstable_phi_m_version)
        if args.unstable_phi_m_version == "extended" and reference_fields is not None:
            plot_snapshot_ustar(
                standard_ustar, anisotropic_ustar, number(cfg, "lx"), number(cfg, "ly"),
                args.output_dir, step, level, reference_fields,
                args.unstable_phi_m_version,
                filename="snapshot_ustar_comparison_previous_scales.png")
        summary = {"simulation": str(args.sim_dir), "snapshot_step": step,
                   "replayed_wall_call_step": step+1,
                   "anisotropy_checkpoint": str(yb_path),
                   "surface_checkpoint": str(surface_path),
                   "wall_model": cfg["wall_model"],
                   "run_anisotropy_flag": cfg["anisotropy_flag"],
                   "comparison_is_counterfactual": True,
                   "anisotropy_surface_level": level,
                   "run_anisotropy_surface_level": run_level,
                   "unstable_anisotropic_phi_m_version": (
                       "dynamic_psi" if args.unstable_phi_m_version == "extended" else "clipped"),
                   "unstable_anisotropic_psi_m_limit": (
                       dynamic_psi_m_limit if args.unstable_phi_m_version == "extended"
                       else float(bruts_psi_m(np.array([0.41 ** -3]))[0])),
                   "unstable_anisotropic_psi_h_limit": (
                       dynamic_psi_h_limit if args.unstable_phi_m_version == "extended" else None),
                   "momentum_cells_at_dynamic_psi_ceiling": (
                       int(np.count_nonzero(np.isclose(anisotropic["psi_m"],
                                                           dynamic_psi_m_limit, rtol=0, atol=1e-9)))
                       if args.unstable_phi_m_version == "extended" else None),
                   "heat_cells_at_dynamic_psi_ceiling": (
                       int(np.count_nonzero(np.isclose(anisotropic["psi_h"],
                                                           dynamic_psi_h_limit, rtol=0, atol=1e-9)))
                       if args.unstable_phi_m_version == "extended" else None),
                   "fixed_plot_scales_from": str(reference_path) if reference_path else None,
                   "difference_definition": "anisotropic minus standard",
                   "ratio_definition": "standard divided by anisotropic",
                   "units": {"ustar": "m/s"}, "metrics": diagnostics}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Saved snapshot fields, plots and summary in {args.output_dir}")
        return
    if args.output_dir is None:
        args.output_dir = (Path(__file__).resolve().parent / "LESustarheat_output" /
                           (f"mean_{args.start_step:09d}_{args.end_step:09d}_yB_level_{level:02d}_dynamic_psi"
                            if args.unstable_phi_m_version == "extended" else
                            f"mean_{args.start_step:09d}_{args.end_step:09d}_yB_level_{level:02d}_clipped_phi_m"
                            if args.unstable_phi_m_version == "clipped" else
                            f"mean_{args.start_step:09d}_{args.end_step:09d}_yB_level_{level:02d}_both"))
    steps = available_steps(args.sim_dir, args.start_step, args.end_step)
    if not steps:
        raise FileNotFoundError(f"No snapshots in {args.sim_dir / 'output_snap'} between "
                                f"{args.start_step} and {args.end_step}")
    if steps[0] + 1 < int(number(cfg, "scalar_init")):
        raise ValueError("Selected snapshots occur before active-scalar initialization")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    standard_sum = np.zeros((nx, ny))
    versions = (("extended", "clipped") if args.unstable_phi_m_version == "both"
                else (args.unstable_phi_m_version,))
    anisotropic_sums = {version: np.zeros((nx, ny)) for version in versions}
    yb_sum = np.zeros((nx, ny))
    snapshot_differences = {version: np.empty((len(steps), nx, ny))
                            for version in versions}
    u_scale = number(cfg, "u_scale")
    for index, step in enumerate(steps):
        snap = args.sim_dir / "output_snap" / f"snapshots_{step:09d}.dat"
        surface_path = checkpoint / f"checkpoint_wsc_{step:09d}_sfc"
        yb_path = checkpoint / f"checkpoint_aniso_{step:09d}"
        missing = [path for path in (snap, surface_path, yb_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required saved state:\n" +
                                    "\n".join(str(path) for path in missing))
        u, v, _w, sc = read_fields(snap, (nx, ny, nz), 4)
        surface = read_surface(surface_path, nx, ny)
        yb = read_yb(yb_path, nx, ny, nz, level)
        if not np.all(np.isfinite(yb)):
            raise ValueError(f"yB contains non-finite values at step {step}")
        standard_ustar = calculate(u[:, :, 0], v[:, :, 0], sc[:, :, 0],
                                   surface, yb, cfg, False)["ustar_nd"] * u_scale
        standard_sum += standard_ustar
        for version in versions:
            anisotropic_ustar = calculate(u[:, :, 0], v[:, :, 0], sc[:, :, 0],
                                          surface, yb, cfg, True, version)["ustar_nd"] * u_scale
            anisotropic_sums[version] += anisotropic_ustar
            snapshot_differences[version][index] = anisotropic_ustar - standard_ustar
        yb_sum += yb
        print(f"Processed snapshot {step}")
    standard_mean = standard_sum / len(steps)
    anisotropic_means = {version: anisotropic_sums[version] / len(steps)
                         for version in versions}
    yb_mean = yb_sum / len(steps)
    summaries = {}
    for version in versions:
        label = "dynamic_psi" if version == "extended" else "clipped_phi_m"
        version_output = args.output_dir / label if len(versions) > 1 else args.output_dir
        version_output.mkdir(parents=True, exist_ok=True)
        anisotropic_mean = anisotropic_means[version]
        comparison_anisotropic = (anisotropic_means[next(v for v in versions if v != version)]
                                  if len(versions) > 1 else None)
        comparison_metrics = plot_snapshot_ustar(
            standard_mean, anisotropic_mean, number(cfg, "lx"), number(cfg, "ly"),
            version_output, steps[-1], level, version_label=version,
            filename="mean_ustar_comparison.png",
            title_override=(f"Mean of {len(steps)} snapshots ({steps[0]}–{steps[-1]}); "
                            f"$y_B$ level {level}; {label.replace('_', ' ')}; own scales"),
            save_snapshot_fields=False)
        if comparison_anisotropic is not None:
            plot_snapshot_ustar(
                standard_mean, anisotropic_mean, number(cfg, "lx"), number(cfg, "ly"),
                version_output, steps[-1], level, version_label=version,
                filename="mean_ustar_comparison_shared_scales.png",
                comparison_anisotropic=comparison_anisotropic,
                title_override=(f"Mean of {len(steps)} snapshots ({steps[0]}–{steps[-1]}); "
                                f"$y_B$ level {level}; {label.replace('_', ' ')}; shared scales"),
                save_snapshot_fields=False)
        diagnostics = plot_averaged_ustar(standard_mean, anisotropic_mean,
                                          number(cfg, "lx"), number(cfg, "ly"),
                                          version_output, level, steps)
        ratio_aniso_over_standard = np.full_like(standard_mean, np.nan)
        ratio_standard_over_aniso = np.full_like(standard_mean, np.nan)
        np.divide(anisotropic_mean, standard_mean, out=ratio_aniso_over_standard,
                  where=np.abs(standard_mean) > 1e-12)
        np.divide(standard_mean, anisotropic_mean, out=ratio_standard_over_aniso,
                  where=np.abs(anisotropic_mean) > 1e-12)
        np.savez_compressed(version_output / "mean_ustar_fields.npz",
                            snapshot_steps=np.asarray(steps), yB_mean=yb_mean,
                            ustar_standard_mean=standard_mean,
                            ustar_anisotropic_mean=anisotropic_mean,
                            ustar_anisotropic_minus_standard_mean=anisotropic_mean-standard_mean,
                            ustar_anisotropic_over_standard_mean=ratio_aniso_over_standard,
                            ustar_standard_over_anisotropic_mean=ratio_standard_over_aniso)
        np.savez_compressed(version_output / "snapshot_ustar_difference_fields.npz",
                            snapshot_steps=np.asarray(steps),
                            ustar_anisotropic_minus_standard=snapshot_differences[version])
        summary = {"simulation": str(args.sim_dir),
                   "snapshot_steps": steps,
                   "number_of_snapshots": len(steps),
                   "requested_step_range": [args.start_step, args.end_step],
                   "wall_model": cfg["wall_model"],
                   "run_anisotropy_flag": cfg["anisotropy_flag"],
                   "comparison_is_counterfactual": not flag(cfg, "anisotropy_flag"),
                   "anisotropy_surface_level": level,
                   "run_anisotropy_surface_level": run_level,
                   "unstable_anisotropic_phi_m_version": label,
                   "unstable_anisotropic_psi_m_limit": (
                       dynamic_psi_m_limit if version == "extended" else
                       float(bruts_psi_m(np.array([0.41 ** -3]))[0])),
                   "unstable_anisotropic_psi_h_limit": (
                       dynamic_psi_h_limit if version == "extended" else None),
                   "difference_definition": "mean anisotropic minus mean standard",
                   "comparison_plot_ratio_definition": "mean standard divided by mean anisotropic",
                   "units": {"ustar": "m/s"},
                   "eight_panel_metrics": comparison_metrics,
                   "averaged_ustar": diagnostics,
                   "standard_mean_m_s": float(np.mean(standard_mean)),
                   "anisotropic_mean_m_s": float(np.mean(anisotropic_mean))}
        (version_output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        summaries[label] = {"output_dir": str(version_output),
                            "metrics": comparison_metrics}
        print(f"Saved {label} averaged maps, PDFs, arrays and summary in {version_output}")
    if len(versions) > 1:
        (args.output_dir / "summary.json").write_text(json.dumps({
            "snapshot_steps": steps,
            "number_of_snapshots": len(steps),
            "shared_plot_scales": True,
            "formulations": summaries}, indent=2)+"\n")


if __name__ == "__main__":
    main()
