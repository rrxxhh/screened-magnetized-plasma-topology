from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import scipy.linalg as la

# Try to import joblib's Parallel/delayed; if unavailable, provide a lightweight fallback.
try:
    from joblib import Parallel, delayed  # type: ignore
except Exception:
    import os
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    warnings.warn("joblib not available, falling back to concurrent.futures ThreadPoolExecutor")

    def delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    def Parallel(n_jobs=1, **_kwargs):
        class _Parallel:
            def __init__(self, n_jobs):
                self.n_jobs = n_jobs

            def __call__(self, tasks):
                tasks = list(tasks)
                if not tasks:
                    return []
                if self.n_jobs == 1:
                    return [task() for task in tasks]
                max_workers = os.cpu_count() if self.n_jobs == -1 or self.n_jobs is None else self.n_jobs
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(task) for task in tasks]
                    return [future.result() for future in futures]

        return _Parallel(n_jobs)




# ============================================================
# Chebyshev differentiation matrix on [-1, 1]
# ============================================================
def chebD(N):
    """
    Return Chebyshev differentiation matrix D and grid y.
    N is polynomial order, so the matrix size is (N+1) x (N+1).
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])

    k = np.arange(N + 1)
    y = np.cos(np.pi * k / N)

    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * (-1) ** k

    Y = np.tile(y.reshape(-1, 1), (1, N + 1))
    dY = Y - Y.T

    D = np.outer(c, 1 / c) / (dY + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))

    return D, y


# ============================================================
# Parameters
# ============================================================
@dataclass(frozen=True)
class SolverConfig:
    kz: float = 0
    nu: float = 0 #-0.5
    nint: int = 180
    lmap: float = 2.5
    ky_min: float = -3.0
    ky_max: float = 3.0
    n_ky: int = 101
    use_parallel: bool = True
    n_jobs: int = -1
    plot_cloud: bool = True
    include_nu: bool = False
    save_figures: bool = True
    show_figures: bool = True
    output_dir: Path = Path("figures")
    dpi: int = 200
    nu=0.1


def mu_fun(x):
    return np.tanh(x)


# def mu_fun(x):
#     return x

CONFIG = SolverConfig(
    # Edit SolverConfig above to change parameters. Keeping CONFIG empty avoids
    # accidentally overriding a changed default such as kz.
)


# ============================================================
# 0) Chebyshev differentiation and mapping x = L y / sqrt(1-y^2)
# ============================================================
def build_problem(
    config: SolverConfig,
    mu: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, np.ndarray | int | None]:
    """Build matrices that do not depend on ky."""
    if mu is None:
        mu = mu_fun

    n_cheb = config.nint + 2

    dy_full, y_full = chebD(n_cheb - 1)

    # Drop endpoints y = +/- 1.
    idx = np.arange(1, n_cheb - 1)
    y = y_full[idx]
    dy = dy_full[np.ix_(idx, idx)]
    d2y = dy @ dy

    # Mapping y -> x.
    x = config.lmap * y / np.sqrt(1 - y**2)

    # Chain rule factors.
    a = (1 - y**2) ** 1.5 / config.lmap
    b = -3 * y * (1 - y**2) ** 2 / (config.lmap**2)

    dx = np.diag(a) @ dy
    dxx = np.diag(a**2) @ d2y + np.diag(b) @ dy

    n = len(y)
    eye = np.eye(n)
    zero = np.zeros((n, n), dtype=complex)
    mu_diag = np.diag(mu(x)).astype(complex)

    # eta block: M = -d_x^2 + 1.
    mat_m = -dxx + eye

    # M^{-1} is reused for every ky. This removes two linear solves per ky.
    minv = la.solve(mat_m, eye, check_finite=False)
    minv_d = minv @ dx
    nu=config.nu

    anu = None
    if config.include_nu:
        anu = config.nu * np.block([
            [1j * eye, zero,      zero,      zero],
            [zero,     1j * eye, zero,      zero],
            [zero,     zero,     1j * eye, zero],
            [zero,     zero,     zero,      zero],
        ])

    return {
        "dim": 4 * n,
        "I": eye,
        "Z": zero,
        "Mu": mu_diag,
        "Dx": dx,
        "MinvI": minv,
        "MinvD": minv_d,
        "Bkz": config.kz * eye,
        "MinvBkz": config.kz * minv,
        "nu": nu,
        "Anu": anu,
    }


# ============================================================
# 1) Function to compute spectrum for one ky
# ============================================================
def compute_spectrum_for_ky(
    ky: float,
    problem: dict[str, np.ndarray | int | None],
) -> np.ndarray:
    eye = problem["I"]
    zero = problem["Z"]
    mu_diag = problem["Mu"]
    dx = problem["Dx"]
    minv_i = problem["MinvI"]
    minv_d = problem["MinvD"]
    bkz = problem["Bkz"]
    minv_bkz = problem["MinvBkz"]
    nu=problem["nu"]

    if not all(
        isinstance(item, np.ndarray)
        for item in (eye, zero, mu_diag, dx, minv_i, minv_d, bkz, minv_bkz)
    ):
        raise TypeError("Problem matrices are missing")

    bky = ky * eye
    minv_bky = ky * minv_i

    A0 = np.block([
        [1j*nu*eye,           -1j * mu_diag, zero,      -1j * dx],
        [1j * mu_diag,    1j*nu*eye,         zero,       bky],
        [zero,            zero,         1j*nu*eye,       bkz],
        [-1j * minv_d,    minv_bky,     minv_bkz,   zero],
    ])

    anu = problem["Anu"]
    A = A0 + anu if isinstance(anu, np.ndarray) else A0

    eigvals = la.eigvals(A, check_finite=False)
    order = np.argsort(np.real(eigvals))

    return eigvals[order]


# ============================================================
# 2) Compute full spectrum
# ============================================================
def compute_full_spectrum(
    config: SolverConfig | None = None,
    mu: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if config is None:
        config = CONFIG

    problem = build_problem(config, mu)
    ky_list = np.linspace(config.ky_min, config.ky_max, config.n_ky)

    if config.use_parallel:
        parallel = Parallel(n_jobs=config.n_jobs, prefer="threads")
        spectra = parallel(
            delayed(compute_spectrum_for_ky)(float(ky), problem) for ky in ky_list
        )
    else:
        spectra = [compute_spectrum_for_ky(float(ky), problem) for ky in ky_list]

    omega_all = np.column_stack(spectra)
    return ky_list, omega_all, int(problem["dim"])


def _format_float(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def parameter_tag(config: SolverConfig) -> str:
    return "_".join([
        f"N{config.nint}",
        f"L{_format_float(config.lmap)}",
        f"kz{_format_float(config.kz)}",
        f"nu{_format_float(config.nu)}",
        f"ky{_format_float(config.ky_min)}to{_format_float(config.ky_max)}",
        f"nky{config.n_ky}",
        f"incnu{int(config.include_nu)}",
    ])


def _save_figure(fig, output_dir: Path, filename: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _subplots_with_backend_fallback(plt, show_figures: bool, **kwargs):
    try:
        fig, ax = plt.subplots(**kwargs)
    except Exception as exc:
        if not show_figures:
            raise
        print(f"Interactive Matplotlib backend unavailable; saving figures only. ({type(exc).__name__})")
        plt.close("all")
        plt.switch_backend("Agg")
        fig, ax = plt.subplots(**kwargs)
        show_figures = False

    return fig, ax, show_figures


# ============================================================
# 3) Plot spectrum
# ============================================================
def plot_spectrum(
    ky_list: np.ndarray,
    omega_all: np.ndarray,
    config: SolverConfig | None = None,
) -> list[Path]:
    if config is None:
        config = CONFIG

    if not config.show_figures:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    show_figures = config.show_figures

    # Clear old figures first; this prevents IDE/interactive runs from showing stale plots.
    plt.close("all")

    dim = omega_all.shape[0]
    ky_flat = np.repeat(ky_list, dim)
    omega_flat = omega_all.ravel(order="F")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    prefix = f"{run_id}_{parameter_tag(config)}"
    output_dir = Path(config.output_dir)
    saved_paths: list[Path] = []

    fig_re, ax_re, show_figures = _subplots_with_backend_fallback(plt, show_figures, figsize=(6, 4))
    ax_re.scatter(ky_flat, np.real(omega_flat), s=2, marker=".", linewidths=0)
    ax_re.set_xlabel(r"$k_y$")
    ax_re.set_ylabel(r"$\mathrm{Re}(\omega)$")
    ax_re.set_title(r"Full spectrum: $\mathrm{Re}(\omega)$ vs $k_y$")
    fig_re.tight_layout()
    if config.save_figures:
        saved_paths.append(_save_figure(fig_re, output_dir, f"{prefix}_real.png", config.dpi))

    fig_im, ax_im, show_figures = _subplots_with_backend_fallback(plt, show_figures, figsize=(6, 4))
    ax_im.scatter(ky_flat, np.imag(omega_flat), s=2, marker=".", linewidths=0)
    ax_im.set_xlabel(r"$k_y$")
    ax_im.set_ylabel(r"$\mathrm{Im}(\omega)$")
    ax_im.set_title(r"Full spectrum: $\mathrm{Im}(\omega)$ vs $k_y$")
    fig_im.tight_layout()
    if config.save_figures:
        saved_paths.append(_save_figure(fig_im, output_dir, f"{prefix}_imag.png", config.dpi))

    if config.plot_cloud:
        fig_cloud, ax_cloud, show_figures = _subplots_with_backend_fallback(
            plt,
            show_figures,
            figsize=(5.5, 5),
        )
        sc = ax_cloud.scatter(
            np.real(omega_flat),
            np.imag(omega_flat),
            c=ky_flat,
            s=2,
            marker=".",
            linewidths=0,
        )
        ax_cloud.set_xlabel(r"$\mathrm{Re}(\omega)$")
        ax_cloud.set_ylabel(r"$\mathrm{Im}(\omega)$")
        ax_cloud.set_title("Complex-plane spectral cloud")
        fig_cloud.colorbar(sc, ax=ax_cloud, label=r"$k_y$")
        fig_cloud.tight_layout()
        if config.save_figures:
            saved_paths.append(_save_figure(fig_cloud, output_dir, f"{prefix}_cloud.png", config.dpi))

    if show_figures:
        plt.show()

    plt.close("all")
    return saved_paths


def main(config: SolverConfig | None = None) -> None:
    if config is None:
        config = CONFIG

    print(
        "Using parameters: "
        f"kz={config.kz}, nu={config.nu}, Nint={config.nint}, "
        f"Lmap={config.lmap}, ky=[{config.ky_min}, {config.ky_max}], n_ky={config.n_ky}"
    )
    ky_list, omega_all, dim = compute_full_spectrum(config)
    saved_paths = plot_spectrum(ky_list, omega_all, config)

    print(f"Computed {config.n_ky} ky values with {dim} eigenvalues each.")
    for path in saved_paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
