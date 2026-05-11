import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Try to import joblib's Parallel/delayed; if unavailable, provide a lightweight fallback
try:
    from joblib import Parallel, delayed  # type: ignore
except Exception:
    import warnings
    warnings.warn("joblib not available, falling back to concurrent.futures ThreadPoolExecutor")

    from concurrent.futures import ThreadPoolExecutor
    import os

    def delayed(func):
        # delayed(func)(*args, **kwargs) should return a zero-argument callable that runs func(*args, **kwargs)
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    def Parallel(n_jobs=1):
        class _Parallel:
            def __init__(self, n_jobs):
                self.n_jobs = n_jobs

            def __call__(self, tasks):
                tasks = list(tasks)
                if not tasks:
                    return []
                # If n_jobs == -1 use all CPUs
                if self.n_jobs == 1:
                    return [t() for t in tasks]
                max_workers = os.cpu_count() if self.n_jobs == -1 or self.n_jobs is None else self.n_jobs
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(t) for t in tasks]
                    return [f.result() for f in futures]

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
kz = 0.5
nu = -0.5

def mu_fun(x):
    return 0.3*np.tanh(x)

# def mu_fun(x):
#     return x

Nint = 180
Lmap = 2.5

ky_list = np.linspace(-3, 3, 101)
nKy = len(ky_list)

use_parallel = True
plot_cloud = True
include_nu = False    # MATLAB parfor branch includes nu term; set False if not needed


# ============================================================
# 0) Chebyshev differentiation and mapping x = L y / sqrt(1-y^2)
# ============================================================
Ncheb = Nint + 2

Dy_full, y_full = chebD(Ncheb - 1)

# Drop endpoints y = ±1
idx = np.arange(1, Ncheb - 1)
y = y_full[idx]
Dy = Dy_full[np.ix_(idx, idx)]
D2y = Dy @ Dy

# Mapping y -> x
x = Lmap * y / np.sqrt(1 - y**2)

# Chain rule factors
a = (1 - y**2) ** 1.5 / Lmap
b = -3 * y * (1 - y**2) ** 2 / (Lmap**2)

Dx = np.diag(a) @ Dy
Dxx = np.diag(a**2) @ D2y + np.diag(b) @ Dy

N = len(y)
I = np.eye(N)
Z = np.zeros((N, N), dtype=complex)

Mu = np.diag(mu_fun(x))

# eta block: M = -d_x^2 + 1
M = -Dxx + I

# Precompute M^{-1} Dx
MinvD = la.solve(M, Dx)

dim = 4 * N


# ============================================================
# 1) Function to compute spectrum for one ky
# ============================================================
def compute_spectrum_for_ky(ky):
    Bky = ky * I
    Bkz = kz * I

    MinvBky = la.solve(M, Bky)
    MinvBkz = la.solve(M, Bkz)

    A0 = np.block([
        [Z,             -1j * Mu,     Z,        -1j * Dx],
        [1j * Mu,        Z,           Z,         Bky],
        [Z,              Z,           Z,         Bkz],
        [-1j * MinvD,    MinvBky,     MinvBkz,   Z]
    ])

    if include_nu:
        Anu = nu * np.block([
            [1j * I,   Z,       Z,       Z],
            [Z,        1j * I,  Z,       Z],
            [Z,        Z,       1j * I,  Z],
            [Z,        Z,       Z,       Z]
        ])
        A = A0 + Anu
    else:
        A = A0

    eigvals = la.eigvals(A)
    order = np.argsort(np.real(eigvals))

    return eigvals[order]


# ============================================================
# 2) Compute full spectrum
# ============================================================
if use_parallel:
    spectra = Parallel(n_jobs=-1)(
        delayed(compute_spectrum_for_ky)(ky) for ky in ky_list
    )
else:
    spectra = [compute_spectrum_for_ky(ky) for ky in ky_list]

omega_all = np.column_stack(spectra)


# ============================================================
# 3) Plot Re(omega) vs ky
# ============================================================
plt.figure(figsize=(6, 4))
for j, ky in enumerate(ky_list):
    plt.scatter(
        ky * np.ones(dim),
        np.real(omega_all[:, j]),
        s=2,
        marker='.'
    )

plt.xlabel(r"$k_y$")
plt.ylabel(r"$\mathrm{Re}(\omega)$")
plt.title(r"Full spectrum: $\mathrm{Re}(\omega)$ vs $k_y$")
plt.tight_layout()
plt.show()


# ============================================================
# 4) Plot Im(omega) vs ky
# ============================================================
plt.figure(figsize=(6, 4))
for j, ky in enumerate(ky_list):
    plt.scatter(
        ky * np.ones(dim),
        np.imag(omega_all[:, j]),
        s=2,
        marker='.'
    )

plt.xlabel(r"$k_y$")
plt.ylabel(r"$\mathrm{Im}(\omega)$")
plt.title(r"Full spectrum: $\mathrm{Im}(\omega)$ vs $k_y$")
plt.tight_layout()
plt.show()


# ============================================================
# 5) Complex-plane spectral cloud colored by ky
# ============================================================
if plot_cloud:
    omega_flat = omega_all.ravel(order="F")
    ky_flat = np.repeat(ky_list, dim)

    plt.figure(figsize=(5.5, 5))
    sc = plt.scatter(
        np.real(omega_flat),
        np.imag(omega_flat),
        c=ky_flat,
        s=2,
        marker='.'
    )
    plt.xlabel(r"$\mathrm{Re}(\omega)$")
    plt.ylabel(r"$\mathrm{Im}(\omega)$")
    plt.title("Complex-plane spectral cloud")
    plt.colorbar(sc, label=r"$k_y$")
    plt.tight_layout()
    plt.show()