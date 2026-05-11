import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


# ============================================================
# Adjustable parameter
# ============================================================
kz = 0.9


# ============================================================
# Symbol matrix h = eta^{-1} H
# ============================================================
def h_symbol(mu, kperp, kz=0.9):
    """
    Weyl symbol:
        h(mu,kx,ky;kz) = eta^{-1} H

    Because the spectrum depends only on k_perp = sqrt(kx^2 + ky^2),
    we choose kx = kperp, ky = 0.
    """
    kx = kperp
    ky = 0.0

    k2 = kx**2 + ky**2 + kz**2

    h = np.array([
        [0.0,       -1j * mu,   0.0,   kx],
        [1j * mu,    0.0,       0.0,   ky],
        [0.0,        0.0,       0.0,   kz],
        [kx/(1+k2),  ky/(1+k2), kz/(1+k2), 0.0]
    ], dtype=complex)

    return h


# ============================================================
# Eigenvalues sorted by real part
# ============================================================
def eigenvalues(mu, kperp, kz=0.9):
    h = h_symbol(mu, kperp, kz=kz)

    # h is generally pseudo-Hermitian, not Hermitian, so use eigvals
    vals = la.eigvals(h)

    # The physical spectrum should be real up to numerical errors
    vals = np.real_if_close(vals, tol=1000)
    vals = np.real(vals)

    # Sort from high to low, consistent with omega_1 >= ... >= omega_4
    vals = np.sort(vals)[::-1]

    return vals


# ============================================================
# Build grid in (mu, k_perp)
# ============================================================
mu_min, mu_max = -2.0, 2.0
kp_min, kp_max = -1, 1

n_mu = 160
n_kp = 160

mu_list = np.linspace(mu_min, mu_max, n_mu)
kp_list = np.linspace(kp_min, kp_max, n_kp)

MU, KP = np.meshgrid(mu_list, kp_list, indexing="ij")

OMEGA = np.zeros((4, n_mu, n_kp))

for i in range(n_mu):
    for j in range(n_kp):
        vals = eigenvalues(MU[i, j], KP[i, j], kz=kz)
        OMEGA[:, i, j] = vals


# ============================================================
# 3D surface plot
# ============================================================
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for band in range(4):
    ax.plot_surface(
        MU,
        KP,
        OMEGA[band],
        rstride=3,
        cstride=3,
        alpha=0.75,
        linewidth=0,
        antialiased=True
    )

ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$k_\perp$")
ax.set_zlabel(r"$\omega$")
ax.set_title(rf"Eigenvalue surfaces of $\eta^{{-1}}H$, $k_z={kz}$")

plt.tight_layout()
plt.show()


# ============================================================
# Optional: contour plot of each branch
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)

for band, ax in enumerate(axes.ravel()):
    cs = ax.contourf(MU, KP, OMEGA[band], levels=60)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$k_\perp$")
    ax.set_title(rf"$\omega_{band+1}$")

plt.show()