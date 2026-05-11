import numpy as np
import scipy.linalg as la


# ============================================================
# Model parameters and mu(x)
# ============================================================
kz = 0.9


def mu_fun(x):
    """
    Define mu(x).
    You can change this profile.
    """
    return x
    # return np.tanh(x)
    # return 0.3 * np.tanh(x) + 1.0


def dmu_fun(x):
    """
    Derivative d mu / dx.
    Must be consistent with mu_fun.
    """
    return 1.0
    # return 1.0 / np.cosh(x)**2
    # return 0.3 / np.cosh(x)**2


# ============================================================
# Hamiltonian H(x,kx,ky)
# ============================================================
def H_matrix(x, kx, ky, kz=0.9):
    mu = mu_fun(x)

    H = np.array([
        [0.0,       -1j * mu,   0.0,   kx],
        [1j * mu,    0.0,       0.0,   ky],
        [0.0,        0.0,       0.0,   kz],
        [kx,         ky,        kz,    0.0]
    ], dtype=complex)

    return H
def etainverse_H_matrix(x, kx, ky, kz=0.9):
    mu = mu_fun(x)
    k2 = kx**2 + ky**2 + kz**2

    H = np.array([
        [0.0,       -1j * mu,   0.0,   kx],
        [1j * mu,    0.0,       0.0,   ky],
        [0.0,        0.0,       0.0,   kz],
        [kx/(1+k2),         ky/(1+k2),        kz/(1+k2),    0.0]
    ], dtype=complex)

    return H


# ============================================================
# Derivatives: grad H = (dH/dx, dH/dkx, dH/dky)
# ============================================================
def dH_matrices(x, kx, ky, kz=0.9):
    dmu = dmu_fun(x)

    dH_dx = np.array([
        [0.0,        -1j * dmu,  0.0,  0.0],
        [1j * dmu,    0.0,       0.0,  0.0],
        [0.0,         0.0,       0.0,  0.0],
        [0.0,         0.0,       0.0,  0.0]
    ], dtype=complex)

    dH_dkx = np.array([
        [0.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0],
        [1.0,  0.0,  0.0,  0.0]
    ], dtype=complex)

    dH_dky = np.array([
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.0]
    ], dtype=complex)

    return [dH_dx, dH_dkx, dH_dky]
def detaH_matrices(x, kx, ky, kz=0.9):
    dmu = dmu_fun(x)

    k2 = kx**2 + ky**2 + kz**2
    s = 1.0 + k2
    s2 = s**2

    dH_dx = np.array([
        [0.0,        -1j * dmu,  0.0,  0.0],
        [1j * dmu,    0.0,       0.0,  0.0],
        [0.0,         0.0,       0.0,  0.0],
        [0.0,         0.0,       0.0,  0.0]
    ], dtype=complex)

    dH_dkx = np.array([
        [0.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0],
        [
            (s - 2.0 * kx**2) / s2,
            -2.0 * kx * ky / s2,
            -2.0 * kx * kz / s2,
            0.0
        ]
    ], dtype=complex)

    dH_dky = np.array([
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.0],
        [
            -2.0 * ky * kx / s2,
            (s - 2.0 * ky**2) / s2,
            -2.0 * ky * kz / s2,
            0.0
        ]
    ], dtype=complex)

    return [dH_dx, dH_dkx, dH_dky]


# ============================================================
# Berry curvature / Kubo curvature for one band
# Coordinates are q = (x, kx, ky)
# ============================================================
def berry_curvature_band(x, kx, ky, band, kz=0.9, eps=1e-10):
    """
    Compute Berry curvature vector Omega_i(q) for one band.

    band:
        Python index, 0,1,2,3.
        band=1 corresponds to mathematical i=2.

    Formula used:
        Omega_i = - Im sum_{j != i}
        [ <i|grad H|j> x <j|grad H|i> ] / (omega_i - omega_j)^2

    For this Hermitian H, left and right eigenvectors are the same.
    """
    etaH = etainverse_H_matrix(x, kx, ky, kz=kz)
    H = H_matrix(x, kx, ky, kz=kz)
    # dHs = dH_matrices(x, kx, ky, kz=kz)
    dHs=detaH_matrices(x, kx, ky, kz=kz)
    eigvals, eigvecsR, eigvecsL, S_check = biorthogonal_eigenvectors(etaH)
    omega_i = eigvals[band]
    psi_iL = eigvecsL[:, band]
    psi_iR = eigvecsR[:, band]
    


    Omega = np.zeros(3, dtype=float)

    for j in range(4):
        if j == band:
            continue

        omega_j = eigvals[j]
        psi_jL = eigvecsL[:, j]
        psi_jR = eigvecsR[:, j]

        denom = (omega_i - omega_j) ** 2

        if abs(denom) < eps:
            # Degeneracy lies on or very near the integration surface.
            # The Chern number is ill-defined in this case.
            continue

        A = np.array([
            np.vdot(psi_iL, dH @ psi_jR) for dH in dHs
        ], dtype=complex)

        B = np.array([
            np.vdot(psi_jL, dH @ psi_iR) for dH in dHs
        ], dtype=complex)

        cross_AB = np.cross(A, B)

        Omega += -np.imag(cross_AB / denom)

    return Omega


# ============================================================
# Integrate over a sphere S_b in (x,kx,ky)
# ============================================================
def integrate_chern_on_sphere(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    bands=(0, 1, 2),
    kz=0.9,
    n_theta=80,
    n_phi=160,
    verbose=True,
):
    """
    Compute C_i for selected bands on a sphere in q=(x,kx,ky).

    center:
        (x0, kx0, ky0)

    radius:
        sphere radius

    bands:
        Python band indices.
        bands=(0,1,2) corresponds to mathematical i=1,2,3.

    Integration:
        Uses Gauss-Legendre quadrature in u=cos(theta)
        and uniform quadrature in phi.

    Surface parametrization:
        q = q0 + R * n
        n = (sqrt(1-u^2) cos phi,
             sqrt(1-u^2) sin phi,
             u)

    dS vector:
        n R^2 du dphi
    """
    x0, kx0, ky0 = center
    R = radius

    # Gauss-Legendre nodes and weights for u in [-1,1]
    u_nodes, u_weights = np.polynomial.legendre.leggauss(n_theta)

    phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    C_band = np.zeros(len(bands), dtype=float)

    for iu, u in enumerate(u_nodes):
        wu = u_weights[iu]
        sin_theta = np.sqrt(max(0.0, 1.0 - u**2))

        for phi in phi_nodes:
            normal = np.array([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                u
            ], dtype=float)

            q = np.array([x0, kx0, ky0], dtype=float) + R * normal

            x, kx, ky = q

            dS_weight = R**2 * wu * dphi

            for ib, band in enumerate(bands):
                Omega = berry_curvature_band(x, kx, ky, band, kz=kz)
                C_band[ib] += np.dot(Omega, normal) * dS_weight

    C_band = C_band / (2.0 * np.pi)
    C_total = np.sum(C_band)

    if verbose:
        print("============================================")
        print("Sphere center =", center)
        print("Sphere radius =", radius)
        print("kz =", kz)
        print("bands Python indices =", bands)
        print("bands mathematical indices =", tuple(b + 1 for b in bands))
        print("C_i =")
        for band, Ci in zip(bands, C_band):
            print(f"  band i = {band + 1}: C_{band + 1} = {Ci:.12f}")
        print("C_gap^b =", C_total)
        print("============================================")

    return C_total, C_band
def biorthogonal_eigenvectors(H, sort_by_real=True):
    """
    Compute right and left eigenvectors of a non-Hermitian matrix H.

    Right eigenvectors:
        H |R_i> = omega_i |R_i>

    Left eigenvectors are obtained as right eigenvectors of H^dagger:
        H^dagger |L_i> = omega_i^* |L_i>

    Normalization:
        <L_i | R_j> = delta_ij

    Returns:
        eigvals : eigenvalues omega_i
        VR      : right eigenvectors, columns are |R_i>
        VL      : left eigenvectors in ket form, columns are |L_i>
    """

    eigvalsR, VR = la.eig(H)
    eigvalsL_raw, VL_raw = la.eig(H.conj().T)

    # Optional sorting of right eigenvalues
    if sort_by_real:
        idxR = np.argsort(np.real(eigvalsR))
        eigvalsR = eigvalsR[idxR]
        VR = VR[:, idxR]

    # Match left eigenvectors to right eigenvalues
    # H^dagger eigenvalue should be omega_i^*
    used = set()
    idxL = []

    for wR in eigvalsR:
        distances = np.abs(eigvalsL_raw.conj() - wR)

        # Avoid using the same left eigenvector twice
        for used_idx in used:
            distances[used_idx] = np.inf

        idx = np.argmin(distances)
        idxL.append(idx)
        used.add(idx)

    VL = VL_raw[:, idxL]
    eigvalsL = eigvalsL_raw[idxL]

    # Biorthogonal normalization
    # First form overlap matrix S_ij = <L_i|R_j>
    S = VL.conj().T @ VR

    # General robust method:
    # Transform left vectors so that VL_new^dagger VR = I
    #
    # Want:
    #   VL_new^dagger VR = I
    #
    # Let VL_new = VL @ A.
    # Then:
    #   VL_new^dagger VR = A^dagger VL^dagger VR = A^dagger S.
    # Need:
    #   A^dagger S = I
    # Hence:
    #   A^dagger = S^{-1}
    #   A = (S^{-1})^dagger
    #
    VL = VL @ la.inv(S).conj().T

    # Check normalization
    S_check = VL.conj().T @ VR

    return eigvalsR, VR, VL, S_check


# ============================================================
# Example
# ============================================================
if __name__ == "__main__":

    # Example sphere in (x,kx,ky)
    center = (-1, 0.0, 0.0)
    radius = 1.0

    C_total, C_band = integrate_chern_on_sphere(
        center=center,
        radius=radius,
        bands=(0, 1, 2),     # mathematical i=1,2,3
        kz=kz,
        n_theta=80,
        n_phi=160,
        verbose=True,
    )