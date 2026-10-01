"""
beam.py — VSH Beam Module for SOFA
Vector Spherical-Harmonic beam expansion, image theory ground-plane model,
and frequency-polynomial parameterization.

Hansen/M-N convention (Jackson 3rd ed. Ch. 9):
  M_lm = [1/sqrt(l(l+1))] * r_hat x grad(Y_lm)   (TE / magnetic multipole)
  N_lm = (1/k) curl(M_lm)  → far-field: N_lm = i * (r_hat x M_lm)  (TM / electric)

scipy spherical harmonics API (scipy >= 1.15):
  scipy.special.sph_harm_y(n, m, theta, phi)
  where theta = polar (colatitude) [0,pi], phi = azimuthal [0,2pi]
  — standard physics convention; n=l (degree), m=order.
"""

import numpy as np
import scipy.special
import healpy


# ---------------------------------------------------------------------------
# dYlm/dtheta recurrence
# ---------------------------------------------------------------------------

def dYlm_dtheta(m, l, phi, theta):
    """Compute partial derivative of Y_lm with respect to theta.

    Uses the L+ operator recurrence:
        dY_lm/dtheta = m*cot(theta)*Y_lm + exp(-i*phi)*sqrt((l-m)(l+m+1))*Y_{l,m+1}

    Args:
        m:     integer order (-l <= m <= l)
        l:     integer degree (l >= 1)
        phi:   azimuthal angle(s) [rad]  — broadcastable array
        theta: polar angle(s)     [rad]  — broadcastable array

    Returns:
        dY/dtheta: complex array same shape as phi/theta

    Validation:
        dYlm_dtheta(0, 1, phi, pi/2) ≈ -0.4886  (real, negative)
        dYlm_dtheta(-1, 1, phi, pi/2) != 0       (m=-l=-1 < l=1 → True)
    """
    Y_ml = scipy.special.sph_harm_y(l, m, theta, phi)

    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(np.abs(np.sin(theta)) > 1e-10,
                         m / np.tan(theta) * Y_ml,
                         0.0 + 0.0j)

    # NOTE: condition is "m < l", NOT "abs(m) < l"
    # This correctly includes m=-l (e.g. -1 < 1 is True)
    if m < l:
        term2 = (np.sqrt((l - m) * (l + m + 1))
                 * np.exp(-1j * phi)
                 * scipy.special.sph_harm_y(l, m + 1, theta, phi))
    else:
        term2 = 0.0

    # NOTE: return term1 + term2 (NOT term1 - term2; sign from L+ derivation)
    return term1 + term2


# ---------------------------------------------------------------------------
# VSH basis pre-computation
# ---------------------------------------------------------------------------

def vsh_basis(lmax=5, nside=64, theta_override=None, phi_override=None):
    """Pre-compute M_lm and N_lm VSH basis vectors on upper-hemisphere pixels.

    When theta_override and phi_override are provided, those angle arrays are
    used directly (e.g. for the image basis at pi - theta_u) without re-doing
    the HEALPix pixel lookup. The upper_mask is always built from a real
    HEALPix pixel lookup so it has shape (n_pix_full,).

    Args:
        lmax:           maximum spherical-harmonic degree (default 5)
        nside:          HEALPix nside (default 64)
        theta_override: optional array of polar angles [rad] to use instead
                        of the HEALPix upper-hemisphere theta values
        phi_override:   optional array of azimuthal angles [rad] to use
                        instead of the HEALPix upper-hemisphere phi values

    Returns:
        dict with keys:
            'M_lm':       (n_lm, n_pix_upper, 2)  complex; axis-2 = (theta, phi)
            'N_lm':       (n_lm, n_pix_upper, 2)  complex; axis-2 = (theta, phi)
            'theta':      (n_pix_upper,)            polar angles used
            'phi':        (n_pix_upper,)            azimuthal angles used
            'upper_mask': (n_pix_full,)             bool; True for upper hemisphere
        where n_lm = lmax^2 + 2*lmax  (35 for lmax=5)

    VSH upper-hemi norm check:
        M10 = basis['M_lm'][1]           # (l=1, m=0) at index 1
        domega = 4*pi / nside2npix(nside)
        ortho = (abs(M10[:,0])**2 + abs(M10[:,1])**2).sum() * domega
        assert abs(ortho - 0.5) < 0.01   # 0.5 (half of full sphere = 1)
    """
    n_pix = healpy.nside2npix(nside)
    theta_all, phi_all = healpy.pix2ang(nside, np.arange(n_pix))
    upper = theta_all < np.pi / 2          # upper_mask: shape (n_pix_full,)
    theta_u = theta_all[upper]
    phi_u   = phi_all[upper]

    # Use override angles if provided (e.g. pi - theta_u for image basis)
    if theta_override is not None and phi_override is not None:
        theta_eval = theta_override
        phi_eval   = phi_override
    else:
        theta_eval = theta_u
        phi_eval   = phi_u

    # Safe sine for pole pixels (avoids divide-by-zero)
    safe_sin = np.where(np.sin(theta_eval) > 1e-10, np.sin(theta_eval), 1.0)

    M_lm_arr = []
    N_lm_arr = []

    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Y    = scipy.special.sph_harm_y(l, m, theta_eval, phi_eval)  # (n_pix_upper,)
            dYdt = dYlm_dtheta(m, l, phi_eval, theta_eval)               # (n_pix_upper,)
            norm = np.sqrt(l * (l + 1))

            # M_lm = [1/norm] * (r_hat x grad Y_lm)
            # grad Y_lm = theta_hat * dY/dtheta + phi_hat * (im/sintheta) * Y
            # r_hat x theta_hat = phi_hat,  r_hat x phi_hat = -theta_hat
            M_theta = -1j * m * Y / (safe_sin * norm)   # (n_pix_upper,)
            M_phi   = dYdt / norm                         # (n_pix_upper,)

            # N_lm = i * (r_hat x M_lm)  [far-field limit]
            N_theta = -1j * M_phi          # = -i * dYdt / norm
            N_phi   =  1j * M_theta        # = m * Y / (safe_sin * norm)

            M_lm_arr.append(np.stack([M_theta, M_phi], axis=-1))
            N_lm_arr.append(np.stack([N_theta, N_phi], axis=-1))

    return {
        'M_lm':       np.array(M_lm_arr),    # (n_lm, n_pix_upper, 2)
        'N_lm':       np.array(N_lm_arr),    # (n_lm, n_pix_upper, 2)
        'theta':      theta_eval,             # (n_pix_upper,)
        'phi':        phi_eval,               # (n_pix_upper,)
        'upper_mask': upper,                  # (n_pix_full,) bool
    }


# ---------------------------------------------------------------------------
# Coefficient packing / unpacking
# ---------------------------------------------------------------------------

def pack_coeffs(alpha, beta):
    """Pack complex VSH coefficients into a real optimization vector.

    Args:
        alpha: (n_lm, P+1) complex  — M_lm coefficients
        beta:  (n_lm, P+1) complex  — N_lm coefficients

    Returns:
        x: real array of length 4 * n_lm * (P+1)
           [alpha.real | alpha.imag | beta.real | beta.imag]
    """
    return np.concatenate([alpha.real.ravel(), alpha.imag.ravel(),
                           beta.real.ravel(),  beta.imag.ravel()])


def unpack_coeffs(x, n_lm=35, P=2):
    """Unpack real optimization vector into complex VSH coefficients.

    Args:
        x:    real array of length 4 * n_lm * (P+1)
        n_lm: number of VSH modes (default 35 for lmax=5)
        P:    polynomial degree (default 2)

    Returns:
        alpha: (n_lm, P+1) complex  — M_lm coefficients
        beta:  (n_lm, P+1) complex  — N_lm coefficients

    Validation:
        unpack_coeffs(pack_coeffs(alpha, beta)) == (alpha, beta)
    """
    n = n_lm * (P + 1)                                          # 35*3 = 105
    alpha = (x[    :  n].reshape(n_lm, P + 1)
             + 1j * x[  n:2*n].reshape(n_lm, P + 1))
    beta  = (x[2*n:3*n].reshape(n_lm, P + 1)
             + 1j * x[3*n:4*n].reshape(n_lm, P + 1))
    return alpha, beta


# ---------------------------------------------------------------------------
# Beam computation
# ---------------------------------------------------------------------------

def beam_from_x(x, h_m, freq_mhz_arr, basis, basis_image, domega,
                nu_ref=75.0, P=2, n_lm=35):
    """Compute normalized beam power pattern from real optimization vector.

    Implements image theory at height h_m above PEC ground plane:
        E_tot = E_direct + R * E_image(pi-theta, phi) * exp(2i*k*h*cos(theta))
        R = diag(+1, -1) on (E_theta, E_phi): TM preserved, TE flipped

    Args:
        x:            real array length 4*n_lm*(P+1) = 420 (from pack_coeffs)
        h_m:          antenna height above PEC ground plane [m]
        freq_mhz_arr: array of frequencies [MHz], shape (n_freq,)
        basis:        dict from vsh_basis() evaluated at theta_u (upper hemisphere)
        basis_image:  dict from vsh_basis() evaluated at (pi - theta_u)
                      — pre-computed once alongside basis
        domega:       solid angle per pixel = 4*pi / nside2npix(nside) [sr]
        nu_ref:       reference frequency for polynomial [MHz] (default 75)
        P:            polynomial degree (default 2; P+1 = 3 terms)
        n_lm:         number of VSH modes (default 35 for lmax=5)

    Returns:
        Btilde: (n_freq, n_pix_upper) — beam power normalized so
                (Btilde * domega).sum(axis=-1) = 2*pi for all frequencies

    Normalization check:
        (Btilde * domega).sum(axis=-1) ≈ 2*pi  (< 0.1% error)
    """
    assert n_lm == len(basis['M_lm']) == len(basis_image['M_lm']), (
        f"n_lm mismatch: {n_lm} vs {len(basis['M_lm'])} vs {len(basis_image['M_lm'])}"
    )

    alpha, beta = unpack_coeffs(x, n_lm=n_lm, P=P)

    # Frequency polynomial: (nu/nu_ref)^p for p=0..P
    # p_powers shape: (n_freq, P+1)
    p_powers = (freq_mhz_arr[:, None] / nu_ref) ** np.arange(P + 1)[None, :]

    # a_nu, b_nu shape: (n_lm, n_freq)
    # alpha @ p_powers.T contracts over polynomial order p
    a_nu = alpha @ p_powers.T    # (n_lm, n_freq)
    b_nu = beta  @ p_powers.T   # (n_lm, n_freq)

    # Direct E-field on upper hemisphere pixels
    # Einsum: 'lf,lq->fq'  (l=mode, f=freq, q=pixel)
    E_theta = (np.einsum('lf,lq->fq', a_nu, basis['M_lm'][:, :, 0])
             + np.einsum('lf,lq->fq', b_nu, basis['N_lm'][:, :, 0]))  # (n_freq, n_pix_upper)
    E_phi   = (np.einsum('lf,lq->fq', a_nu, basis['M_lm'][:, :, 1])
             + np.einsum('lf,lq->fq', b_nu, basis['N_lm'][:, :, 1]))  # (n_freq, n_pix_upper)

    # Image E-field: basis_image evaluated at (pi - theta_u), same phi_u
    E_img_theta = (np.einsum('lf,lq->fq', a_nu, basis_image['M_lm'][:, :, 0])
                 + np.einsum('lf,lq->fq', b_nu, basis_image['N_lm'][:, :, 0]))
    E_img_phi   = (np.einsum('lf,lq->fq', a_nu, basis_image['M_lm'][:, :, 1])
                 + np.einsum('lf,lq->fq', b_nu, basis_image['N_lm'][:, :, 1]))

    # Path-length phase factor from height h_m above PEC
    k     = 2 * np.pi * freq_mhz_arr * 1e6 / 3e8               # (n_freq,)
    phase = np.exp(2j * k[:, None] * h_m * np.cos(basis['theta'])[None, :])

    # Image theory: R_theta = +1 (TM/E_theta preserved), R_phi = -1 (TE/E_phi flipped)
    E_tot_theta = E_theta + E_img_theta * phase
    E_tot_phi   = E_phi   - E_img_phi   * phase

    # Power pattern and normalization to 2*pi over upper hemisphere
    B      = np.abs(E_tot_theta)**2 + np.abs(E_tot_phi)**2          # (n_freq, n_pix_upper)
    Btilde = B * (2 * np.pi / (B * domega).sum(axis=-1, keepdims=True))
    return Btilde


# ---------------------------------------------------------------------------
# Coefficient initialization
# ---------------------------------------------------------------------------

def initialize_coeffs(n_lm=35, P=2, seed=42):
    """Initialize VSH coefficients for optimization starting point.

    Sets beta[1, 0] = 1.0 (VED-like N_10 mode) with small complex perturbations
    on all coefficients.

    Args:
        n_lm: number of VSH modes (default 35 for lmax=5)
        P:    polynomial degree (default 2)
        seed: random seed (default 42)

    Returns:
        x0: real array of length 4 * n_lm * (P+1) = 420

    Mode index convention:
        Index 0: (l=1, m=-1)
        Index 1: (l=1, m= 0)  ← beta[1, 0] = 1.0
        Index 2: (l=1, m=+1)
        Index 3: (l=2, m=-2)
        ...
    """
    alpha = np.zeros((n_lm, P + 1), dtype=complex)
    beta  = np.zeros((n_lm, P + 1), dtype=complex)

    # VED-like starting point: N_10 mode (index 1 = (l=1, m=0))
    beta[1, 0] = 1.0

    rng = np.random.default_rng(seed)

    # Perturb real and imaginary parts independently
    alpha += 0.01 * (rng.standard_normal(alpha.shape)
                     + 1j * rng.standard_normal(alpha.shape))
    beta  += 0.01 * (rng.standard_normal(beta.shape)
                     + 1j * rng.standard_normal(beta.shape))

    return pack_coeffs(alpha, beta)    # length 4 * n_lm * (P+1) = 420
