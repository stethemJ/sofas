"""Ionospheric correction module for SOFA.

Implements thin-screen refraction and plasma reflection correction.
Valid for zenith angles < 70 degrees. For theta_z > 70 deg, multiply
delta_rho by 1/cos(theta_z) (~2.9x at 70 deg).

Units check for refraction formula:
    40.3 [m^3 s^-2] * TEC [m^-2] / (nu^2 [s^-2] * H [m]) = dimensionless ✓

Numerical check:
    TEC=5 TECU, nu=10 MHz, theta_z=60 deg, H=300 km:
    delta_rho = (40.3 * 5e16 * sin(60 deg)) / ((10e6)^2 * 3e5)
              = (40.3 * 5e16 * 0.866) / (1e14 * 3e5)
              = 1.745e18 / 3e19
              = 0.058 rad ✓
"""

import numpy as np
import healpy


def apply_ionospheric_correction(sky_map, freq_hz, tec_zen_tecu, n_e_max,
                                 h_screen=300e3, t_ground=290.0, nside=64):
    """Apply thin-screen ionospheric correction to a sky map.

    Refracts upper-hemisphere pixels toward zenith according to the
    thin-screen approximation. Valid for zenith angles < 70 degrees.

    Args:
        sky_map      : (n_pix,) sky brightness temperature map [K]; must match nside.
        freq_hz      : observing frequency [Hz].
        tec_zen_tecu : zenith TEC in TECU (1 TECU = 10^16 m^-2).
        n_e_max      : peak electron density [m^-3]; sets plasma cutoff frequency.
        h_screen     : ionospheric thin-screen height [m] (default 300 km).
        t_ground     : ground temperature [K] returned when freq_hz <= nu_p (default 290 K).
        nside        : HEALPix nside parameter (default 64).

    Returns:
        result : (n_pix,) corrected sky map [K]. When freq_hz <= nu_p (plasma
                 cutoff), returns a uniform map filled with t_ground.
    """
    # Convert TEC from TECU to SI units [m^-2]
    TEC_zen_SI = tec_zen_tecu * 1e16    # 1 TECU = 10^16 m^-2

    # Plasma cutoff frequency [Hz]; n_e_max in m^-3 → nu_p in Hz
    nu_p = 8.98 * np.sqrt(n_e_max)

    # Below plasma frequency: sky is opaque, return ground temperature
    if freq_hz <= nu_p:
        return np.full(healpy.nside2npix(nside), t_ground)

    # Pixel angles for all pixels
    n_pix = healpy.nside2npix(nside)
    theta_pix, phi_pix = healpy.pix2ang(nside, np.arange(n_pix))

    # Upper hemisphere mask (theta < pi/2 means above horizon)
    mask = theta_pix < np.pi / 2
    theta_z = theta_pix[mask]    # zenith angles for upper-hemisphere pixels

    # Refraction toward zenith [rad]
    # delta_rho = 40.3 * TEC_zen_SI * sin(theta_z) / (freq_hz^2 * h_screen)
    delta_rho = (40.3 * TEC_zen_SI * np.sin(theta_z)) / (freq_hz**2 * h_screen)

    # Shifted zenith angles (refraction moves pixels toward zenith)
    theta_new = theta_z - delta_rho

    # Interpolate sky map at new (refracted) positions
    result = sky_map.copy()
    result[mask] = healpy.get_interp_val(sky_map, theta_new, phi_pix[mask])

    return result
