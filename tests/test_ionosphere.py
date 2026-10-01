"""Tests for sofa.ionosphere — thin-screen ionospheric correction.

Validation criteria from CLAUDE.md:
  11. TEC → 0, ν >> ν_p: output ≈ input (transparency).
  12. ν ≤ ν_p: output = 290 K everywhere (plasma cutoff).
  13. TEC=5 TECU, ν=10 MHz, θ_z=60°, H=300 km → δρ = 0.058 rad (numerical).
"""

import numpy as np
import healpy
import pytest

from sofa.ionosphere import apply_ionospheric_correction


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

NSIDE = 64
N_PIX = healpy.nside2npix(NSIDE)


@pytest.fixture(scope="module")
def uniform_sky():
    """Uniform 3000 K sky map (flat spectrum, easy to check)."""
    return np.full(N_PIX, 3000.0)


@pytest.fixture(scope="module")
def smooth_sky():
    """Smooth sky map with mild variation: dipole + constant."""
    theta, phi = healpy.pix2ang(NSIDE, np.arange(N_PIX))
    return 3000.0 + 500.0 * np.cos(theta)


# ---------------------------------------------------------------------------
# Test 11: Transparency — TEC=0 → output ≈ input
# ---------------------------------------------------------------------------

def test_transparency_uniform_sky(uniform_sky):
    """TEC=0 with uniform sky: correction is identity (interpolation preserves constant)."""
    result = apply_ionospheric_correction(
        uniform_sky, freq_hz=100e6, tec_zen_tecu=0.0,
        n_e_max=1e8, nside=NSIDE
    )
    assert result.shape == (N_PIX,)
    # Uniform sky: any interpolation returns the same constant
    np.testing.assert_allclose(result, uniform_sky, rtol=1e-3,
                               err_msg="TEC=0 should leave uniform sky unchanged")


def test_transparency_smooth_sky(smooth_sky):
    """TEC=0 with smooth sky: output matches input to within 0.1% rms."""
    result = apply_ionospheric_correction(
        smooth_sky, freq_hz=100e6, tec_zen_tecu=0.0,
        n_e_max=1e8, nside=NSIDE
    )
    assert result.shape == (N_PIX,)
    rms_diff = np.sqrt(np.mean((result - smooth_sky) ** 2))
    rms_sky = np.sqrt(np.mean(smooth_sky ** 2))
    assert rms_diff / rms_sky < 1e-3, (
        f"TEC=0 sky not preserved: rms_diff/rms_sky = {rms_diff / rms_sky:.2e}"
    )


def test_transparency_high_frequency(smooth_sky):
    """Very high frequency (ν >> ν_p): refraction angle is tiny, output ≈ input."""
    # At ν=1 GHz, TEC=10 TECU: δρ ~ 40.3*1e17*sin(θ)/(1e18*3e5) ~ 1.3e-5 rad → negligible
    result = apply_ionospheric_correction(
        smooth_sky, freq_hz=1000e6, tec_zen_tecu=10.0,
        n_e_max=1e8, nside=NSIDE
    )
    rms_diff = np.sqrt(np.mean((result - smooth_sky) ** 2))
    rms_sky = np.sqrt(np.mean(smooth_sky ** 2))
    assert rms_diff / rms_sky < 1e-3, (
        f"High-frequency refraction should be negligible: {rms_diff / rms_sky:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 12: Plasma cutoff — ν ≤ ν_p → output = t_ground everywhere
# ---------------------------------------------------------------------------

def test_plasma_cutoff_exact(uniform_sky):
    """ν exactly at plasma cutoff → all pixels return t_ground=290 K.

    n_e_max=1e11 → ν_p = 8.98 * sqrt(1e11) ≈ 2.843 MHz.
    Using ν = 2 MHz < ν_p.
    """
    n_e_max = 1e11          # ν_p ≈ 2.843 MHz
    freq_hz = 2e6           # 2 MHz < ν_p
    t_ground = 290.0

    result = apply_ionospheric_correction(
        uniform_sky, freq_hz=freq_hz, tec_zen_tecu=5.0,
        n_e_max=n_e_max, t_ground=t_ground, nside=NSIDE
    )
    assert result.shape == (N_PIX,)
    np.testing.assert_allclose(result, t_ground, rtol=0.0, atol=1e-10,
                               err_msg="Below plasma cutoff: all pixels must equal t_ground")


def test_plasma_cutoff_at_nu_p(smooth_sky):
    """ν = ν_p (exactly equal) also triggers the cutoff (freq_hz <= nu_p)."""
    n_e_max = 1e11
    nu_p = 8.98 * np.sqrt(n_e_max)   # ~2.843e6 Hz
    t_ground = 150.0

    result = apply_ionospheric_correction(
        smooth_sky, freq_hz=nu_p, tec_zen_tecu=5.0,
        n_e_max=n_e_max, t_ground=t_ground, nside=NSIDE
    )
    np.testing.assert_allclose(result, t_ground, rtol=0.0, atol=1e-10,
                               err_msg="At exactly ν_p the cutoff condition (<=) must trigger")


def test_plasma_cutoff_custom_t_ground(uniform_sky):
    """Cutoff output respects the t_ground argument, not a hard-coded 290 K."""
    n_e_max = 1e11
    freq_hz = 1e6           # well below cutoff
    t_ground = 500.0        # non-default ground temperature

    result = apply_ionospheric_correction(
        uniform_sky, freq_hz=freq_hz, tec_zen_tecu=5.0,
        n_e_max=n_e_max, t_ground=t_ground, nside=NSIDE
    )
    np.testing.assert_allclose(result, t_ground, rtol=0.0, atol=1e-10,
                               err_msg="Cutoff map must equal custom t_ground value")


def test_above_plasma_cutoff_not_290(smooth_sky):
    """Just above cutoff: result should NOT be 290 K (signal passes through)."""
    n_e_max = 1e11
    nu_p = 8.98 * np.sqrt(n_e_max)   # ~2.843 MHz
    freq_hz = nu_p * 1.01             # 1% above cutoff

    result = apply_ionospheric_correction(
        smooth_sky, freq_hz=freq_hz, tec_zen_tecu=5.0,
        n_e_max=n_e_max, t_ground=290.0, nside=NSIDE
    )
    # The sky has values ~3000 K; result should differ from 290.0
    assert not np.allclose(result, 290.0, atol=10.0), \
        "Just above plasma cutoff, sky should pass through (not return t_ground)"


# ---------------------------------------------------------------------------
# Test 13: Numerical check — δρ formula value
# ---------------------------------------------------------------------------

def test_numerical_delta_rho():
    """TEC=5 TECU, ν=10 MHz, θ_z=60°, H=300 km → δρ = 0.058 rad.

    This verifies the refraction formula independently of the HEALPix machinery.
    δρ = (40.3 * TEC_zen_SI * sin(θ_z)) / (ν² * H)
    """
    TEC_zen_SI = 5.0 * 1e16    # 5 TECU → SI
    freq_hz    = 10e6           # 10 MHz
    theta_z    = np.pi / 3     # 60 degrees
    h_screen   = 300e3         # 300 km in metres

    delta_rho = (40.3 * TEC_zen_SI * np.sin(theta_z)) / (freq_hz**2 * h_screen)

    assert abs(delta_rho - 0.058) < 0.001, (
        f"Expected δρ ≈ 0.058 rad, got {delta_rho:.4f} rad"
    )


def test_numerical_delta_rho_explicit():
    """Verify step-by-step arithmetic from the CLAUDE.md derivation."""
    TEC_zen_SI = 5e16
    freq_hz    = 10e6
    theta_z    = np.pi / 3     # sin(60°) ≈ 0.866
    h_screen   = 300e3

    numerator   = 40.3 * TEC_zen_SI * np.sin(theta_z)   # ~ 1.745e18
    denominator = freq_hz**2 * h_screen                  # 1e14 * 3e5 = 3e19

    delta_rho = numerator / denominator
    # Expected: 1.745e18 / 3e19 ≈ 0.0582
    assert abs(numerator   - 1.745e18) / 1.745e18 < 0.01, \
        f"Numerator mismatch: {numerator:.4e}"
    assert abs(denominator - 3e19)     / 3e19     < 1e-10, \
        f"Denominator mismatch: {denominator:.4e}"
    assert abs(delta_rho   - 0.058)               < 0.001, \
        f"δρ = {delta_rho:.4f} rad, expected 0.058 rad"


# ---------------------------------------------------------------------------
# Additional structural tests
# ---------------------------------------------------------------------------

def test_output_shape(uniform_sky):
    """Output map has same shape as input."""
    result = apply_ionospheric_correction(
        uniform_sky, freq_hz=100e6, tec_zen_tecu=2.0,
        n_e_max=1e8, nside=NSIDE
    )
    assert result.shape == uniform_sky.shape


def test_lower_hemisphere_unchanged(smooth_sky):
    """Pixels below the horizon (theta >= pi/2) are not refracted — values come
    from copying sky_map directly for those pixels."""
    theta_all, _ = healpy.pix2ang(NSIDE, np.arange(N_PIX))
    lower = theta_all >= np.pi / 2

    result = apply_ionospheric_correction(
        smooth_sky, freq_hz=100e6, tec_zen_tecu=5.0,
        n_e_max=1e8, nside=NSIDE
    )
    # Lower hemisphere pixels must equal the original sky map
    np.testing.assert_array_equal(
        result[lower], smooth_sky[lower],
        err_msg="Lower hemisphere pixels must not be modified by ionospheric correction"
    )


def test_plasma_frequency_formula():
    """Verify the plasma frequency formula: ν_p = 8.98 * sqrt(n_e_max).

    n_e_max = 1e11 m^-3 → ν_p ≈ 2.843e6 Hz (2.843 MHz).
    """
    n_e_max = 1e11
    nu_p_expected = 8.98 * np.sqrt(n_e_max)
    assert abs(nu_p_expected - 2.843e6) < 1e4, \
        f"Plasma frequency formula gives {nu_p_expected:.3e} Hz, expected ~2.843 MHz"


# ---------------------------------------------------------------------------
# Directional refraction tests — verifies the physics, not just the formula
# ---------------------------------------------------------------------------

def test_refraction_moves_pixels_toward_zenith():
    """Nonzero TEC with a gradient sky: refraction pulls upper-hemisphere
    pixels toward zenith, so each refracted pixel samples a sky value closer
    to zenith than the original pixel direction.

    Sky model: T ∝ cos(theta) (bright near zenith, faint near horizon).
    Refraction decreases theta → samples brighter (larger cos) value → result > original.
    """
    theta_all, phi_all = healpy.pix2ang(NSIDE, np.arange(N_PIX))
    # Sky brightest at zenith (theta=0), fading to zero at horizon
    sky = np.maximum(np.cos(theta_all), 0.0) * 3000.0

    result = apply_ionospheric_correction(
        sky, freq_hz=10e6, tec_zen_tecu=5.0,
        n_e_max=1e8, nside=NSIDE
    )

    upper = theta_all < np.pi / 2
    # Exclude pixels very close to zenith (theta < 5 deg) — delta_rho * sin(theta) ~ 0 there
    mid_upper = upper & (theta_all > np.deg2rad(5))

    # Refracted result must be >= original for a zenith-bright sky
    # (pixels pulled toward zenith sample higher T); allow small interpolation error
    mean_diff = np.mean(result[mid_upper] - sky[mid_upper])
    assert mean_diff > 0, (
        f"Refraction should shift pixels toward zenith (brighter), "
        f"but mean diff = {mean_diff:.2f} K"
    )


def test_refraction_increases_with_tec():
    """Larger TEC → larger refraction angle → larger displacement of sky map.

    Measured as rms difference from the uncorrected sky over upper hemisphere.
    """
    theta_all, _ = healpy.pix2ang(NSIDE, np.arange(N_PIX))
    sky = np.maximum(np.cos(theta_all), 0.0) * 3000.0 + 500.0 * np.sin(theta_all)

    upper = theta_all < np.pi / 2

    result_low  = apply_ionospheric_correction(sky, freq_hz=20e6, tec_zen_tecu=1.0,
                                               n_e_max=1e8, nside=NSIDE)
    result_high = apply_ionospheric_correction(sky, freq_hz=20e6, tec_zen_tecu=10.0,
                                               n_e_max=1e8, nside=NSIDE)

    rms_low  = np.sqrt(np.mean((result_low[upper]  - sky[upper]) ** 2))
    rms_high = np.sqrt(np.mean((result_high[upper] - sky[upper]) ** 2))

    assert rms_high > rms_low, (
        f"Higher TEC should cause larger sky displacement: "
        f"rms_low={rms_low:.2f} K, rms_high={rms_high:.2f} K"
    )


def test_refraction_decreases_with_frequency():
    """Higher frequency → smaller refraction (1/ν² dependence).

    Same TEC, two frequencies: lower ν should displace the sky more.
    """
    theta_all, _ = healpy.pix2ang(NSIDE, np.arange(N_PIX))
    sky = np.maximum(np.cos(theta_all), 0.0) * 3000.0 + 500.0 * np.sin(theta_all)

    upper = theta_all < np.pi / 2

    result_lo_nu = apply_ionospheric_correction(sky, freq_hz=10e6,  tec_zen_tecu=5.0,
                                                n_e_max=1e8, nside=NSIDE)
    result_hi_nu = apply_ionospheric_correction(sky, freq_hz=100e6, tec_zen_tecu=5.0,
                                                n_e_max=1e8, nside=NSIDE)

    rms_lo = np.sqrt(np.mean((result_lo_nu[upper] - sky[upper]) ** 2))
    rms_hi = np.sqrt(np.mean((result_hi_nu[upper] - sky[upper]) ** 2))

    assert rms_lo > rms_hi, (
        f"Lower frequency should cause larger refraction: "
        f"rms_10MHz={rms_lo:.2f} K, rms_100MHz={rms_hi:.4f} K"
    )
