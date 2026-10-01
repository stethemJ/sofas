"""
Tests for antenna_temp.py — LST bins, coordinate conversion, and beam-sky integral.

Validation criteria:
  1. Flat-sky: uniform sky → T_ant = const (ν, LST)
  2. Coordinate conversion: HEALPix phi → AltAz az formula correctness
  3. setup_observation: correct site coordinates and 24 LST bins
"""

import numpy as np
import healpy
import pytest

from sofa.antenna_temp import (
    setup_observation,
    compute_antenna_temperature,
)


# ---------------------------------------------------------------------------
# Test 1: Flat-sky → T_ant constant in frequency and LST
# ---------------------------------------------------------------------------

def test_flat_sky_uniform_tant(nside, domega):
    """Uniform sky temperature → T_ant identical at all frequencies and LST bins.

    When every upper-hemisphere pixel has the same brightness (T0) at every
    frequency, the beam-weighted integral reduces to T0 regardless of the beam
    shape or time of observation.  We check that the resulting T_ant array is
    uniform to machine precision.
    """
    T0 = 3000.0   # arbitrary non-zero constant [K]
    n_freq = 20
    n_lst = 24

    n_pix_full = healpy.nside2npix(nside)
    theta_all, _ = healpy.pix2ang(nside, np.arange(n_pix_full))
    upper_mask = theta_all < np.pi / 2
    n_pix_upper = int(upper_mask.sum())

    # Uniform beam (constant power in all directions) — any beam works for flat sky,
    # but uniform is the simplest and avoids beam normalization constraints.
    Btilde = np.ones((n_freq, n_pix_upper), dtype=float)
    # Normalize so (Btilde * domega).sum(axis=-1) = 2π, as required by the integral
    Btilde *= 2 * np.pi / (Btilde * domega).sum(axis=-1, keepdims=True)

    # Flat sky: identical temperature at every pixel and every LST bin
    T_sky_upper = np.full(n_pix_upper, T0, dtype=float)
    T_sky_lst = [T_sky_upper for _ in range(n_lst)]

    T_ant = compute_antenna_temperature(Btilde, T_sky_lst, domega)

    assert T_ant.shape == (n_freq, n_lst)
    # All values should equal T0 to within floating-point rounding
    np.testing.assert_allclose(
        T_ant, T0,
        rtol=1e-10, atol=1e-10,
        err_msg="Flat sky should give perfectly uniform T_ant(ν, LST).",
    )


def test_flat_sky_non_uniform_beam(nside, domega):
    """Flat sky with a non-trivial beam still returns uniform T_ant.

    Even a strongly directional beam integrates to a constant when the sky
    brightness is spatially and spectrally uniform.
    """
    T0 = 1500.0
    n_freq = 10
    n_lst = 5

    n_pix_full = healpy.nside2npix(nside)
    theta_all, _ = healpy.pix2ang(nside, np.arange(n_pix_full))
    upper_mask = theta_all < np.pi / 2
    n_pix_upper = int(upper_mask.sum())
    theta_upper = theta_all[upper_mask]

    # sin²θ beam shape (VED-like)
    rng = np.random.default_rng(0)
    B_raw = np.sin(theta_upper) ** 2 + 0.01   # avoid zeros
    # Broadcast over frequencies with small per-frequency variation
    B_multi = B_raw[np.newaxis, :] * (1.0 + 0.1 * rng.standard_normal((n_freq, 1)))
    B_multi = np.abs(B_multi)
    Btilde = B_multi * (2 * np.pi / (B_multi * domega).sum(axis=-1, keepdims=True))

    T_sky_upper = np.full(n_pix_upper, T0)
    T_sky_lst = [T_sky_upper for _ in range(n_lst)]

    T_ant = compute_antenna_temperature(Btilde, T_sky_lst, domega)
    np.testing.assert_allclose(
        T_ant, T0,
        rtol=1e-8, atol=1e-8,
        err_msg="Non-uniform beam on flat sky must still give T_ant = T0.",
    )


# ---------------------------------------------------------------------------
# Test 2: Coordinate conversion — HEALPix φ → AltAz azimuth formula
# ---------------------------------------------------------------------------

class TestAzimuthConversion:
    """Verify az_rad = (π/2 − φ_hp) % (2π) gives the correct AltAz azimuth.

    HEALPix uses phi increasing counter-clockwise (CCW) from East.
    Astropy AltAz uses azimuth increasing clockwise (CW) from North.
    The formula (π/2 − φ_hp) % (2π) handles the convention difference.
    """

    @staticmethod
    def hp_phi_to_az(phi_hp):
        return (np.pi / 2 - phi_hp) % (2 * np.pi)

    def test_east_phi_zero(self):
        """phi_hp = 0 (East in HEALPix CCW) → az = π/2 (East in AltAz CW from North)."""
        phi_hp = 0.0
        az = self.hp_phi_to_az(phi_hp)
        assert abs(az - np.pi / 2) < 1e-12, (
            f"East (phi_hp=0) should map to az=π/2, got {az:.6f}"
        )

    def test_north_phi_half_pi(self):
        """phi_hp = π/2 (North in HEALPix CCW) → az = 0 (North in AltAz CW from North)."""
        phi_hp = np.pi / 2
        az = self.hp_phi_to_az(phi_hp)
        assert abs(az - 0.0) < 1e-12, (
            f"North (phi_hp=π/2) should map to az=0, got {az:.6f}"
        )

    def test_west_phi_pi(self):
        """phi_hp = π (West in HEALPix CCW) → az = 3π/2 (West in AltAz CW from North)."""
        phi_hp = np.pi
        az = self.hp_phi_to_az(phi_hp)
        assert abs(az - 3 * np.pi / 2) < 1e-12, (
            f"West (phi_hp=π) should map to az=3π/2, got {az:.6f}"
        )

    def test_south_phi_three_halves_pi(self):
        """phi_hp = 3π/2 (South in HEALPix CCW) → az = π (South in AltAz CW from North)."""
        phi_hp = 3 * np.pi / 2
        az = self.hp_phi_to_az(phi_hp)
        assert abs(az - np.pi) < 1e-12, (
            f"South (phi_hp=3π/2) should map to az=π, got {az:.6f}"
        )

    def test_two_pi_wrapping(self):
        """Result is always in [0, 2π); no negative values or values ≥ 2π."""
        phi_values = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        az_values = self.hp_phi_to_az(phi_values)
        assert np.all(az_values >= 0.0), "Azimuth should be non-negative."
        assert np.all(az_values < 2 * np.pi), "Azimuth should be < 2π."

    def test_vectorised_array(self):
        """Formula works element-wise on arrays (as used in get_sky_at_lst)."""
        phi_hp = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        expected_az = np.array([np.pi / 2, 0.0, 3 * np.pi / 2, np.pi])
        az = self.hp_phi_to_az(phi_hp)
        np.testing.assert_allclose(az, expected_az, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 3: setup_observation — site and time grid
# ---------------------------------------------------------------------------

class TestSetupObservation:
    """Verify setup_observation returns the MRO site and 24 hourly UTC times."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.t0, self.utc_times, self.site = setup_observation()

    def test_24_lst_bins(self):
        assert len(self.utc_times) == 24, (
            f"Expected 24 UTC time bins, got {len(self.utc_times)}"
        )

    def test_site_latitude(self):
        assert abs(self.site.lat.deg - (-26.70)) < 0.01, (
            f"Site latitude should be -26.70°, got {self.site.lat.deg:.4f}°"
        )

    def test_site_longitude(self):
        assert abs(self.site.lon.deg - 116.60) < 0.01, (
            f"Site longitude should be +116.60°, got {self.site.lon.deg:.4f}°"
        )

    def test_site_altitude(self):
        assert abs(self.site.height.to('m').value - 377.0) < 1.0, (
            f"Site altitude should be ~377 m, got {self.site.height.to('m').value:.1f} m"
        )

    def test_start_epoch(self):
        """t0 should be 2026-10-21 00:00:00 UTC."""
        from astropy.time import Time
        t_expected = Time('2026-10-21 00:00:00', scale='utc')
        assert abs(self.t0.unix - t_expected.unix) < 1.0, (
            f"t0 should be 2026-10-21 00:00:00 UTC, got {self.t0.iso}"
        )

    def test_hourly_spacing(self):
        """Consecutive UTC times should be separated by exactly 1 hour."""
        import astropy.units as u
        for i in range(1, 24):
            dt = (self.utc_times[i] - self.utc_times[i - 1]).to(u.hr).value
            assert abs(dt - 1.0) < 1e-6, (
                f"Time step {i} should be 1 hr, got {dt:.8f} hr"
            )

    def test_returns_three_values(self):
        """Function must return exactly (t0, utc_times, site)."""
        result = setup_observation()
        assert len(result) == 3
