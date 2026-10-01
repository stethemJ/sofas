"""tests/test_skymap.py — Validation tests for sofa.skymap (Component 2).

Tests cover:
  1. sky_temperature at reference frequency equals amplitude.
  2. sky_temperature power-law scaling.
  3. Smoke test: build_sky_model creates cache and returns 3 arrays.
"""

import os
import tempfile

import numpy as np
import pytest

from sofa.skymap import sky_temperature

# ---------------------------------------------------------------------------
# pygdsm availability guard (used for build_sky_model smoke test)
# ---------------------------------------------------------------------------
try:
    from pygdsm import GlobalSkyModel2016  # noqa: F401
    pygdsm_available = True
except ImportError:
    pygdsm_available = False

# Shared RNG (seed 42 per project convention)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Test 1: sky_temperature at reference frequency equals amplitude
# ---------------------------------------------------------------------------

def test_sky_temperature_at_reference_frequency():
    """T(freq_ref) = A for all pixels (nu/nu_ref = 1, exp(C*0) = 1)."""
    A = np.ones(100)
    beta = -2.5 * np.ones(100)
    C = np.zeros(100)
    T = sky_temperature(A, beta, C, freq_mhz=408.0)
    assert np.allclose(T, 1.0), (
        f"sky_temperature at reference frequency should equal A=1; "
        f"max deviation = {np.abs(T - 1.0).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2: sky_temperature power-law scaling
# ---------------------------------------------------------------------------

def test_sky_temperature_power_law_scaling():
    """Ratio T(230)/T(408) equals (230/408)^beta for pure power law (C=0)."""
    A = np.ones(100)
    beta = -2.5 * np.ones(100)
    C = np.zeros(100)

    T_230 = sky_temperature(A, beta, C, freq_mhz=230.0)
    T_408 = sky_temperature(A, beta, C, freq_mhz=408.0)

    ratio = T_230 / T_408
    expected = (230.0 / 408.0) ** (-2.5)

    assert np.allclose(ratio, expected, rtol=1e-5), (
        f"Power-law ratio mismatch: got {ratio[0]:.6f}, expected {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: build_sky_model smoke test (skipped if pygdsm not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not pygdsm_available, reason="pygdsm not installed")
def test_build_sky_model_cache():
    """build_sky_model creates a cache file and returns 3 arrays of shape (n_pix,)."""
    import healpy
    from sofa.skymap import build_sky_model

    nside = 16  # small nside for speed
    n_pix = healpy.nside2npix(nside)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "sky_model_test.npy")

        # First call: should generate and cache
        result = build_sky_model(nside=nside, cache_path=cache_path)

        assert len(result) == 3, (
            f"build_sky_model should return 3 arrays (A, beta_bin, C); got {len(result)}"
        )

        A_map, beta_bin_map, C_map = result

        for name, arr in [("A_map", A_map), ("beta_bin_map", beta_bin_map), ("C_map", C_map)]:
            assert arr.shape == (n_pix,), (
                f"{name} shape {arr.shape} != expected ({n_pix},)"
            )

        assert os.path.exists(cache_path), "Cache file was not created."

        # Second call: should load from cache and produce identical results
        result_cached = build_sky_model(nside=nside, cache_path=cache_path)
        A_c, beta_c, C_c = result_cached

        assert np.allclose(A_map, A_c), "Cached A_map differs from original."
        assert np.allclose(beta_bin_map, beta_c), "Cached beta_bin_map differs from original."
        assert np.allclose(C_map, C_c), "Cached C_map differs from original."
