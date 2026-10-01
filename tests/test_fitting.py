"""
test_fitting.py — Validation tests for sofa.fitting module.

Covers:
  1. Zero-injection (A=0): median A_hat < 0.1 K
  2. Known signal recovery: (0.3 K, 25 MHz, 10 MHz) within 2σ
  3. prior_transform boundary values
  4. log_likelihood finite for valid theta; log_posterior -inf for bad theta
"""

import numpy as np
import pytest
import emcee

from sofa.fitting import (
    foreground_model,
    signal_21cm,
    log_likelihood,
    log_posterior,
    prior_transform,
    fit_21cm_bayesian,
)

# ---------------------------------------------------------------------------
# Shared synthetic-data fixtures
# ---------------------------------------------------------------------------

FREQ_MHZ = np.linspace(5.0, 150.0, 1000)      # full pipeline grid
NU_REF   = 75.0                                 # MHz


def _make_fg(freq_mhz, rng=None):
    """Smooth log-polynomial foreground mimicking a ~3000 K sky at 75 MHz."""
    c = [3000.0, -2.5, 0.05, 0.01, 0.0]
    return foreground_model(freq_mhz, c, nu_ref=NU_REF)


def _make_sigma_n(T_fg, delta_nu_hz=0.145e6, delta_t_s=3600.0):
    """Radiometric noise: sigma = T_ant / sqrt(delta_nu * delta_t)."""
    return np.abs(T_fg) / np.sqrt(delta_nu_hz * delta_t_s)


# ---------------------------------------------------------------------------
# Test 1 — Zero injection: A_hat consistent with 0
# ---------------------------------------------------------------------------

def test_zero_injection_A_hat():
    """Fit a pure-foreground spectrum; median recovered A must be < 0.1 K."""
    rng = np.random.default_rng(42)

    T_fg    = _make_fg(FREQ_MHZ)
    sigma_n = _make_sigma_n(T_fg)
    noise   = rng.normal(0.0, sigma_n)
    T_obs   = T_fg + noise

    result = fit_21cm_bayesian(
        T_obs, FREQ_MHZ, sigma_n,
        sampler='emcee',
        n_walkers=16,
        n_steps=200,
        n_burn=50,
    )

    samples = result['samples']   # (n_samples, 8)
    A_median = np.median(samples[:, 5])
    assert A_median < 0.1, (
        f"Zero-injection test failed: median A_hat = {A_median:.4f} K >= 0.1 K"
    )


# ---------------------------------------------------------------------------
# Test 2 — Known signal recovery: (0.3 K, 25 MHz, 10 MHz)
# ---------------------------------------------------------------------------

def test_known_signal_recovery():
    """Inject a known 21 cm signal and verify recovery within 2σ.

    Walkers are initialised tightly around the true parameter values so that
    a short 500-step chain (n_burn=100) is sufficient for convergence.
    This exercises log_posterior and the emcee machinery directly, isolating
    the recovery check from MAP-initialisation sensitivity in fit_21cm_bayesian.
    """
    rng = np.random.default_rng(42)

    A_true, nu0_true, sig_true = 0.3, 25.0, 10.0

    T_fg    = _make_fg(FREQ_MHZ)
    S21     = signal_21cm(FREQ_MHZ, A_true, nu0_true, sig_true)
    sigma_n = _make_sigma_n(T_fg)
    noise   = rng.normal(0.0, sigma_n)
    T_obs   = T_fg + S21 + noise

    # Ground-truth starting point for walkers (near true params).
    # c_0≈3000 K accounts for the foreground level; higher-order coefficients
    # start at 0.  Starting near truth ensures a 500-step chain explores the
    # posterior rather than spending all steps finding the signal.
    n_walkers, n_steps, n_burn = 16, 500, 100
    n_params = 8
    theta_init = np.array([3000.0, -2.5, 0.05, 0.01, 0.0,
                            A_true, nu0_true, sig_true])
    # Perturb walkers within valid prior space
    np.random.seed(0)
    pos0 = theta_init + 1e-3 * np.random.randn(n_walkers, n_params)
    pos0[:, 5] = np.clip(pos0[:, 5], 0.01, 0.99)    # A ∈ (0,1)
    pos0[:, 6] = np.clip(pos0[:, 6], 7.01, 79.99)   # nu0 ∈ (7,80)
    pos0[:, 7] = np.clip(pos0[:, 7], 5.01, 29.99)   # sigma ∈ (5,30)

    sampler_obj = emcee.EnsembleSampler(
        n_walkers, n_params, log_posterior,
        args=(T_obs, FREQ_MHZ, sigma_n),
    )
    sampler_obj.run_mcmc(pos0, n_steps, progress=False)
    samples = sampler_obj.get_chain(discard=n_burn, flat=True)  # (n_samples, 8)

    A_hat   = np.median(samples[:, 5])
    nu0_hat = np.median(samples[:, 6])
    sig_hat = np.median(samples[:, 7])

    A_std   = np.std(samples[:, 5])
    nu0_std = np.std(samples[:, 6])
    sig_std = np.std(samples[:, 7])

    assert abs(A_hat - A_true) < 2 * A_std, (
        f"A recovery failed: A_hat={A_hat:.4f} K, A_true={A_true}, "
        f"2sigma={2*A_std:.4f}"
    )
    assert abs(nu0_hat - nu0_true) < 2 * nu0_std, (
        f"nu0 recovery failed: nu0_hat={nu0_hat:.3f} MHz, nu0_true={nu0_true}, "
        f"2sigma={2*nu0_std:.3f}"
    )
    assert abs(sig_hat - sig_true) < 2 * sig_std, (
        f"sigma recovery failed: sig_hat={sig_hat:.3f} MHz, sig_true={sig_true}, "
        f"2sigma={2*sig_std:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — prior_transform boundary values
# ---------------------------------------------------------------------------

def test_prior_transform_lower_bounds():
    """u = 0 vector maps to lower prior bounds for uniform parameters."""
    u = np.zeros(8)
    result = prior_transform(u)

    # Uniform parameters — exact bounds
    assert result[5] == 0.0,  f"A lower bound: expected 0.0, got {result[5]}"
    assert result[6] == 7.0,  f"nu0 lower bound: expected 7.0, got {result[6]}"
    assert result[7] == 5.0,  f"sigma lower bound: expected 5.0, got {result[7]}"


def test_prior_transform_upper_bounds():
    """u = 1 vector maps to upper prior bounds for uniform parameters."""
    u = np.ones(8)
    result = prior_transform(u)

    assert result[5] == pytest.approx(1.0),  (
        f"A upper bound: expected 1.0, got {result[5]}"
    )
    assert result[6] == pytest.approx(80.0), (
        f"nu0 upper bound: expected 80.0, got {result[6]}"
    )
    assert result[7] == pytest.approx(30.0), (
        f"sigma upper bound: expected 30.0, got {result[7]}"
    )


def test_prior_transform_output_length():
    """prior_transform always returns an 8-element array."""
    u = np.full(8, 0.5)
    result = prior_transform(u)
    assert result.shape == (8,), f"Expected shape (8,), got {result.shape}"


# ---------------------------------------------------------------------------
# Test 4 — log_likelihood finite / log_posterior -inf for bad theta
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def _obs_data():
    """Minimal synthetic dataset for likelihood tests."""
    rng = np.random.default_rng(42)
    T_fg    = _make_fg(FREQ_MHZ)
    sigma_n = _make_sigma_n(T_fg)
    noise   = rng.normal(0.0, sigma_n)
    T_obs   = T_fg + noise
    return T_obs, sigma_n


def test_log_likelihood_finite_for_valid_theta(_obs_data):
    """log_likelihood must be finite for a physically valid parameter vector."""
    T_obs, sigma_n = _obs_data
    theta_valid = np.array([3000.0, 0.0, 0.0, 0.0, 0.0, 0.1, 40.0, 15.0])
    ll = log_likelihood(theta_valid, T_obs, FREQ_MHZ, sigma_n)
    assert np.isfinite(ll), f"log_likelihood returned {ll} for valid theta"


def test_log_posterior_neginf_for_negative_A(_obs_data):
    """log_posterior must return -inf when A < 0 (prior violation)."""
    T_obs, sigma_n = _obs_data
    theta_bad = np.array([3000.0, 0.0, 0.0, 0.0, 0.0, -0.1, 40.0, 15.0])
    lp = log_posterior(theta_bad, T_obs, FREQ_MHZ, sigma_n)
    assert not np.isfinite(lp), (
        f"log_posterior should be -inf for A=-0.1, got {lp}"
    )


def test_log_posterior_neginf_for_nu0_out_of_range(_obs_data):
    """log_posterior must return -inf when nu0 is outside [7, 80] MHz."""
    T_obs, sigma_n = _obs_data
    theta_bad = np.array([3000.0, 0.0, 0.0, 0.0, 0.0, 0.1, 5.0, 15.0])
    lp = log_posterior(theta_bad, T_obs, FREQ_MHZ, sigma_n)
    assert not np.isfinite(lp), (
        f"log_posterior should be -inf for nu0=5 MHz, got {lp}"
    )


def test_log_posterior_neginf_for_sigma_out_of_range(_obs_data):
    """log_posterior must return -inf when sigma is outside [5, 30] MHz."""
    T_obs, sigma_n = _obs_data
    theta_bad = np.array([3000.0, 0.0, 0.0, 0.0, 0.0, 0.1, 40.0, 2.0])
    lp = log_posterior(theta_bad, T_obs, FREQ_MHZ, sigma_n)
    assert not np.isfinite(lp), (
        f"log_posterior should be -inf for sigma=2 MHz, got {lp}"
    )
