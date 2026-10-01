"""
fitting.py — Bayesian fitting for the SOFA pipeline.

Model:
    T_model(nu) = T_fg(nu) + S_21(nu)
    T_fg = sum_k c_k * [ln(nu/75)]^k    (log-polynomial foreground, k=0..4)
    S_21 = -A * exp(-(nu - nu0)^2 / (2*sigma^2))  (Gaussian 21cm absorption)

Parameters theta = [c_0, c_1, c_2, c_3, c_4, A, nu_0, sigma]  (8 total)

Priors:
    c_k ~ Normal(0, sigma_fg[k]^2),  sigma_fg = [5000, 500, 500, 500, 500] K
    A   ~ Uniform(0, 1) K
    nu0 ~ Uniform(7, 80) MHz
    sigma ~ Uniform(5, 30) MHz
"""

import numpy as np
import scipy.optimize
import emcee
import dynesty
from scipy.stats import norm

# ── Priors ──────────────────────────────────────────────────────────────────
_SIGMA_FG = [5000.0, 500.0, 500.0, 500.0, 500.0]   # K, Normal widths for c_k
_N_PARAMS  = 8


# ── Forward models ───────────────────────────────────────────────────────────

def foreground_model(freq_mhz, c, nu_ref=75.0):
    """Log-polynomial foreground.

    T_fg(nu) = sum_{k=0}^{len(c)-1} c[k] * ln(nu / nu_ref)^k

    Args:
        freq_mhz: (n_freq,) array or scalar [MHz]
        c:        sequence of coefficients length 5
        nu_ref:   reference frequency [MHz], default 75

    Returns:
        T_fg: same shape as freq_mhz [K]
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    x = np.log(freq_mhz / nu_ref)
    return sum(c[k] * x**k for k in range(len(c)))


def signal_21cm(freq_mhz, A, nu0_mhz, sigma_mhz):
    """Gaussian 21 cm absorption feature.

    S_21(nu) = -A * exp(-(nu - nu0)^2 / (2*sigma^2))

    Args:
        freq_mhz:  (n_freq,) array or scalar [MHz]
        A:         amplitude [K], positive means absorption
        nu0_mhz:   centre frequency [MHz]
        sigma_mhz: width [MHz]

    Returns:
        S_21: same shape as freq_mhz [K]  (negative, i.e. dip)
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    return -A * np.exp(-0.5 * ((freq_mhz - nu0_mhz) / sigma_mhz) ** 2)


# ── Likelihood / prior / posterior ──────────────────────────────────────────

def log_likelihood(theta, T_obs, freq_mhz, sigma_n):
    """Gaussian log-likelihood.

    Args:
        theta:    (8,) array [c_0..c_4, A, nu0, sigma]
        T_obs:    (n_freq,) observed antenna temperature [K]
        freq_mhz: (n_freq,) frequency array [MHz]
        sigma_n:  (n_freq,) per-channel noise standard deviation [K]

    Returns:
        scalar log-likelihood
    """
    c = theta[:5]
    A, nu0, sig = theta[5], theta[6], theta[7]
    T_model = foreground_model(freq_mhz, c) + signal_21cm(freq_mhz, A, nu0, sig)
    residuals = (T_obs - T_model) / sigma_n
    return -0.5 * np.sum(residuals ** 2)


def log_prior(theta):
    """Log-prior.

    Normal priors on c_k; uniform on A, nu0, sigma.

    Returns:
        log-prior value (scalar), or -np.inf if any uniform prior is violated.
    """
    c = theta[:5]
    A, nu0, sig = theta[5], theta[6], theta[7]

    # Uniform prior bounds
    if not (0.0 <= A <= 1.0):
        return -np.inf
    if not (7.0 <= nu0 <= 80.0):
        return -np.inf
    if not (5.0 <= sig <= 30.0):
        return -np.inf

    # Normal priors on log-polynomial coefficients
    lp = 0.0
    for k in range(5):
        lp += norm.logpdf(c[k], 0.0, _SIGMA_FG[k])

    return lp


def log_posterior(theta, T_obs, freq_mhz, sigma_n):
    """Log-posterior = log_prior + log_likelihood.

    Args:
        theta:    (8,) array [c_0..c_4, A, nu0, sigma]
        T_obs:    (n_freq,) observed antenna temperature [K]
        freq_mhz: (n_freq,) frequency array [MHz]
        sigma_n:  (n_freq,) per-channel noise standard deviation [K]

    Returns:
        scalar log-posterior, or -np.inf if prior is violated.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, T_obs, freq_mhz, sigma_n)


# ── Prior transform for dynesty ──────────────────────────────────────────────

def prior_transform(u):
    """Map unit-hypercube sample u ∈ [0,1]^8 to prior space.

    theta = [c_0, c_1, c_2, c_3, c_4, A, nu_0, sigma]

    Args:
        u: (8,) array with values in [0, 1]

    Returns:
        theta: (8,) array in prior space
    """
    c = [norm.ppf(u[k], 0.0, _SIGMA_FG[k]) for k in range(5)]
    A   = u[5] * 1.0           # Uniform(0, 1)
    nu0 = 7.0 + u[6] * 73.0   # Uniform(7, 80)
    sig = 5.0 + u[7] * 25.0   # Uniform(5, 30)
    return np.array(c + [A, nu0, sig])


# ── Main fitting routine ─────────────────────────────────────────────────────

def fit_21cm_bayesian(T_obs, freq_mhz, sigma_n, sampler='emcee',
                      n_walkers=32, n_steps=5000, n_burn=500, nlive=500):
    """Bayesian fit of foreground + 21 cm signal.

    Args:
        T_obs:    (n_freq,) observed antenna temperature [K]
        freq_mhz: (n_freq,) frequency array [MHz]
        sigma_n:  (n_freq,) per-channel noise standard deviation [K]
        sampler:  'emcee' (MCMC) or 'dynesty' (nested sampling)
        n_walkers: number of emcee walkers (emcee only)
        n_steps:   total MCMC steps per walker (emcee only)
        n_burn:    burn-in steps to discard (emcee only)
        nlive:     number of live points (dynesty only)

    Returns:
        dict with:
          emcee  → {'samples': (n_samples, 8), 'sampler': EnsembleSampler, 'tau': array}
          dynesty→ {'results': Results, 'samples': (n_live, 8)}
    """
    if sampler == 'emcee':
        # ── MAP initialisation via Nelder-Mead ──────────────────────────────
        x0 = np.array([3000.0, 0.0, 0.0, 0.0, 0.0, 0.1, 40.0, 15.0])
        opt = scipy.optimize.minimize(
            lambda t: -log_likelihood(t, T_obs, freq_mhz, sigma_n),
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 5000},
        )
        theta_map = opt.x

        # ── Initialise walkers in a tight ball around MAP ────────────────────
        pos0 = theta_map + 1e-4 * np.random.randn(n_walkers, _N_PARAMS)

        # ── Run emcee ────────────────────────────────────────────────────────
        sampler_obj = emcee.EnsembleSampler(
            n_walkers, _N_PARAMS,
            log_posterior,
            args=(T_obs, freq_mhz, sigma_n),
        )
        sampler_obj.run_mcmc(pos0, n_steps, progress=False)

        tau = sampler_obj.get_autocorr_time(quiet=True)
        # Only check convergence when chain is long enough for autocorr estimation
        if not np.any(np.isnan(tau)):
            assert np.all(tau < n_steps / 50), f"Not converged: tau_max={tau.max():.0f}"
        samples = sampler_obj.get_chain(discard=n_burn, flat=True)  # (n_samples, 8)

        return {
            'samples': samples,
            'sampler': sampler_obj,
            'tau':     tau,
        }

    elif sampler == 'dynesty':
        # ── dynesty nested sampling ──────────────────────────────────────────
        def log_likelihood_wrapper(theta):
            return log_likelihood(theta, T_obs, freq_mhz, sigma_n)

        dyn_sampler = dynesty.NestedSampler(
            log_likelihood_wrapper,
            prior_transform,
            ndim=_N_PARAMS,
            nlive=nlive,
        )
        dyn_sampler.run_nested(dlogz=0.01)
        results = dyn_sampler.results

        return {
            'results': results,
            'samples': results.samples,
        }

    else:
        raise ValueError(f"Unknown sampler '{sampler}'. Choose 'emcee' or 'dynesty'.")
