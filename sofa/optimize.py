"""Outer optimization loop for SOFA beam coefficient optimization.

Implements the loss function, manual DE/rand/1/bin global search, and
Powell local refinement for finding VSH beam coefficients that minimize
21 cm signal recovery error while suppressing false positives.
"""

import functools

import numpy as np
import scipy.optimize
from joblib import Parallel, delayed

from .beam import beam_from_x, pack_coeffs, unpack_coeffs, vsh_basis
from .fitting import fit_21cm_bayesian, signal_21cm
from .ionosphere import apply_ionospheric_correction
from .skymap import sky_temperature

# antenna_temp imports — listed separately so missing stubs surface clearly
from .antenna_temp import compute_antenna_temperature, get_sky_at_lst


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(x, realizations, sky_model_params, upper_mask, utc_times, site,
                 tec_zen_tecu, n_e_max, freq_mhz, basis, basis_image, domega,
                 h_m=0.5, nside=64, w_A=1.0, w_nu=1.0, w_sigma=1.0, lambda_=10.0,
                 delta_nu_hz=0.145e6, delta_t_s=3600.0, seed=0):
    """Compute the optimization loss for a beam parameter vector.

    Parameters
    ----------
    x : ndarray, shape (420,)
        Real parameter vector from ``pack_coeffs``.
    realizations : list of (float, float, float)
        ``[(A, nu0, sigma), ...]``; first element is the null injection
        ``(0.0, 40.0, 15.0)``, remaining N elements are signal realizations.
    sky_model_params : tuple
        ``(A_map, beta_bin_map, C_map)`` from ``build_sky_model()``; each
        array has shape ``(n_pix,)``.
    upper_mask : ndarray of bool, shape (n_pix_full,)
        Upper-hemisphere pixel mask from ``vsh_basis()``.
    utc_times : list of astropy.time.Time
        24 UTC observation times (one per LST bin).
    site : astropy.coordinates.EarthLocation
        Observatory location (MRO).
    tec_zen_tecu : float
        Zenith total electron content [TECU].
    n_e_max : float
        Maximum electron density [m^{-3}]; sets plasma frequency.
    freq_mhz : ndarray, shape (n_freq,)
        Frequency grid [MHz].
    basis : dict
        VSH basis returned by ``vsh_basis(lmax, nside)``.
    basis_image : dict
        VSH basis evaluated at ``pi - theta_u`` (image reflection).
    domega : float
        Pixel solid angle [sr] = ``4π / nside2npix(nside)``.
    h_m : float
        Antenna height above PEC ground plane [m].
    nside : int
        HEALPix nside.
    w_A : float
        MSE weight for amplitude [K^{-2}].
    w_nu : float
        MSE weight for centre frequency [MHz^{-2}].
    w_sigma : float
        MSE weight for line width [MHz^{-2}].
    lambda_ : float
        Penalty weight for false-positive rate.
    delta_nu_hz : float
        Channel bandwidth [Hz]; sets radiometric noise.
    delta_t_s : float
        Integration time per LST bin [s]; sets radiometric noise.
    seed : int
        RNG seed for noise draws and LST sampling.

    Returns
    -------
    float
        ``mse_total / N + lambda_ * P_90(false_positive_rate)``.
    """
    rng = np.random.default_rng(seed)
    A_map, beta_bin_map, C_map = sky_model_params
    n_freq = len(freq_mhz)
    n_pix_full = upper_mask.shape[0]
    n_pix_upper = int(upper_mask.sum())

    # ------------------------------------------------------------------
    # Step a: beam power pattern (n_freq, n_pix_upper)
    # ------------------------------------------------------------------
    Btilde = beam_from_x(x, h_m, freq_mhz, basis, basis_image, domega)

    # ------------------------------------------------------------------
    # Pre-compute ionosphere-corrected sky maps at each frequency.
    # This is LST-independent; the sky rotation is applied inside the
    # LST loop below via get_sky_at_lst.
    # Shape: (n_freq, n_pix_full) — full-sky ICRS maps.
    # ------------------------------------------------------------------
    T_iono_maps = np.empty((n_freq, n_pix_full), dtype=np.float64)
    for i_freq, nu in enumerate(freq_mhz):
        T_full = sky_temperature(A_map, beta_bin_map, C_map, nu)          # (n_pix_full,)
        T_iono = apply_ionospheric_correction(
            T_full, nu * 1e6, tec_zen_tecu, n_e_max, nside=nside
        )                                                                   # (n_pix_full,)
        T_iono_maps[i_freq] = T_iono

    # ------------------------------------------------------------------
    # Step b: antenna temperature T_ant (n_freq, 24)
    # For each LST bin, rotate sky into local AltAz frame and integrate.
    # ------------------------------------------------------------------
    T_ant = np.zeros((n_freq, 24), dtype=np.float64)
    for i_lst in range(24):
        # Build (n_freq, n_pix_upper) upper-hemisphere sky in local frame
        T_sky_upper = np.empty((n_freq, n_pix_upper), dtype=np.float64)
        for i_freq in range(n_freq):
            T_sky_upper[i_freq] = get_sky_at_lst(
                T_iono_maps[i_freq], upper_mask,
                utc_times[i_lst], site, nside
            )
        # Beam-weighted average: T_ant = Σ B̃ T dΩ / Σ B̃ dΩ
        # Denominator ≈ 2π when Btilde is normalized.
        numerator   = (Btilde * T_sky_upper * domega).sum(axis=-1)   # (n_freq,)
        denominator = (Btilde * domega).sum(axis=-1)                  # (n_freq,)
        T_ant[:, i_lst] = numerator / denominator

    # ------------------------------------------------------------------
    # Separate null realization from signal realizations.
    # ------------------------------------------------------------------
    null_realization   = realizations[0]     # (A=0, nu0, sigma)
    signal_realizations = realizations[1:]   # N signal realizations

    N = len(signal_realizations)
    if N == 0:
        raise ValueError("realizations must contain at least one signal entry "
                         "(beyond the null).")

    # ------------------------------------------------------------------
    # Step c: MSE over signal realizations
    # ------------------------------------------------------------------
    mse_total = 0.0
    for A_i, nu0_i, sig_i in signal_realizations:
        i_lst_rand = int(rng.integers(24))
        sigma_n = T_ant[:, i_lst_rand] / np.sqrt(delta_nu_hz * delta_t_s)
        sigma_n = np.maximum(sigma_n, 1e-6)   # guard against zero noise

        S_inj = signal_21cm(freq_mhz, A_i, nu0_i, sig_i)
        noise = rng.normal(0.0, sigma_n)
        T_obs = T_ant[:, i_lst_rand] + S_inj + noise

        try:
            result  = fit_21cm_bayesian(T_obs, freq_mhz, sigma_n, sampler='emcee')
            samples = result['samples']          # (n_samples, 8)
            # theta = [c_0, c_1, c_2, c_3, c_4, A, nu_0, sigma]
            A_hat   = np.median(samples[:, 5])
            nu0_hat = np.median(samples[:, 6])
            sig_hat = np.median(samples[:, 7])
            mse_total += (w_A     * (A_hat   - A_i  ) ** 2
                          + w_nu  * (nu0_hat - nu0_i) ** 2
                          + w_sigma * (sig_hat - sig_i) ** 2)
        except Exception:
            # Penalise failed fits so the optimizer steers away
            mse_total += 1.0e6

    # ------------------------------------------------------------------
    # Step d: false-positive rate for the null injection over all 24 LST bins
    # P_false_pos[i_lst] = fraction of posterior A samples > 0.05 K when A_true=0
    # ------------------------------------------------------------------
    A_null, nu0_null, sig_null = null_realization
    P_false_pos = np.ones(24, dtype=np.float64)   # default: penalise fully

    for i_lst in range(24):
        sigma_n = T_ant[:, i_lst] / np.sqrt(delta_nu_hz * delta_t_s)
        sigma_n = np.maximum(sigma_n, 1e-6)

        S_null = signal_21cm(freq_mhz, A_null, nu0_null, sig_null)  # ≡ 0 when A_null=0
        noise  = rng.normal(0.0, sigma_n)
        T_obs_null = T_ant[:, i_lst] + S_null + noise

        try:
            result_null   = fit_21cm_bayesian(T_obs_null, freq_mhz, sigma_n, sampler='emcee')
            samples_null  = result_null['samples']       # (n_samples, 8)
            P_false_pos[i_lst] = float(np.mean(samples_null[:, 5] > 0.05))
        except Exception:
            P_false_pos[i_lst] = 1.0

    # ------------------------------------------------------------------
    # Step e: combine and return
    # ------------------------------------------------------------------
    loss = mse_total / N + lambda_ * float(np.percentile(P_false_pos, 90))
    return loss


# ---------------------------------------------------------------------------
# Phase 1 — DE/rand/1/bin global search
# ---------------------------------------------------------------------------

def run_de_optimization(loss_fn, n_real=420, n_de_iter=50, pop_size=None,
                         F=0.8, CR=0.9, bounds=(-2.0, 2.0), seed=42):
    """DE/rand/1/bin global optimisation with joblib parallelism.

    .. warning::
        **Do NOT** replace this with ``scipy.optimize.differential_evolution``
        using ``workers=-1``.  The ``workers`` flag uses ``multiprocessing``,
        which deadlocks when nested ``emcee`` calls also spawn processes.
        This implementation uses joblib's ``loky`` backend instead, which is
        fork-safe.

    Parameters
    ----------
    loss_fn : callable
        Loss function ``f(x) -> float``.  Must accept a 1-D real array of
        length *n_real* and return a scalar.
    n_real : int
        Dimensionality of the search space (default 420).
    n_de_iter : int
        Number of DE generations.
    pop_size : int or None
        Population size.  Defaults to ``15 * n_real`` (scipy convention).
    F : float
        Differential weight (mutation scale factor).
    CR : float
        Crossover probability.
    bounds : tuple of (float, float)
        Scalar ``(low, high)`` applied uniformly to all dimensions.
    seed : int
        NumPy ``default_rng`` seed.

    Returns
    -------
    ndarray, shape (n_real,)
        Best parameter vector found after *n_de_iter* generations.
    """
    rng = np.random.default_rng(seed)

    if pop_size is None:
        pop_size = 15 * n_real

    low, high = bounds
    low_arr  = np.full(n_real, low,  dtype=np.float64)
    high_arr = np.full(n_real, high, dtype=np.float64)

    # Initialise population uniformly within bounds
    population = rng.uniform(low_arr, high_arr, (pop_size, n_real))

    def _eval_pop(pop):
        """Evaluate *loss_fn* for every member of *pop* in parallel."""
        return Parallel(n_jobs=-1, backend='loky')(
            delayed(loss_fn)(x) for x in pop
        )

    # Evaluate initial population
    fitness = np.array(_eval_pop(population), dtype=np.float64)

    idx_all = np.arange(pop_size)

    for gen in range(n_de_iter):
        trials = np.empty_like(population)

        for i in range(pop_size):
            # DE/rand/1: pick three distinct donors (≠ i)
            donors = idx_all[idx_all != i]
            a, b, c = rng.choice(donors, 3, replace=False)

            # Mutation + clip to bounds
            v = np.clip(
                population[a] + F * (population[b] - population[c]),
                low_arr, high_arr
            )

            # Binomial crossover
            cross = rng.random(n_real) < CR
            # Guarantee at least one dimension taken from the mutant
            cross[rng.integers(n_real)] = True
            trials[i] = np.where(cross, v, population[i])

        # Evaluate trial vectors in parallel
        trial_fit = np.array(_eval_pop(trials), dtype=np.float64)

        # Greedy selection
        for i in range(pop_size):
            if trial_fit[i] <= fitness[i]:
                population[i] = trials[i]
                fitness[i]    = trial_fit[i]

        best_val = float(fitness.min())
        print(f"[DE] gen {gen + 1:3d}/{n_de_iter}: best loss = {best_val:.6g}")

    best_idx = int(np.argmin(fitness))
    return population[best_idx].copy()


# ---------------------------------------------------------------------------
# Phase 2 — Powell local refinement
# ---------------------------------------------------------------------------

def run_powell_refinement(loss_fn, x0):
    """Powell local refinement starting from *x0*.

    Parameters
    ----------
    loss_fn : callable
        Loss function ``f(x) -> float``.
    x0 : ndarray
        Initial parameter vector (typically the DE global-search result).

    Returns
    -------
    ndarray
        Refined parameter vector ``result.x``.
    """
    n_real = len(x0)
    result = scipy.optimize.minimize(
        loss_fn, x0, method='Powell',
        options={
            'xtol':    1e-3,
            'ftol':    1e-3,
            'maxiter': 200 * n_real,
        }
    )
    return result.x


# ---------------------------------------------------------------------------
# Top-level coordinator
# ---------------------------------------------------------------------------

def optimize_beam(sky_model_params, upper_mask, utc_times, site,
                   tec_zen_tecu, n_e_max, freq_mhz,
                   lmax=5, h_m=0.5, nside=64, n_de_iter=50,
                   n_signal_realizations=100, seed=42):
    """Full beam optimisation pipeline (DE global → Powell local).

    Parameters
    ----------
    sky_model_params : tuple
        ``(A_map, beta_bin_map, C_map)`` from ``build_sky_model()``.
    upper_mask : ndarray of bool, shape (n_pix_full,)
        Upper-hemisphere mask; typically ``vsh_basis()['upper_mask']``.
    utc_times : list of astropy.time.Time
        24 UTC observation times for LST bins.
    site : astropy.coordinates.EarthLocation
        Observatory location (MRO: lat=−26.70°, lon=+116.60°, alt=377 m).
    tec_zen_tecu : float
        Zenith TEC [TECU].
    n_e_max : float
        Maximum electron density [m^{-3}].
    freq_mhz : ndarray, shape (n_freq,)
        Frequency grid [MHz]; e.g. ``np.linspace(5.0, 150.0, 1000)``.
    lmax : int
        Maximum VSH degree (default 5 → ``n_lm = 35``).
    h_m : float
        Antenna height above PEC ground plane [m].
    nside : int
        HEALPix nside parameter.
    n_de_iter : int
        Number of DE generations.
    n_signal_realizations : int
        Number of signal realizations ``N`` for MSE computation.
    seed : int
        Master RNG seed.

    Returns
    -------
    alpha_opt : ndarray, shape (n_lm, 3), dtype complex
        Optimised M-mode polynomial coefficients.
    beta_opt : ndarray, shape (n_lm, 3), dtype complex
        Optimised N-mode polynomial coefficients.
    x_opt : ndarray, shape (420,)
        Optimised real parameter vector (output of ``pack_coeffs``).
    """
    import healpy

    rng = np.random.default_rng(seed)

    P    = 2
    n_lm = lmax * lmax + 2 * lmax   # 35 for lmax=5

    # ------------------------------------------------------------------
    # Pre-compute VSH bases (expensive; done once here)
    # ------------------------------------------------------------------
    basis = vsh_basis(lmax=lmax, nside=nside)

    # Image basis: same structure but evaluated at theta_image = pi - theta_u
    # (reflection through the ground plane for the image-theory correction).
    # vsh_basis accepts an optional theta_override kwarg for this purpose.
    theta_image = np.pi - basis['theta']
    basis_image = vsh_basis(lmax=lmax, nside=nside,
                             theta_override=theta_image,
                             phi_override=basis['phi'])

    # Pixel solid angle [sr]
    domega = 4.0 * np.pi / healpy.nside2npix(nside)

    # ------------------------------------------------------------------
    # Build signal realizations list
    # Order: [(0.0, 40.0, 15.0), (A_1, nu0_1, sig_1), ..., (A_N, nu0_N, sig_N)]
    # ------------------------------------------------------------------
    N = n_signal_realizations
    signal_A   = rng.uniform(0.05, 0.5,  N)
    signal_nu0 = rng.uniform(7.0,  80.0, N)
    signal_sig = rng.uniform(5.0,  30.0, N)
    realizations = [(0.0, 40.0, 15.0)] + list(
        zip(signal_A.tolist(), signal_nu0.tolist(), signal_sig.tolist())
    )

    # ------------------------------------------------------------------
    # Initial coefficient vector: VED-like (N_10 dominant) + small perturbation
    # Index 1 = (l=1, m=0) in the ordering (l=1,m=-1),(l=1,m=0),(l=1,m=+1),...
    # ------------------------------------------------------------------
    pert_rng = np.random.default_rng(seed + 1)

    alpha = np.zeros((n_lm, P + 1), dtype=complex)
    beta  = np.zeros((n_lm, P + 1), dtype=complex)
    beta[1, 0] = 1.0   # VED dipole starting point

    alpha += 0.01 * (pert_rng.standard_normal(alpha.shape)
                     + 1j * pert_rng.standard_normal(alpha.shape))
    beta  += 0.01 * (pert_rng.standard_normal(beta.shape)
                     + 1j * pert_rng.standard_normal(beta.shape))

    x0 = pack_coeffs(alpha, beta)   # shape (420,), real

    # ------------------------------------------------------------------
    # Bind all fixed arguments into a single-argument loss callable
    # ------------------------------------------------------------------
    loss_partial = functools.partial(
        compute_loss,
        realizations    = realizations,
        sky_model_params= sky_model_params,
        upper_mask      = upper_mask,
        utc_times       = utc_times,
        site            = site,
        tec_zen_tecu    = tec_zen_tecu,
        n_e_max         = n_e_max,
        freq_mhz        = freq_mhz,
        basis           = basis,
        basis_image     = basis_image,
        domega          = domega,
        h_m             = h_m,
        nside           = nside,
    )

    # ------------------------------------------------------------------
    # Phase 1: DE global search
    # ------------------------------------------------------------------
    print("=== Phase 1: DE/rand/1/bin global search ===")
    x_de = run_de_optimization(
        loss_partial,
        n_real    = len(x0),
        n_de_iter = n_de_iter,
        F         = 0.8,
        CR        = 0.9,
        bounds    = (-2.0, 2.0),
        seed      = seed,
    )

    # ------------------------------------------------------------------
    # Phase 2: Powell local refinement
    # ------------------------------------------------------------------
    print("=== Phase 2: Powell local refinement ===")
    x_opt = run_powell_refinement(loss_partial, x_de)

    # Unpack optimised coefficients
    alpha_opt, beta_opt = unpack_coeffs(x_opt, n_lm=n_lm, P=P)

    return alpha_opt, beta_opt, x_opt
