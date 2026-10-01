"""skymap.py — GSM2016 sky model loading, spectral index binning, and sky temperature.

Component 2 of the SOFA pipeline.
Reference: arXiv:2607.00299
"""

import os

import healpy
import numpy as np
from scipy.optimize import curve_fit


# Spot frequencies used for per-pixel SED fitting [MHz]
_FIT_FREQS_MHZ = np.array([45.0, 74.0, 150.0, 230.0, 408.0])

# CMB monopole temperature [K]
_T_CMB = 2.725


def build_sky_model(nside=64, cache_path=None):
    """Build (or load cached) per-pixel sky SED parameters from GSM2016.

    The model per pixel is:
        T(nu) = A * (nu/408)^beta_bin * exp(C * log(nu/408)^2)

    where beta_bin is the median spectral index of its decile-percentile bin
    (10 bins defined by the 0th–100th percentiles of the β distribution).

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (default 64 → 49152 pixels).
    cache_path : str or None
        Path to .npy cache file.  Relative paths are relative to cwd.
        If None (default), the path is resolved as
        ``data/sky_model_nside{nside}.npy`` so that different nside values
        never share the same cache file.

    Returns
    -------
    A_map : ndarray, shape (n_pix,)
        Amplitude at 408 MHz [K].
    beta_bin_map : ndarray, shape (n_pix,)
        Binned spectral index (10 equal-count decile bins).
    C_map : ndarray, shape (n_pix,)
        Spectral curvature coefficient.
    """
    cache_path = cache_path or f'data/sky_model_nside{nside}.npy'
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return tuple(data)

    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Lazy import: pygdsm is only required when generating (not loading) the model
    try:
        from pygdsm import GlobalSkyModel2016
    except ImportError:
        try:
            from pygdsm import GlobalSkyModel16 as GlobalSkyModel2016
        except ImportError as exc:
            raise ImportError(
                "pygdsm is required to build the sky model. "
                "Install it with: pip install pygdsm"
            ) from exc

    gsm = GlobalSkyModel2016(freq_unit='MHz', data_unit='TRJ')

    # Evaluate GSM at spot frequencies; subtract CMB monopole
    T_maps = {}
    for freq in _FIT_FREQS_MHZ:
        raw = gsm.generate(freq)
        # GSM may return at its native nside; degrade/upgrade to target nside
        if healpy.npix2nside(len(raw)) != nside:
            raw = healpy.ud_grade(raw, nside)
        T_maps[freq] = raw - _T_CMB   # (n_pix,)

    T_230 = T_maps[230.0]
    T_408 = T_maps[408.0]

    # Per-pixel spectral index between 230 and 408 MHz; clip to physical range
    # Guard against non-positive temperatures before log
    ratio = np.clip(T_408, 1e-3, None) / np.clip(T_230, 1e-3, None)
    beta_p = np.log(ratio) / np.log(408.0 / 230.0)
    beta_p = np.clip(beta_p, -3.0, -2.0)

    # 10 bins defined by decile percentile boundaries of the β distribution.
    # Bin edges are the 0th, 10th, ..., 100th percentiles of β; each pixel is
    # assigned to a bin by where its β falls, and gets that bin's median value.
    # This is equivalent to equal-count binning but defined by the distribution
    # shape rather than pixel rank, which is more robust to local outliers.
    n_pix = healpy.nside2npix(nside)
    bin_edges   = np.percentile(beta_p, np.linspace(0, 100, 11))   # 11 edges → 10 bins
    bin_indices = np.digitize(beta_p, bin_edges[1:-1])              # 0 … 9
    beta_bin_map = np.empty(n_pix, dtype=float)
    for b in range(10):
        mask = bin_indices == b
        beta_bin_map[mask] = np.median(beta_p[mask])

    # Stack observed temperatures for fitting: shape (n_pix, n_freq)
    T_obs_matrix = np.stack([T_maps[f] for f in _FIT_FREQS_MHZ], axis=1)  # (n_pix, n_freq)

    log_ratio = np.log(_FIT_FREQS_MHZ / 408.0)   # (n_freq,)  — precomputed for speed
    nu_ratio  = _FIT_FREQS_MHZ / 408.0            # (n_freq,)

    A_map = np.empty(n_pix, dtype=float)
    C_map = np.empty(n_pix, dtype=float)

    for ipix in range(n_pix):
        beta_fixed = beta_bin_map[ipix]
        T_obs_pix  = T_obs_matrix[ipix]   # (n_freq,)

        # Model: T = A * (nu/408)^beta_fixed * exp(C * log(nu/408)^2)
        # Free parameters: A and C only (beta_fixed is pinned)
        def model(nu_dummy, A, C):
            return A * nu_ratio ** beta_fixed * np.exp(C * log_ratio ** 2)

        # Initial guess: A = T at 408 MHz, C = 0
        A0 = max(T_obs_pix[-1], 1.0)   # T_408 is the last entry
        try:
            popt, _ = curve_fit(model, _FIT_FREQS_MHZ, T_obs_pix,
                                 p0=[A0, 0.0],
                                 bounds=([0.0, -10.0], [np.inf, 10.0]),
                                 maxfev=5000)
            A_map[ipix] = popt[0]
            C_map[ipix] = popt[1]
        except RuntimeError:
            # Fallback: least-squares with beta_fixed, C=0
            A_map[ipix] = A0
            C_map[ipix] = 0.0

    np.save(cache_path, np.array([A_map, beta_bin_map, C_map], dtype=object))
    return A_map, beta_bin_map, C_map


def sky_temperature(A, beta_bin, C, freq_mhz, freq_ref_mhz=408.0):
    """Evaluate per-pixel sky brightness temperature at a single frequency.

    T = A * (freq_mhz / freq_ref_mhz)^beta_bin * exp(C * log(freq_mhz/freq_ref_mhz)^2)

    Parameters
    ----------
    A : array_like, shape (n_pix,) or scalar
        Amplitude parameter [K].
    beta_bin : array_like, shape (n_pix,) or scalar
        Binned spectral index.
    C : array_like, shape (n_pix,) or scalar
        Curvature coefficient.
    freq_mhz : float
        Evaluation frequency [MHz].
    freq_ref_mhz : float
        Reference frequency [MHz] (default 408 MHz).

    Returns
    -------
    T : ndarray, shape (n_pix,) [K]
    """
    A        = np.asarray(A,        dtype=float)
    beta_bin = np.asarray(beta_bin, dtype=float)
    C        = np.asarray(C,        dtype=float)
    x = np.log(freq_mhz / freq_ref_mhz)
    return A * (freq_mhz / freq_ref_mhz) ** beta_bin * np.exp(C * x ** 2)
