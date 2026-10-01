#!/usr/bin/env python3
"""
plot_outputs.py — Run a demonstration SOFA pipeline pass and save all plots.

Outputs saved to outputs/ (created if absent):
  beam_3d.html          — interactive 3D beam pattern (Plotly)
  tant_waterfall.png    — T_ant(freq, LST) waterfall
  signal_recovery.png   — foreground + 21 cm injection and recovery
  posterior_corner.png  — emcee posterior corner plot
  sky_model_408.png     — GSM2016 sky at 408 MHz (skipped if pygdsm absent)
  spectral_index.png    — spectral index decile map (skipped if pygdsm absent)

Uses the VED-like beam initialization (not optimized). Frequency grid and
nside are reduced from production values for speed.
"""

import os
import sys
import warnings
import numpy as np
import healpy

from sofa.beam import vsh_basis, initialize_coeffs, beam_from_x
from sofa.antenna_temp import setup_observation, get_sky_at_lst, compute_antenna_temperature
from sofa.ionosphere import apply_ionospheric_correction
from sofa.fitting import fit_21cm_bayesian, foreground_model, signal_21cm
from sofa.plot import (
    plot_beam_3d,
    plot_tant_waterfall,
    plot_signal_recovery,
    plot_posterior_corner,
    plot_sky_model,
    plot_spectral_index_map,
)

# ── Config ────────────────────────────────────────────────────────────────────
OUTDIR  = os.path.join(os.path.dirname(__file__), 'outputs')
NSIDE   = 32           # 64 in production; 32 for demo speed
LMAX    = 5
H_M     = 0.5          # antenna height above PEC [m]
NU_REF  = 75.0         # reference frequency [MHz]

# Frequency grid — use 100 points for demo (1000 in production)
FREQ_MHZ = np.linspace(5.0, 150.0, 100)

# Ionosphere
TEC_ZEN_TECU = 5.0
N_E_MAX      = 1e11    # plasma freq ≈ 2.84 MHz

# Signal injection for demo fit
A_TRUE   = 0.20   # K
NU0_TRUE = 78.0   # MHz
SIG_TRUE = 18.0   # MHz

# Short emcee run for demo (use 5000 steps in production)
N_WALKERS = 32
N_STEPS   = 800
N_BURN    = 200


def make_synthetic_sky(nside, freq_mhz, beta=-2.55, T_ref=3000.0, nu_ref=408.0):
    """Power-law sky fallback when pygdsm is unavailable."""
    n_pix = healpy.nside2npix(nside)
    T = T_ref * (freq_mhz / nu_ref) ** beta * np.ones(n_pix)
    return T


def load_or_build_sky_model(nside):
    """Return (sky_fn, sky_label) where sky_fn(freq_mhz) → (n_pix,) [K]."""
    try:
        from sofa.skymap import build_sky_model, sky_temperature
        cache = os.path.join(os.path.dirname(__file__), 'data',
                             f'sky_model_nside{nside}.npy')
        A_map, beta_bin, C_map = build_sky_model(nside=nside, cache_path=cache)

        def sky_fn(freq_mhz):
            return sky_temperature(A_map, beta_bin, C_map, freq_mhz)

        return sky_fn, A_map, beta_bin, 'GSM2016'
    except Exception as exc:
        warnings.warn(f'pygdsm unavailable ({exc}); using synthetic power-law sky.')
        beta_val = -2.55
        T_ref    = 3000.0
        nu_ref   = 408.0
        n_pix    = healpy.nside2npix(nside)
        beta_map = np.full(n_pix, beta_val)

        def sky_fn(freq_mhz):
            return make_synthetic_sky(nside, freq_mhz, beta=beta_val,
                                      T_ref=T_ref, nu_ref=nu_ref)

        return sky_fn, None, beta_map, 'Synthetic power-law'


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'Outputs → {OUTDIR}')

    # ── 1. VSH basis ──────────────────────────────────────────────────────────
    print('Computing VSH basis … ', end='', flush=True)
    domega = 4 * np.pi / healpy.nside2npix(NSIDE)
    basis  = vsh_basis(lmax=LMAX, nside=NSIDE)
    upper_mask = basis['upper_mask']

    # Image basis: reflect theta through equatorial plane
    theta_img = np.pi - basis['theta']
    basis_image = vsh_basis(lmax=LMAX, nside=NSIDE,
                            theta_override=theta_img,
                            phi_override=basis['phi'])
    print('done')

    # ── 2. Initial beam (VED-like, not optimized) ──────────────────────────────
    print('Computing initial beam … ', end='', flush=True)
    x0     = initialize_coeffs(n_lm=35, P=2, seed=42)
    Btilde = beam_from_x(x0, H_M, FREQ_MHZ, basis, basis_image, domega)
    print(f'done  shape={Btilde.shape}')

    # ── 3. Sky model ──────────────────────────────────────────────────────────
    print('Loading sky model … ', end='', flush=True)
    sky_fn, A_map, beta_map, sky_label = load_or_build_sky_model(NSIDE)
    print(f'{sky_label}')

    # ── 4. Observation setup ──────────────────────────────────────────────────
    _, utc_times, site = setup_observation()

    # ── 5. Antenna temperature over all 24 LST bins ───────────────────────────
    print('Computing T_ant (24 LST bins × 100 frequencies) …')
    T_ant = np.zeros((len(FREQ_MHZ), 24))

    for i_lst, t_i in enumerate(utc_times):
        print(f'  LST bin {i_lst:2d}/23', end='\r', flush=True)
        T_sky_upper = np.zeros((len(FREQ_MHZ), upper_mask.sum()))

        for i_f, nu in enumerate(FREQ_MHZ):
            sky_full = sky_fn(nu)
            sky_iono = apply_ionospheric_correction(
                sky_full, nu * 1e6, TEC_ZEN_TECU, N_E_MAX, nside=NSIDE)
            T_sky_upper[i_f] = get_sky_at_lst(sky_iono, upper_mask, t_i, site,
                                               nside=NSIDE)

        for i_f in range(len(FREQ_MHZ)):
            T_ant[i_f, i_lst] = (
                (Btilde[i_f] * T_sky_upper[i_f] * domega).sum()
                / (Btilde[i_f] * domega).sum()
            )

    print(f'\nT_ant range: {T_ant.min():.1f} – {T_ant.max():.1f} K')

    # ── 6. Bayesian fit with injected 21 cm signal ────────────────────────────
    print('Running Bayesian fit (emcee) … ', end='', flush=True)
    i_lst_demo = 12   # midday LST bin
    delta_nu_hz = (FREQ_MHZ[1] - FREQ_MHZ[0]) * 1e6
    sigma_n = T_ant[:, i_lst_demo] / np.sqrt(delta_nu_hz * 3600.0)

    rng   = np.random.default_rng(0)
    S_inj = signal_21cm(FREQ_MHZ, A_TRUE, NU0_TRUE, SIG_TRUE)
    T_obs = T_ant[:, i_lst_demo] + S_inj + rng.normal(0.0, sigma_n)

    result  = fit_21cm_bayesian(T_obs, FREQ_MHZ, sigma_n,
                                sampler='emcee',
                                n_walkers=N_WALKERS,
                                n_steps=N_STEPS,
                                n_burn=N_BURN)
    samples = result['samples']

    # Median recovered parameters
    c_med   = np.median(samples[:, :5], axis=0)
    A_rec   = np.median(samples[:, 5])
    nu0_rec = np.median(samples[:, 6])
    sig_rec = np.median(samples[:, 7])
    print(f'done  A={A_rec:.3f} K  ν₀={nu0_rec:.1f} MHz  σ={sig_rec:.1f} MHz')
    print(f'      (true: A={A_TRUE} K  ν₀={NU0_TRUE} MHz  σ={SIG_TRUE} MHz)')

    T_fg_fit  = foreground_model(FREQ_MHZ, c_med)
    S_21_rec  = signal_21cm(FREQ_MHZ, A_rec, nu0_rec, sig_rec)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print('Saving plots …')

    # Beam 3D (HTML)
    fig_beam = plot_beam_3d(Btilde, basis, FREQ_MHZ,
                            output_path=os.path.join(OUTDIR, 'beam_3d.html'))
    print('  beam_3d.html')

    # T_ant waterfall
    fig_tant = plot_tant_waterfall(T_ant, FREQ_MHZ,
                                   output_path=os.path.join(OUTDIR, 'tant_waterfall.png'))
    print('  tant_waterfall.png')

    # Signal recovery
    fig_rec = plot_signal_recovery(
        FREQ_MHZ, T_obs, T_fg_fit, S_inj, S_21_rec,
        output_path=os.path.join(OUTDIR, 'signal_recovery.png'))
    print('  signal_recovery.png')

    # Corner plot
    fig_corner = plot_posterior_corner(
        samples, output_path=os.path.join(OUTDIR, 'posterior_corner.png'))
    print('  posterior_corner.png')

    # Sky model maps (only if GSM2016 was loaded)
    if A_map is not None:
        sky_408 = sky_fn(408.0)
        fig_sky = plot_sky_model(
            sky_408, nside=NSIDE, title='GSM2016 — 408 MHz',
            output_path=os.path.join(OUTDIR, 'sky_model_408.png'))
        print('  sky_model_408.png')

        fig_beta = plot_spectral_index_map(
            beta_map, nside=NSIDE,
            output_path=os.path.join(OUTDIR, 'spectral_index.png'))
        print('  spectral_index.png')
    else:
        print('  (sky model plots skipped — pygdsm not available)')

    print(f'\nAll outputs written to {OUTDIR}/')


if __name__ == '__main__':
    main()
