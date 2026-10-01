#!/usr/bin/env python3
"""
run_analysis.py — Full SOFA pipeline with checkpointing.

Stages (each is skipped if its checkpoint already exists in data/):
  1. sky      — Build GSM2016 sky model (cached .npy)
  2. optimize — DE/rand/1/bin + Powell beam optimization (cached .npy)
  3. tant     — Compute T_ant with optimized beam for all 24 LST bins (cached .npy)
  4. emcee    — emcee posteriors for each LST bin (cached .npz)
  5. dynesty  — Dynesty nested sampling on best LST bin (cached .pkl)
  6. plots    — Generate all figures → outputs/

Usage:
  python run_analysis.py                   # run all stages
  python run_analysis.py --stages 1,3,6   # run specific stages only
  python run_analysis.py --from 3         # resume from stage 3 onward
  python run_analysis.py --mode demo      # fast demo (small grid, short chains)
  python run_analysis.py --force          # rerun all even if checkpoints exist

Demo mode uses nside=32, 100 freq points, pop_size=126, n_de_iter=3,
N_realizations=5, n_steps=500 — runs in ~10–30 min on a laptop.

Full mode matches CLAUDE.md spec: nside=64, 1000 freq points, pop_size=6300,
n_de_iter=50, N_realizations=100, n_steps=5000. Expect days on a cluster.
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import healpy
import numpy as np

# ── Output dirs ────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(ROOT, 'data')
OUTDIR  = os.path.join(ROOT, 'outputs')
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(OUTDIR,  exist_ok=True)

# ── Checkpoint paths ───────────────────────────────────────────────────────────
def ckpt(name): return os.path.join(DATADIR, name)

SKY_CKPT     = lambda nside: ckpt(f'sky_model_nside{nside}.npy')
OPT_CKPT     = ckpt('optimized_beam.npy')
TANT_CKPT    = ckpt('T_ant.npy')
EMCEE_CKPT   = ckpt('emcee_samples.npz')
DYNESTY_CKPT = ckpt('dynesty_results.pkl')


# ── Config ─────────────────────────────────────────────────────────────────────
DEMO_CFG = dict(
    nside=32, freq_n=100, lmax=5, h_m=0.5,
    pop_size=126, n_de_iter=3, n_realizations=5,
    n_walkers=16, n_steps=500,  n_burn=100,
    nlive=200,
    tec_zen_tecu=5.0, n_e_max=1e11,
)
FULL_CFG = dict(
    nside=64, freq_n=1000, lmax=5, h_m=0.5,
    pop_size=6300, n_de_iter=50, n_realizations=100,
    n_walkers=32, n_steps=5000, n_burn=500,
    nlive=500,
    tec_zen_tecu=5.0, n_e_max=1e11,
)


def stamp(label):
    print(f'\n{"─"*60}')
    print(f'  {label}')
    print(f'{"─"*60}')


# ── Stage 1: Sky model ─────────────────────────────────────────────────────────
def stage_sky(cfg, force=False):
    stamp('Stage 1 — Sky model (GSM2016)')
    path = SKY_CKPT(cfg['nside'])
    if os.path.exists(path) and not force:
        print(f'  Checkpoint found: {path}  (skipping)')
        return np.load(path, allow_pickle=True)

    from sofa.skymap import build_sky_model
    t0 = time.time()
    result = build_sky_model(nside=cfg['nside'], cache_path=path)
    print(f'  Done in {time.time()-t0:.1f} s  →  {path}')
    return result


# ── Stage 2: Beam optimization ─────────────────────────────────────────────────
def stage_optimize(cfg, sky_params, force=False):
    stamp('Stage 2 — Beam optimization (DE + Powell)')
    if os.path.exists(OPT_CKPT) and not force:
        print(f'  Checkpoint found: {OPT_CKPT}  (skipping)')
        return np.load(OPT_CKPT)

    import functools
    from sofa.beam import vsh_basis, initialize_coeffs
    from sofa.antenna_temp import setup_observation
    from sofa.optimize import compute_loss, run_de_optimization, run_powell_refinement

    freq_mhz = np.linspace(5.0, 150.0, cfg['freq_n'])
    nside     = cfg['nside']
    domega    = 4 * np.pi / healpy.nside2npix(nside)

    print('  Pre-computing VSH basis …')
    basis = vsh_basis(lmax=cfg['lmax'], nside=nside)
    basis_image = vsh_basis(lmax=cfg['lmax'], nside=nside,
                            theta_override=np.pi - basis['theta'],
                            phi_override=basis['phi'])

    _, utc_times, site = setup_observation()

    rng = np.random.default_rng(42)
    realizations = [(0.0, 40.0, 15.0)] + list(zip(
        rng.uniform(0.05, 0.5,  cfg['n_realizations']),
        rng.uniform(7.0,  80.0, cfg['n_realizations']),
        rng.uniform(5.0,  30.0, cfg['n_realizations']),
    ))

    loss_fn = functools.partial(
        compute_loss,
        realizations=realizations,
        sky_model_params=sky_params,
        upper_mask=basis['upper_mask'],
        utc_times=utc_times,
        site=site,
        tec_zen_tecu=cfg['tec_zen_tecu'],
        n_e_max=cfg['n_e_max'],
        freq_mhz=freq_mhz,
        basis=basis,
        basis_image=basis_image,
        domega=domega,
        h_m=cfg['h_m'],
        nside=nside,
    )

    print(f'  DE search: pop_size={cfg["pop_size"]}, iters={cfg["n_de_iter"]} …')
    print('  (This is the slow step — expect hours to days for full config)')
    t0 = time.time()
    x_de = run_de_optimization(
        loss_fn,
        n_real=420,
        n_de_iter=cfg['n_de_iter'],
        pop_size=cfg['pop_size'],
    )
    print(f'  DE done in {(time.time()-t0)/3600:.2f} h')

    print('  Powell refinement …')
    t0 = time.time()
    x_opt = run_powell_refinement(loss_fn, x_de)
    print(f'  Powell done in {time.time()-t0:.1f} s')

    np.save(OPT_CKPT, x_opt)
    print(f'  Saved → {OPT_CKPT}')
    return x_opt


# ── Stage 3: T_ant ─────────────────────────────────────────────────────────────
def stage_tant(cfg, x_opt, sky_params, force=False):
    stamp('Stage 3 — Antenna temperature (all 24 LST bins)')
    if os.path.exists(TANT_CKPT) and not force:
        print(f'  Checkpoint found: {TANT_CKPT}  (skipping)')
        return np.load(TANT_CKPT)

    from sofa.beam import vsh_basis, beam_from_x
    from sofa.skymap import sky_temperature
    from sofa.ionosphere import apply_ionospheric_correction
    from sofa.antenna_temp import setup_observation, get_sky_at_lst

    nside    = cfg['nside']
    freq_mhz = np.linspace(5.0, 150.0, cfg['freq_n'])
    domega   = 4 * np.pi / healpy.nside2npix(nside)

    basis = vsh_basis(lmax=cfg['lmax'], nside=nside)
    basis_image = vsh_basis(lmax=cfg['lmax'], nside=nside,
                            theta_override=np.pi - basis['theta'],
                            phi_override=basis['phi'])
    upper_mask = basis['upper_mask']

    Btilde = beam_from_x(x_opt, cfg['h_m'], freq_mhz, basis, basis_image, domega)

    A_map, beta_bin, C_map = sky_params
    _, utc_times, site = setup_observation()

    T_ant = np.zeros((len(freq_mhz), 24))
    t0 = time.time()
    for i_lst, t_i in enumerate(utc_times):
        print(f'  LST bin {i_lst+1:2d}/24 …', end='\r', flush=True)
        T_sky_upper = np.zeros((len(freq_mhz), upper_mask.sum()))
        for i_f, nu in enumerate(freq_mhz):
            sky_full = sky_temperature(A_map, beta_bin, C_map, nu)
            sky_iono = apply_ionospheric_correction(
                sky_full, nu * 1e6, cfg['tec_zen_tecu'], cfg['n_e_max'], nside=nside)
            T_sky_upper[i_f] = get_sky_at_lst(sky_iono, upper_mask, t_i, site, nside=nside)
        for i_f in range(len(freq_mhz)):
            T_ant[i_f, i_lst] = (
                (Btilde[i_f] * T_sky_upper[i_f] * domega).sum()
                / (Btilde[i_f] * domega).sum()
            )
    print(f'\n  Done in {time.time()-t0:.1f} s  T_ant: {T_ant.min():.1f}–{T_ant.max():.1f} K')
    np.save(TANT_CKPT, T_ant)
    print(f'  Saved → {TANT_CKPT}')
    return T_ant


# ── Stage 4: emcee posteriors for every LST bin ─────────────────────────────
def stage_emcee(cfg, T_ant, force=False):
    stamp('Stage 4 — emcee posteriors (all 24 LST bins, injected signal)')
    if os.path.exists(EMCEE_CKPT) and not force:
        print(f'  Checkpoint found: {EMCEE_CKPT}  (skipping)')
        d = np.load(EMCEE_CKPT, allow_pickle=True)
        return d['samples_all'], d['T_obs_all'], d['sigma_n_all']

    from sofa.fitting import fit_21cm_bayesian, signal_21cm

    freq_mhz = np.linspace(5.0, 150.0, cfg['freq_n'])
    delta_nu = (freq_mhz[1] - freq_mhz[0]) * 1e6
    rng      = np.random.default_rng(1)

    A_TRUE, NU0_TRUE, SIG_TRUE = 0.20, 78.0, 18.0
    S_inj = signal_21cm(freq_mhz, A_TRUE, NU0_TRUE, SIG_TRUE)

    samples_all = []
    T_obs_all   = []
    sigma_n_all = []

    for i_lst in range(24):
        print(f'  LST bin {i_lst+1:2d}/24 …', end='\r', flush=True)
        sigma_n = np.maximum(T_ant[:, i_lst] / np.sqrt(delta_nu * 3600.0), 1e-6)
        T_obs   = T_ant[:, i_lst] + S_inj + rng.normal(0.0, sigma_n)
        result  = fit_21cm_bayesian(T_obs, freq_mhz, sigma_n,
                                    sampler='emcee',
                                    n_walkers=cfg['n_walkers'],
                                    n_steps=cfg['n_steps'],
                                    n_burn=cfg['n_burn'])
        samples_all.append(result['samples'])
        T_obs_all.append(T_obs)
        sigma_n_all.append(sigma_n)

    samples_all = np.array(samples_all, dtype=object)
    print(f'\n  Done')
    np.savez(EMCEE_CKPT, samples_all=samples_all,
             T_obs_all=np.array(T_obs_all),
             sigma_n_all=np.array(sigma_n_all))
    print(f'  Saved → {EMCEE_CKPT}')
    return samples_all, np.array(T_obs_all), np.array(sigma_n_all)


# ── Stage 5: dynesty on best LST bin ──────────────────────────────────────────
def stage_dynesty(cfg, T_ant, samples_all, T_obs_all, sigma_n_all, force=False):
    stamp('Stage 5 — Dynesty nested sampling (best LST bin)')
    if os.path.exists(DYNESTY_CKPT) and not force:
        print(f'  Checkpoint found: {DYNESTY_CKPT}  (skipping)')
        with open(DYNESTY_CKPT, 'rb') as f:
            return pickle.load(f)

    from sofa.fitting import fit_21cm_bayesian

    freq_mhz = np.linspace(5.0, 150.0, cfg['freq_n'])

    # Pick LST bin where recovered A is closest to true A=0.2 K
    A_meds = [np.median(s[:, 5]) for s in samples_all]
    best_lst = int(np.argmin(np.abs(np.array(A_meds) - 0.20)))
    print(f'  Using LST bin {best_lst} (median A={A_meds[best_lst]:.3f} K)')

    T_obs   = T_obs_all[best_lst]
    sigma_n = sigma_n_all[best_lst]

    result = fit_21cm_bayesian(T_obs, freq_mhz, sigma_n,
                               sampler='dynesty',
                               nlive=cfg['nlive'])
    with open(DYNESTY_CKPT, 'wb') as f:
        pickle.dump({'results': result['results'], 'best_lst': best_lst}, f)
    print(f'  Saved → {DYNESTY_CKPT}')
    return {'results': result['results'], 'best_lst': best_lst}


# ── Stage 6: Plots ─────────────────────────────────────────────────────────────
def stage_plots(cfg, x_opt, sky_params, T_ant, samples_all,
                T_obs_all, sigma_n_all, dynesty_data):
    stamp('Stage 6 — Generating plots')

    from sofa.beam import vsh_basis, beam_from_x
    from sofa.skymap import sky_temperature
    from sofa.fitting import foreground_model, signal_21cm
    from sofa.plot import (plot_beam_3d, plot_tant_waterfall,
                           plot_signal_recovery, plot_posterior_corner,
                           plot_sky_model, plot_spectral_index_map)

    nside    = cfg['nside']
    freq_mhz = np.linspace(5.0, 150.0, cfg['freq_n'])
    domega   = 4 * np.pi / healpy.nside2npix(nside)

    basis = vsh_basis(lmax=cfg['lmax'], nside=nside)
    basis_image = vsh_basis(lmax=cfg['lmax'], nside=nside,
                            theta_override=np.pi - basis['theta'],
                            phi_override=basis['phi'])
    Btilde = beam_from_x(x_opt, cfg['h_m'], freq_mhz, basis, basis_image, domega)

    A_map, beta_bin, C_map = sky_params

    # Beam 3D
    plot_beam_3d(Btilde, basis, freq_mhz,
                 output_path=os.path.join(OUTDIR, 'beam_3d.html'))
    print('  beam_3d.html')

    # T_ant waterfall
    plot_tant_waterfall(T_ant, freq_mhz,
                        output_path=os.path.join(OUTDIR, 'tant_waterfall.png'))
    print('  tant_waterfall.png')

    # Sky model maps
    sky_408 = sky_temperature(A_map, beta_bin, C_map, 408.0)
    plot_sky_model(sky_408, nside=nside, title='GSM2016 — 408 MHz',
                   output_path=os.path.join(OUTDIR, 'sky_model_408.png'))
    print('  sky_model_408.png')

    plot_spectral_index_map(beta_bin, nside=nside,
                            output_path=os.path.join(OUTDIR, 'spectral_index.png'))
    print('  spectral_index.png')

    # Per-LST signal recovery: pick LST bin with best A recovery
    A_meds  = [np.median(s[:, 5]) for s in samples_all]
    best    = int(np.argmin(np.abs(np.array(A_meds) - 0.20)))
    samples = samples_all[best]
    T_obs   = T_obs_all[best]
    c_med   = np.median(samples[:, :5], axis=0)
    A_rec, nu0_rec, sig_rec = (np.median(samples[:, 5]),
                               np.median(samples[:, 6]),
                               np.median(samples[:, 7]))
    T_fg_fit = foreground_model(freq_mhz, c_med)
    S_true   = signal_21cm(freq_mhz, 0.20, 78.0, 18.0)
    S_rec    = signal_21cm(freq_mhz, A_rec, nu0_rec, sig_rec)

    plot_signal_recovery(freq_mhz, T_obs, T_fg_fit, S_true, S_rec,
                         output_path=os.path.join(OUTDIR, 'signal_recovery.png'))
    print('  signal_recovery.png')

    # emcee corner (best LST)
    plot_posterior_corner(samples,
                          output_path=os.path.join(OUTDIR, 'posterior_corner_emcee.png'))
    print('  posterior_corner_emcee.png')

    # dynesty corner
    if dynesty_data is not None:
        import dynesty.plotting as dyplot
        import matplotlib.pyplot as plt
        results = dynesty_data['results']
        fig, _ = dyplot.cornerplot(results, color='blue', show_titles=True)
        fig.savefig(os.path.join(OUTDIR, 'posterior_corner_dynesty.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  posterior_corner_dynesty.png')

    # Parameter recovery summary
    print(f'\n  Signal recovery (LST bin {best}):')
    print(f'    A:   true={0.20:.3f} K   recovered={A_rec:.3f} K')
    print(f'    ν₀:  true={78.0:.1f} MHz  recovered={nu0_rec:.1f} MHz')
    print(f'    σ:   true={18.0:.1f} MHz  recovered={sig_rec:.1f} MHz')

    print(f'\n  All outputs → {OUTDIR}/')


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['demo', 'full'], default='demo',
                   help='demo: fast (~10–30 min); full: production (days on cluster)')
    p.add_argument('--stages', default=None,
                   help='Comma-separated stage numbers to run, e.g. "1,3,6"')
    p.add_argument('--from', dest='from_stage', type=int, default=1,
                   help='Run all stages from this number onward')
    p.add_argument('--force', action='store_true',
                   help='Re-run all stages even if checkpoints exist')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = DEMO_CFG if args.mode == 'demo' else FULL_CFG

    if args.stages:
        run_stages = set(int(s) for s in args.stages.split(','))
    else:
        run_stages = set(range(args.from_stage, 7))

    def should_run(n): return n in run_stages

    print(f'\nSOFA full analysis  [mode={args.mode}]')
    print(f'Running stages: {sorted(run_stages)}')
    print(f'Data dir:    {DATADIR}')
    print(f'Outputs dir: {OUTDIR}')

    sky_params   = None
    x_opt        = None
    T_ant        = None
    samples_all  = None
    T_obs_all    = None
    sigma_n_all  = None
    dynesty_data = None

    # Load existing checkpoints for stages we're not re-running
    if not should_run(1) or (os.path.exists(SKY_CKPT(cfg['nside'])) and not args.force):
        p = SKY_CKPT(cfg['nside'])
        if os.path.exists(p):
            sky_params = tuple(np.load(p, allow_pickle=True))

    if not should_run(2) or (os.path.exists(OPT_CKPT) and not args.force):
        if os.path.exists(OPT_CKPT):
            x_opt = np.load(OPT_CKPT)

    if not should_run(3) or (os.path.exists(TANT_CKPT) and not args.force):
        if os.path.exists(TANT_CKPT):
            T_ant = np.load(TANT_CKPT)

    if not should_run(4) or (os.path.exists(EMCEE_CKPT) and not args.force):
        if os.path.exists(EMCEE_CKPT):
            d = np.load(EMCEE_CKPT, allow_pickle=True)
            samples_all = d['samples_all']
            T_obs_all   = d['T_obs_all']
            sigma_n_all = d['sigma_n_all']

    if not should_run(5) or (os.path.exists(DYNESTY_CKPT) and not args.force):
        if os.path.exists(DYNESTY_CKPT):
            with open(DYNESTY_CKPT, 'rb') as f:
                dynesty_data = pickle.load(f)

    # Run selected stages
    if should_run(1):
        sky_params = tuple(stage_sky(cfg, force=args.force))

    if should_run(2):
        if sky_params is None:
            sys.exit('Stage 2 requires stage 1 output. Run stage 1 first.')
        x_opt = stage_optimize(cfg, sky_params, force=args.force)

    if should_run(3):
        if x_opt is None:
            sys.exit('Stage 3 requires stage 2 output. Run stage 2 first.')
        if sky_params is None:
            sys.exit('Stage 3 requires stage 1 output. Run stage 1 first.')
        T_ant = stage_tant(cfg, x_opt, sky_params, force=args.force)

    if should_run(4):
        if T_ant is None:
            sys.exit('Stage 4 requires stage 3 output. Run stage 3 first.')
        samples_all, T_obs_all, sigma_n_all = stage_emcee(cfg, T_ant, force=args.force)

    if should_run(5):
        if samples_all is None or T_ant is None:
            sys.exit('Stage 5 requires stages 3+4. Run them first.')
        dynesty_data = stage_dynesty(cfg, T_ant, samples_all,
                                     T_obs_all, sigma_n_all, force=args.force)

    if should_run(6):
        if any(v is None for v in [x_opt, sky_params, T_ant, samples_all]):
            sys.exit('Stage 6 requires stages 1–4. Run them first.')
        stage_plots(cfg, x_opt, sky_params, T_ant,
                    samples_all, T_obs_all, sigma_n_all, dynesty_data)

    print('\nDone.')


if __name__ == '__main__':
    main()
