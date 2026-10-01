"""Visualization functions for the SOFA pipeline.

Provides corner plots, 3D beam visualization, sky maps, T_ant waterfall,
signal recovery diagnostics, and VSH mode polar surfaces.
"""

import os
import pathlib

import numpy as np
import healpy
import corner
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
import plotly.graph_objects as go

# Apply sofa.mplstyle — look next to the package root (one level up from sofa/)
_style_path = pathlib.Path(__file__).parent.parent / 'sofa.mplstyle'
if _style_path.exists():
    plt.style.use(str(_style_path))



def plot_posterior_corner(samples, labels=None, output_path=None):
    """Corner plot of posterior samples from Bayesian fitting.

    Args:
        samples:     (n_samples, 8) array from emcee or dynesty.
        labels:      List of 8 parameter labels. Defaults to standard SOFA labels.
        output_path: If given, save figure to this path (PNG/PDF).

    Returns:
        matplotlib Figure.
    """
    if labels is None:
        labels = [
            r'$c_0$', r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$',
            r'$A$ [K]', r'$\nu_0$ [MHz]', r'$\sigma$ [MHz]',
        ]
    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_kwargs={'fontsize': 12},
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_beam_3d(Btilde, basis, freq_mhz_arr, freq_indices=None, output_path=None):
    """Interactive 3D Plotly visualization of the normalized beam pattern.

    Displays the beam at three frequencies (default: 10, 50, 100 MHz) as
    Scatter3d traces on the upper hemisphere, colored by beam power.

    Args:
        Btilde:       (n_freq, n_pix_upper) normalized beam power array.
        basis:        Dict returned by vsh_basis(); must contain 'theta' and 'phi'.
        freq_mhz_arr: (n_freq,) frequency array [MHz].
        freq_indices: List of integer indices into freq_mhz_arr to plot. If None,
                      the indices closest to [10, 50, 100] MHz are used.
        output_path:  If given, write interactive HTML to this path.

    Returns:
        plotly Figure.
    """
    freq_mhz_arr = np.asarray(freq_mhz_arr)
    theta_u = basis['theta']
    phi_u   = basis['phi']

    x = np.sin(theta_u) * np.cos(phi_u)
    y = np.sin(theta_u) * np.sin(phi_u)
    z = np.cos(theta_u)

    if freq_indices is None:
        target_freqs = [10.0, 50.0, 100.0]
        freq_indices = [int(np.argmin(np.abs(freq_mhz_arr - f))) for f in target_freqs]

    traces = []
    for idx in freq_indices:
        freq_label = f'{freq_mhz_arr[idx]:.1f} MHz'
        b_vals = Btilde[idx]
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=b_vals,
                colorscale='Viridis',
                colorbar=dict(title=r'B̃', thickness=15),
                showscale=(idx == freq_indices[-1]),  # only one colorbar
            ),
            name=freq_label,
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Beam Pattern B̃',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        legend=dict(title='Frequency'),
    )

    if output_path is not None:
        fig.write_html(output_path)
    return fig


def plot_sky_model(sky_map, nside=64, title='GSM2016 Sky Model',
                   log_scale=True, output_path=None):
    """Mollweide projection of a HEALPix sky temperature map.

    Args:
        sky_map:     (n_pix,) HEALPix map [K].
        nside:       HEALPix nside parameter.
        title:       Figure title string.
        log_scale:   Use logarithmic colour stretch (default True). Recommended
                     for sky maps — the Galactic plane dynamic range (~100:1)
                     makes a linear scale collapse everything else to the floor.
        output_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig = plt.figure(figsize=(10, 5))
    norm = 'log' if log_scale else None
    healpy.mollview(sky_map, title=title, fig=fig.number, unit='K', norm=norm)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_spectral_index_map(beta_bin_map, nside=64, output_path=None):
    """Mollweide projection of the decile-binned spectral index map.

    Args:
        beta_bin_map: (n_pix,) HEALPix map of spectral indices.
        nside:        HEALPix nside parameter.
        output_path:  If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig = plt.figure(figsize=(10, 5))
    healpy.mollview(
        beta_bin_map,
        title='Spectral Index (Decile Bins)',
        fig=fig.number,
        unit=r'$\beta$',
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_spectral_index_bins(beta_bin_map, sky_map_408=None, output_path=None):
    """Mollweide map of spectral-index decile regions overlaid on sky structure.

    Each pixel is coloured by its decile bin (10 discrete colours).  If a
    408 MHz sky map is provided, log-brightness contours are drawn in black on
    top so you can see the Galactic morphology underneath the bin regions.

    Args:
        beta_bin_map: (n_pix,) HEALPix map of binned spectral indices (10 unique values).
        sky_map_408:  (n_pix,) HEALPix sky temperature map at 408 MHz [K], or None.
        output_path:  If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    # Integer bin index 0..9 and the corresponding median beta per bin
    unique_betas = np.sort(np.unique(beta_bin_map))
    n_bins = len(unique_betas)
    bin_index = np.searchsorted(unique_betas, beta_bin_map).astype(float)

    # 10-colour discrete palette (cool→warm = steep→flat spectral index)
    palette = plt.colormaps['RdYlBu_r'](np.linspace(0.05, 0.95, n_bins))
    disc_cmap = mcolors.ListedColormap(palette, name='sofa_bins')

    # 2D Mollweide projections via healpy (returns masked arrays)
    proj_bins = healpy.mollview(bin_index, return_projected_map=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor('white')

    # --- Bin region fill ---
    bins_filled = np.ma.filled(proj_bins, np.nan)
    im = ax.imshow(
        bins_filled,
        cmap=disc_cmap,
        vmin=-0.5, vmax=n_bins - 0.5,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
    )

    # --- Galaxy contours from 408 MHz sky brightness ---
    if sky_map_408 is not None:
        log_sky = np.log10(np.clip(sky_map_408, 1.0, None))
        proj_sky = healpy.mollview(log_sky, return_projected_map=True)
        plt.close()
        sky_filled = np.ma.filled(proj_sky, np.nan)
        # 7 contour levels spanning 10th–90th percentile of valid pixels
        valid = sky_filled[np.isfinite(sky_filled)]
        levels = np.linspace(np.percentile(valid, 10),
                             np.percentile(valid, 90), 7)
        ax.contour(sky_filled, levels=levels,
                   colors='k', linewidths=0.6, alpha=0.45)

    ax.set_axis_off()
    ax.set_title('Spectral-index decile regions', fontsize=13, pad=8)

    # --- Discrete colorbar with β labels ---
    bounds = np.arange(n_bins + 1) - 0.5
    cb_norm = mcolors.BoundaryNorm(bounds, disc_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=disc_cmap, norm=cb_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, ticks=np.arange(n_bins),
                      fraction=0.025, pad=0.02,
                      label='Spectral index  β (median per bin)')
    cb.ax.set_yticklabels([f'{b:+.3f}' for b in unique_betas], fontsize=9)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_tant_waterfall(T_ant, freq_mhz, output_path=None):
    """Waterfall (imshow) of antenna temperature vs frequency and LST.

    Args:
        T_ant:       (n_freq, 24) antenna temperature array [K].
        freq_mhz:    (n_freq,) frequency array [MHz].
        output_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    freq_mhz = np.asarray(freq_mhz)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        T_ant,
        aspect='auto',
        origin='lower',
        extent=[0, 24, freq_mhz[0], freq_mhz[-1]],
    )
    ax.set_xlabel('LST bin')
    ax.set_ylabel('Frequency [MHz]')
    ax.set_title('Antenna Temperature')
    plt.colorbar(im, ax=ax, label='$T_{ant}$ [K]')
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_signal_recovery(freq_mhz, T_obs, T_fg_fit, S_21_true,
                         S_21_recovered=None, output_path=None):
    """Two-panel diagnostic plot for 21 cm signal recovery.

    Top panel: observed data, foreground fit, true total model, and
    (optionally) recovered total model.
    Bottom panel: residuals after subtracting foreground + recovered signal
    (or foreground alone if S_21_recovered is None).

    Args:
        freq_mhz:      (n_freq,) frequency array [MHz].
        T_obs:         (n_freq,) observed antenna temperature [K].
        T_fg_fit:      (n_freq,) fitted foreground model [K].
        S_21_true:     (n_freq,) true 21 cm signal [K].
        S_21_recovered:(n_freq,) recovered 21 cm signal [K], or None.
        output_path:   If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    freq_mhz = np.asarray(freq_mhz)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    ax_top, ax_bot = axes

    # --- Top panel ---
    ax_top.plot(freq_mhz, T_obs, color='black', lw=0.8, label=r'$T_{obs}$', alpha=0.7)
    ax_top.plot(freq_mhz, T_fg_fit, color='royalblue', lw=1.5, label=r'$T_{fg}$ (fit)')
    ax_top.plot(freq_mhz, T_fg_fit + S_21_true, color='orange', lw=1.5,
                label=r'$T_{fg}+S_{21}^{true}$', linestyle='--')
    if S_21_recovered is not None:
        ax_top.plot(freq_mhz, T_fg_fit + S_21_recovered, color='crimson', lw=1.5,
                    label=r'$T_{fg}+S_{21}^{rec}$', linestyle=':')
    ax_top.set_ylabel('Temperature [K]')
    ax_top.set_title('21 cm Signal Recovery')
    ax_top.legend(fontsize=10)

    # --- Bottom panel (residuals) ---
    if S_21_recovered is not None:
        residuals = T_obs - T_fg_fit - S_21_recovered
        res_label = r'$T_{obs} - T_{fg} - S_{21}^{rec}$'
    else:
        residuals = T_obs - T_fg_fit
        res_label = r'$T_{obs} - T_{fg}$'

    ax_bot.plot(freq_mhz, residuals, color='gray', lw=0.8)
    ax_bot.axhline(0, color='black', lw=0.8, linestyle='--')
    ax_bot.set_xlabel('Frequency [MHz]')
    ax_bot.set_ylabel(res_label)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_vsh_modes(modes=None, n_theta=120, n_phi=240, output_path=None):
    """3D polar surfaces for a selection of VSH modes (M and N).

    For each mode (l, m) the radius at each direction (θ, φ) is the magnitude
    of the VSH vector:
        r(θ,φ) = |M_lm(θ,φ)| = sqrt(|M_theta|² + |M_phi|²)

    The surface is colored by the same magnitude so structure is visible even
    where the deformation from a sphere is small.  M modes are shown on the
    left column, N modes on the right.

    Args:
        modes:       List of (l, m) tuples to display.  Defaults to a
                     representative selection covering low orders and mixed orders.
        n_theta:     Number of polar-angle grid points (default 120).
        n_phi:       Number of azimuthal grid points (default 240).
        output_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    import scipy.special

    if modes is None:
        modes = [
            (1, 0),   # dipole, axisymmetric
            (1, 1),   # dipole, m=l
            (2, 0),   # quadrupole, axisymmetric
            (2, 1),   # mixed
            (2, 2),   # quadrupole, m=l
            (3, 1),   # mixed, higher order
        ]

    # Fine grid on full sphere
    theta = np.linspace(1e-3, np.pi - 1e-3, n_theta)   # avoid exact poles
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')  # (n_theta, n_phi)

    sin_t  = np.sin(THETA)
    safe_sin = np.where(sin_t > 1e-10, sin_t, 1.0)

    def _vsh_magnitude(l, m):
        """Return |M_lm|(theta,phi) and |N_lm|(theta,phi), each (n_theta, n_phi)."""
        Y    = scipy.special.sph_harm_y(l, m, THETA, PHI)
        # dY/dtheta via L+ recurrence
        term1 = np.where(np.abs(sin_t) > 1e-10,
                         m / np.tan(THETA) * Y, 0.0 + 0.0j)
        if m < l:
            term2 = (np.sqrt((l - m) * (l + m + 1))
                     * np.exp(-1j * PHI)
                     * scipy.special.sph_harm_y(l, m + 1, THETA, PHI))
        else:
            term2 = 0.0
        dYdt = term1 + term2

        norm    = np.sqrt(l * (l + 1))
        M_theta = -1j * m * Y / (safe_sin * norm)
        M_phi   = dYdt / norm
        N_theta = -1j * M_phi
        N_phi   =  1j * M_theta

        mag_M = np.sqrt(np.abs(M_theta)**2 + np.abs(M_phi)**2)
        mag_N = np.sqrt(np.abs(N_theta)**2 + np.abs(N_phi)**2)
        return mag_M.real, mag_N.real

    def _polar_surface(ax, r, title, cmap='viridis'):
        """Plot a 3D polar surface with radius r(theta,phi)."""
        peak = r.max()
        median = np.median(r)
        r_norm = r / peak if (peak > 0 and peak > 1e-6 * max(median, 1e-30)) else np.zeros_like(r)

        x = r_norm * np.sin(THETA) * np.cos(PHI)
        y = r_norm * np.sin(THETA) * np.sin(PHI)
        z = r_norm * np.cos(THETA)

        surf = ax.plot_surface(x, y, z, facecolors=plt.colormaps[cmap](r_norm),
                               rstride=1, cstride=1, linewidth=0,
                               antialiased=False, shade=False)
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_axis_off()
        # Equal aspect ratio
        lim = 0.8
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.view_init(elev=25, azim=45)
        return surf

    n_modes = len(modes)
    fig = plt.figure(figsize=(4 * n_modes, 8))
    fig.suptitle('VSH Modes: radius = |VSH(θ,φ)|', fontsize=13, y=1.01)

    for col, (l, m) in enumerate(modes):
        mag_M, mag_N = _vsh_magnitude(l, m)

        ax_M = fig.add_subplot(2, n_modes, col + 1, projection='3d')
        _polar_surface(ax_M, mag_M,
                       title=rf'$M_{{{l},{m}}}$  (TE)',
                       cmap='Blues')

        ax_N = fig.add_subplot(2, n_modes, n_modes + col + 1, projection='3d')
        _polar_surface(ax_N, mag_N,
                       title=rf'$N_{{{l},{m}}}$  (TM)',
                       cmap='Oranges')

    # Row labels
    fig.text(0.01, 0.75, 'M modes (TE)', va='center', rotation='vertical',
             fontsize=11, fontweight='bold')
    fig.text(0.01, 0.25, 'N modes (TM)', va='center', rotation='vertical',
             fontsize=11, fontweight='bold')

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_vsh_image_theory(l=1, m=0, heights_m=None, freq_mhz=75.0,
                           n_theta=120, n_phi=240, output_path=None):
    """3D polar surfaces showing beam power modified by image theory at varying heights.

    For a single VSH mode (l, m), shows the total power pattern
        B(θ,φ) = |E_tot_theta|² + |E_tot_phi|²
    at several heights h above a PEC ground plane (upper hemisphere only).
    The radius at each (θ,φ) is B/B_max so lobe shape is the focus.
    Top row: M mode; bottom row: N mode.

    Image theory:
        E_tot_theta = E_theta + E_img_theta * exp(2i k h cosθ)   R_theta = +1
        E_tot_phi   = E_phi   - E_img_phi   * exp(2i k h cosθ)   R_phi   = -1
    where E_img is the VSH evaluated at (π−θ, φ).

    Args:
        l, m:         VSH mode (default l=1, m=0 — the N_10 VED mode).
        heights_m:    Heights above PEC [m]. Defaults to [0, λ/8, λ/4, λ/2].
        freq_mhz:     Reference frequency for image-theory phase [MHz].
        n_theta:      Polar grid resolution.
        n_phi:        Azimuthal grid resolution.
        output_path:  If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    import scipy.special

    wavelength_m = 3e8 / (freq_mhz * 1e6)
    if heights_m is None:
        heights_m = [0.0, wavelength_m / 8, wavelength_m / 4, wavelength_m / 2]

    # Upper hemisphere: θ ∈ (0, π/2)
    theta = np.linspace(1e-3, np.pi / 2 - 1e-3, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')   # (n_theta, n_phi)

    sin_t      = np.sin(THETA)
    safe_sin   = np.where(sin_t > 1e-10, sin_t, 1.0)
    cos_t      = np.cos(THETA)
    THETA_img  = np.pi - THETA                             # reflected polar angle
    safe_sin_i = np.where(np.abs(np.sin(THETA_img)) > 1e-10,
                          np.sin(THETA_img), 1.0)
    k = 2 * np.pi * freq_mhz * 1e6 / 3e8

    def _ylm(ll, mm, TH, PH):
        return scipy.special.sph_harm_y(ll, mm, TH, PH)

    def _dYdt(ll, mm, TH, PH):
        Y  = _ylm(ll, mm, TH, PH)
        t1 = np.where(np.abs(np.sin(TH)) > 1e-10,
                      mm / np.tan(TH) * Y, 0.0 + 0.0j)
        if mm < ll:
            t2 = (np.sqrt((ll - mm) * (ll + mm + 1))
                  * np.exp(-1j * PH)
                  * _ylm(ll, mm + 1, TH, PH))
        else:
            t2 = 0.0
        return t1 + t2

    norm = np.sqrt(l * (l + 1))

    # Direct VSH on upper hemisphere
    M_th_d = -1j * m * _ylm(l, m, THETA, PHI)     / (safe_sin * norm)
    M_ph_d =             _dYdt(l, m, THETA, PHI)   / norm
    N_th_d = -1j * M_ph_d
    N_ph_d =  1j * M_th_d

    # Image VSH on reflected (lower) hemisphere pixels
    M_th_i = -1j * m * _ylm(l, m, THETA_img, PHI)   / (safe_sin_i * norm)
    M_ph_i =             _dYdt(l, m, THETA_img, PHI) / norm
    N_th_i = -1j * M_ph_i
    N_ph_i =  1j * M_th_i

    n_h = len(heights_m)
    fig = plt.figure(figsize=(3.5 * n_h, 8))
    fig.suptitle(
        rf'Image theory: $M_{{{l},{m}}}$ (top) & $N_{{{l},{m}}}$ (bottom)'
        f' at {freq_mhz:.0f} MHz',
        fontsize=12, y=1.01)

    def _h_label(h):
        if h == 0:
            return 'h = 0'
        ratio = wavelength_m / h
        if abs(ratio - round(ratio)) < 0.05:
            return rf'h = λ/{round(ratio):.0f}'
        return f'h = {h:.2f} m'

    def _plot_upper(ax, r, title, cmap):
        # Use relative threshold: treat pattern as zero if peak < 1e-6 × median.
        # This avoids amplifying floating-point noise when TE/TM destructively
        # cancels (e.g. M_10 at h=0 where image term exactly cancels direct term).
        peak = r.max()
        median = np.median(r)
        r_norm = r / peak if (peak > 0 and peak > 1e-6 * max(median, 1e-30)) else np.zeros_like(r)
        x = r_norm * np.sin(THETA) * np.cos(PHI)
        y = r_norm * np.sin(THETA) * np.sin(PHI)
        z = r_norm * np.cos(THETA)
        # Ground plane indicator
        phi_disc = np.linspace(0, 2 * np.pi, 60)
        ax.plot(0.9 * np.cos(phi_disc), 0.9 * np.sin(phi_disc),
                np.zeros(60), color='k', lw=0.8, alpha=0.4)
        ax.plot_surface(x, y, z,
                        facecolors=plt.colormaps[cmap](r_norm),
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False)
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_axis_off()
        lim = 0.85
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-0.1, lim)
        ax.view_init(elev=20, azim=45)

    for col, h in enumerate(heights_m):
        phase = np.exp(2j * k * h * cos_t)

        B_M = (np.abs(M_th_d + M_th_i * phase)**2
             + np.abs(M_ph_d - M_ph_i * phase)**2)
        B_N = (np.abs(N_th_d + N_th_i * phase)**2
             + np.abs(N_ph_d - N_ph_i * phase)**2)

        ax_M = fig.add_subplot(2, n_h, col + 1, projection='3d')
        _plot_upper(ax_M, B_M,
                    title=rf'$M_{{{l},{m}}}$  {_h_label(h)}',
                    cmap='Blues')

        ax_N = fig.add_subplot(2, n_h, n_h + col + 1, projection='3d')
        _plot_upper(ax_N, B_N,
                    title=rf'$N_{{{l},{m}}}$  {_h_label(h)}',
                    cmap='Oranges')

    fig.text(0.01, 0.75, 'M mode (TE)', va='center', rotation='vertical',
             fontsize=10, fontweight='bold')
    fig.text(0.01, 0.25, 'N mode (TM)', va='center', rotation='vertical',
             fontsize=10, fontweight='bold')

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
