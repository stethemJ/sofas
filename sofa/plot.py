"""Visualization functions for the SOFA pipeline.

Provides corner plots, 3D beam visualization, sky maps, T_ant waterfall,
and signal recovery diagnostics.
"""

import numpy as np
import healpy
import corner
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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


def plot_sky_model(sky_map, nside=64, title='GSM2016 Sky Model', output_path=None):
    """Mollweide projection of a HEALPix sky temperature map.

    Args:
        sky_map:     (n_pix,) HEALPix map [K].
        nside:       HEALPix nside parameter.
        title:       Figure title string.
        output_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig = plt.figure(figsize=(10, 5))
    healpy.mollview(sky_map, title=title, fig=fig.number, unit='K')
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
