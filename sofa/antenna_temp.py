"""
antenna_temp.py — LST bins, sky rotation ICRS→AltAz, beam-sky integral

Computes antenna temperature T_ant(freq, LST) by integrating the normalized
beam pattern against the rotated sky temperature map for each LST bin.
"""

import numpy as np
import healpy

from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u


def setup_observation():
    """Set up observation parameters for MRO site over 24 LST bins.

    Returns
    -------
    t0 : astropy.time.Time
        Start epoch: 2026-10-21 00:00:00 UTC
    utc_times : list of Time
        24 hourly UTC times spanning one sidereal day
    site : EarthLocation
        MRO site (lat=-26.70 deg, lon=+116.60 deg, alt=377 m)
    """
    t0 = Time('2026-10-21 00:00:00', scale='utc')
    utc_times = [t0 + i * u.hr for i in range(24)]
    site = EarthLocation(lat=-26.70 * u.deg, lon=116.60 * u.deg, height=377 * u.m)
    return t0, utc_times, site


def get_sky_at_lst(sky_map, upper_mask, t_i, site, nside=64):
    """Rotate sky map to local AltAz frame and return temperatures for upper hemisphere.

    Maps upper-hemisphere HEALPix beam pixels to sky coordinates for the given
    observation time, then interpolates the sky brightness temperature map.

    Coordinate convention: beam frame uses AltAz with phi=0 pointing East (HEALPix)
    converted to az=0 pointing North (astropy AltAz), i.e.:
        az_rad = (pi/2 - phi_hp) % (2*pi)
    Cross-check: phi_hp=0 (East) → az=pi/2 (East) ✓
                 phi_hp=pi/2 (North) → az=0 (North) ✓

    Parameters
    ----------
    sky_map : array (n_pix_full,)
        Full-sky temperature map [K] in HEALPix RING ordering.
    upper_mask : array (n_pix_full,) bool
        True for upper-hemisphere pixels (theta < pi/2).
    t_i : astropy.time.Time
        UTC time for this LST bin.
    site : EarthLocation
        Observer location.
    nside : int
        HEALPix nside (default 64).

    Returns
    -------
    T_upper : array (n_pix_upper,)
        Sky brightness temperature [K] for each upper-hemisphere pixel,
        rotated to match local AltAz at time t_i.
    """
    ipix_upper = np.where(upper_mask)[0]
    theta_hp, phi_hp = healpy.pix2ang(nside, ipix_upper)

    # Convert HEALPix angular coords to AltAz
    alt_rad = np.pi / 2 - theta_hp
    az_rad = (np.pi / 2 - phi_hp) % (2 * np.pi)  # CCW (HEALPix) → CW from North (astropy)

    # Transform AltAz → ICRS
    altaz_frame = AltAz(obstime=t_i, location=site)
    coords = SkyCoord(
        alt=alt_rad * u.rad,
        az=az_rad * u.rad,
        frame=altaz_frame
    ).icrs

    # ICRS → HEALPix colatitude/longitude for sky map interpolation
    theta_sky = np.pi / 2 - coords.dec.rad   # colatitude in [0, pi]
    phi_sky = coords.ra.rad                   # RA as longitude in [0, 2pi]

    return healpy.get_interp_val(sky_map, theta_sky, phi_sky)


def compute_antenna_temperature(Btilde, T_sky_lst, domega):
    """Compute beam-weighted antenna temperature for all frequencies and LST bins.

    Each LST bin is computed independently. The denominator equals 2pi when
    Btilde is normalized so that (Btilde * domega).sum(axis=-1) = 2pi.

    Parameters
    ----------
    Btilde : array (n_freq, n_pix_upper)
        Normalized beam power pattern. Should satisfy
        (Btilde * domega).sum(axis=-1) = 2pi for all frequencies.
    T_sky_lst : list of 24 arrays, each (n_pix_upper,)
        Sky temperature [K] for upper-hemisphere pixels at each LST bin.
    domega : float
        Solid angle per HEALPix pixel [sr] = 4*pi / nside2npix(nside).

    Returns
    -------
    T_ant : array (n_freq, 24)
        Antenna temperature [K] for each frequency and LST bin.
    """
    n_freq = Btilde.shape[0]
    n_lst = len(T_sky_lst)
    T_ant = np.empty((n_freq, n_lst), dtype=float)

    for i_lst in range(n_lst):
        T_sky = T_sky_lst[i_lst]   # (n_pix_upper,)
        # Beam-weighted integral: T_ant = integral(B * T * dOmega) / integral(B * dOmega)
        numerator = (Btilde * T_sky[np.newaxis, :] * domega).sum(axis=-1)   # (n_freq,)
        denominator = (Btilde * domega).sum(axis=-1)                          # (n_freq,)
        T_ant[:, i_lst] = numerator / denominator

    return T_ant
