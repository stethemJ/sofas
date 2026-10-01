# SOFA: Project Pipeline for Claude Code Implementation

**Project:** vector Spherical-harmonic beam Optimization for Forward modeled Antennas

This document is the implementation guide for the SOFA codebase. It describes every
component's scientific basis, Python API, data flow, and validation criteria with enough
precision that each module can be implemented without further clarification.

---

## Scientific Context

- **Target signal:** Global 21 cm absorption dip. Dark Ages (z~30–200, ν~7–47 MHz); Cosmic
  Dawn (z~8–30, ν~47–150 MHz).
- **Signal model:** `S_21 = −A × exp[−(ν−ν_0)²/(2σ²)]` [K]; A~0.05–0.5 K, ν_0~7–80 MHz,
  σ~5–30 MHz.
- **Primary systematic:** Chromatic beam × anisotropic foreground → spurious spectral
  features at 21 cm amplitude.
- **Frequency grid:** `freq_mhz = np.linspace(5.0, 150.0, 1000)`, Δν ≈ 0.145 MHz.
- **Reference frequency:** `ν_ref = 75 MHz` — used consistently for beam polynomial and
  foreground model throughout.
- **System temperature:** T_sys ≈ T_ant(ν) (sky-dominated; ~1000 K at 75 MHz, ∝ (ν/75)^{-2.5}).
- **Observation epoch:** 24 LST bins, 2026-10-21 00:00 UTC to 2026-10-22 00:00 UTC.
- **Site:** MRO, lat = −26.70°, lon = +116.60°, alt = 377 m.

---

## Repository Structure

```
sofa/
├── sofa/
│   ├── __init__.py
│   ├── beam.py          # VSH expansion, image theory, ground-plane model
│   ├── skymap.py        # GSM2016 loading, spectral index binning, sky model
│   ├── ionosphere.py    # Thin-screen refraction, plasma reflection, TEC
│   ├── antenna_temp.py  # LST bins, sky rotation ICRS→AltAz, beam-sky integral
│   ├── fitting.py       # Log-polynomial foreground + Gaussian 21cm Bayesian
│   ├── optimize.py      # Outer loop: loss function, global + local optimization
│   └── plot.py          # Corner plots, 3D beam, sky maps, T_ant waterfall
├── tests/
├── notebooks/
├── data/                # Cached GSM params (.npy); ionosphere profiles
└── pyproject.toml
```

**Key Libraries:** `pygdsm`, `healpy`, `astropy`, `numpy`, `scipy`, `joblib`, `emcee`,
`dynesty`, `corner`, `matplotlib`, `plotly`, `pytest`

---

## Component 1: VSH Beam (`beam.py`)

### VSH Definitions (Hansen/M-N convention; Jackson 3rd ed. Ch. 9)

| Symbol  | Definition                          | Type                   |
|---------|-------------------------------------|------------------------|
| `M_lm`  | `[1/√(l(l+1))] × r̂ × ∇Y_lm`        | TE (magnetic multipole) |
| `N_lm`  | `(1/k) ∇ × M_lm`                   | TM (electric multipole) |

Far-field limit: `N_lm = i × (r̂ × M_lm)`.

**Phase convention:** `scipy.special.sph_harm` uses Condon–Shortley (−1)^m for m > 0.
Jackson uses the same convention. No extra factor needed.

### scipy Argument Order — CRITICAL

`scipy.special.sph_harm` was **removed in scipy ≥ 1.15**. Use `sph_harm_y`:

```python
# Correct (scipy >= 1.15): standard physics convention — theta=polar, phi=azimuthal
Y_lm = scipy.special.sph_harm_y(l, m, theta_rad, phi_rad)  # n=l first, theta before phi

# OLD (scipy < 1.15, now removed):
# scipy.special.sph_harm(m, l, phi_rad, theta_rad)  # m first, phi first
```

### ∂Y_lm/∂θ Recurrence (from L+ operator; required for M_lm computation)

```python
def dYlm_dtheta(m, l, phi, theta):
    """∂Y_lm/∂θ = m cotθ Y_lm + e^{-iφ} √[(l-m)(l+m+1)] Y_{l,m+1}
    Derived from L+: e^{iφ}(∂/∂θ + i cotθ ∂/∂φ)Y_lm = e^{-iφ}√[(l-m)(l+m+1)] Y_{l,m+1}
    """
    Y_ml = scipy.special.sph_harm_y(l, m, theta, phi)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(np.abs(np.sin(theta)) > 1e-10,
                         m / np.tan(theta) * Y_ml, 0.0)
    if m < l:   # NOTE: 'm < l' not 'abs(m) < l' — correctly includes m=-l (e.g. -1 < 1)
        term2 = (np.sqrt((l - m) * (l + m + 1))
                 * np.exp(-1j * phi)
                 * scipy.special.sph_harm_y(l, m + 1, theta, phi))
    else:
        term2 = 0.0
    return term1 + term2   # NOTE: +, not − (sign from L+ derivation)

# Validation:
# dYlm_dtheta(0, 1, phi, pi/2) = e^{-iφ}×√2×Y_{1,1}(pi/2)
#   Y_{1,1}(pi/2) = -√(3/8π)e^{iφ} → result = −√(3/4π) ≈ −0.4886 ✓
# dYlm_dtheta(-1, 1, phi, theta) is nonzero (m<l=-1<1 True) ✓
```

### Anchor Table (l=1, m=0)

| Mode   | Far-field E-field   | Power pattern |
|--------|---------------------|---------------|
| `M_10` | ∝ **−φ̂ sinθ**      | ∝ sin²θ       |
| `N_10` | ∝ **+i sinθ θ̂**   | ∝ sin²θ       |

Derivation: Y_10 = √(3/4π)cosθ → ∂Y_10/∂θ = −√(3/4π)sinθ → M_10 ∝ −φ̂ sinθ →
N_10 = i(r̂ × M_10) = +i sinθ θ̂. Only |E|² = B enters the power pattern and T_ant
computation; the imaginary phase in N_10 has no effect on observables.

### VSH Basis on HEALPix Upper Hemisphere

```python
def vsh_basis(lmax=5, nside=64):
    """Pre-compute M_lm, N_lm on upper-hemisphere pixels once; cache for reuse.

    Returns dict with keys 'M_lm', 'N_lm' each of shape (n_lm, n_pix_upper, 2),
    where axis-2 = (θ̂-component, φ̂-component). Also 'theta', 'phi', 'upper_mask'.
    """
    n_pix = healpy.nside2npix(nside)
    theta_all, phi_all = healpy.pix2ang(nside, np.arange(n_pix))
    upper = theta_all < np.pi / 2
    theta_u, phi_u = theta_all[upper], phi_all[upper]
    # Safe sine for pole pixels (Y_lm → 0 faster than sinθ for |m|≥1; m=0 zeros numerator)
    safe_sin = np.where(np.sin(theta_u) > 1e-10, np.sin(theta_u), 1.0)

    M_lm_arr, N_lm_arr = [], []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Y    = scipy.special.sph_harm_y(l, m, theta_u, phi_u)  # (n_pix_upper,)
            dYdt = dYlm_dtheta(m, l, phi_u, theta_u)               # (n_pix_upper,)
            norm = np.sqrt(l * (l + 1))

            # M_lm = [1/norm] r̂×∇Y_lm; ∇Y_lm = θ̂ dY/dθ + φ̂ (im/sinθ) Y
            # r̂×θ̂ = φ̂, r̂×φ̂ = -θ̂
            M_theta = -1j * m * Y / (safe_sin * norm)   # (n_pix_upper,)
            M_phi   = dYdt / norm                         # (n_pix_upper,)

            # N_lm = i(r̂ × M_lm)
            N_theta = -1j * M_phi          # = -i dYdt / norm
            N_phi   =  1j * M_theta        # = m Y / (safe_sin * norm)

            M_lm_arr.append(np.stack([M_theta, M_phi], axis=-1))
            N_lm_arr.append(np.stack([N_theta, N_phi], axis=-1))

    return {
        'M_lm':       np.array(M_lm_arr),    # (n_lm, n_pix_upper, 2)
        'N_lm':       np.array(N_lm_arr),    # (n_lm, n_pix_upper, 2)
        'theta':      theta_u,               # (n_pix_upper,)
        'phi':        phi_u,                 # (n_pix_upper,)
        'upper_mask': upper,                 # (n_pix_full,) bool
    }
# VSH orthonormality check (upper hemisphere):
# ∫_upper |M_10|² dΩ = 0.5  (half of full-sphere = 1, by symmetry)
# Analytically: M_10_phi = -√(3/4π)sinθ/√2 → (3/4) × ∫₀^{π/2} sin³θ dθ = 0.5
# M10 = basis['M_lm'][1]   # (l=1,m=0) at index 1
# ortho = (np.abs(M10[:,0])**2 + np.abs(M10[:,1])**2).sum() * domega
# assert abs(ortho - 0.5) < 0.01   # ← 0.5, not 1.0
```

### Coefficient Indexing and Frequency Parameterization

```
Order: (l=1,m=-1), (l=1,m=0), (l=1,m=+1), (l=2,m=-2), ..., (L_max, +L_max)
n_lm = L_max² + 2·L_max   [L_max=5 → n_lm=35]

a_lm(ν) = Σ_{p=0}^{2} α_{lm,p} × (ν/75)^p,   α ∈ ℂ
b_lm(ν) = Σ_{p=0}^{2} β_{lm,p} × (ν/75)^p,   β ∈ ℂ
Total real parameters: 4 × 35 × 3 = 420  (α.real, α.imag, β.real, β.imag)
```

### Complex↔Real Parameter Conversion (scipy optimizers require real inputs)

```python
def pack_coeffs(alpha, beta):
    """(n_lm, P+1) complex → real vector length 420."""
    return np.concatenate([alpha.real.ravel(), alpha.imag.ravel(),
                           beta.real.ravel(),  beta.imag.ravel()])

def unpack_coeffs(x, n_lm=35, P=2):
    """Real vector length 420 → complex alpha, beta each (n_lm, P+1)."""
    n = n_lm * (P + 1)   # 105
    alpha = x[  :  n].reshape(n_lm, P+1) + 1j * x[  n:2*n].reshape(n_lm, P+1)
    beta  = x[2*n:3*n].reshape(n_lm, P+1) + 1j * x[3*n:4*n].reshape(n_lm, P+1)
    return alpha, beta
```

### Image Theory at Height h above PEC

```
E_tot = E_direct + R · E_image(π−θ, φ) · exp(2i·k·h·cosθ),   k = 2πν_Hz/c
R = diag(+1, −1) on (E_θ, E_φ): TM (E_θ) preserved, TE (E_φ) flipped
```

Physical checks: N_10, h→0 → |E_tot| = 2sinθ (constructive). N_10, h=λ/4 at
75 MHz → null at zenith (θ=0), maximum at horizon (θ=π/2). ✓

### Power Pattern and Normalization

```python
# domega = 4π/N_pix_full (HEALPix equal-area pixels)
# n_pix_upper ≈ N_pix_full/2 → Σ domega over upper hemisphere ≈ 2π ✓
domega = 4 * np.pi / healpy.nside2npix(nside)

B      = np.abs(E_theta)**2 + np.abs(E_phi)**2             # (n_freq, n_pix_upper)
Btilde = B * (2 * np.pi / (B * domega).sum(axis=-1, keepdims=True))
# → (Btilde * domega).sum(axis=-1) = 2π for all frequencies ✓
```

### Beam from Real Parameter Vector

```python
def beam_from_x(x, h_m, freq_mhz_arr, basis, basis_image, domega,
                nu_ref=75.0, P=2, n_lm=35):
    """Compute normalized beam Btilde from real optimization vector x.

    Args:
        x:            real array length 420 (from pack_coeffs)
        basis:        from vsh_basis(lmax, nside) evaluated at theta_u
        basis_image:  from vsh_basis(lmax, nside) evaluated at pi - theta_u
                      (same nside, same n_pix_upper; pre-compute once alongside basis)
        domega:       4π / nside2npix(nside)

    Returns:
        Btilde: (n_freq, n_pix_upper)  ← beam power, normalized so Σ B̃ dΩ = 2π
    """
    assert n_lm == len(basis['M_lm']) == len(basis_image['M_lm']), \
        f"n_lm mismatch: {n_lm} vs {len(basis['M_lm'])}"

    alpha, beta = unpack_coeffs(x, n_lm=n_lm, P=P)
    # Frequency polynomial: (nu/nu_ref)^p for p=0..P
    p_powers = (freq_mhz_arr[:, None] / nu_ref) ** np.arange(P + 1)[None, :]
    a_nu = alpha @ p_powers.T    # (n_lm, n_freq); contracts polynomial order p
    b_nu = beta  @ p_powers.T   # (n_lm, n_freq)

    # Direct E-field on upper hemisphere pixels (l=mode, f=freq, q=pixel)
    E_theta = (np.einsum('lf,lq->fq', a_nu, basis['M_lm'][:, :, 0])
             + np.einsum('lf,lq->fq', b_nu, basis['N_lm'][:, :, 0]))  # (n_freq, n_pix_upper)
    E_phi   = (np.einsum('lf,lq->fq', a_nu, basis['M_lm'][:, :, 1])
             + np.einsum('lf,lq->fq', b_nu, basis['N_lm'][:, :, 1]))

    # Image E-field (basis_image evaluated at pi-theta_u, same phi_u)
    E_img_theta = (np.einsum('lf,lq->fq', a_nu, basis_image['M_lm'][:, :, 0])
                 + np.einsum('lf,lq->fq', b_nu, basis_image['N_lm'][:, :, 0]))
    E_img_phi   = (np.einsum('lf,lq->fq', a_nu, basis_image['M_lm'][:, :, 1])
                 + np.einsum('lf,lq->fq', b_nu, basis_image['N_lm'][:, :, 1]))

    # Path-length phase from height h_m above PEC
    k     = 2 * np.pi * freq_mhz_arr * 1e6 / 3e8    # (n_freq,)
    phase = np.exp(2j * k[:, None] * h_m * np.cos(basis['theta'])[None, :])

    E_tot_theta = E_theta + E_img_theta * phase    # R_theta = +1 (TM preserved)
    E_tot_phi   = E_phi   - E_img_phi   * phase    # R_phi   = −1 (TE flipped)

    B      = np.abs(E_tot_theta)**2 + np.abs(E_tot_phi)**2
    Btilde = B * (2 * np.pi / (B * domega).sum(axis=-1, keepdims=True))
    return Btilde
```

### Initialization

```python
n_lm, P = 35, 2
alpha = np.zeros((n_lm, P + 1), dtype=complex)
beta  = np.zeros((n_lm, P + 1), dtype=complex)
beta[1, 0] = 1.0   # index 1 = (l=1, m=0); VED-like starting point
rng = np.random.default_rng(42)
# Perturb real and imaginary parts independently:
alpha += 0.01 * (rng.standard_normal(alpha.shape) + 1j * rng.standard_normal(alpha.shape))
beta  += 0.01 * (rng.standard_normal(beta.shape)  + 1j * rng.standard_normal(beta.shape))
x0 = pack_coeffs(alpha, beta)   # length 420, real, ready for optimizer
```

### Validation

- N_10 only (set `beta[1,0]=1`, all others 0), h→0: B ∝ sin²θ < 1% rms.
- N_10, h=λ/4 at 75 MHz: max at θ=90°, null at θ=0°.
- `dYlm_dtheta(0, 1, phi, pi/2)` ≈ −0.4886 (real, negative).
- `dYlm_dtheta(-1, 1, phi, pi/2)` ≠ 0 (m=-l mode, condition `m<l` = True).
- `unpack_coeffs(pack_coeffs(alpha, beta)) == (alpha, beta)` ✓
- `(Btilde * domega).sum(axis=-1)` ≈ 2π for all ν (< 0.1%).
- VSH upper-hemi norm: `(|M10[:,0]|² + |M10[:,1]|²).sum() × domega ≈ 0.5`.

---

## Component 2: Sky Model (`skymap.py`)

**Caching:** GSM2016 generation is slow (~30 s). Save/load `(A_map, beta_bin_map, C_map)`
as `data/sky_model_nside{nside}.npy`.

```python
from pygdsm import GlobalSkyModel2016

def build_sky_model(nside=64, cache_path='data/sky_model.npy'):
    """Returns (A_map, beta_bin_map, C_map), each (n_pix,). Loads cache if present."""
    if os.path.exists(cache_path):
        return tuple(np.load(cache_path, allow_pickle=True))
    gsm = GlobalSkyModel2016(freq_unit='MHz', data_unit='TRJ')
    T_230 = gsm.generate(230) - 2.725    # subtract T_CMB [K]
    T_408 = gsm.generate(408) - 2.725
    # Per-pixel spectral index
    beta_p = np.log(T_408 / T_230) / np.log(408 / 230)
    beta_p = np.clip(beta_p, -3.0, -2.0)
    # 10 equal-count decile bins
    idx = np.argsort(beta_p)
    beta_bin = np.zeros_like(beta_p)
    for b in np.array_split(idx, 10):
        beta_bin[b] = beta_p[b].mean()
    # Per-pixel fit: T = A*(nu/408)^beta_bin * exp(C*log(nu/408)^2)  [arXiv:2607.00299]
    # Use scipy.optimize.curve_fit over ≥5 GSM spot frequencies per pixel
    # (e.g. 45, 74, 150, 230, 408 MHz)
    ...
    np.save(cache_path, np.array([A_map, beta_bin, C_map], dtype=object))
    return A_map, beta_bin, C_map

def sky_temperature(A, beta_bin, C, freq_mhz, freq_ref_mhz=408.0):
    """T = A*(f/f_ref)^beta * exp(C*log(f/f_ref)^2). Returns (n_pix,) [K]."""
    x = np.log(freq_mhz / freq_ref_mhz)
    return A * (freq_mhz / freq_ref_mhz)**beta_bin * np.exp(C * x**2)
```

**Validation:** T_model(230 MHz) and T_model(408 MHz) reproduce GSM2016 maps to < 1%
rms per pixel.

---

## Component 3: Ionospheric Correction (`ionosphere.py`)

```python
def apply_ionospheric_correction(sky_map, freq_hz, tec_zen_tecu, n_e_max,
                                 h_screen=300e3, t_ground=290.0, nside=64):
    """Thin-screen ionospheric correction. Valid for theta_z < 70°.
    For theta_z > 70°: multiply delta_rho by 1/cos(theta_z) (~2.9× at 70°).
    Returns corrected sky map (n_pix,) [K].
    """
    TEC_zen_SI = tec_zen_tecu * 1e16    # 1 TECU = 10^16 m^{-2}; convert at entry
    nu_p = 8.98 * np.sqrt(n_e_max)     # plasma frequency [Hz]; N_e=1e11 → 2.84 MHz
    if freq_hz <= nu_p:
        return np.full(healpy.nside2npix(nside), t_ground)

    n_pix = healpy.nside2npix(nside)
    theta_pix, phi_pix = healpy.pix2ang(nside, np.arange(n_pix))
    mask = theta_pix < np.pi / 2
    theta_z = theta_pix[mask]

    # Refraction toward zenith [rad]:
    delta_rho = (40.3 * TEC_zen_SI * np.sin(theta_z)) / (freq_hz**2 * h_screen)
    theta_new = theta_z - delta_rho

    result = sky_map.copy()
    result[mask] = healpy.get_interp_val(sky_map, theta_new, phi_pix[mask])
    return result
```

**Units:** `40.3 [m³s⁻²] × TEC [m⁻²] / (ν² [s⁻²] × H [m]) = dimensionless` ✓

**Validation:**
- TEC → 0, ν >> ν_p: output = input.
- ν ≤ ν_p: output = 290 K.
- Numerical check: TEC=5 TECU (=5×10¹⁶ m⁻²), ν=10 MHz=10⁷ Hz, θ_z=60°, H=3×10⁵ m:
  `δρ = (40.3 × 5e16 × 0.866) / (1e14 × 3e5) = 0.058 rad` ✓

---

## Component 4: Antenna Temperature (`antenna_temp.py`)

**Beam coordinate frame:** AltAz, θ = zenith angle, φ = **clockwise from North**
(astropy AltAz convention: 0=N, π/2=E, π=S). This is opposite to HEALPix phi (CCW).

### HEALPix → AltAz → ICRS Coordinate Chain

```python
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u

# Upper hemisphere HEALPix pixels → AltAz
theta_hp, phi_hp = healpy.pix2ang(nside, ipix_upper)    # radians
alt_rad = np.pi / 2 - theta_hp
az_rad  = (np.pi / 2 - phi_hp) % (2 * np.pi)            # CCW→CW from North
# Cross-check: phi_hp=0 (E) → az=π/2 (E) ✓; phi_hp=π/2 (N) → az=0 (N) ✓

# AltAz → ICRS
coords = SkyCoord(alt=alt_rad * u.rad, az=az_rad * u.rad,
                  frame=AltAz(obstime=t_i, location=site)).icrs

# ICRS → HEALPix sky angles for sky map interpolation
theta_sky = np.pi / 2 - coords.dec.rad    # colatitude [0, π]
phi_sky   = coords.ra.rad                  # RA as longitude [0, 2π]
T_pix = healpy.get_interp_val(sky_map, theta_sky, phi_sky)
```

### Antenna Temperature per LST Bin

```python
# Beam-weighted integral
T_ant[i_freq, i_lst] = (Btilde[i_freq] * T_iono * domega).sum() \
                      / (Btilde[i_freq] * domega).sum()
# Denominator = 2π when Btilde is normalized over upper hemisphere ✓
```

**Setup:**

```python
t0 = Time('2026-10-21 00:00:00', scale='utc')
utc_times = [t0 + i * u.hr for i in range(24)]
site = EarthLocation(lat=-26.70 * u.deg, lon=116.60 * u.deg, height=377 * u.m)
```

**Output:** `T_ant` shape `(n_freq, 24)`. Each LST bin is fit **independently** — do not
average over LST bins before fitting.

**Validation:**
- Uniform flat-spectrum sky → T_ant = const (ν, LST).
- GC (RA≈266°, Dec≈−29°): T_ant peaks when GC transits near zenith at MRO.

---

## Component 5: Bayesian Fitting (`fitting.py`)

### Model

```
T_model(ν) = T_fg(ν) + S_21(ν)
T_fg = Σ_{k=0}^{4} c_k × [ln(ν/75)]^k        [ν in MHz, ν_ref=75 MHz]
S_21 = −A × exp[−(ν−ν_0)²/(2σ²)]             [K, absorption dip]
theta = [c_0, c_1, c_2, c_3, c_4, A, nu_0, sigma]   # n_params = 8
```

### Priors

- `c_k ~ Normal(0, sigma_fg[k]²)`: `sigma_fg = [5000, 500, 500, 500, 500]` K
  (covers ~3000 K sky brightness at 75 MHz with margin for higher-order terms)
- `A ~ Uniform(0, 1 K)`
- `nu_0 ~ Uniform(7, 80 MHz)` — covers Dark Ages + lower Cosmic Dawn
- `sigma ~ Uniform(5, 30 MHz)`

### Noise (per-LST, per-frequency)

```python
delta_nu_hz = 0.145e6    # Hz (frequency resolution)
delta_t_s   = 3600.0     # 1 hr per LST bin
sigma_n = T_ant[:, i_lst] / np.sqrt(delta_nu_hz * delta_t_s)  # (n_freq,)
```

### emcee (used during optimization loop)

```python
import emcee

n_walkers, n_steps, n_burn = 32, 5000, 500

# Walker initialization: tight ball around MAP estimate
theta_map = scipy.optimize.minimize(
    lambda t: -log_likelihood(t, T_obs, freq_mhz, sigma_n),
    x0=[3000, 0, 0, 0, 0, 0.1, 40.0, 15.0],
    method='Nelder-Mead', options={'maxiter': 5000}
).x
pos0 = theta_map + 1e-4 * np.random.randn(n_walkers, n_params)

sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior)
sampler.run_mcmc(pos0, n_steps, progress=False)
tau = sampler.get_autocorr_time(quiet=True)
assert np.all(tau < n_steps / 50), f"Not converged: tau_max={tau.max():.0f}"
samples = sampler.get_chain(discard=n_burn, flat=True)   # (n_samples, 8)
```

### dynesty (final posteriors only)

```python
import dynesty
from scipy.stats import norm

def prior_transform(u):
    """Maps U[0,1]^8 → prior space. theta=[c_0,...,c_4, A, nu_0, sigma]."""
    sigma_fg = [5000, 500, 500, 500, 500]
    c = [norm.ppf(u[k], 0, sigma_fg[k]) for k in range(5)]
    A   = u[5] * 1.0         # Uniform(0, 1)
    nu0 = 7.0 + u[6] * 73.0  # Uniform(7, 80)
    sig = 5.0 + u[7] * 25.0  # Uniform(5, 30)
    return np.array(c + [A, nu0, sig])

sampler = dynesty.NestedSampler(log_likelihood_dynesty, prior_transform,
                                 ndim=8, nlive=500)
sampler.run_nested(dlogz=0.01)
results = sampler.results   # .samples, .logwt → evidence Z
```

**P(Â > 0.05 K | A=0) estimator:**

```python
# Fit null injection with emcee; A is parameter index 5
result_null = fit_21cm_bayesian(T_ant_nu, freq_mhz, sigma_n, sampler='emcee')
P_false_pos = np.mean(result_null['samples'][:, 5] > 0.05)
```

**Validation:**
- Zero injection (A=0): A_hat consistent with 0 at 95% CL.
- Known signal (0.3K, 25 MHz, 10 MHz): recovered within 1σ.
- Pure-FG fit: residuals < σ_n.

---

## Component 6: Outer Optimization (`optimize.py`)

### Loss Function

```
L(θ_beam) = (1/N) Σ_{i=1}^{N} [w_A(Â_i−A_i)² + w_ν(ν̂_0,i−ν_0,i)² + w_σ(σ̂_i−σ_i)²]
            + λ × P_90(Â > 0.05 K | A=0)
```

**Hyperparameters:**
- `w_A=1.0 K⁻²`, `w_ν=1.0 MHz⁻²`, `w_σ=1.0 MHz⁻²`
- `λ=10.0`, `N=100` realizations
- `P_90` = 90th percentile of P(Â>0.05|A=0) over 24 LST bins

### Signal Realizations

```python
rng = np.random.default_rng(seed)
realizations = [(0.0, 40.0, 15.0)] + list(zip(
    rng.uniform(0.05, 0.5,  N),   # A [K]
    rng.uniform(7.0,  80.0, N),   # nu_0 [MHz]
    rng.uniform(5.0,  30.0, N),   # sigma [MHz]
))
```

### compute_loss — Inner Loop Data Flow

```python
def compute_loss(x, realizations, sky_model, site, tec_zen_tecu, freq_mhz,
                 basis, basis_image, domega, lmax=5, h_m=0.5, nside=64):
    """
    1. unpack_coeffs(x) → alpha, beta (complex)
    2. beam_from_x(x, h_m, freq_mhz, basis, basis_image, domega) → Btilde (n_freq, n_pix_upper)
    3. Loop i_lst=0..23:
         sky rotation ICRS→AltAz for utc_times[i_lst]
         apply_ionospheric_correction → T_iono (n_pix)
         T_ant[:,i_lst] = beam-weighted integral (n_freq,)
    4. For each (A_i, nu0_i, sig_i) in realizations:
         pick random i_lst
         sigma_n = T_ant[:,i_lst] / sqrt(delta_nu_hz * delta_t_s)  [per-ν, per-LST]
         T_obs = T_ant[:,i_lst] + S_21(freq_mhz, A_i, nu0_i, sig_i) + rng.normal(0, sigma_n)
         samples = fit_21cm_bayesian(T_obs, freq_mhz, sigma_n, sampler='emcee').samples
         accumulate MSE: sum(w*(param_hat - param_true)^2)
    5. Null (A=0): P_false_pos[i_lst] = mean(samples[:,5] > 0.05) for each LST
    6. return mse_total/N + lambda_ * np.percentile(P_false_pos, 90)
    """
```

### Phase 1: DE/Rand/1/Bin Global Search (joblib parallel, fork-safe)

```python
from joblib import Parallel, delayed

n_real    = 420         # real optimization dimension
pop_size  = 15 * n_real  # 6300 (15 per dimension, scipy convention)
F, CR     = 0.8, 0.9
low  = np.full(n_real, -2.0)
high = np.full(n_real,  2.0)
population = rng.uniform(low, high, (pop_size, n_real))

def eval_pop(pop):
    return Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_loss)(x, ...) for x in pop
    )

# ⚠ Do NOT use scipy.optimize.differential_evolution(workers=-1)
# → deadlock when nested emcee calls spawn their own processes

fitness = eval_pop(population)
for gen in range(n_de_iter):
    trials = np.empty_like(population)
    idx = np.arange(pop_size)
    for i in range(pop_size):
        a, b, c = rng.choice(idx[idx != i], 3, replace=False)
        v = np.clip(population[a] + F * (population[b] - population[c]), low, high)
        # Binomial crossover
        cross = rng.random(n_real) < CR
        cross[rng.integers(n_real)] = True   # guarantee ≥ 1 dimension from mutant
        trials[i] = np.where(cross, v, population[i])
    trial_fit = eval_pop(trials)
    for i in range(pop_size):
        if trial_fit[i] <= fitness[i]:
            population[i], fitness[i] = trials[i], trial_fit[i]

best = population[np.argmin(fitness)]
```

### Phase 2: Powell Local Refinement

```python
result = scipy.optimize.minimize(
    lambda x: compute_loss(x, ...),
    x0=best, method='Powell',
    options={'xtol': 1e-3, 'ftol': 1e-3, 'maxiter': 200 * n_real}
)
alpha_opt, beta_opt = unpack_coeffs(result.x)
```

---

## Component 7: Visualization (`plot.py`)

- `corner.corner(samples, labels=['c₀',...,'A','ν₀','σ'])` — posterior distributions
- `plotly` surface of `B̃(θ,φ,ν)` at 10, 50, 100 MHz; animate over height h
- `healpy.mollview` for GSM2016 map, decile spectral index map
- `matplotlib.imshow` of `T_ant` waterfall (n_freq × 24 LST bins)

---

## Key Validation Tests

| #  | Test                      | Pass Criterion                                        |
|----|---------------------------|-------------------------------------------------------|
| 1  | Sky model fidelity        | T_model(230 MHz) < 1% rms from T_GSM(230 MHz)        |
| 2  | Beam normalization        | `(Btilde*domega).sum(axis=-1)` ≈ 2π (< 0.1%)        |
| 3  | VED dipole h→0            | B ∝ sin²θ < 1% rms                                   |
| 4  | Quarter-wave monopole     | Max at θ=90°, null at θ=0° at 75 MHz                 |
| 5  | dYlm/dθ value             | `dYlm_dtheta(0,1,phi,pi/2)` ≈ −0.4886                |
| 6  | dYlm m=−l mode            | `dYlm_dtheta(-1,1,phi,pi/2)` ≠ 0                    |
| 7  | pack/unpack roundtrip     | `unpack(pack(α,β)) == (α,β)`                         |
| 8  | VSH upper-hemi norm       | `sum(|M_10|²)*domega` ≈ 0.5 (not 1.0)               |
| 9  | Flat-sky T_ant            | Uniform sky → T_ant = const (ν, LST)                 |
| 10 | GC transit peak           | T_ant peaks at GC transit LST bin                    |
| 11 | Iono transparency         | TEC→0, ν>>ν_p: output = input                        |
| 12 | Plasma cutoff             | ν ≤ ν_p: output = 290 K                              |
| 13 | Iono numerical            | TEC=5, 10 MHz, 60°, 300 km → δρ = 0.058 rad          |
| 14 | Zero-injection            | A_hat consistent with 0 at 95% CL                    |
| 15 | Known signal recovery     | (0.3K, 25 MHz, 10 MHz): recovered within 1σ          |
| 16 | E2E integration           | Dipole beam + injected signal → recovered             |

---

## Implementation Notes

**numpy/scipy versions:** Use `np.random.default_rng(seed)` (not `np.random.seed`).
`scipy.special.sph_harm` was **removed in scipy ≥ 1.15**. Use `scipy.special.sph_harm_y(l, m, theta, phi)`
which follows standard physics convention: `theta` = polar (colatitude), `phi` = azimuthal.

**Performance:** `vsh_basis` and `build_sky_model` are expensive. Call each once and
cache. `beam_from_x` dominates inner loop; the einsum contractions vectorize over all
frequencies simultaneously.

**Parallelism:** The `loky` backend in joblib is fork-safe and works correctly with
emcee's internal threading. Never nest `multiprocessing`-backed parallel calls.
