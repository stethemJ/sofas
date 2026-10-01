"""
tests/test_beam.py — Validation tests for sofa.beam module.

Covers all validation criteria from CLAUDE.md:
  1. dYlm_dtheta(0, 1, phi, pi/2) ≈ -0.4886  (scipy arg-order trap)
  2. dYlm_dtheta(-1, 1, phi, pi/2) != 0       (m=-l mode, m<l condition)
  3. pack/unpack roundtrip for complex coefficients
  4. Beam normalization: (Btilde * domega).sum(axis=-1) ≈ 2π
  5. VED dipole h→0: B ∝ sin²θ, rms error < 1%
  6. VSH upper-hemisphere orthonormality: sum(|M_10|²) * domega ≈ 0.5
"""

import numpy as np
import pytest
from sofa.beam import (
    dYlm_dtheta,
    pack_coeffs,
    unpack_coeffs,
    beam_from_x,
)


# ---------------------------------------------------------------------------
# Test 1: dYlm/dtheta value at theta=pi/2 (scipy argument-order check)
# ---------------------------------------------------------------------------

class TestDYlmDtheta:
    """Validate dYlm_dtheta recurrence against analytic anchor values."""

    def test_dY10_dtheta_at_equator_real(self):
        """dYlm_dtheta(0, 1, phi, pi/2) must be real and ≈ -0.4886.

        Derivation:
            Y_10 = sqrt(3/4pi) * cos(theta)
            dY_10/dtheta = -sqrt(3/4pi) * sin(theta)
            At theta=pi/2: -sqrt(3/4pi) = -sqrt(3/(4*pi)) ≈ -0.4886
        """
        phi = 0.5  # arbitrary azimuthal angle
        result = dYlm_dtheta(0, 1, phi, np.pi / 2)
        assert abs(result.real - (-0.4886)) < 0.001, (
            f"dYlm_dtheta(0,1,phi,pi/2).real = {result.real:.6f}, expected ≈ -0.4886"
        )
        assert abs(result.imag) < 1e-10, (
            f"dYlm_dtheta(0,1,phi,pi/2).imag = {result.imag:.2e}, expected ≈ 0"
        )

    def test_dY10_dtheta_phi_independence(self):
        """For m=0, dY_10/dtheta is real and phi-independent at any theta."""
        theta = np.pi / 3  # 60 degrees
        for phi in [0.0, 0.5, 1.0, np.pi]:
            result = dYlm_dtheta(0, 1, phi, theta)
            assert abs(result.imag) < 1e-10, (
                f"dYlm_dtheta(0,1,{phi},theta).imag = {result.imag:.2e}, expected 0"
            )

    def test_dYlm_m_equals_negative_l_nonzero(self):
        """dYlm_dtheta(-1, 1, phi, theta) must be nonzero for theta != pi/2.

        The condition in the recurrence is 'm < l' (not 'abs(m) < l').
        For m=-1, l=1: -1 < 1 is True → term2 is included.
        Bug: 'abs(m) < l' → abs(-1)=1 < 1 is False → term2=0 → wrong result.

        NOTE: theta=pi/2 is avoided here because at that specific angle,
        term1=m*cot(pi/2)*Y=0 and term2 ∝ Y_10(pi/2)=cos(pi/2)=0, making
        the result identically zero for both the correct and buggy implementations.
        theta=pi/3 gives a nonzero result only when term2 is correctly included.

        Analytic check:
            Y_{1,-1}(theta, phi) = sqrt(3/8pi) * sin(theta) * exp(-i*phi)
            => dY_{1,-1}/dtheta = sqrt(3/8pi) * cos(theta) * exp(-i*phi)
        """
        phi = 0.5
        theta = np.pi / 3  # cos(pi/3)=0.5 → analytic result is nonzero

        result = dYlm_dtheta(-1, 1, phi, theta)

        # Must be clearly nonzero — the bug (abs(m)<l) produces a different value
        assert abs(result) > 0.01, (
            f"|dYlm_dtheta(-1,1,phi,pi/3)| = {abs(result):.6f}, "
            "expected > 0.01"
        )

        # Cross-check against analytic formula
        analytic = np.sqrt(3 / (8 * np.pi)) * np.cos(theta) * np.exp(-1j * phi)
        assert abs(result - analytic) < 0.001, (
            f"dYlm_dtheta(-1,1,phi,pi/3) = {result}, "
            f"analytic = {analytic}, "
            f"error = {abs(result - analytic):.2e}"
        )

    def test_dYlm_m_equals_l_no_term2(self):
        """dYlm_dtheta(l, l, phi, theta): term2 should be absent (m == l → m < l is False)."""
        # For m=l, there is no Y_{l,l+1} term; result is purely term1.
        # This is a boundary condition check, not a bug trigger, but ensures
        # the 'm < l' guard works correctly for both boundary cases.
        phi = 1.0
        theta = np.pi / 4
        result_mm = dYlm_dtheta(1, 1, phi, theta)   # m=l: only term1
        result_m0 = dYlm_dtheta(0, 1, phi, theta)   # m<l: term1 + term2
        # They should be different values (no single assertion about magnitude)
        assert result_mm != result_m0, "m=l and m<l modes should give different results"


# ---------------------------------------------------------------------------
# Test 2+3: pack/unpack coefficient roundtrip
# ---------------------------------------------------------------------------

class TestPackUnpackCoeffs:
    """Validate that pack_coeffs and unpack_coeffs are exact inverses."""

    def test_roundtrip_identity(self):
        """unpack_coeffs(pack_coeffs(alpha, beta)) == (alpha, beta) exactly."""
        rng = np.random.default_rng(42)
        n_lm, P = 35, 2
        alpha = rng.standard_normal((n_lm, P + 1)) + 1j * rng.standard_normal((n_lm, P + 1))
        beta  = rng.standard_normal((n_lm, P + 1)) + 1j * rng.standard_normal((n_lm, P + 1))

        x = pack_coeffs(alpha, beta)
        assert len(x) == 420, f"packed length = {len(x)}, expected 420"
        assert x.dtype in (np.float64, np.float32), f"packed array must be real, got {x.dtype}"

        alpha2, beta2 = unpack_coeffs(x, n_lm=n_lm, P=P)
        assert np.allclose(alpha, alpha2), "alpha mismatch after pack/unpack roundtrip"
        assert np.allclose(beta, beta2),  "beta mismatch after pack/unpack roundtrip"

    def test_vector_length(self):
        """pack_coeffs produces exactly 4 * n_lm * (P+1) = 420 real values."""
        rng = np.random.default_rng(0)
        n_lm, P = 35, 2
        alpha = rng.standard_normal((n_lm, P + 1)) + 1j * rng.standard_normal((n_lm, P + 1))
        beta  = rng.standard_normal((n_lm, P + 1)) + 1j * rng.standard_normal((n_lm, P + 1))
        x = pack_coeffs(alpha, beta)
        assert len(x) == 4 * n_lm * (P + 1)

    def test_segment_ordering(self):
        """Verify the internal ordering: [alpha.real | alpha.imag | beta.real | beta.imag]."""
        n_lm, P = 35, 2
        n = n_lm * (P + 1)  # 105
        alpha = np.ones((n_lm, P + 1)) + 2j * np.ones((n_lm, P + 1))
        beta  = 3 * np.ones((n_lm, P + 1)) + 4j * np.ones((n_lm, P + 1))
        x = pack_coeffs(alpha, beta)
        assert np.allclose(x[:n],     1.0), "first segment must be alpha.real"
        assert np.allclose(x[n:2*n],  2.0), "second segment must be alpha.imag"
        assert np.allclose(x[2*n:3*n], 3.0), "third segment must be beta.real"
        assert np.allclose(x[3*n:],   4.0), "fourth segment must be beta.imag"

    def test_zero_roundtrip(self):
        """All-zero coefficients pack and unpack correctly."""
        n_lm, P = 35, 2
        alpha = np.zeros((n_lm, P + 1), dtype=complex)
        beta  = np.zeros((n_lm, P + 1), dtype=complex)
        x = pack_coeffs(alpha, beta)
        assert np.all(x == 0.0)
        alpha2, beta2 = unpack_coeffs(x)
        assert np.allclose(alpha, alpha2)
        assert np.allclose(beta, beta2)


# ---------------------------------------------------------------------------
# Test 4: Beam normalization  (Btilde * domega).sum ≈ 2π
# ---------------------------------------------------------------------------

class TestBeamNormalization:
    """Verify that the normalized beam integrates to 2π over the upper hemisphere."""

    def test_normalization_N10_h0(self, basis, basis_image, domega, freq_mhz):
        """N10-only beam at h=0 must integrate to 2π at every frequency."""
        n_lm, P = 35, 2
        alpha = np.zeros((n_lm, P + 1), dtype=complex)
        beta  = np.zeros((n_lm, P + 1), dtype=complex)
        beta[1, 0] = 1.0  # N_10 mode only (l=1, m=0 at index 1)

        x = pack_coeffs(alpha, beta)
        Btilde = beam_from_x(x, h_m=0.0, freq_mhz_arr=freq_mhz,
                             basis=basis, basis_image=basis_image, domega=domega)

        integrals = (Btilde * domega).sum(axis=-1)  # (n_freq,)
        frac_err  = np.abs(integrals - 2 * np.pi) / (2 * np.pi)
        assert np.all(frac_err < 0.001), (
            f"Beam normalization error: max frac error = {frac_err.max():.4e}, "
            "expected < 0.1%"
        )

    def test_normalization_shape(self, basis, basis_image, domega, freq_mhz):
        """beam_from_x returns shape (n_freq, n_pix_upper)."""
        n_lm, P = 35, 2
        beta = np.zeros((n_lm, P + 1), dtype=complex)
        beta[1, 0] = 1.0
        alpha = np.zeros_like(beta)
        x = pack_coeffs(alpha, beta)
        Btilde = beam_from_x(x, h_m=0.0, freq_mhz_arr=freq_mhz,
                             basis=basis, basis_image=basis_image, domega=domega)
        n_pix_upper = basis['M_lm'].shape[1]
        assert Btilde.shape == (len(freq_mhz), n_pix_upper), (
            f"Btilde shape {Btilde.shape} != ({len(freq_mhz)}, {n_pix_upper})"
        )

    def test_normalization_nonzero_height(self, basis, basis_image, domega, freq_mhz):
        """Beam normalization holds for nonzero height (h = 0.5 m)."""
        n_lm, P = 35, 2
        beta = np.zeros((n_lm, P + 1), dtype=complex)
        beta[1, 0] = 1.0
        alpha = np.zeros_like(beta)
        x = pack_coeffs(alpha, beta)
        Btilde = beam_from_x(x, h_m=0.5, freq_mhz_arr=freq_mhz,
                             basis=basis, basis_image=basis_image, domega=domega)
        integrals = (Btilde * domega).sum(axis=-1)
        frac_err  = np.abs(integrals - 2 * np.pi) / (2 * np.pi)
        assert np.all(frac_err < 0.001), (
            f"Beam normalization (h=0.5m): max frac error = {frac_err.max():.4e}"
        )


# ---------------------------------------------------------------------------
# Test 5: VED dipole pattern h→0: B ∝ sin²θ, rms < 1%
# ---------------------------------------------------------------------------

class TestVEDDipolePattern:
    """Verify that N_10-only beam at h=0 produces the correct sin²θ pattern."""

    def test_sin2_pattern_correlation(self, basis, basis_image, domega, freq_mhz):
        """N10 beam (h=0) must be proportional to sin²θ to < 1% rms.

        Method: normalize both B and sin²θ to unit mean, then check rms of residuals.
        The correlation check is frequency-independent for a spectrally flat coefficient.
        """
        n_lm, P = 35, 2
        alpha = np.zeros((n_lm, P + 1), dtype=complex)
        beta  = np.zeros((n_lm, P + 1), dtype=complex)
        beta[1, 0] = 1.0  # N_10 only

        x = pack_coeffs(alpha, beta)
        Btilde = beam_from_x(x, h_m=0.0, freq_mhz_arr=freq_mhz,
                             basis=basis, basis_image=basis_image, domega=domega)

        theta_u  = basis['theta']               # (n_pix_upper,)
        sin2_ref = np.sin(theta_u) ** 2         # analytic pattern

        # Test at mid-band frequency to avoid numerical edge effects
        i_mid = len(freq_mhz) // 2
        B_row = Btilde[i_mid]

        # Normalize both to mean = 1 for shape comparison
        B_norm   = B_row   / B_row.mean()
        sin2_norm = sin2_ref / sin2_ref.mean()

        rms_err = np.sqrt(np.mean((B_norm - sin2_norm) ** 2))
        assert rms_err < 0.01, (
            f"VED dipole pattern rms error = {rms_err:.4f}, expected < 0.01 (1%)"
        )

    def test_sin2_pattern_all_freqs(self, basis, basis_image, domega, freq_mhz):
        """N10 beam (h=0) shape must be sin²θ at all frequencies (frequency-independent)."""
        n_lm, P = 35, 2
        alpha = np.zeros((n_lm, P + 1), dtype=complex)
        beta  = np.zeros((n_lm, P + 1), dtype=complex)
        beta[1, 0] = 1.0

        x = pack_coeffs(alpha, beta)
        Btilde = beam_from_x(x, h_m=0.0, freq_mhz_arr=freq_mhz,
                             basis=basis, basis_image=basis_image, domega=domega)

        theta_u   = basis['theta']
        sin2_ref  = np.sin(theta_u) ** 2
        sin2_norm = sin2_ref / sin2_ref.mean()  # (n_pix_upper,)

        for i_freq in range(len(freq_mhz)):
            B_row  = Btilde[i_freq]
            B_norm = B_row / B_row.mean()
            rms_err = np.sqrt(np.mean((B_norm - sin2_norm) ** 2))
            assert rms_err < 0.01, (
                f"VED dipole pattern rms error at freq index {i_freq} "
                f"({freq_mhz[i_freq]:.1f} MHz) = {rms_err:.4f}, expected < 0.01"
            )


# ---------------------------------------------------------------------------
# Test 6: VSH upper-hemisphere orthonormality
# ---------------------------------------------------------------------------

class TestVSHOrthonormality:
    """Verify the upper-hemisphere integral of |M_10|² is 0.5 (half of full sphere)."""

    def test_M10_upper_hemi_norm(self, basis, domega):
        """(|M_10_theta|² + |M_10_phi|²).sum() * domega ≈ 0.5.

        M_10 is the (l=1, m=0) mode; it sits at index 1 in the mode ordering:
            (l=1, m=-1), (l=1, m=0), (l=1, m=+1), ...

        Full-sphere orthonormality: ∫ |M_lm|² dΩ = 1
        Upper-hemisphere symmetry:  ∫_upper |M_10|² dΩ = 0.5  (by symmetry)
        """
        M10 = basis['M_lm'][1]  # (n_pix_upper, 2); index 1 = (l=1, m=0)
        ortho = (np.abs(M10[:, 0]) ** 2 + np.abs(M10[:, 1]) ** 2).sum() * domega
        assert abs(ortho - 0.5) < 0.01, (
            f"VSH upper-hemi norm for M_10 = {ortho:.6f}, expected ≈ 0.5 ± 0.01"
        )

    def test_M10_index_is_l1_m0(self, basis):
        """Confirm that mode index 1 corresponds to (l=1, m=0) via M_10 pattern.

        M_10 has zero theta-component (M_theta ∝ m*Y, m=0) and nonzero phi-component
        (M_phi ∝ dY_10/dtheta ∝ -sinθ). So |M_theta| should be much smaller than |M_phi|.
        """
        M10 = basis['M_lm'][1]  # (n_pix_upper, 2)
        M_theta_rms = np.sqrt(np.mean(np.abs(M10[:, 0]) ** 2))
        M_phi_rms   = np.sqrt(np.mean(np.abs(M10[:, 1]) ** 2))
        # For m=0: M_theta ∝ m*Y = 0; M_phi_rms should dominate
        assert M_phi_rms > 10 * M_theta_rms, (
            f"M_10 theta-component rms {M_theta_rms:.4e} should be negligible "
            f"compared to phi-component rms {M_phi_rms:.4e}"
        )

    def test_N10_upper_hemi_norm(self, basis, domega):
        """N_10 upper-hemisphere norm should also be ≈ 0.5 (same by construction)."""
        N10 = basis['N_lm'][1]  # (n_pix_upper, 2); index 1 = (l=1, m=0)
        ortho = (np.abs(N10[:, 0]) ** 2 + np.abs(N10[:, 1]) ** 2).sum() * domega
        assert abs(ortho - 0.5) < 0.01, (
            f"VSH upper-hemi norm for N_10 = {ortho:.6f}, expected ≈ 0.5 ± 0.01"
        )

    def test_basis_shape(self, basis, lmax):
        """Verify basis array shapes are consistent with n_lm = lmax² + 2*lmax."""
        n_lm_expected = lmax ** 2 + 2 * lmax  # 35 for lmax=5
        n_lm_actual   = basis['M_lm'].shape[0]
        assert n_lm_actual == n_lm_expected, (
            f"n_lm = {n_lm_actual}, expected {n_lm_expected} for lmax={lmax}"
        )
        assert basis['M_lm'].shape[2] == 2, "M_lm axis-2 must be 2 (theta, phi components)"
        assert basis['N_lm'].shape    == basis['M_lm'].shape, "N_lm and M_lm must have same shape"
