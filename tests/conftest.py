import numpy as np
import pytest
import healpy


@pytest.fixture(scope="session")
def nside():
    return 64


@pytest.fixture(scope="session")
def lmax():
    return 5


@pytest.fixture(scope="session")
def freq_mhz():
    return np.linspace(5.0, 150.0, 50)


@pytest.fixture(scope="session")
def domega(nside):
    return 4 * np.pi / healpy.nside2npix(nside)


@pytest.fixture(scope="session")
def basis(nside, lmax):
    from sofa.beam import vsh_basis
    return vsh_basis(lmax=lmax, nside=nside)


@pytest.fixture(scope="session")
def basis_image(nside, lmax):
    import numpy as np
    import healpy
    from sofa.beam import vsh_basis
    n_pix = healpy.nside2npix(nside)
    theta_all, phi_all = healpy.pix2ang(nside, np.arange(n_pix))
    upper_mask = theta_all < np.pi / 2
    theta_u = theta_all[upper_mask]
    phi_u = phi_all[upper_mask]
    theta_img = np.pi - theta_u
    return vsh_basis(lmax=lmax, nside=nside, theta_override=theta_img, phi_override=phi_u)
