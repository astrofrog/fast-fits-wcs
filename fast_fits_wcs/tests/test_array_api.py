"""The low-level API is array-API compliant: it computes in, and returns, the
input's array namespace. Here we check numpy-in/numpy-out and jax-in/jax-out
agree with each other and with astropy.wcs.
"""
import numpy as np
import jax.numpy as jnp
import pytest
from astropy.wcs import WCS as AstropyWCS
from fast_fits_wcs import WCS as FastWCS

CASES = [
    dict(ctype=("RA---TAN", "DEC--TAN"), crval=[266.4, -29.0]),
    dict(ctype=("RA---SIN", "DEC--SIN"), crval=[150.0, 40.0]),
    dict(ctype=("GLON-CAR", "GLAT-CAR"), crval=[30.0, 10.0]),
]


def _build(case):
    a = AstropyWCS(naxis=2)
    a.wcs.ctype = list(case["ctype"]); a.wcs.crval = case["crval"]
    a.wcs.crpix = [128.0, 128.0]; a.wcs.cdelt = [-0.01, 0.01]
    f = FastWCS(naxis=2)
    f.ctype = list(case["ctype"]); f.crval = case["crval"]
    f.crpix = [128.0, 128.0]; f.cdelt = [-0.01, 0.01]
    return a, f


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["ctype"][0][:4])
def test_namespace_preserved_and_matches_astropy(case):
    a, f = _build(case)
    rng = np.random.default_rng(0)
    px = rng.uniform(1, 256, 1000); py = rng.uniform(1, 256, 1000)

    # numpy in -> numpy out
    wn = f.pixel_to_world_values(px, py)
    assert all(isinstance(v, np.ndarray) for v in wn)

    # jax in -> jax out
    wj = f.pixel_to_world_values(jnp.asarray(px), jnp.asarray(py))
    assert all(isinstance(v, jnp.ndarray) for v in wj)

    # both agree with astropy
    aw = a.wcs_pix2world(px, py, 0)
    for k in (0, 1):
        assert np.max(np.abs(np.asarray(wn[k]) - aw[k])) < 1e-9
        assert np.max(np.abs(np.asarray(wj[k]) - aw[k])) < 1e-9

    # the world->pixel direction preserves the namespace too
    assert all(isinstance(v, np.ndarray) for v in f.world_to_pixel_values(*wn))
    assert all(isinstance(v, jnp.ndarray) for v in f.world_to_pixel_values(*wj))


if __name__ == "__main__":
    for _c in CASES:
        test_namespace_preserved_and_matches_astropy(_c)
    print("OK: low-level API is array-API compliant")
