"""The low-level API is array-API compliant: it computes in, and returns, the
input's array namespace. Here we check numpy-in/numpy-out and jax-in/jax-out
agree with each other and with astropy.wcs.
"""
import numpy as np
import jax.numpy as jnp
from astropy.wcs import WCS as AstropyWCS
from fast_fits_wcs import WCS as FastWCS

rng = np.random.default_rng(0)

cases = [
    dict(ctype=("RA---TAN", "DEC--TAN"), crval=[266.4, -29.0]),
    dict(ctype=("RA---SIN", "DEC--SIN"), crval=[150.0, 40.0]),
    dict(ctype=("GLON-CAR", "GLAT-CAR"), crval=[30.0, 10.0]),
]

max_err = 0.0
for c in cases:
    a = AstropyWCS(naxis=2)
    a.wcs.ctype = list(c["ctype"]); a.wcs.crval = c["crval"]
    a.wcs.crpix = [128.0, 128.0]; a.wcs.cdelt = [-0.01, 0.01]

    f = FastWCS(naxis=2)
    f.ctype = list(c["ctype"]); f.crval = c["crval"]
    f.crpix = [128.0, 128.0]; f.cdelt = [-0.01, 0.01]

    px = rng.uniform(1, 256, 1000); py = rng.uniform(1, 256, 1000)

    # numpy in -> numpy out
    wn = f.pixel_to_world_values(px, py)
    assert all(isinstance(v, np.ndarray) for v in wn), "numpy in should give numpy out"

    # jax in -> jax out
    wj = f.pixel_to_world_values(jnp.asarray(px), jnp.asarray(py))
    assert all(isinstance(v, jnp.ndarray) for v in wj), "jax in should give jax out"

    # both agree with astropy
    aw = a.wcs_pix2world(px, py, 0)
    for k in (0, 1):
        max_err = max(max_err,
                      np.max(np.abs(np.asarray(wn[k]) - aw[k])),
                      np.max(np.abs(np.asarray(wj[k]) - aw[k])))

    # round-trip preserves namespace
    bn = f.world_to_pixel_values(*wn)
    bj = f.world_to_pixel_values(*wj)
    assert all(isinstance(v, np.ndarray) for v in bn)
    assert all(isinstance(v, jnp.ndarray) for v in bj)
    print(f"{c['ctype']}: numpy & jax both match astropy, namespaces preserved")

print(f"\nMAX error vs astropy (numpy & jax): {max_err:.2e} deg")
assert max_err < 1e-9
print("OK: low-level API is array-API compliant")
