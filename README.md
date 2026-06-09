# fast-fits-wcs

A minimal, [APE 14](https://github.com/astropy/astropy-APEs/blob/main/APE14.rst)-compliant
celestial FITS WCS whose numeric core is pure [JAX](https://jax.readthedocs.io),
so the pixel↔world transform is `jit`/`vmap`/`grad`-able and runs on CPU/GPU/TPU.
Parameters are set as plain attributes and it is a drop-in for `astropy.wcs.WCS`
for the cases it covers: 2-D celestial WCS with the TAN, STG, SIN, ARC, ZEA, and
CAR projections, in equatorial/galactic/ecliptic frames.

```python
from fast_fits_wcs import WCS

w = WCS(naxis=2)
w.ctype = 'RA---TAN', 'DEC--TAN'
w.crpix = [128.0, 128.0]
w.crval = [10.0, 20.0]
w.cdelt = [-0.001, 0.001]

ra, dec = w.pixel_to_world_values([130.0, 200.0], [132.0, 50.0])
sky = w.pixel_to_world(130.0, 132.0)   # -> SkyCoord (needs astropy)

# or build one straight from a FITS header
from astropy.io import fits
w = WCS.from_header(fits.getheader("image.fits"))
```

The low-level API (`pixel_to_world_values` / `world_to_pixel_values`) is
[Array API](https://data-apis.org/array-api/) compliant: it computes in, and
returns, the input's array namespace, so numpy in → numpy out, jax in → jax out
(jitted), cupy in → cupy out, all on the input's device. The high-level
(`SkyCoord`) API is numpy-backed, as `SkyCoord` requires.

See `examples/reproject_example.py` for reprojecting a real galactic-center
image and checking it against `astropy.wcs`.
