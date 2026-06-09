# fast-fits-wcs

A minimal, [APE 14](https://github.com/astropy/astropy-APEs/blob/main/APE14.rst)-compliant
celestial FITS WCS whose numeric core is written against the
[Python array API](https://data-apis.org/array-api/), so the pixel↔world
transform runs in whatever array library you pass it — numpy (CPU),
[JAX](https://jax.readthedocs.io) (`jit`/`vmap`/`grad`, CPU/GPU/TPU), or CuPy
(GPU) — the same code, results staying on the input's device. Parameters are set
as plain attributes, and because it implements the APE 14 interface it can be
used anywhere an APE 14 WCS is accepted (WCSAxes, reproject, ...) — though its
own construction API differs from `astropy.wcs.WCS`. It currently covers 2-D
celestial WCS with the TAN, STG, SIN, ARC, ZEA, and CAR projections, in
equatorial/galactic/ecliptic frames.

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

## Array API: one call, any backend

The low-level API (`pixel_to_world_values` / `world_to_pixel_values`) computes
in — and returns — the input's array namespace, on that array's device. The
*same call* runs on numpy, jax, or cupy and the result stays in that library:

```pycon
>>> import numpy as np, jax.numpy as jnp, cupy as cp
>>> from fast_fits_wcs import WCS
>>> w = WCS(naxis=2)
>>> w.ctype = 'RA---TAN', 'DEC--TAN'
>>> w.crpix = [1000., 1000.]; w.crval = [266., -29.]; w.cdelt = [-1e-3, 1e-3]
>>> N = 1_000_000
>>> x = np.random.uniform(1, 2000, N); y = np.random.uniform(1, 2000, N)

>>> # numpy in -> numpy out (CPU)
>>> type(w.pixel_to_world_values(x, y)[0])
<class 'numpy.ndarray'>
>>> %timeit w.pixel_to_world_values(x, y)
149 ms ± 0.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> # jax in -> jax out (GPU, jitted)
>>> jx, jy = jnp.asarray(x), jnp.asarray(y)
>>> lon, lat = w.pixel_to_world_values(jx, jy)
>>> type(lon), lon.devices()
(<class 'jaxlib._jax.ArrayImpl'>, {CudaDevice(id=0)})
>>> %timeit w.pixel_to_world_values(jx, jy)[0].block_until_ready()
6.44 ms ± 0.06 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> # cupy in -> cupy out (GPU)
>>> cx, cy = cp.asarray(x), cp.asarray(y)
>>> lon, lat = w.pixel_to_world_values(cx, cy)
>>> type(lon), lon.device
(<class 'cupy.ndarray'>, <CUDA Device 0>)
>>> from cupyx.profiler import benchmark
>>> print(benchmark(lambda: w.pixel_to_world_values(cx, cy), n_repeat=100))
<lambda>:  CPU: 977 us   GPU-0: 7455 us
```

At 10⁶ points the GPU backends are ~20× faster than numpy here (jax 6.4 ms,
cupy 7.5 ms vs numpy 149 ms) — and that's on an old GeForce GTX 1060, whose
float64 throughput is heavily throttled; a modern datacenter GPU widens the gap.
(numpy uses `array-api-compat` for older versions; cupy/torch need it installed.)
The high-level (`SkyCoord`) API is numpy-backed, as `SkyCoord` requires.

See `examples/reproject_example.py` for reprojecting a real galactic-center
image and checking it against `astropy.wcs`.
