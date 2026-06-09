"""Benchmark fast_fits_wcs vs astropy.wcs for pixel->world over a range of sizes."""
import time
import numpy as np
import jax
import jax.numpy as jnp

from astropy.wcs import WCS as AstropyWCS
from fast_fits_wcs import WCS as FastWCS

SIZES = [1, 100, 10_000, 1_000_000, 10_000_000]
REPEATS = 7


def make_astropy():
    a = AstropyWCS(naxis=2)
    a.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    a.wcs.crval = [10.0, 20.0]
    a.wcs.crpix = [128.0, 128.0]
    a.wcs.cdelt = [-0.001, 0.001]
    return a


def make_jax():
    j = FastWCS(naxis=2)
    j.ctype = ["RA---TAN", "DEC--TAN"]
    j.crval = [10.0, 20.0]
    j.crpix = [128.0, 128.0]
    j.cdelt = [-0.001, 0.001]
    return j


def timeit(fn, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


a = make_astropy()
j = make_jax()

# Pull out the device-resident jitted core for the best-case jax number.
from fast_fits_wcs.core import _jit_transforms
lng, lat, code, crpix, cd, pole, phi_p = j._param_arrays()
crpix_j, cd_j, pole_j = jnp.asarray(crpix), jnp.asarray(cd), jnp.asarray(pole)
fwd, _ = _jit_transforms(code, lng, lat)

# The low-level API now preserves the input's array namespace, so numpy in ->
# numpy out (eager), jax in -> jax out (jitted). We time three things:
#   astropy      : astropy.wcs, the reference
#   ffw numpy    : fast_fits_wcs with numpy arrays (what a numpy/reproject user gets)
#   ffw jax-core : the device-resident jitted transform (WCS as a node in a jax graph)
print(f"{'N':>12} | {'astropy ms':>12} | {'ffw numpy ms':>13} | "
      f"{'ffw jax-core ms':>16} | {'speedup(core)':>13}")
print("-" * 78)

for n in SIZES:
    rng = np.random.default_rng(1)
    px = rng.uniform(1, 256, size=n)
    py = rng.uniform(1, 256, size=n)

    # astropy low-level values API (0-based)
    t_astropy = timeit(lambda: a.pixel_to_world_values(px, py))

    # fast_fits_wcs with numpy arrays: numpy in, numpy out, eager
    t_numpy = timeit(lambda: j.pixel_to_world_values(px, py))

    # fast_fits_wcs device-resident core: inputs already jax arrays on device,
    # outputs kept on device (block_until_ready). This is the number that
    # matters when WCS is one node in a larger jax computation.
    pix_dev = jnp.stack([jnp.asarray(px), jnp.asarray(py)])
    fwd(pix_dev, crpix_j, cd_j, pole_j, phi_p).block_until_ready()  # warmup/compile

    def core():
        fwd(pix_dev, crpix_j, cd_j, pole_j, phi_p).block_until_ready()

    t_jax_core = timeit(core)

    print(f"{n:>12} | {t_astropy*1e3:>12.4f} | {t_numpy*1e3:>13.4f} | "
          f"{t_jax_core*1e3:>16.4f} | {t_astropy / t_jax_core:>12.2f}x")
