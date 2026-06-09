"""Benchmark jaxwcs vs astropy.wcs for pixel->world over a range of sizes."""
import time
import numpy as np
import jax
import jax.numpy as jnp

from astropy.wcs import WCS as AstropyWCS
from jaxwcs import WCS as JaxWCS

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
    j = JaxWCS(naxis=2)
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
from jaxwcs.core import _build_transforms
lng, lat, code, frame, crpix, crval, cd, phi_p = j._params()
fwd, _ = _build_transforms(code, lng, lat)

print(f"{'N':>12} | {'astropy ms':>12} | {'jax API ms':>12} | "
      f"{'jax core ms':>12} | {'speedup(core)':>13}")
print("-" * 72)

for n in SIZES:
    rng = np.random.default_rng(1)
    px = rng.uniform(1, 256, size=n)
    py = rng.uniform(1, 256, size=n)

    # astropy low-level values API (0-based)
    t_astropy = timeit(lambda: a.pixel_to_world_values(px, py))

    # jaxwcs full APE-14 API: host arrays in, host arrays out
    j.pixel_to_world_values(px, py)  # warmup/compile for this shape
    t_jax_api = timeit(lambda: j.pixel_to_world_values(px, py))

    # jaxwcs device-resident core: inputs already jax arrays on device,
    # outputs kept on device (block_until_ready). This is the number that
    # matters when WCS is one node in a larger jax computation.
    pix_dev = jnp.stack([jnp.asarray(px), jnp.asarray(py)])
    out = fwd(pix_dev, crpix, cd, crval, phi_p)
    out.block_until_ready()

    def core():
        fwd(pix_dev, crpix, cd, crval, phi_p).block_until_ready()

    t_jax_core = timeit(core)

    print(f"{n:>12} | {t_astropy*1e3:>12.4f} | {t_jax_api*1e3:>12.4f} | "
          f"{t_jax_core*1e3:>12.4f} | {t_astropy / t_jax_core:>12.2f}x")
