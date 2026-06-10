"""Benchmark fast_fits_wcs pixel->world across array sizes and array backends.

Times the public low-level API (``pixel_to_world_values``) for a scalar and a
range of array sizes, on every backend installed -- numpy (CPU), jax (CPU/GPU),
cupy (GPU) -- with astropy.wcs as the reference. The same call is used for every
backend; only the input array's library differs (numpy in -> numpy out, jax in
-> jax out, cupy in -> cupy out), which is the point of the array-API core.

Run:  python benchmarks/performance.py [max_log10_N]
e.g.  python benchmarks/performance.py 6     # cap sizes at 1e6 (small GPUs)

jax/cupy timings are device-resident: inputs are placed on the device once and
the compute is synchronised before the timer stops (block_until_ready for jax,
deviceSynchronize for cupy), so they reflect on-device compute, not transfers.

Note on threading: the XLA CPU runtime is multi-threaded (jax-cpu uses several
cores here), whereas numpy and astropy/wcslib are effectively single-threaded.
So jax-cpu's large-N speedup is partly multi-core, not a single-thread win.
"""
import os
import sys
import gc
import time
import statistics

# Don't let jax preallocate the GPU -- it has to share VRAM with cupy here.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from astropy.wcs import WCS as AstropyWCS
from fast_fits_wcs import WCS as FastWCS

# scalar, then array sizes
CASES = ["scalar", 100, 10_000, 1_000_000, 10_000_000]
if len(sys.argv) > 1:
    cap = 10 ** int(sys.argv[1])
    CASES = ["scalar"] + [n for n in CASES[1:] if n <= cap]


# --- optional backends -------------------------------------------------------
def _load_backends():
    """ordered name -> (to_device, sync) for each available array placement."""
    backends = {"numpy": (lambda a: a, lambda out: None)}
    try:
        import jax
        print("jax devices:", jax.devices())

        def jsync(out):
            for v in out:
                v.block_until_ready()

        # jax can place arrays on a chosen device; the jitted transform runs
        # there, so we can time CPU and GPU separately in the same process.
        for plat in ("cpu", "gpu"):
            try:
                dev = jax.devices(plat)[0]
            except RuntimeError:
                continue  # that platform isn't available
            backends[f"jax-{plat}"] = (
                lambda a, d=dev: jax.device_put(np.asarray(a), d), jsync)
    except ImportError:
        print("jax: not installed, skipped")
    try:
        import cupy as cp

        def csync(out):
            cp.cuda.runtime.deviceSynchronize()

        backends["cupy"] = (lambda a: cp.asarray(a), csync)
    except ImportError:
        print("cupy: not installed, skipped")
    return backends


def _free_cupy():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _inputs(case):
    if case == "scalar":
        return np.float64(123.4), np.float64(567.8)
    rng = np.random.default_rng(0)
    return rng.uniform(1, 2000, case), rng.uniform(1, 2000, case)


def _repeats(case):
    n = 1 if case == "scalar" else case
    if n <= 10_000:
        return 100
    if n <= 1_000_000:
        return 20
    return 7


def _measure(call, sync, repeats):
    out = call(); sync(out)                      # warm up (JIT compile / kernels)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = call(); sync(out)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _fmt(seconds):
    if seconds is None:
        s = "-"
    else:
        us = seconds * 1e6
        s = f"{us:.2f} us" if us < 1000 else f"{us / 1000:.2f} ms"
    return f"{s:>11}"


def _width(col):
    return 11 if col == "astropy" else 18  # backends also carry a speedup field


def _cell(col, seconds, astropy_seconds):
    """A table cell: just the time for astropy, time + speedup vs astropy else."""
    t = _fmt(seconds)
    if col == "astropy":
        return t
    if seconds is None or astropy_seconds is None:
        return f"{t:>{_width(col)}}"
    return f"{t} {astropy_seconds / seconds:5.1f}x"


def main():
    a = AstropyWCS(naxis=2)
    a.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    a.wcs.crval = [266.0, -29.0]; a.wcs.crpix = [1000.0, 1000.0]; a.wcs.cdelt = [-1e-3, 1e-3]
    w = FastWCS(naxis=2)
    w.ctype = ["RA---TAN", "DEC--TAN"]
    w.crval = [266.0, -29.0]; w.crpix = [1000.0, 1000.0]; w.cdelt = [-1e-3, 1e-3]

    backends = _load_backends()
    cols = ["astropy", "numpy"] + [b for b in ("jax-cpu", "jax-gpu", "cupy")
                                   if b in backends]

    header = f"{'N':>12} | " + " | ".join(f"{c:>{_width(c)}}" for c in cols)
    print("\n" + header)
    print("-" * len(header))

    for case in CASES:
        x_np, y_np = _inputs(case)
        reps = _repeats(case)
        row = {}

        # astropy reference (CPU, host arrays)
        row["astropy"] = _measure(lambda: a.pixel_to_world_values(x_np, y_np),
                                   lambda out: None, reps)

        # one backend's device arrays resident at a time (VRAM-friendly)
        for name in cols[1:]:
            to_dev, sync = backends[name]
            try:
                xd, yd = to_dev(x_np), to_dev(y_np)
                row[name] = _measure(lambda: w.pixel_to_world_values(xd, yd), sync, reps)
                del xd, yd
            except Exception as e:
                row[name] = None
                print(f"  ({name} @ N={case}: {type(e).__name__})")
            gc.collect()
            if name == "cupy":
                _free_cupy()

        label = "scalar" if case == "scalar" else f"{case:,}"
        print(f"{label:>12} | " + " | ".join(_cell(c, row[c], row["astropy"]) for c in cols))


if __name__ == "__main__":
    main()
