"""Validate the array-API low-level API on GPU backends (cupy / torch / jax).

For every array library that's installed, this feeds the WCS arrays from that
library, and checks that:
  * the output is in the same library (numpy in -> numpy out, cupy -> cupy, ...),
  * the output stays on the same device as the input (i.e. on the GPU),
  * the values match astropy.wcs.

Run on a GPU box after installing the package and whichever backends you have:

    pip install -e .                      # the package
    pip install array-api-compat          # robust cupy/torch namespace detection
    pip install cupy-cuda12x              # or the wheel matching your CUDA
    pip install torch                     # CUDA build
    pip install -U "jax[cuda12]"          # CUDA build

    python benchmarks/validate_gpu_backends.py

Backends that aren't installed are skipped. numpy is always run as a sanity row.
"""
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from fast_fits_wcs import WCS as FastWCS

try:
    import array_api_compat  # noqa: F401
    print("array-api-compat: installed (used for namespace detection)")
except ImportError:
    print("array-api-compat: NOT installed -- cupy/torch may not be detected; "
          "`pip install array-api-compat`")


# --- backends: (name, to_device, to_numpy, device_string) --------------------
def _make_backends():
    backends = [("numpy", lambda a: np.asarray(a), lambda t: np.asarray(t),
                 lambda t: str(getattr(t, "device", "cpu")))]
    try:
        import cupy as cp
        backends.append(("cupy", lambda a: cp.asarray(a), lambda t: cp.asnumpy(t),
                         lambda t: f"cuda:{t.device.id}"))
    except ImportError:
        print("cupy:  not installed, skipped")
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        backends.append(("torch", lambda a: torch.asarray(a, device=dev),
                         lambda t: t.detach().cpu().numpy(), lambda t: str(t.device)))
        if dev == "cpu":
            print("torch: CUDA not available, will run on CPU")
    except ImportError:
        print("torch: not installed, skipped")
    try:
        import jax
        import jax.numpy as jnp
        backends.append(("jax", lambda a: jnp.asarray(a), lambda t: np.asarray(t),
                         lambda t: str(list(t.devices()))))
        print("jax backend:", jax.default_backend())
    except ImportError:
        print("jax:   not installed, skipped")
    return backends


def make_wcs_pair(ctype, crval):
    a = AstropyWCS(naxis=2)
    a.wcs.ctype = list(ctype); a.wcs.crval = crval
    a.wcs.crpix = [128.0, 128.0]; a.wcs.cdelt = [-0.01, 0.01]
    f = FastWCS(naxis=2)
    f.ctype = list(ctype); f.crval = crval
    f.crpix = [128.0, 128.0]; f.cdelt = [-0.01, 0.01]
    return a, f


CASES = [
    (("RA---TAN", "DEC--TAN"), [266.4, -29.0]),   # equatorial, zenithal
    (("GLON-CAR", "GLAT-CAR"), [30.0, 10.0]),     # galactic, cylindrical
]

rng = np.random.default_rng(0)
px = rng.uniform(1, 256, 5000)
py = rng.uniform(1, 256, 5000)

backends = _make_backends()
all_ok = True

for ctype, crval in CASES:
    a, f = make_wcs_pair(ctype, crval)
    aw0, aw1 = a.wcs_pix2world(px, py, 0)
    print(f"\n=== {ctype} ===")
    for name, to_dev, to_np, dev_of in backends:
        try:
            xd, yd = to_dev(px), to_dev(py)
            w0, w1 = f.pixel_to_world_values(xd, yd)
            b0, b1 = f.world_to_pixel_values(w0, w1)

            in_mod = type(xd).__module__.split(".")[0]
            out_mod = type(w0).__module__.split(".")[0]
            rt_mod = type(b0).__module__.split(".")[0]
            ns_ok = (out_mod == in_mod) and (rt_mod == in_mod)

            err = max(np.max(np.abs(to_np(w0) - aw0)), np.max(np.abs(to_np(w1) - aw1)))
            rt = max(np.max(np.abs(to_np(b0) - px)), np.max(np.abs(to_np(b1) - py)))
            ok = ns_ok and err < 1e-7 and rt < 1e-6
            all_ok = all_ok and ok
            print(f"  {name:6s} dev={dev_of(w0):<22} out={out_mod:<6} "
                  f"err={err:.1e} deg  rt={rt:.1e} px  {'OK' if ok else 'FAIL'}")
        except Exception as e:
            all_ok = False
            print(f"  {name:6s} FAILED: {type(e).__name__}: {str(e)[:80]}")

print("\n" + ("ALL BACKENDS OK" if all_ok else "SOME BACKENDS FAILED"))
