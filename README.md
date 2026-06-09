# jaxwcs

A minimal, [APE 14](https://github.com/astropy/astropy-APEs/blob/main/APE14.rst)-compliant
celestial FITS WCS whose numeric core is pure [JAX](https://jax.readthedocs.io),
so the pixel↔world transform is `jit`-able, `vmap`-able, `grad`-able, and runs
on CPU/GPU/TPU. It is meant as a drop-in alternative to `astropy.wcs.WCS` for the
subset of cases it covers (currently TAN; designed to extend to the other
zenithal/cylindrical projections).

## Usage

Parameters are plain attributes, mirroring `astropy.wcs.WCS().wcs`:

```python
from jaxwcs import WCS

w = WCS(naxis=2)
w.ctype = 'RA---TAN', 'DEC--TAN'
w.crpix = [128.0, 128.0]
w.crval = [10.0, 20.0]
w.cdelt = [-0.001, 0.001]
w.pc    = [[1, 0], [0, 1]]          # optional (identity default); or set w.cd

# APE 14 low-level (pure numeric, no astropy needed)
ra, dec = w.pixel_to_world_values([130.0, 200.0], [132.0, 50.0])
x, y    = w.world_to_pixel_values(ra, dec)

# APE 14 high-level (needs astropy) -> returns a SkyCoord
sky = w.pixel_to_world(130.0, 132.0)
```

`w` subclasses astropy's `BaseLowLevelWCS`/`HighLevelWCSMixin`, so it is a real
APE 14 object: pass it anywhere that consumes the interface (WCSAxes, reproject,
NDData, …).

## Design

The pipeline is the standard FITS one (Calabretta & Greisen 2002), split into
three projection-agnostic stages plus one projection-specific stage:

1. ref-pixel offset + linear `CD` matrix → intermediate world coords (`core.py`)
2. **projection**: plane ↔ native spherical — the only per-projection code (`projections.py`)
3. native ↔ celestial spherical rotation from `CRVAL`/`LONPOLE` (`celestial.py`)

Only stage 2 is projection-specific. To add a projection, write two JAX
functions and register them by FITS code — see `_tan_plane_to_native` /
`_tan_native_to_plane` in `projections.py` and the `register(...)` call beside
them. Everything else (`core.py`, `celestial.py`) is reused unchanged.

The WCS parameters are passed to the jitted transforms as *arguments*, not
closed over, so editing `CRVAL`/`CRPIX`/`CDELT` never recompiles — only changing
the projection or the input array shape does.

### Precision

WCS needs 64-bit: in float32 the forward transform is only ~0.02″ accurate and
the world→pixel inverse degrades to tens of pixels. `jaxwcs` therefore enables
`jax_enable_x64` on import (set `JAXWCS_X64=0` to opt out). With x64 it agrees
with `astropy.wcs` to ~1e-8 arcsec — verified over random WCS/points in
`benchmarks/test_correctness.py`.

## Performance vs astropy.wcs

`pixel_to_world` on CPU (16-core box, single XLA thread; see
`benchmarks/performance.py`):

| N points | astropy | jaxwcs (full API) | jaxwcs (device core) | speedup (core) |
|---------:|--------:|------------------:|---------------------:|---------------:|
| 1        | 0.012 ms | 0.72 ms | 0.011 ms | ~1x |
| 100      | 0.021 ms | 0.72 ms | 0.028 ms | 0.8x |
| 10⁴      | 0.99 ms | 1.1 ms | 0.32 ms | ~3x |
| 10⁶      | 118 ms | 27 ms | 19 ms | ~6x |
| 10⁷      | 1210 ms | 323 ms | 213 ms | ~6x |

Reading: `astropy`'s wcslib is hard to beat for tiny inputs — there is a fixed
~0.7 ms JAX dispatch/host-transfer cost per call that dominates the **full API**
column below ~10⁴ points. The **device core** column (inputs already on-device,
output left on-device — the realistic case when WCS is one node in a larger JAX
graph) wins from ~10⁴ points up, reaching ~6× at 10⁶–10⁷ on CPU purely from XLA
vectorization/fusion. On GPU/TPU the large-N gap should widen substantially.

The point of `jaxwcs` is less raw CPU speed than that the transform is
**differentiable and composable**: `jax.grad` through the full TAN pipeline
(e.g. ∂RA/∂CRVAL for WCS fitting/calibration) and `jax.vmap` over a stack of
WCS both work — see the demo at the bottom of this repo's history.

## Status / limitations

- TAN only so far; 2-D celestial only; no SIP/distortions; `LONPOLE` honored,
  `LATPOLE` ignored (not needed for zenithal). All of these are natural
  extensions of the structure above.
