"""Reproject a synthetic image using fast_fits_wcs, and check it against astropy.wcs.

This exercises the APE 14 interface end-to-end: the ``reproject`` package drives
``pixel_to_world_values`` / ``world_to_pixel_values`` on the WCS objects we hand
it, with no knowledge that they're fast_fits_wcs rather than astropy. We reproject the
same image twice -- once with fast_fits_wcs WCS objects, once with the equivalent
astropy WCS objects -- and confirm the results agree, which they must if fast_fits_wcs
is a faithful drop-in.

Run:  python examples/reproject_example.py
"""
import numpy as np
from reproject import reproject_interp

from fast_fits_wcs import WCS as FastWCS
from astropy.wcs import WCS as AstropyWCS


def _pc(rot_deg):
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    return np.array([[c, -s], [s, c]])


def make_jax_wcs(crval, crpix, cdelt, rot_deg=0.0):
    w = FastWCS(naxis=2)
    w.ctype = ["RA---TAN", "DEC--TAN"]
    w.crval = list(crval)
    w.crpix = list(crpix)
    w.cdelt = list(cdelt)
    w.pc = _pc(rot_deg)
    return w


def make_astropy_wcs(crval, crpix, cdelt, rot_deg=0.0):
    w = AstropyWCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = list(cdelt)
    w.wcs.pc = _pc(rot_deg)
    return w


def synthetic_image(ny, nx):
    """A smooth image with structure, so interpolation differences would show."""
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    img = np.zeros((ny, nx))
    for (cy, cx, amp, sig) in [(60, 70, 5.0, 12.0), (140, 120, 3.0, 20.0)]:
        img += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))
    img += 0.01 * x + 0.005 * y  # gentle gradient
    return img


def main():
    ny, nx = 200, 200
    data = synthetic_image(ny, nx)

    # Input frame and a rotated/shifted output frame.
    in_params = dict(crval=[150.0, 2.0], crpix=[100.0, 100.0], cdelt=[-0.01, 0.01])
    out_params = dict(crval=[150.05, 2.02], crpix=[100.0, 100.0],
                      cdelt=[-0.01, 0.01], rot_deg=15.0)
    shape_out = (ny, nx)

    # --- reproject with fast_fits_wcs ------------------------------------------------
    jax_out, jax_fp = reproject_interp(
        (data, make_jax_wcs(**in_params)),
        make_jax_wcs(**out_params),
        shape_out=shape_out,
    )

    # --- reproject with astropy (reference) ----------------------------------
    ap_out, ap_fp = reproject_interp(
        (data, make_astropy_wcs(**in_params)),
        make_astropy_wcs(**out_params),
        shape_out=shape_out,
    )

    # --- compare --------------------------------------------------------------
    both_valid = (jax_fp == 1) & (ap_fp == 1)
    n_both = int(both_valid.sum())
    data_diff = np.max(np.abs(jax_out[both_valid] - ap_out[both_valid]))

    # Footprints may disagree only on a handful of edge pixels, where a ~1e-8
    # deg WCS difference flips a sample just across the input boundary.
    fp_disagree = int(np.sum(jax_fp != ap_fp))
    fp_frac = fp_disagree / jax_fp.size

    in_flux = np.nansum(data)
    jax_flux = np.nansum(jax_out)

    print(f"valid (both) pixels        : {n_both}")
    print(f"max |jax - astropy| value  : {data_diff:.3e}")
    print(f"footprint pixels disagree  : {fp_disagree} ({fp_frac:.2%})")
    print(f"input flux / jax out flux  : {in_flux:.2f} / {jax_flux:.2f}")

    assert data_diff < 1e-6, "reprojected values disagree with astropy"
    assert fp_frac < 1e-3, "footprints disagree on more than 0.1% of pixels"
    print("\nOK: reproject with fast_fits_wcs matches reproject with astropy.wcs")


if __name__ == "__main__":
    main()
