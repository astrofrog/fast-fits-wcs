"""Reproject the MSX galactic-center image into the 2MASS frame, twice.

This reproduces the classic ``reproject`` getting-started example, but does the
reprojection two ways and compares them:

* the input MSX image is ``GLON-CAR`` (galactic, cylindrical),
* the output 2MASS frame is ``RA---TAN`` (equatorial, gnomonic),

so it exercises both a cylindrical and a zenithal WCS, in different coordinate
frames, entirely through the APE 14 interface. We build the WCS objects once
with ``astropy.wcs.WCS`` and once with ``fast_fits_wcs.WCS.from_header``, feed
each pair to ``reproject_interp``, and confirm the results agree. A WCSAxes
figure shows the original, both reprojections, and their difference.

Run:  python examples/reproject_example.py
Data: the MSX/2MASS galactic-center cutouts bundled as astropy example data.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, don't open a window
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import PercentileInterval
from astropy.wcs import WCS as AstropyWCS
from reproject import reproject_interp

from fast_fits_wcs import WCS as FastWCS

OUT_PNG = __file__.replace(".py", ".png")


def load():
    twomass = fits.open(get_pkg_data_filename("galactic_center/gc_2mass_k.fits"))[0]
    msx = fits.open(get_pkg_data_filename("galactic_center/gc_msx_e.fits"))[0]
    return twomass, msx


def main():
    twomass, msx = load()
    shape_out = twomass.data.shape

    # Reproject MSX (GLON-CAR) onto the 2MASS (RA---TAN) frame, each WCS built
    # by both libraries.
    msx_astropy, _ = reproject_interp(
        (msx.data, AstropyWCS(msx.header)),
        AstropyWCS(twomass.header),
        shape_out=shape_out,
    )
    msx_fast, _ = reproject_interp(
        (msx.data, FastWCS.from_header(msx.header)),
        FastWCS.from_header(twomass.header),
        shape_out=shape_out,
    )

    # --- compare --------------------------------------------------------------
    valid = np.isfinite(msx_astropy) & np.isfinite(msx_fast)
    max_abs = np.max(np.abs(msx_astropy[valid] - msx_fast[valid]))
    scale = np.nanpercentile(np.abs(msx_astropy[valid]), 99)
    print(f"reprojected pixels (finite both): {int(valid.sum())}")
    print(f"max |astropy - fast_fits_wcs|   : {max_abs:.3e}")
    print(f"  (relative to 99th pct {scale:.3e}): {max_abs / scale:.2e}")
    assert max_abs / scale < 1e-9, "reprojections disagree"

    # --- plot (WCSAxes, on the 2MASS frame) ----------------------------------
    twcs = AstropyWCS(twomass.header)
    interval = PercentileInterval(99.0)
    vmin, vmax = interval.get_limits(twomass.data)
    mvmin, mvmax = interval.get_limits(msx_astropy[np.isfinite(msx_astropy)])

    fig = plt.figure(figsize=(16, 4.2))
    panels = [
        ("2MASS K (original)", twomass.data, vmin, vmax, "afmhot"),
        ("MSX reprojected (astropy.wcs)", msx_astropy, mvmin, mvmax, "afmhot"),
        ("MSX reprojected (fast_fits_wcs)", msx_fast, mvmin, mvmax, "afmhot"),
        ("difference (fast - astropy)", msx_fast - msx_astropy, -1e-12, 1e-12, "RdBu"),
    ]
    for i, (title, data, lo, hi, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, i, projection=twcs)
        ax.imshow(data, origin="lower", vmin=lo, vmax=hi, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.coords[0].set_axislabel("RA")
        ax.coords[1].set_axislabel("Dec")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"\nwrote {OUT_PNG}")
    print("OK: fast_fits_wcs reprojection matches astropy.wcs (CAR -> TAN, galactic -> equatorial)")


if __name__ == "__main__":
    main()
