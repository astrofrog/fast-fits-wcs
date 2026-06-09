"""Reproject the MSX galactic-center image (GLON-CAR) into the 2MASS frame
(RA---TAN), with astropy.wcs and with fast_fits_wcs, and compare the results.

Run:  python examples/reproject_example.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write a PNG
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import PercentileInterval
from astropy.wcs import WCS as AstropyWCS
from reproject import reproject_interp

from fast_fits_wcs import WCS as FastWCS

OUT_PNG = __file__.replace(".py", ".png")

twomass = fits.open(get_pkg_data_filename("galactic_center/gc_2mass_k.fits"))[0]
msx = fits.open(get_pkg_data_filename("galactic_center/gc_msx_e.fits"))[0]
shape_out = twomass.data.shape


def reproject_with(make_wcs):
    out, _ = reproject_interp((msx.data, make_wcs(msx.header)),
                              make_wcs(twomass.header), shape_out=shape_out)
    return out


ref = reproject_with(AstropyWCS)            # reference: astropy.wcs
act = reproject_with(FastWCS.from_header)   # this package

# Relative difference, restricted to pixels with real signal.
m = np.isfinite(ref) & np.isfinite(act) & (np.abs(ref) > 0.01 * np.nanmax(np.abs(ref)))
ratio = (act[m] - ref[m]) / ref[m]
print(f"max |(act - ref) / ref| = {np.abs(ratio).max():.2e}  ({m.sum()} pixels)")

twcs = AstropyWCS(twomass.header)
fig = plt.figure(figsize=(13, 4))
for i, (title, data) in enumerate([("2MASS K (original)", twomass.data),
                                   ("MSX (astropy.wcs)", ref),
                                   ("MSX (fast_fits_wcs)", act)], start=1):
    ax = fig.add_subplot(1, 4, i, projection=twcs)
    lo, hi = PercentileInterval(99.0).get_limits(data[np.isfinite(data)])
    ax.imshow(data, origin="lower", vmin=lo, vmax=hi, cmap="afmhot")
    ax.set_title(title, fontsize=9)
    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")

ax = fig.add_subplot(1, 4, 4)
ax.hist(ratio, bins=100)
ax.set_yscale("log")
ax.set_title("(fast - astropy) / astropy", fontsize=9)
ax.set_xlabel("relative difference")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110)
print("wrote", OUT_PNG)
