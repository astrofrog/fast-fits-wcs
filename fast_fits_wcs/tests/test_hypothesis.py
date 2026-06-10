"""Property-based tests of fast_fits_wcs against astropy.wcs (Hypothesis).

For random WCS (every supported projection, both axis orders, random
crval/crpix/cdelt/rotation, equatorial and galactic) and random pixels, the
pixel->world and world->pixel transforms must agree with astropy.wcs. Run with
``pytest benchmarks/test_hypothesis.py`` or directly as a script.
"""
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from astropy.wcs import WCS as AstropyWCS
from fast_fits_wcs import WCS as FastWCS

CODES = ["TAN", "STG", "SIN", "ARC", "ZEA", "CAR"]
NAMES = {"eq": ("RA", "DEC"), "gal": ("GLON", "GLAT")}

# Forward (pixel->world) agrees to ~5e-10 arcsec. The inverse (world->pixel)
# multiplies world error by inv(CD) ~ 1/cdelt (up to 1e4 here), so its pixel
# precision is looser by that factor (~1e-5 px observed) -- still far below
# anything a real bug would produce.
TOL_ARCSEC = 1e-7
TOL_PIXEL = 1e-3


def _floats(lo, hi):
    return st.floats(min_value=lo, max_value=hi, allow_nan=False,
                     allow_infinity=False, width=64)


@st.composite
def wcs_and_pixels(draw):
    code = draw(st.sampled_from(CODES))
    lon_name, lat_name = NAMES[draw(st.sampled_from(list(NAMES)))]
    swap = draw(st.booleans())  # put latitude on the first axis

    crval_lon = draw(_floats(0.0, 359.999))
    crval_lat = draw(_floats(-80.0, 80.0))           # stay away from the poles
    crpix = [draw(_floats(-200.0, 2000.0)), draw(_floats(-200.0, 2000.0))]
    mag = draw(_floats(1e-4, 5e-3))                   # deg/pixel
    cdelt = [draw(st.sampled_from([-1.0, 1.0])) * mag,
             draw(st.sampled_from([-1.0, 1.0])) * mag]
    rot = draw(_floats(0.0, 359.999))

    # Build CTYPE in the canonical 8-char form, e.g. 'RA---TAN', 'GLON-TAN'.
    lon_ct = lon_name + "-" * (8 - len(code) - len(lon_name)) + code
    lat_ct = lat_name + "-" * (8 - len(code) - len(lat_name)) + code

    if swap:
        ctype = [lat_ct, lon_ct]
        crval = [crval_lat, crval_lon]
        lng, lat = 1, 0
    else:
        ctype = [lon_ct, lat_ct]
        crval = [crval_lon, crval_lat]
        lng, lat = 0, 1

    # a handful of pixels within a bounded field (mag * offset <= ~7.5 deg)
    n = draw(st.integers(min_value=1, max_value=8))
    offs = draw(st.lists(st.tuples(_floats(-1500.0, 1500.0), _floats(-1500.0, 1500.0)),
                         min_size=n, max_size=n))
    px = np.array([crpix[0] + dx for dx, _ in offs])
    py = np.array([crpix[1] + dy for _, dy in offs])

    return dict(ctype=ctype, crval=crval, crpix=crpix, cdelt=cdelt, rot=rot,
                lng=lng, lat=lat, px=px, py=py)


def _pc(rot):
    c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    return np.array([[c, -s], [s, c]])


def _build(p):
    a = AstropyWCS(naxis=2)
    a.wcs.ctype = p["ctype"]; a.wcs.crval = p["crval"]; a.wcs.crpix = p["crpix"]
    a.wcs.cdelt = p["cdelt"]; a.wcs.pc = _pc(p["rot"])
    f = FastWCS(naxis=2)
    f.ctype = list(p["ctype"]); f.crval = list(p["crval"]); f.crpix = list(p["crpix"])
    f.cdelt = list(p["cdelt"]); f.pc = _pc(p["rot"])
    return a, f


def _angsep_arcsec(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))) * 3600.0


@settings(max_examples=400, deadline=None)
@given(p=wcs_and_pixels())
def test_pixel_to_world_matches_astropy(p):
    a, f = _build(p)
    lng, lat = p["lng"], p["lat"]
    aw = a.wcs_pix2world(p["px"], p["py"], 0)
    assume(np.all(np.isfinite(aw[0])) and np.all(np.isfinite(aw[1])))
    fw = f.pixel_to_world_values(p["px"], p["py"])
    sep = _angsep_arcsec(np.asarray(aw[lng]), np.asarray(aw[lat]),
                         np.asarray(fw[lng]), np.asarray(fw[lat]))
    assert np.max(sep) < TOL_ARCSEC, f"{np.max(sep):.3e} arcsec for {p['ctype']}"


@settings(max_examples=400, deadline=None)
@given(p=wcs_and_pixels())
def test_world_to_pixel_matches_astropy(p):
    a, f = _build(p)
    aw = a.wcs_pix2world(p["px"], p["py"], 0)
    assume(np.all(np.isfinite(aw[0])) and np.all(np.isfinite(aw[1])))
    ap = a.wcs_world2pix(aw[0], aw[1], 0)
    fp = f.world_to_pixel_values(aw[0], aw[1])
    err = max(np.max(np.abs(np.asarray(fp[0]) - ap[0])),
              np.max(np.abs(np.asarray(fp[1]) - ap[1])))
    assert err < TOL_PIXEL, f"{err:.3e} px for {p['ctype']}"


if __name__ == "__main__":
    test_pixel_to_world_matches_astropy()
    test_world_to_pixel_matches_astropy()
    print("OK: fast_fits_wcs matches astropy.wcs across random WCS (hypothesis)")
