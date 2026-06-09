"""Check jaxwcs against astropy.wcs over random WCS and random points."""
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from jaxwcs import WCS as JaxWCS

rng = np.random.default_rng(0)


def make_pair(crval, crpix, cdelt, rot_deg, ctype=("RA---TAN", "DEC--TAN")):
    ca, sa = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    pc = np.array([[ca, -sa], [sa, ca]])

    a = AstropyWCS(naxis=2)
    a.wcs.ctype = list(ctype)
    a.wcs.crval = crval
    a.wcs.crpix = crpix
    a.wcs.cdelt = cdelt
    a.wcs.pc = pc

    j = JaxWCS(naxis=2)
    j.ctype = list(ctype)
    j.crval = list(crval)
    j.crpix = list(crpix)
    j.cdelt = list(cdelt)
    j.pc = pc
    return a, j


cases = [
    (dict(crval=[10.0, 20.0], crpix=[128.0, 128.0], cdelt=[-0.001, 0.001], rot_deg=0.0)),
    (dict(crval=[266.4, -29.0], crpix=[512.5, 512.5], cdelt=[-2e-4, 2e-4], rot_deg=23.5)),
    (dict(crval=[0.0, 89.5], crpix=[50.0, 50.0], cdelt=[-0.01, 0.01], rot_deg=0.0)),
    (dict(crval=[123.0, -89.7], crpix=[1.0, 1.0], cdelt=[-0.05, 0.05], rot_deg=-40.0)),
    (dict(crval=[200.0, 0.0], crpix=[256.0, 256.0], cdelt=[-3e-3, 3e-3], rot_deg=90.0)),
    (dict(crval=[200.0, 0.0], crpix=[256.0, 256.0], cdelt=[-3e-3, 3e-3], rot_deg=90.0,
          ctype=("GLON-TAN", "GLAT-TAN"))),
    # swapped axis order: DEC first
    (dict(crval=[20.0, 10.0], crpix=[128.0, 128.0], cdelt=[0.001, -0.001], rot_deg=15.0,
          ctype=("DEC--TAN", "RA---TAN"))),
]

max_world_err = 0.0
max_pix_err = 0.0

for params in cases:
    a, j = make_pair(**params)

    px = rng.uniform(1, 256, size=2000)
    py = rng.uniform(1, 256, size=2000)

    aw0, aw1 = a.wcs_pix2world(px, py, 0)
    jw0, jw1 = j.pixel_to_world_values(px, py)

    # angular separation on the sky (robust to the lon/lat axis order)
    lng = 0 if params.get("ctype", ("RA---TAN",))[0].split("-")[0] in ("RA", "GLON") else 1
    lat = 1 - lng
    aw = [aw0, aw1]
    jw = [jw0, jw1]
    dlon = np.deg2rad(aw[lng] - jw[lng])
    lat_r = np.deg2rad(aw[lat])
    sep = np.hypot(
        np.cos(lat_r) * dlon,
        np.deg2rad(aw[lat] - jw[lat]),
    )
    world_err = np.max(np.abs(np.rad2deg(sep))) * 3600.0  # arcsec
    max_world_err = max(max_world_err, world_err)

    # round-trip world->pixel
    ap0, ap1 = a.wcs_world2pix(aw0, aw1, 0)
    jp0, jp1 = j.world_to_pixel_values(jw0, jw1)
    pix_err = max(np.max(np.abs(ap0 - jp0)), np.max(np.abs(ap1 - jp1)))
    max_pix_err = max(max_pix_err, pix_err)

    print(f"ctype={params.get('ctype', ('RA---TAN','DEC--TAN'))} "
          f"world_err={world_err:.3e} arcsec  pix_err={pix_err:.3e} px")

print()
print(f"MAX world error: {max_world_err:.3e} arcsec")
print(f"MAX pixel error: {max_pix_err:.3e} px")
assert max_world_err < 1e-6, "world values disagree with astropy"
assert max_pix_err < 1e-6, "pixel values disagree with astropy"
print("OK: matches astropy.wcs")
