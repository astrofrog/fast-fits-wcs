"""A minimal, JAX-optimized, APE 14-compliant celestial FITS WCS.

Parameters are plain attributes, mirroring ``astropy.wcs.WCS().wcs``::

    w = WCS(naxis=2)
    w.ctype = ['RA---TAN', 'DEC--TAN']
    w.crpix = [128.0, 128.0]
    w.crval = [10.0, 20.0]
    w.cdelt = [-0.001, 0.001]
    w.pc    = [[1, 0], [0, 1]]      # optional; identity by default

The numeric core is a pair of pure JAX functions built lazily from the
projection (chosen by CTYPE) and cached. The WCS parameters are passed to those
functions as *arguments*, so changing CRVAL/CRPIX/etc. never triggers a
recompile -- only changing the projection or the input array shape does.
"""

from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp

from . import projections
from .celestial import (
    native_to_celestial,
    celestial_to_native,
    default_lonpole,
)

# Recognised celestial axis name -> (role, frame). Extend as needed.
_LON_NAMES = {"RA": "icrs", "GLON": "galactic", "ELON": "barycentrictrueecliptic"}
_LAT_NAMES = {"DEC": "icrs", "GLAT": "galactic", "ELAT": "barycentrictrueecliptic"}

_PHYSICAL_TYPES = {
    "icrs": ("pos.eq.ra", "pos.eq.dec"),
    "galactic": ("pos.galactic.lon", "pos.galactic.lat"),
    "barycentrictrueecliptic": ("pos.ecliptic.lon", "pos.ecliptic.lat"),
}


def _parse_ctype(ctype):
    """('RA---TAN', 'DEC--TAN') -> (lng_index, lat_index, proj_code, frame)."""
    roles = []  # (axis_index, 'lon'|'lat', frame, code)
    for i, ct in enumerate(ctype):
        # CTYPE is 'NAME' then '-' fill then a 3-char projection code,
        # e.g. 'RA---TAN' or 'GLON-TAN'.
        body = ct.replace("-", " ").split()
        if len(body) != 2:
            continue
        axis_name, code = body[0], body[1]
        if axis_name in _LON_NAMES:
            roles.append((i, "lon", _LON_NAMES[axis_name], code))
        elif axis_name in _LAT_NAMES:
            roles.append((i, "lat", _LAT_NAMES[axis_name], code))
    if len(roles) != 2:
        raise ValueError(f"Expected one lon and one lat celestial axis, got {ctype!r}")
    by_role = {r[1]: r for r in roles}
    if "lon" not in by_role or "lat" not in by_role:
        raise ValueError(f"CTYPE must contain a lon and a lat axis: {ctype!r}")
    lng, lat = by_role["lon"][0], by_role["lat"][0]
    code = by_role["lon"][3]
    if by_role["lat"][3] != code:
        raise ValueError("lon and lat axes must use the same projection")
    frame = by_role["lon"][2]
    return lng, lat, code, frame


@lru_cache(maxsize=None)
def _build_transforms(code, lng, lat):
    """Return jitted (pixel_to_world, world_to_pixel) for a projection layout.

    Signatures (all arrays are jnp):
        pixel_to_world(pix, crpix, cd, crval, phi_p) -> world
        world_to_pixel(world, crpix, cd_inv, crval, phi_p) -> pix
    where ``pix``/``world`` are stacked on axis 0: shape (2, *S).
    """
    proj = projections.get(code)
    plane_to_native = proj.plane_to_native
    native_to_plane = proj.native_to_plane

    def _expand(vec, ndim):
        # (2,) -> (2, 1, 1, ...) to broadcast against a (2, *S) stack
        return vec.reshape((2,) + (1,) * (ndim - 1))

    @jax.jit
    def pixel_to_world(pix, crpix, cd, crval, phi_p):
        q = pix - _expand(crpix - 1.0, pix.ndim)          # 1-based offset from ref pixel
        m = jnp.einsum("ij,j...->i...", cd, q)            # intermediate world coords (deg)
        phi, theta = plane_to_native(m[lng], m[lat])
        alpha, delta = native_to_celestial(
            phi, theta, crval[lng], crval[lat], phi_p
        )
        out = [None, None]
        out[lng], out[lat] = alpha, delta
        return jnp.stack(out)

    @jax.jit
    def world_to_pixel(world, crpix, cd_inv, crval, phi_p):
        phi, theta = celestial_to_native(
            world[lng], world[lat], crval[lng], crval[lat], phi_p
        )
        x, y = native_to_plane(phi, theta)
        m = [None, None]
        m[lng], m[lat] = x, y
        m = jnp.stack(m)
        q = jnp.einsum("ij,j...->i...", cd_inv, m)
        return q + _expand(crpix - 1.0, world.ndim)

    return pixel_to_world, world_to_pixel


try:
    from astropy.wcs.wcsapi import BaseLowLevelWCS, HighLevelWCSMixin
    _BASES = (BaseLowLevelWCS, HighLevelWCSMixin)
except Exception:  # astropy is optional for the low-level numeric API
    _BASES = (object,)


class WCS(*_BASES):
    """A celestial FITS WCS with JAX-backed transforms and the APE 14 API.

    Subclasses astropy's ``BaseLowLevelWCS``/``HighLevelWCSMixin`` when astropy
    is available, so ``wcs.pixel_to_world(...)`` returns a ``SkyCoord`` and the
    object is a drop-in for anything that consumes the APE 14 interface.
    """

    def __init__(self, naxis=2):
        if naxis != 2:
            raise NotImplementedError("Only 2-D celestial WCS is supported")
        self.naxis = naxis
        self.ctype = ["RA---TAN", "DEC--TAN"]
        self.crpix = [0.0, 0.0]
        self.crval = [0.0, 0.0]
        self.cdelt = [1.0, 1.0]
        self.pc = np.eye(2)
        self._cd = None  # if set, overrides cdelt @ pc
        self.lonpole = None  # None -> FITS default

    # --- parameter handling --------------------------------------------------

    @property
    def cd(self):
        if self._cd is not None:
            return np.asarray(self._cd, dtype=float)
        return np.asarray(self.cdelt, dtype=float)[:, None] * np.asarray(self.pc, dtype=float)

    @cd.setter
    def cd(self, value):
        self._cd = np.asarray(value, dtype=float)

    def _layout(self):
        return _parse_ctype(tuple(self.ctype))

    def _params(self):
        """Pack current attributes into jnp arrays for the transform functions."""
        lng, lat, code, frame = self._layout()
        crpix = jnp.asarray(self.crpix, dtype=float)
        crval = jnp.asarray(self.crval, dtype=float)
        cd = jnp.asarray(self.cd, dtype=float)
        if self.lonpole is None:
            phi_p = default_lonpole(float(self.crval[lat]), projections.get(code).theta0)
        else:
            phi_p = float(self.lonpole)
        phi_p = jnp.asarray(phi_p, dtype=float)
        return lng, lat, code, frame, crpix, crval, cd, phi_p

    # --- raw numeric transforms (accept/return stacked jnp arrays) -----------

    def _pixel_to_world_stacked(self, pix):
        lng, lat, code, frame, crpix, crval, cd, phi_p = self._params()
        fwd, _ = _build_transforms(code, lng, lat)
        return fwd(pix, crpix, cd, crval, phi_p)

    def _world_to_pixel_stacked(self, world):
        lng, lat, code, frame, crpix, crval, cd, phi_p = self._params()
        _, inv = _build_transforms(code, lng, lat)
        cd_inv = jnp.linalg.inv(cd)
        return inv(world, crpix, cd_inv, crval, phi_p)

    # --- APE 14: BaseLowLevelWCS -------------------------------------------

    @property
    def pixel_n_dim(self):
        return 2

    @property
    def world_n_dim(self):
        return 2

    @property
    def world_axis_physical_types(self):
        lng, lat, code, frame = self._layout()
        lon_t, lat_t = _PHYSICAL_TYPES[frame]
        out = [None, None]
        out[lng], out[lat] = lon_t, lat_t
        return out

    @property
    def world_axis_units(self):
        return ["deg", "deg"]

    @property
    def pixel_axis_names(self):
        return ["", ""]

    @property
    def world_axis_names(self):
        return list(self.ctype)

    def pixel_to_world_values(self, *pixel_arrays):
        pix = jnp.stack([jnp.asarray(a, dtype=float) for a in pixel_arrays])
        world = self._pixel_to_world_stacked(pix)
        return tuple(np.asarray(world[i]) for i in range(2))

    def world_to_pixel_values(self, *world_arrays):
        world = jnp.stack([jnp.asarray(a, dtype=float) for a in world_arrays])
        pix = self._world_to_pixel_stacked(world)
        return tuple(np.asarray(pix[i]) for i in range(2))

    @property
    def world_axis_object_components(self):
        lng, lat, code, frame = self._layout()
        out = [None, None]
        out[lng] = ("celestial", 0, "spherical.lon.degree")
        out[lat] = ("celestial", 1, "spherical.lat.degree")
        return out

    @property
    def world_axis_object_classes(self):
        from astropy.coordinates import SkyCoord

        lng, lat, code, frame = self._layout()
        return {
            "celestial": (
                SkyCoord,
                (),
                {"frame": frame, "unit": "deg"},
            )
        }

    @property
    def pixel_shape(self):
        return None

    @property
    def pixel_bounds(self):
        return None

    @property
    def axis_correlation_matrix(self):
        return np.ones((2, 2), dtype=bool)

    def __repr__(self):
        return (
            f"<fast_fits_wcs.WCS ctype={self.ctype} crval={self.crval} "
            f"crpix={self.crpix} cdelt={self.cdelt}>"
        )
