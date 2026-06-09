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
    compute_celestial_pole,
)

# Recognised celestial axis name -> coordinate-frame family. The exact
# equatorial frame (icrs/fk5/fk4) is resolved later from RADESYS/EQUINOX.
_LON_NAMES = {"RA": "equatorial", "GLON": "galactic", "ELON": "ecliptic"}
_LAT_NAMES = {"DEC": "equatorial", "GLAT": "galactic", "ELAT": "ecliptic"}

_PHYSICAL_TYPES = {
    "equatorial": ("pos.eq.ra", "pos.eq.dec"),
    "galactic": ("pos.galactic.lon", "pos.galactic.lat"),
    "ecliptic": ("pos.ecliptic.lon", "pos.ecliptic.lat"),
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
    def pixel_to_world(pix, crpix, cd, pole, phi_p):
        q = pix - _expand(crpix - 1.0, pix.ndim)          # 1-based offset from ref pixel
        m = jnp.einsum("ij,j...->i...", cd, q)            # intermediate world coords (deg)
        phi, theta = plane_to_native(m[lng], m[lat])
        alpha, delta = native_to_celestial(
            phi, theta, pole[0], pole[1], phi_p
        )
        out = [None, None]
        out[lng], out[lat] = alpha, delta
        return jnp.stack(out)

    @jax.jit
    def world_to_pixel(world, crpix, cd_inv, pole, phi_p):
        phi, theta = celestial_to_native(
            world[lng], world[lat], pole[0], pole[1], phi_p
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
        self.latpole = None  # None -> FITS default (+90)
        self.radesys = None  # equatorial frame, e.g. 'FK5'/'ICRS'; None -> infer
        self.equinox = None  # e.g. 2000.0; used to infer frame when RADESYS absent

    @classmethod
    def from_header(cls, header):
        """Build a WCS from a FITS header (an ``astropy`` Header or a dict).

        Reads the standard 2-D celestial keywords: CTYPEi, CRPIXi, CRVALi, and
        the linear transform as either a CDi_j matrix or CDELTi with PCi_j /
        CROTA2. LONPOLE is honoured if present. Only the projections this
        package supports are accepted (see ``projections``).
        """
        # Normalise to a plain uppercase-key lookup so dicts and astropy
        # Headers behave the same.
        h = {str(k).upper(): v for k, v in dict(header).items()}

        def need(key):
            if key not in h:
                raise ValueError(f"header is missing required keyword {key!r}")
            return h[key]

        naxis = int(h.get("WCSAXES", h.get("NAXIS", 2)))
        w = cls(naxis=naxis)
        w.ctype = [need(f"CTYPE{i}") for i in (1, 2)]
        w.crpix = [float(need(f"CRPIX{i}")) for i in (1, 2)]
        w.crval = [float(need(f"CRVAL{i}")) for i in (1, 2)]

        if any(f"CD{i}_{j}" in h for i in (1, 2) for j in (1, 2)):
            w.cd = [[float(h.get(f"CD{i}_{j}", 0.0)) for j in (1, 2)] for i in (1, 2)]
        else:
            cdelt = [float(h.get(f"CDELT{i}", 1.0)) for i in (1, 2)]
            if any(f"PC{i}_{j}" in h for i in (1, 2) for j in (1, 2)):
                pc = [[float(h.get(f"PC{i}_{j}", 1.0 if i == j else 0.0))
                       for j in (1, 2)] for i in (1, 2)]
                w.cdelt = cdelt
                w.pc = pc
            elif "CROTA2" in h:
                # Old AIPS-style rotation -> equivalent CD matrix.
                rho = float(h["CROTA2"]) * np.pi / 180.0
                c, s = np.cos(rho), np.sin(rho)
                w.cd = [[cdelt[0] * c, -cdelt[1] * s],
                        [cdelt[0] * s, cdelt[1] * c]]
            else:
                w.cdelt = cdelt

        if "LONPOLE" in h:
            w.lonpole = float(h["LONPOLE"])
        if "LATPOLE" in h:
            w.latpole = float(h["LATPOLE"])
        if "RADESYS" in h:
            w.radesys = str(h["RADESYS"])
        if "EQUINOX" in h:
            w.equinox = float(h["EQUINOX"])
        elif "EPOCH" in h:
            w.equinox = float(h["EPOCH"])
        return w

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
        """Pack current attributes into jnp arrays for the transform functions.

        The celestial pole ``(alpha_p, delta_p)`` and ``phi_p`` (LONPOLE) are
        computed here on the host once per call -- cheap scalar work -- so the
        jitted transforms stay projection-rotation-agnostic.
        """
        lng, lat, code, frame = self._layout()
        theta0 = projections.get(code).theta0
        alpha0, delta0 = float(self.crval[lng]), float(self.crval[lat])

        if self.lonpole is None:
            phi_p = default_lonpole(delta0, theta0)
        else:
            phi_p = float(self.lonpole)
        latpole = 90.0 if self.latpole is None else float(self.latpole)
        alpha_p, delta_p = compute_celestial_pole(alpha0, delta0, theta0, phi_p, latpole)

        crpix = jnp.asarray(self.crpix, dtype=float)
        cd = jnp.asarray(self.cd, dtype=float)
        pole = jnp.asarray([alpha_p, delta_p], dtype=float)
        phi_p = jnp.asarray(phi_p, dtype=float)
        return lng, lat, code, frame, crpix, pole, cd, phi_p

    # --- raw numeric transforms (accept/return stacked jnp arrays) -----------

    def _pixel_to_world_stacked(self, pix):
        lng, lat, code, frame, crpix, pole, cd, phi_p = self._params()
        fwd, _ = _build_transforms(code, lng, lat)
        return fwd(pix, crpix, cd, pole, phi_p)

    def _world_to_pixel_stacked(self, world):
        lng, lat, code, frame, crpix, pole, cd, phi_p = self._params()
        _, inv = _build_transforms(code, lng, lat)
        cd_inv = jnp.linalg.inv(cd)
        return inv(world, crpix, cd_inv, pole, phi_p)

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

    def _frame_kwargs(self, family):
        """SkyCoord frame kwargs for a coordinate-frame family. The equatorial
        frame is resolved from RADESYS/EQUINOX following FITS conventions."""
        if family == "galactic":
            return {"frame": "galactic"}
        if family == "ecliptic":
            return {"frame": "barycentrictrueecliptic"}
        # equatorial
        if self.radesys is not None:
            name = self.radesys.strip().lower()
        elif self.equinox is not None:
            name = "fk5" if float(self.equinox) >= 1984.0 else "fk4"
        else:
            name = "icrs"
        kwargs = {"frame": name}
        if name in ("fk5", "fk4") and self.equinox is not None:
            prefix = "J" if name == "fk5" else "B"
            kwargs["equinox"] = f"{prefix}{float(self.equinox)}"
        return kwargs

    @property
    def world_axis_object_classes(self):
        from astropy.coordinates import SkyCoord

        lng, lat, code, frame = self._layout()
        return {
            "celestial": (
                SkyCoord,
                (),
                {"unit": "deg", **self._frame_kwargs(frame)},
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
