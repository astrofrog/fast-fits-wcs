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


def _expand(xp, vec, ndim):
    # (N,) -> (N, 1, 1, ...) to broadcast against an (N, *S) stack
    return xp.reshape(vec, (vec.shape[0],) + (1,) * (ndim - 1))


def _apply_matrix(xp, mat, q):
    # mat @ q over the leading axis: (N, N) applied to (N, *S) -> (N, *S).
    # Reshape to 2-D so we use only standard-array-API matmul (no einsum).
    n = q.shape[0]
    trailing = q.shape[1:]
    q2 = xp.reshape(q, (n, -1))
    m2 = xp.matmul(mat, q2)
    return xp.reshape(m2, (n,) + trailing)


@lru_cache(maxsize=None)
def _make_transforms(code, lng, lat):
    """Backend-agnostic (pixel_to_world, world_to_pixel) for a projection layout.

    Both take the array namespace ``xp`` first; ``pix``/``world`` are stacked on
    axis 0 with shape ``(2, *S)`` and the WCS parameters are arrays in ``xp``::

        pixel_to_world(xp, pix,   crpix, cd,     pole, phi_p) -> world
        world_to_pixel(xp, world, crpix, cd_inv, pole, phi_p) -> pix
    """
    proj = projections.get(code)
    plane_to_native = proj.plane_to_native
    native_to_plane = proj.native_to_plane

    def pixel_to_world(xp, pix, crpix, cd, pole, phi_p):
        q = pix - _expand(xp, crpix - 1.0, pix.ndim)      # 1-based offset from ref pixel
        m = _apply_matrix(xp, cd, q)                      # intermediate world coords (deg)
        phi, theta = plane_to_native(xp, m[lng], m[lat])
        alpha, delta = native_to_celestial(xp, phi, theta, pole[0], pole[1], phi_p)
        out = [None, None]
        out[lng], out[lat] = alpha, delta
        return xp.stack(out)

    def world_to_pixel(xp, world, crpix, cd_inv, pole, phi_p):
        phi, theta = celestial_to_native(xp, world[lng], world[lat], pole[0], pole[1], phi_p)
        x, y = native_to_plane(xp, phi, theta)
        m = [None, None]
        m[lng], m[lat] = x, y
        m = xp.stack(m)
        q = _apply_matrix(xp, cd_inv, m)
        return q + _expand(xp, crpix - 1.0, world.ndim)

    return pixel_to_world, world_to_pixel


@lru_cache(maxsize=None)
def _jit_transforms(code, lng, lat):
    """jax-jitted wrappers of the generic transforms, for jax-array inputs."""
    import jax
    import jax.numpy as jnp

    p2w, w2p = _make_transforms(code, lng, lat)
    fwd = jax.jit(lambda pix, crpix, cd, pole, phi_p: p2w(jnp, pix, crpix, cd, pole, phi_p))
    inv = jax.jit(lambda world, crpix, cd_inv, pole, phi_p: w2p(jnp, world, crpix, cd_inv, pole, phi_p))
    return fwd, inv


def _array_namespace(x):
    """Array-API namespace for ``x``; numpy for lists / Python scalars.

    Prefers ``array-api-compat`` when installed (robust cupy/torch/jax support),
    falling back to the array's own ``__array_namespace__`` (numpy>=2, jax).
    """
    try:
        import array_api_compat
    except ImportError:
        array_api_compat = None
    if array_api_compat is not None:
        try:
            return array_api_compat.array_namespace(x)
        except TypeError:  # lists / Python scalars
            return np
    if hasattr(x, "__array_namespace__"):
        return x.__array_namespace__()
    return np


def _is_jax(xp):
    return "jax" in getattr(xp, "__name__", "")


def _place(xp, host_array, ref):
    """Put a host (numpy) parameter array into ``xp`` on ``ref``'s device/dtype."""
    dtype = ref.dtype
    device = getattr(ref, "device", None)
    try:
        return xp.asarray(host_array, dtype=dtype, device=device)
    except TypeError:  # backend's asarray doesn't take device=
        return xp.asarray(host_array, dtype=dtype)

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

    def _param_arrays(self):
        """Current parameters as host (numpy) arrays plus the layout.

        The celestial pole ``(alpha_p, delta_p)`` and ``phi_p`` (LONPOLE) are
        computed here on the host -- cheap scalar work, independent of the input
        array's backend -- and moved onto the input's device by the callers.
        Returns ``(lng, lat, code, crpix, cd, pole, phi_p)``.
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

        crpix = np.asarray(self.crpix, dtype=float)
        cd = np.asarray(self.cd, dtype=float)
        pole = np.asarray([alpha_p, delta_p], dtype=float)
        return lng, lat, code, crpix, cd, pole, float(phi_p)

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
        # Array-API: compute in (and return) the input's namespace, so passing
        # numpy/jax/cupy arrays gives numpy/jax/cupy back, on the same device.
        xp = _array_namespace(pixel_arrays[0])
        lng, lat, code, crpix, cd, pole, phi_p = self._param_arrays()
        pix = xp.stack([xp.asarray(a) for a in pixel_arrays])
        if _is_jax(xp):
            fwd, _ = _jit_transforms(code, lng, lat)
            world = fwd(pix, xp.asarray(crpix, dtype=pix.dtype), xp.asarray(cd, dtype=pix.dtype),
                        xp.asarray(pole, dtype=pix.dtype), phi_p)
        else:
            p2w, _ = _make_transforms(code, lng, lat)
            world = p2w(xp, pix, _place(xp, crpix, pix), _place(xp, cd, pix),
                        _place(xp, pole, pix), phi_p)
        return tuple(world[i] for i in range(2))

    def world_to_pixel_values(self, *world_arrays):
        xp = _array_namespace(world_arrays[0])
        lng, lat, code, crpix, cd, pole, phi_p = self._param_arrays()
        cd_inv = np.linalg.inv(cd)
        world = xp.stack([xp.asarray(a) for a in world_arrays])
        if _is_jax(xp):
            _, inv = _jit_transforms(code, lng, lat)
            pix = inv(world, xp.asarray(crpix, dtype=world.dtype), xp.asarray(cd_inv, dtype=world.dtype),
                      xp.asarray(pole, dtype=world.dtype), phi_p)
        else:
            _, w2p = _make_transforms(code, lng, lat)
            pix = w2p(xp, world, _place(xp, crpix, world), _place(xp, cd_inv, world),
                      _place(xp, pole, world), phi_p)
        return tuple(pix[i] for i in range(2))

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
