"""Spherical map projections, expressed as pure JAX functions.

Each projection is a pair of functions that convert between the *projection
plane* coordinates ``(x, y)`` (intermediate world coordinates, in degrees) and
the *native spherical* coordinates ``(phi, theta)`` (in degrees), following the
conventions of Calabretta & Greisen (2002), "Representations of celestial
coordinates in FITS" (Paper II).

To add a projection, write the two functions and register them with the
three-letter FITS code via :func:`register`. Everything downstream (the linear
transform, the spherical rotation, the WCS class) is projection-agnostic.
"""

import jax.numpy as jnp

DEG2RAD = jnp.pi / 180.0
RAD2DEG = 180.0 / jnp.pi

# Maps a three-letter FITS projection code -> Projection instance.
_REGISTRY = {}


class Projection:
    """A spherical projection as a (plane->native, native->plane) function pair.

    Both functions operate in degrees and must be JAX-traceable (use
    ``jax.numpy`` only, no Python branching on array values).
    """

    def __init__(self, code, plane_to_native, native_to_plane, theta0=90.0):
        self.code = code
        self.plane_to_native = plane_to_native
        self.native_to_plane = native_to_plane
        # Native latitude of the fiducial point. 90 for zenithal projections
        # (TAN, SIN, STG, ARC, ZEA, ...); used to pick the default LONPOLE.
        self.theta0 = theta0


def register(projection):
    _REGISTRY[projection.code] = projection
    return projection


def get(code):
    try:
        return _REGISTRY[code]
    except KeyError:
        raise ValueError(
            f"Projection {code!r} is not supported "
            f"(known: {sorted(_REGISTRY)})"
        )


# --- TAN: gnomonic projection ------------------------------------------------
#
# Zenithal, theta0 = 90. Native radius from the fiducial point:
#     R_theta = (180/pi) * cot(theta)          [degrees]
# so the plane coordinates are
#     x =  R_theta * sin(phi)
#     y = -R_theta * cos(phi)


def _tan_plane_to_native(x, y):
    phi = jnp.arctan2(x, -y) * RAD2DEG
    r = jnp.hypot(x, y)  # degrees
    theta = jnp.arctan2(RAD2DEG, r) * RAD2DEG
    return phi, theta


def _tan_native_to_plane(phi, theta):
    phi_r = phi * DEG2RAD
    r = RAD2DEG / jnp.tan(theta * DEG2RAD)  # degrees
    x = r * jnp.sin(phi_r)
    y = -r * jnp.cos(phi_r)
    return x, y


register(Projection("TAN", _tan_plane_to_native, _tan_native_to_plane))
