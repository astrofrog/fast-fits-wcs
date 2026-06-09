"""Spherical map projections, expressed as pure JAX functions.

Each projection is a pair of functions that convert between the *projection
plane* coordinates ``(x, y)`` (intermediate world coordinates, in degrees) and
the *native spherical* coordinates ``(phi, theta)`` (in degrees), following the
conventions of Calabretta & Greisen (2002), "Representations of celestial
coordinates in FITS" (Paper II).

The zenithal (azimuthal) projections all share the same azimuthal behaviour and
differ only in the radial function R(theta). :func:`zenithal` builds one from a
single radial relation and its inverse, so adding TAN/STG/SIN/ARC/ZEA is a
one-liner each. Everything downstream (the linear transform, the spherical
rotation, the WCS class) is projection-agnostic.
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


# --- zenithal projections ----------------------------------------------------
#
# Native co-latitude maps to a plane radius R(theta); the azimuth is shared:
#     phi = atan2(x, -y),   R = hypot(x, y)
#     x   = R sin(phi),     y = -R cos(phi)
# Each projection supplies only theta_from_R and R_from_theta (both in degrees).


def zenithal(code, theta_from_R, R_from_theta):
    def plane_to_native(x, y):
        phi = jnp.arctan2(x, -y) * RAD2DEG
        R = jnp.hypot(x, y)
        return phi, theta_from_R(R)

    def native_to_plane(phi, theta):
        R = R_from_theta(theta)
        phi_r = phi * DEG2RAD
        return R * jnp.sin(phi_r), -R * jnp.cos(phi_r)

    return register(Projection(code, plane_to_native, native_to_plane, theta0=90.0))


# R in degrees; r = R * pi/180 is the radius in radians.
# TAN gnomonic:              R = (180/pi) cot(theta)
zenithal("TAN",
         lambda R: jnp.arctan2(RAD2DEG, R) * RAD2DEG,
         lambda th: RAD2DEG / jnp.tan(th * DEG2RAD))

# STG stereographic:         R = (360/pi) tan((90-theta)/2)   [conformal]
zenithal("STG",
         lambda R: 90.0 - 2.0 * jnp.arctan(R * DEG2RAD / 2.0) * RAD2DEG,
         lambda th: 2.0 * RAD2DEG * jnp.tan((90.0 - th) * DEG2RAD / 2.0))

# SIN orthographic:          R = (180/pi) cos(theta)
zenithal("SIN",
         lambda R: jnp.arccos(jnp.clip(R * DEG2RAD, -1.0, 1.0)) * RAD2DEG,
         lambda th: RAD2DEG * jnp.cos(th * DEG2RAD))

# ARC zenithal equidistant:  R = 90 - theta   (degrees)
zenithal("ARC",
         lambda R: 90.0 - R,
         lambda th: 90.0 - th)

# ZEA zenithal equal-area:   R = (360/pi) sin((90-theta)/2)
zenithal("ZEA",
         lambda R: 90.0 - 2.0 * jnp.arcsin(jnp.clip(R * DEG2RAD / 2.0, -1.0, 1.0)) * RAD2DEG,
         lambda th: 2.0 * RAD2DEG * jnp.sin((90.0 - th) * DEG2RAD / 2.0))


# --- cylindrical projections -------------------------------------------------
#
# Fiducial point on the native equator (theta0 = 0), so the native<->celestial
# rotation is the general one (see celestial.compute_celestial_pole).

# CAR plate carree: the plane coordinates are the native angles themselves.
register(Projection("CAR",
                    lambda x, y: (x, y),
                    lambda phi, theta: (phi, theta),
                    theta0=0.0))
