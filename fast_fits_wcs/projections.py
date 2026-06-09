"""Spherical map projections, expressed against the Python array API.

Each projection is a pair of functions that convert between the *projection
plane* coordinates ``(x, y)`` (intermediate world coordinates, in degrees) and
the *native spherical* coordinates ``(phi, theta)`` (in degrees), following the
conventions of Calabretta & Greisen (2002), "Representations of celestial
coordinates in FITS" (Paper II).

Every function takes the array namespace ``xp`` as its first argument (the
caller passes the namespace of the input arrays -- numpy, jax, cupy, ...), so
the projections are backend-agnostic and use only standard array-API
operations. The zenithal projections differ only in the radial function
R(theta); :func:`zenithal` builds one from a single radial relation and its
inverse, so adding TAN/STG/SIN/ARC/ZEA is a one-liner each.
"""

import math

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Maps a three-letter FITS projection code -> Projection instance.
_REGISTRY = {}


class Projection:
    """A spherical projection as a (plane->native, native->plane) function pair.

    Both functions take ``xp`` (an array-API namespace) first and otherwise
    operate in degrees using only standard array-API operations.
    """

    def __init__(self, code, plane_to_native, native_to_plane, theta0=90.0):
        self.code = code
        self.plane_to_native = plane_to_native
        self.native_to_plane = native_to_plane
        # Native latitude of the fiducial point. 90 for zenithal projections
        # (TAN, SIN, STG, ARC, ZEA, ...), 0 for cylindrical ones (CAR); used to
        # pick the default LONPOLE and to compute the celestial pole.
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
#     phi = atan2(x, -y),   R = sqrt(x**2 + y**2)
#     x   = R sin(phi),     y = -R cos(phi)
# Each projection supplies only theta_from_R and R_from_theta (both in degrees).


def zenithal(code, theta_from_R, R_from_theta):
    def plane_to_native(xp, x, y):
        phi = xp.atan2(x, -y) * RAD2DEG
        R = xp.sqrt(x * x + y * y)
        return phi, theta_from_R(xp, R)

    def native_to_plane(xp, phi, theta):
        R = R_from_theta(xp, theta)
        phi_r = phi * DEG2RAD
        return R * xp.sin(phi_r), -R * xp.cos(phi_r)

    return register(Projection(code, plane_to_native, native_to_plane, theta0=90.0))


# R in degrees; r = R * pi/180 is the radius in radians.
# TAN gnomonic:              R = (180/pi) cot(theta)
zenithal("TAN",
         lambda xp, R: xp.atan2(RAD2DEG + 0.0 * R, R) * RAD2DEG,
         lambda xp, th: RAD2DEG / xp.tan(th * DEG2RAD))

# STG stereographic:         R = (360/pi) tan((90-theta)/2)   [conformal]
zenithal("STG",
         lambda xp, R: 90.0 - 2.0 * xp.atan(R * DEG2RAD / 2.0) * RAD2DEG,
         lambda xp, th: 2.0 * RAD2DEG * xp.tan((90.0 - th) * DEG2RAD / 2.0))

# SIN orthographic:          R = (180/pi) cos(theta)
zenithal("SIN",
         lambda xp, R: xp.acos(xp.clip(R * DEG2RAD, min=-1.0, max=1.0)) * RAD2DEG,
         lambda xp, th: RAD2DEG * xp.cos(th * DEG2RAD))

# ARC zenithal equidistant:  R = 90 - theta   (degrees)
zenithal("ARC",
         lambda xp, R: 90.0 - R,
         lambda xp, th: 90.0 - th)

# ZEA zenithal equal-area:   R = (360/pi) sin((90-theta)/2)
zenithal("ZEA",
         lambda xp, R: 90.0 - 2.0 * xp.asin(xp.clip(R * DEG2RAD / 2.0, min=-1.0, max=1.0)) * RAD2DEG,
         lambda xp, th: 2.0 * RAD2DEG * xp.sin((90.0 - th) * DEG2RAD / 2.0))


# --- cylindrical projections -------------------------------------------------
#
# Fiducial point on the native equator (theta0 = 0), so the native<->celestial
# rotation is the general one (see celestial.compute_celestial_pole).

# CAR plate carree: the plane coordinates are the native angles themselves.
register(Projection("CAR",
                    lambda xp, x, y: (x, y),
                    lambda xp, phi, theta: (phi, theta),
                    theta0=0.0))
