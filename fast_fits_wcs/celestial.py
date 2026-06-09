"""Rotation between native spherical and celestial coordinates.

The rotation native (phi, theta) <-> celestial (alpha, delta) is fixed by the
celestial coordinates of the native pole ``(alpha_p, delta_p)`` and the native
longitude of the celestial pole ``LONPOLE`` (``phi_p``). For zenithal
projections the fiducial point *is* the native pole, so ``(alpha_p, delta_p) =
CRVAL``; for others (e.g. cylindrical CAR) the pole is computed from CRVAL,
theta0 and LONPOLE/LATPOLE by :func:`compute_celestial_pole`. Formulae are
eqs. (2), (5) and (8)-(10) of Calabretta & Greisen (2002).
"""

import numpy as np

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def _clamp_unit(xp, v):
    """Clamp to [-1, 1] for safe asin/acos, avoiding clip's min/max keywords
    (unavailable on older numpy)."""
    return xp.minimum(xp.maximum(v, -1.0), 1.0)


def native_to_celestial(xp, phi, theta, alpha_p, delta_p, phi_p):
    """(phi, theta) native -> (alpha, delta) celestial, all in degrees."""
    dphi = (phi - phi_p) * DEG2RAD
    theta_r = theta * DEG2RAD
    dp = delta_p * DEG2RAD

    sin_t, cos_t = xp.sin(theta_r), xp.cos(theta_r)
    sin_dp, cos_dp = xp.sin(dp), xp.cos(dp)
    cos_dphi = xp.cos(dphi)

    sin_d = sin_t * sin_dp + cos_t * cos_dp * cos_dphi
    delta = xp.asin(_clamp_unit(xp, sin_d)) * RAD2DEG

    y = -cos_t * xp.sin(dphi)
    x = sin_t * cos_dp - cos_t * sin_dp * cos_dphi
    alpha = alpha_p + xp.atan2(y, x) * RAD2DEG
    return alpha % 360.0, delta


def celestial_to_native(xp, alpha, delta, alpha_p, delta_p, phi_p):
    """(alpha, delta) celestial -> (phi, theta) native, all in degrees."""
    dalpha = (alpha - alpha_p) * DEG2RAD
    delta_r = delta * DEG2RAD
    dp = delta_p * DEG2RAD

    sin_d, cos_d = xp.sin(delta_r), xp.cos(delta_r)
    sin_dp, cos_dp = xp.sin(dp), xp.cos(dp)
    cos_da = xp.cos(dalpha)

    sin_t = sin_d * sin_dp + cos_d * cos_dp * cos_da
    theta = xp.asin(_clamp_unit(xp, sin_t)) * RAD2DEG

    y = -cos_d * xp.sin(dalpha)
    x = sin_d * cos_dp - cos_d * sin_dp * cos_da
    phi = phi_p + xp.atan2(y, x) * RAD2DEG
    # Wrap native longitude to (-180, 180]. Harmless for zenithal projections
    # (which use sin/cos phi) but essential for cylindrical ones where the
    # plane coordinate is phi itself.
    phi = (phi + 180.0) % 360.0 - 180.0
    return phi, theta


def default_lonpole(delta0, theta0):
    """FITS default LONPOLE: 0 if the reference point is at/above the native
    fiducial latitude, else 180 (degrees)."""
    return 0.0 if delta0 >= theta0 else 180.0


def compute_celestial_pole(alpha0, delta0, theta0, phi_p, latpole=90.0, phi0=0.0):
    """Celestial coordinates ``(alpha_p, delta_p)`` of the native pole, in deg.

    Solves the fiducial-point relation (Calabretta & Greisen 2002, eqs 8-10)
    for the pole given the reference point ``(alpha0, delta0) = CRVAL``, the
    native coordinates of the fiducial point ``(phi0, theta0)``, and the native
    longitude of the celestial pole ``phi_p`` (LONPOLE). Where two latitudes
    solve the relation, ``latpole`` (LATPOLE, default +90) selects between them.
    This is plain-numpy host code, run once per WCS rather than per pixel.
    """
    if theta0 == 90.0:
        # Zenithal: the fiducial point is the native pole.
        return alpha0 % 360.0, delta0

    d2r = np.pi / 180.0
    dphi = (phi0 - phi_p) * d2r
    a = np.sin(theta0 * d2r)
    b = np.cos(theta0 * d2r) * np.cos(dphi)
    c = np.sin(delta0 * d2r)
    hyp = np.hypot(a, b)

    gamma = np.arctan2(a, b)
    offset = np.arccos(np.clip(c / hyp, -1.0, 1.0))
    candidates = [(gamma + offset) / d2r, (gamma - offset) / d2r]
    valid = [d for d in candidates if -90.0001 <= d <= 90.0001] or candidates
    delta_p = min(valid, key=lambda d: abs(d - latpole))

    dp = delta_p * d2r
    num = -np.cos(theta0 * d2r) * np.sin(dphi)
    den = np.sin(theta0 * d2r) * np.cos(dp) - np.cos(theta0 * d2r) * np.sin(dp) * np.cos(dphi)
    alpha_p = alpha0 - np.arctan2(num, den) / d2r
    return alpha_p % 360.0, delta_p
