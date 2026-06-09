"""Rotation between native spherical and celestial coordinates.

For a zenithal projection the fiducial point sits at the native pole, so the
celestial coordinates of the native pole are simply ``(CRVAL1, CRVAL2)`` and the
rotation is fixed by those plus the native longitude of the celestial pole,
``LONPOLE`` (``phi_p``). Formulae are eqs. (2) and (5) of Calabretta & Greisen
(2002).
"""

import jax.numpy as jnp

DEG2RAD = jnp.pi / 180.0
RAD2DEG = 180.0 / jnp.pi


def native_to_celestial(phi, theta, alpha_p, delta_p, phi_p):
    """(phi, theta) native -> (alpha, delta) celestial, all in degrees."""
    dphi = (phi - phi_p) * DEG2RAD
    theta_r = theta * DEG2RAD
    dp = delta_p * DEG2RAD

    sin_t, cos_t = jnp.sin(theta_r), jnp.cos(theta_r)
    sin_dp, cos_dp = jnp.sin(dp), jnp.cos(dp)
    cos_dphi = jnp.cos(dphi)

    sin_d = sin_t * sin_dp + cos_t * cos_dp * cos_dphi
    delta = jnp.arcsin(jnp.clip(sin_d, -1.0, 1.0)) * RAD2DEG

    y = -cos_t * jnp.sin(dphi)
    x = sin_t * cos_dp - cos_t * sin_dp * cos_dphi
    alpha = alpha_p + jnp.arctan2(y, x) * RAD2DEG
    return alpha % 360.0, delta


def celestial_to_native(alpha, delta, alpha_p, delta_p, phi_p):
    """(alpha, delta) celestial -> (phi, theta) native, all in degrees."""
    dalpha = (alpha - alpha_p) * DEG2RAD
    delta_r = delta * DEG2RAD
    dp = delta_p * DEG2RAD

    sin_d, cos_d = jnp.sin(delta_r), jnp.cos(delta_r)
    sin_dp, cos_dp = jnp.sin(dp), jnp.cos(dp)
    cos_da = jnp.cos(dalpha)

    sin_t = sin_d * sin_dp + cos_d * cos_dp * cos_da
    theta = jnp.arcsin(jnp.clip(sin_t, -1.0, 1.0)) * RAD2DEG

    y = -cos_d * jnp.sin(dalpha)
    x = sin_d * cos_dp - cos_d * sin_dp * cos_da
    phi = phi_p + jnp.arctan2(y, x) * RAD2DEG
    return phi, theta


def default_lonpole(delta0, theta0):
    """FITS default LONPOLE: 0 if the reference point is at/above the native
    fiducial latitude, else 180 (degrees)."""
    return 0.0 if delta0 >= theta0 else 180.0
