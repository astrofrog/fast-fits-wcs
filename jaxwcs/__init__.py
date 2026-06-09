"""jaxwcs: a minimal, JAX-optimized, APE 14-compliant celestial FITS WCS."""

try:
    from ._version import __version__
except ImportError:  # source checkout without a build; version file not generated
    __version__ = "0.0.0+unknown"

# WCS needs 64-bit precision: float32 gives only ~0.02 arcsec accuracy and
# degrades badly in the world->pixel inverse. Enable it before any jax array is
# created. Set JAXWCS_X64=0 to opt out (e.g. if you share a process with a
# float32 ML pipeline and can tolerate the precision loss).
import os as _os

if _os.environ.get("JAXWCS_X64", "1") != "0":
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

from .core import WCS
from . import projections
from .projections import Projection, register

__all__ = ["WCS", "Projection", "register", "projections"]
