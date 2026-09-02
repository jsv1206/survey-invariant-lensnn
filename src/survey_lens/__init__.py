"""Survey-invariant strong-lens embedding network.

A telescope-invariant neural network that takes a blurred lens image, its PSF
stamp, and the pixel scale, and produces a PSF-invariant deblurred embedding
from which it (a) reconstructs the sharp image and (b) regresses the Einstein
radius theta_E.

Note: import :mod:`survey_lens.env_setup` FIRST in any process that will build
simulations (it must run before lenstronomy/astropy are imported). The CLI
entry points and the simulator-facing modules do this for you.
"""
__version__ = "0.1.0"

__all__ = ["__version__"]
