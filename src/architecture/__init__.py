"""Architecture package (RGP-DEQ, spectral layers, decoders)."""

from src.architecture.ginodeq import GINO_DEQ, GINOBlock, RGP_DEQ, RGPBlock
from src.architecture.spectral_linear import SpectralLinear

__all__ = [
    "RGP_DEQ",
    "RGPBlock",
    "GINO_DEQ",
    "GINOBlock",
    "SpectralLinear",
]
