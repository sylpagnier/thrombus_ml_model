"""Architecture package (lazy ``lora_injection`` avoids import cycles with ``architecture.ginodeq``)."""

from src.architecture.lora_injection import (
    LoRAParametrization,
    SpectralLinear,
    inject_lora_to_kinematics,
    inject_lora_to_spectral_linears,
)

__all__ = [
    "LoRAParametrization",
    "SpectralLinear",
    "inject_lora_to_kinematics",
    "inject_lora_to_spectral_linears",
]
