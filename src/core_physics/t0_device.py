"""CUDA device policy for T0 ladder scripts."""

from __future__ import annotations

import torch


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("[ERR] CUDA GPU required for T0 rung 4.1 (no CPU fallback)")
    return torch.device("cuda")
