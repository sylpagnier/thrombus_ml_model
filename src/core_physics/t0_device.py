"""CUDA device helper for deploy / eval scripts."""

from __future__ import annotations

import torch


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    return torch.device("cuda")
