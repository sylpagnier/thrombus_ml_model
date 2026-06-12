"""Training device policy for clot ML ladder scripts (GPU by default)."""

from __future__ import annotations

import os
import sys

import torch


def resolve_clot_ml_training_device() -> torch.device:
    """Return CUDA device for clot ML training; exit if GPU unavailable.

    Env ``CLOT_ML_DEVICE`` (default ``cuda``): ``cuda`` | ``cpu``.
    Training entrypoints should call this instead of falling back to CPU silently.
    """
    want = (os.environ.get("CLOT_ML_DEVICE") or "cuda").strip().lower()
    if want == "cpu":
        return torch.device("cpu")
    if want not in ("cuda", "gpu", "auto"):
        print(f"[ERR] unknown CLOT_ML_DEVICE={want!r}; use cuda or cpu", file=sys.stderr)
        raise SystemExit(1)
    if not torch.cuda.is_available():
        print(
            "[ERR] CLOT_ML_DEVICE=cuda but torch.cuda.is_available() is False.\n"
            "[i] Install a CUDA PyTorch wheel, e.g. .\\scripts\\install_torch_cuda.ps1",
            file=sys.stderr,
        )
        raise SystemExit(1)
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"[i] clot ML training device: {dev} ({props.name})", flush=True)
    return dev


def resolve_clot_ml_eval_device() -> torch.device:
    """Eval/smoke device: honors ``CLOT_ML_DEVICE``; falls back to CPU if CUDA missing."""
    want = (os.environ.get("CLOT_ML_DEVICE") or "cuda").strip().lower()
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    print("[WARN] cuda unavailable; step5 eval on cpu", flush=True)
    return torch.device("cpu")
