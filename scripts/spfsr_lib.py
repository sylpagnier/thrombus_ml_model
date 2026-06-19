"""Shared loader for the preprocessed COMSOL spf.sr cache (scripts/preprocess_spfsr.py)."""
from __future__ import annotations
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed" / "spfsr_cache"


def has_cache(stem: str) -> bool:
    return (CACHE / f"{stem}.pt").exists()


def load_raw(stem: str, dev):
    return torch.load(CACHE / f"{stem}.pt", map_location=dev, weights_only=False)


def aligned(d, dev, stem: str):
    """Return exact COMSOL fields on graph nodes, aligned to graph frame times d.t.

    Returns dict: sr [T_graph, N] (1/s), dsrx/dsry [T_graph, N] or None, has_deriv.
    """
    c = load_raw(stem, dev)
    et = c["export_times"].to(dev).reshape(-1)             # [NTexp]
    t_s = d.t.to(dev).reshape(-1).float()                  # [T_graph]
    idx = torch.cdist(t_s.view(-1, 1), et.view(-1, 1)).argmin(dim=1)   # nearest export frame
    out = {"sr": c["sr"].to(dev)[idx], "has_deriv": "dsrx" in c}
    if out["has_deriv"]:
        out["dsrx"] = c["dsrx"].to(dev)[idx]
        out["dsry"] = c["dsry"].to(dev)[idx]
    out["frame_idx"] = idx
    return out
