"""Compatibility shim — prefer summarize_frontier_ge2_prec_8h.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("summarize_frontier_ge2_prec_8h.py")
    runpy.run_path(str(target), run_name="__main__")
