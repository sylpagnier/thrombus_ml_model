"""Backward-compatible launcher for :mod:`src.tools.extract_biochem_comsol`.

Prefer::

    python -m src.tools.extract_biochem_comsol
"""

from __future__ import annotations

from src.tools.extract_biochem_comsol import main

if __name__ == "__main__":
    main()
