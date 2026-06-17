"""Legacy wrapper -> train_biochem_gnn_loao.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_TARGET = REPO / "scripts" / "train_biochem_gnn_loao.py"
_spec = importlib.util.spec_from_file_location("train_biochem_gnn_loao", _TARGET)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    print("[i] train_clot_deploy_gnn_loao.py -> train_biochem_gnn_loao.py", flush=True)
    raise SystemExit(_mod.main())
