"""Legacy wrapper -> check_clot_deploy_gnn_gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_TARGET = REPO / "scripts" / "check_clot_deploy_gnn_gate.py"
_spec = importlib.util.spec_from_file_location("check_clot_deploy_gnn_gate", _TARGET)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    raise SystemExit(_mod.main())
