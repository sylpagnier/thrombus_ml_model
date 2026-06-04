"""Metric gate for rung 12 Lane B (clot-phi on gnode11_finish corrector dump)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import importlib.util

from src.utils.paths import get_project_root

_gate_spec = importlib.util.spec_from_file_location(
    "gnode12_gate", _REPO / "scripts" / "_gnode12_gate.py"
)
_gnode12_gate = importlib.util.module_from_spec(_gate_spec)
assert _gate_spec.loader is not None
_gate_spec.loader.exec_module(_gnode12_gate)
run_lane_gate = _gnode12_gate.run_lane_gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-json",
        default="outputs/biochem/passive_species_focus_compare/gnode12_lane_b_clotphi/multi_anchor.jsonl",
    )
    ap.add_argument("--min-clot-min-f1", type=float, default=0.26)
    ap.add_argument(
        "--compare-lane-a-json",
        default="",
        help="Lane A multi_anchor.jsonl for optional p007 compare (warn only).",
    )
    ap.add_argument(
        "--skip-lane-a-compare",
        action="store_true",
        help="Do not warn if p007 F1 is below Lane A.",
    )
    args = ap.parse_args()

    root = get_project_root()
    eval_path = Path(args.eval_json)
    if not eval_path.is_absolute():
        eval_path = root / eval_path

    sys.exit(
        run_lane_gate(
            lane_label="B",
            eval_path=eval_path,
            min_clot_min_f1=args.min_clot_min_f1,
            skip_mu_trend=True,
            max_teacher_mu_log_mae=1.35,
            mu_unlock_run_jsonl="",
            compare_lane_a_json=args.compare_lane_a_json,
            warn_if_below_lane_a_p007=not args.skip_lane_a_compare,
        )
    )


if __name__ == "__main__":
    main()
