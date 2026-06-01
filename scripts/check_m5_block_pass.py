"""M5 block pass gate (M5.3-M5.6): finetune/bridge attempted + best K10 leg metrics.

Pass (default):
  - passive_mu_unlock_best exists
  - at least one m5_k10* run in runs_index / run.jsonl
  - best K10 leg: val mu_log_mae all <= 0.95 AND (viz_clot_frac >= 0.02 OR adjacent/wall improved)

Usage:
  python scripts/check_m5_block_pass.py
  python scripts/check_m5_block_pass.py --summary outputs/biochem/m5_block/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from check_m3_align_gate import _find_run_dir, _load_jsonl, _series


_K10_NOTES = (
    "m5_k10f_wide_from_passive",
    "m5_k10e_narrow_from_passive",
    "m5_k10g_bias_from_passive",
)


def _best_val_row(run_dir: Path) -> dict | None:
    rows = [
        r
        for r in _load_jsonl(run_dir / "run.jsonl")
        if r.get("event") == "val" and isinstance(r.get("epoch"), int)
    ]
    if not rows:
        return None
    return min(rows, key=lambda r: float(r.get("val_mu_log_mae", 1e9)))


def eval_m5_block(summary_path: Path | None = None) -> dict:
    unlock = _REPO / "outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth"
    out: dict = {"ok": False, "legs": {}, "best_k10": None}
    if not unlock.is_file():
        out["error"] = f"missing {unlock.name} (run M5.1 first)"
        return out

    for note in (
        "passive_mu_unlock_finetune",
        "passive_m5_bridge",
        *_K10_NOTES,
    ):
        rd = _find_run_dir(note, quiet=True)
        if rd is None:
            out["legs"][note] = {"found": False}
            continue
        row = _best_val_row(rd)
        if row is None:
            out["legs"][note] = {"found": True, "run_dir": str(rd), "val": None}
            continue
        out["legs"][note] = {
            "found": True,
            "run_dir": str(rd),
            "epoch": row.get("epoch"),
            "val_mu_log_mae": float(row.get("val_mu_log_mae", float("nan"))),
            "val_mu_log_mae_wall": float(row.get("val_mu_log_mae_wall", float("nan"))),
            "val_viz_clot_frac": float(row.get("val_viz_clot_frac", 0.0) or 0.0),
            "val_species_fi_log_mae": float(row.get("val_species_fi_log_mae", float("nan"))),
        }

    k10_candidates = [
        (note, out["legs"].get(note))
        for note in _K10_NOTES
        if out["legs"].get(note, {}).get("val")
    ]
    if not k10_candidates:
        out["error"] = "no K10 leg with val rows (M5.6)"
        return out

    def _score(item: tuple[str, dict]) -> float:
        _, leg = item
        mu = float(leg["val_mu_log_mae"])
        clot = float(leg.get("val_viz_clot_frac") or 0.0)
        return mu - 0.15 * clot

    best_note, best_leg = min(k10_candidates, key=_score)
    out["best_k10"] = {"run_note": best_note, **best_leg}
    mu_ok = float(best_leg["val_mu_log_mae"]) <= 0.95
    clot_ok = float(best_leg.get("val_viz_clot_frac") or 0.0) >= 0.02
    wall = float(best_leg.get("val_mu_log_mae_wall") or 99.0)
    wall_ok = wall < 2.5
    out["ok"] = bool(mu_ok and (clot_ok or wall_ok))
    out["mu_ok"] = mu_ok
    out["clot_ok"] = clot_ok
    out["wall_ok"] = wall_ok
    locked = _REPO / "outputs/biochem/biochem_teacher_passive_m5_clot_locked.pth"
    out["locked_ckpt"] = str(locked) if locked.is_file() else None

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/biochem/m5_block/summary.json")
    args = ap.parse_args()
    summary = eval_m5_block(Path(args.summary))
    print(json.dumps(summary, indent=2))
    if summary.get("ok"):
        print("[OK] M5 block pass")
        if summary.get("locked_ckpt"):
            print(f"[i]  Locked: {summary['locked_ckpt']}")
        print("[i]  Viz: python -m src.evaluation.visualize_pipeline --teacher-only "
              f"--biochem-checkpoint outputs/biochem/biochem_teacher_passive_m5_clot_locked.pth --anchor patient007")
        return 0
    print(f"[FAIL] M5 block: {summary.get('error', 'criteria not met')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
