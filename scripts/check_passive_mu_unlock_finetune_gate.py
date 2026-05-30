"""Gate for passive mu-unlock finetune (wall/high-mu recovery + species stable).

Pass (default):
  - passive_mu_unlock_finetune=1 on last val row
  - val all logMAE improved vs first epoch (any drop)
  - val wall logMAE last <= first * 1.05 OR last < 2.5
  - val high-mu logMAE last <= first * 0.98 (strict improvement)
  - species FI last <= 0.08

Usage:
  python scripts/check_passive_mu_unlock_finetune_gate.py --run-note passive_mu_unlock_finetune
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


def eval_finetune_run(
    run_dir: Path,
    *,
    min_epochs: int,
    species_fi_max: float,
    wall_max_ratio: float,
    wall_abs_max: float,
    high_improve_ratio: float,
) -> dict:
    eps = [
        r
        for r in _load_jsonl(run_dir / "run.jsonl")
        if r.get("event") == "val" and r.get("stage") == "teacher"
    ]
    eps.sort(key=lambda x: int(x["epoch"]))
    mu_all = _series(eps, "val_mu_log_mae")
    mu_wall = _series(eps, "val_mu_log_mae_wall")
    mu_high = _series(eps, "val_mu_log_mae_high_mu")
    fi = _series(eps, "val_species_fi_log_mae")
    finetune_flag = eps[-1].get("passive_mu_unlock_finetune") if eps else None
    finetune_ok = finetune_flag in (1, 1.0, True, "1")

    mu_all_ok = len(mu_all) >= 2 and mu_all[-1] < mu_all[0]
    wall_ok = True
    wall_note = "n/a"
    if len(mu_wall) >= 2 and mu_wall[0] > 0:
        ratio = mu_wall[-1] / mu_wall[0]
        wall_ok = ratio <= wall_max_ratio or mu_wall[-1] <= wall_abs_max
        wall_note = f"wall_ratio={ratio:.3f} last={mu_wall[-1]:.4f}"
    high_ok = True
    high_note = "n/a"
    if len(mu_high) >= 2 and mu_high[0] > 0:
        ratio = mu_high[-1] / mu_high[0]
        high_ok = ratio <= high_improve_ratio
        high_note = f"high_ratio={ratio:.3f} last={mu_high[-1]:.4f}"

    species_ok = bool(fi) and fi[-1] <= species_fi_max
    ok = (
        len(eps) >= min_epochs
        and finetune_ok
        and mu_all_ok
        and wall_ok
        and high_ok
        and species_ok
    )
    return {
        "ok": ok,
        "finetune_ok": finetune_ok,
        "mu_all_ok": mu_all_ok,
        "wall_ok": wall_ok,
        "wall_note": wall_note,
        "high_ok": high_ok,
        "high_note": high_note,
        "species_ok": species_ok,
        "val_mu_log_mae": mu_all,
        "val_mu_log_mae_wall": mu_wall,
        "val_mu_log_mae_high_mu": mu_high,
        "val_species_fi_log_mae": fi,
        "n_val_epochs": len(eps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--min-epochs", type=int, default=4)
    ap.add_argument("--species-fi-max", type=float, default=0.08)
    ap.add_argument("--wall-max-ratio", type=float, default=1.05)
    ap.add_argument("--wall-abs-max", type=float, default=2.5)
    ap.add_argument("--high-improve-ratio", type=float, default=0.98)
    args = ap.parse_args()

    rd = _find_run_dir(run_note=args.run_note)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2

    out = eval_finetune_run(
        rd,
        min_epochs=args.min_epochs,
        species_fi_max=args.species_fi_max,
        wall_max_ratio=args.wall_max_ratio,
        wall_abs_max=args.wall_abs_max,
        high_improve_ratio=args.high_improve_ratio,
    )
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    print(f"{tag} passive_mu_unlock_finetune: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
