"""Gate for passive mu-unlock probe: mu improves, species stable, unlock flag set.

Pass (default):
  - last epoch ``passive_mu_unlock=1``
  - val ``mu_log_mae`` last < first - 0.05 (mu moved)
  - val species FI last <= 0.08 and <= 1.2 * first (no species catastrophe)
  - train species FI mean last <= 0.08 (if logged)

Usage:
  python scripts/check_passive_mu_unlock_gate.py --run-note passive_mu_unlock_probe
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


def eval_mu_unlock_run(
    run_dir: Path,
    *,
    min_epochs: int,
    mu_min_drop: float,
    species_fi_max: float,
    species_max_ratio: float,
) -> dict:
    eps = [
        r
        for r in _load_jsonl(run_dir / "run.jsonl")
        if r.get("event") == "val" and r.get("stage") == "teacher"
    ]
    eps.sort(key=lambda x: int(x["epoch"]))
    mu = _series(eps, "val_mu_log_mae")
    fi = _series(eps, "val_species_fi_log_mae")
    unlock_flag = eps[-1].get("passive_mu_unlock") if eps else None
    unlock_ok = unlock_flag in (1, 1.0, True, "1")
    mu_ok = False
    mu_note = "n/a"
    if len(mu) >= 2:
        drop = mu[0] - mu[-1]
        mu_ok = drop >= mu_min_drop and mu[-1] < mu[0]
        mu_note = f"mu_drop={drop:.4f}>={mu_min_drop}"
    species_ok = True
    species_note = "n/a"
    if len(fi) >= 1:
        species_ok = fi[-1] <= species_fi_max
        if len(fi) >= 2 and fi[0] > 0:
            ratio = fi[-1] / fi[0]
            species_ok = species_ok and ratio <= species_max_ratio
            species_note = f"FI_last={fi[-1]:.4f} ratio={ratio:.3f}"
        else:
            species_note = f"FI_last={fi[-1]:.4f}"
    train_fi = _series(eps, "train_species_fi_log_mae_mean")
    train_ok = True
    if train_fi:
        train_ok = train_fi[-1] <= species_fi_max
    ok = (
        len(eps) >= min_epochs
        and unlock_ok
        and mu_ok
        and species_ok
        and train_ok
    )
    return {
        "ok": ok,
        "unlock_ok": unlock_ok,
        "mu_ok": mu_ok,
        "mu_note": mu_note,
        "species_ok": species_ok,
        "species_note": species_note,
        "train_species_ok": train_ok,
        "val_mu_log_mae": mu,
        "val_species_fi_log_mae": fi,
        "n_val_epochs": len(eps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--min-epochs", type=int, default=4)
    ap.add_argument("--mu-min-drop", type=float, default=0.05)
    ap.add_argument("--species-fi-max", type=float, default=0.08)
    ap.add_argument("--species-max-ratio", type=float, default=1.2)
    args = ap.parse_args()

    rd = _find_run_dir(run_note=args.run_note)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2

    out = eval_mu_unlock_run(
        rd,
        min_epochs=args.min_epochs,
        mu_min_drop=args.mu_min_drop,
        species_fi_max=args.species_fi_max,
        species_max_ratio=args.species_max_ratio,
    )
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    print(f"{tag} passive_mu_unlock: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
