"""Gate for passive step-2 bridge: species co-descent + modest mu val trend (not step-3 multitask).

Pass (default):
  - M3 align checks (L_bio, masked L_ADR_S, mask_n, val species FI)
  - val_mu_log_mae last <= 1.15 * first (mu aux must not blow up)
  - last epoch reports passive_step2_bridge=1

Usage:
  python scripts/check_passive_step2_bridge_gate.py --run-note passive_step2_bridge
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

from check_m3_align_gate import _find_run_dir, _load_jsonl, _series, eval_run as m3_eval_run


def eval_bridge_run(
    run_dir: Path,
    *,
    bio_max_ratio: float,
    adr_max_ratio: float,
    min_mask_n: float,
    min_epochs: int,
    mu_max_ratio: float,
) -> dict:
    out = m3_eval_run(
        run_dir,
        bio_max_ratio=bio_max_ratio,
        adr_max_ratio=adr_max_ratio,
        min_mask_n=min_mask_n,
        min_epochs=min_epochs,
    )
    eps = [
        r
        for r in _load_jsonl(run_dir / "run.jsonl")
        if r.get("event") == "val" and r.get("stage") == "teacher"
    ]
    eps.sort(key=lambda x: int(x["epoch"]))
    mu = _series(eps, "val_mu_log_mae")
    bridge_flag = eps[-1].get("passive_step2_bridge") if eps else None
    bridge_ok = bridge_flag in (1, 1.0, True, "1")
    mu_ok = True
    mu_note = "n/a"
    if len(mu) >= 2 and mu[0] > 0:
        ratio = mu[-1] / mu[0]
        mu_ok = ratio <= mu_max_ratio
        mu_note = f"mu_ratio={ratio:.4f}<={mu_max_ratio}"
    train_sp = _series(eps, "train_species_fi_log_mae_mean")
    train_sp_ok = bool(train_sp) and train_sp[-1] <= max(train_sp[0] * 1.05, train_sp[0] + 0.05)
    train_sp_note = (
        f"train_FI_last={train_sp[-1]:.4f}" if train_sp else "train_species_missing"
    )
    out["mu_ok"] = mu_ok
    out["mu_note"] = mu_note
    out["val_mu_log_mae"] = mu
    out["bridge_ok"] = bridge_ok
    out["train_species_ok"] = train_sp_ok
    out["train_species_note"] = train_sp_note
    out["ok"] = bool(
        out.get("ok")
        and mu_ok
        and bridge_ok
        and (train_sp_ok or not train_sp)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--bio-max-ratio", type=float, default=0.92)
    ap.add_argument("--adr-max-ratio", type=float, default=0.88)
    ap.add_argument("--min-mask-n", type=float, default=32.0)
    ap.add_argument("--min-epochs", type=int, default=6)
    ap.add_argument("--mu-max-ratio", type=float, default=1.15)
    args = ap.parse_args()

    rd = _find_run_dir(run_note=args.run_note)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2

    out = eval_bridge_run(
        rd,
        bio_max_ratio=args.bio_max_ratio,
        adr_max_ratio=args.adr_max_ratio,
        min_mask_n=args.min_mask_n,
        min_epochs=args.min_epochs,
        mu_max_ratio=args.mu_max_ratio,
    )
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    print(f"{tag} passive_step2_bridge: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
