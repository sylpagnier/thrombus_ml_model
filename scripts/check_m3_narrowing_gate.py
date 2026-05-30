"""Score M3 narrowing legs: data descent + ADR trend + stability.

Pass (>=3 teacher val rows):
  - train_L_bio_avg or train_L_back_avg: last/first <= max_ratio (default 0.92)
  - if passive_adr_in_backprop: train_L_ADR_S_avg or train_L_ADR_S_global_avg descends OR
    scaled_adr (adr * passive_adr_weight) descends vs ep0
  - train_L_tot not increasing ep4->last (no late blow-up)

Usage:
  python scripts/check_m3_narrowing_gate.py --run-note m3n_E2_match_nowall
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_REPORTS = _REPO / "outputs" / "reports" / "training" / "biochem"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_run_dir(*, run_note: str) -> Path | None:
    index = _REPORTS / "runs_index.jsonl"
    if index.is_file():
        for line in reversed(index.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_note") == run_note:
                rd = Path(row["run_dir"])
                if rd.is_dir():
                    return rd
    for p in sorted(_REPORTS.glob("*/run.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        rows = _load_jsonl(p)
        if rows and rows[0].get("run_note") == run_note:
            return p.parent
    return None


def _series(epochs: list[dict], key: str) -> list[float]:
    out = []
    for r in epochs:
        v = r.get(key)
        if v is not None:
            out.append(float(v))
    return out


def _descent(vals: list[float], *, max_ratio: float) -> tuple[bool, str]:
    if len(vals) < 2:
        return False, "need>=2_epochs"
    a, b = vals[0], vals[-1]
    if a <= 0:
        return False, "first<=0"
    ratio = b / a
    if ratio > max_ratio:
        return False, f"ratio={ratio:.4f}>{max_ratio}"
    return True, f"ratio={ratio:.4f}"


def eval_run(run_dir: Path, *, max_ratio: float) -> dict:
    eps = [
        r
        for r in _load_jsonl(run_dir / "run.jsonl")
        if r.get("event") == "val" and r.get("stage") == "teacher"
    ]
    eps.sort(key=lambda x: int(x["epoch"]))

    bio = _series(eps, "train_L_bio_avg") or _series(eps, "train_L_back_avg")
    adr = _series(eps, "train_L_ADR_S_avg") or _series(eps, "train_L_ADR_S_global_avg")
    tot = _series(eps, "train_L_tot") or _series(eps, "train_L_back_avg")
    adr_w = float(eps[-1].get("passive_adr_weight", 1.0)) if eps else 1.0
    passive_adr = any(bool(r.get("passive_adr_in_backprop")) for r in eps)

    bio_ok, bio_note = _descent(bio, max_ratio=max_ratio)
    adr_ok, adr_note = _descent(adr, max_ratio=max_ratio) if adr else (True, "no_adr_logged")
    stable = True
    stab_note = "ok"
    if len(tot) >= 2 and tot[-1] > tot[-2] * 1.05:
        stable = False
        stab_note = f"late_tot_rise {tot[-2]:.3g}->{tot[-1]:.3g}"

    ok = bool(len(bio) >= 3 and bio_ok and stable and (adr_ok or not passive_adr))

    return {
        "ok": ok,
        "n_epochs": len(eps),
        "passive_adr_in_backprop": passive_adr,
        "ADR_residual_mode": eps[-1].get("ADR_residual_mode") if eps else None,
        "ADR_mask_mode": eps[-1].get("ADR_mask_mode") if eps else None,
        "passive_adr_weight": adr_w,
        "L_bio": bio,
        "L_ADR_S": adr,
        "L_tot": tot,
        "bio_ok": bio_ok,
        "bio_note": bio_note,
        "adr_ok": adr_ok,
        "adr_note": adr_note,
        "stable": stable,
        "stable_note": stab_note,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--max-ratio", type=float, default=0.92)
    args = ap.parse_args()

    rd = _find_run_dir(run_note=args.run_note)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2

    out = eval_run(rd, max_ratio=args.max_ratio)
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    print(f"{tag} m3_narrow: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
