"""Gate for M3 alignment probe: L_bio + masked ADR co-descent, adequate mask size, species trend.

Pass (default):
  - >= 6 teacher val rows
  - train_L_bio_avg last/first <= 0.90
  - train_L_ADR_S_avg last/first <= 0.85 (masked ADR must fall)
  - data_bio_mask_n >= min_mask_n (default 32) on last epoch
  - optional: val_species_fi_log_mae not worse than ep0 by >5%

Usage:
  python scripts/check_m3_align_gate.py --run-note m3_align_transport_union
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


def _ratio_descent(vals: list[float], *, max_ratio: float) -> tuple[bool, str]:
    if len(vals) < 2:
        return False, "need>=2_epochs"
    a, b = vals[0], vals[-1]
    if a <= 0:
        return False, "first<=0"
    ratio = b / a
    if ratio > max_ratio:
        return False, f"ratio={ratio:.4f}>{max_ratio}"
    return True, f"ratio={ratio:.4f}"


def eval_run(
    run_dir: Path,
    *,
    bio_max_ratio: float,
    adr_max_ratio: float,
    min_mask_n: float,
    min_epochs: int,
) -> dict:
    eps = [
        r
        for r in _load_jsonl(run_dir / "run.jsonl")
        if r.get("event") == "val" and r.get("stage") == "teacher"
    ]
    eps.sort(key=lambda x: int(x["epoch"]))

    bio = _series(eps, "train_L_bio_avg") or _series(eps, "train_L_back_avg")
    adr = _series(eps, "train_L_ADR_S_avg")
    mask_n = (
        _series(eps, "data_bio_mask_n")
        or _series(eps, "val_species_mask_n")
        or _series(eps, "ADR_mask_n")
    )
    adr_mask_n = _series(eps, "ADR_mask_n")
    fi = _series(eps, "val_species_fi_log_mae")

    bio_ok, bio_note = _ratio_descent(bio, max_ratio=bio_max_ratio)
    adr_ok, adr_note = _ratio_descent(adr, max_ratio=adr_max_ratio) if adr else (False, "no_adr")
    mask_ok = bool(mask_n) and mask_n[-1] >= min_mask_n
    mask_note = f"mask_n={mask_n[-1]:.0f}" if mask_n else "mask_n_missing"
    species_ok = True
    species_note = "n/a"
    if len(fi) >= 2:
        species_ok, species_note = _ratio_descent(fi, max_ratio=1.05)

    ok = bool(
        len(eps) >= min_epochs
        and bio_ok
        and adr_ok
        and mask_ok
        and species_ok
    )

    return {
        "ok": ok,
        "n_epochs": len(eps),
        "L_bio": bio,
        "L_ADR_S": adr,
        "data_bio_mask_n": mask_n,
        "ADR_mask_n": adr_mask_n,
        "val_species_fi_log_mae": fi,
        "bio_ok": bio_ok,
        "bio_note": bio_note,
        "adr_ok": adr_ok,
        "adr_note": adr_note,
        "mask_ok": mask_ok,
        "mask_note": mask_note,
        "species_ok": species_ok,
        "species_note": species_note,
        "ADR_residual_mode": eps[-1].get("ADR_residual_mode") if eps else None,
        "ADR_mask_times": eps[-1].get("ADR_mask_times") if eps else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--bio-max-ratio", type=float, default=0.90)
    ap.add_argument("--adr-max-ratio", type=float, default=0.85)
    ap.add_argument("--min-mask-n", type=float, default=32.0)
    ap.add_argument("--min-epochs", type=int, default=6)
    args = ap.parse_args()

    rd = _find_run_dir(run_note=args.run_note)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2

    out = eval_run(
        rd,
        bio_max_ratio=args.bio_max_ratio,
        adr_max_ratio=args.adr_max_ratio,
        min_mask_n=args.min_mask_n,
        min_epochs=args.min_epochs,
    )
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    print(f"{tag} m3_align: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
