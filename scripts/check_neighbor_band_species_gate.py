"""Gate for neighbor-band FI/Mat species teacher (GT flow, fi_mat scope).

Usage:
  python scripts/check_neighbor_band_species_gate.py --run-note neighbor_band_species
  python scripts/check_neighbor_band_species_gate.py --checkpoint outputs/biochem/biochem_teacher_last.pth
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_run_jsonl(run_note: str | None) -> list[dict]:
    if not run_note:
        return []
    path = _REPO / "outputs" / "reports" / "training" / "biochem" / run_note / "run.jsonl"
    if not path.is_file():
        path = _REPO / "outputs" / "reports" / "training" / "biochem" / f"{run_note}.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _series(rows: list[dict], key: str) -> list[tuple[int, float]]:
    out = []
    for r in rows:
        if r.get("event") == "val" and key in r:
            out.append((int(r.get("epoch", -1)), float(r[key])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", default="neighbor_band_species")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--species-fi-max", type=float, default=0.06)
    ap.add_argument("--min-band-f1", type=float, default=0.35)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    fi_max = float(args.species_fi_max) * (1.5 if args.probe else 1.0)
    f1_min = float(args.min_band_f1) * (0.85 if args.probe else 1.0)

    ckpt = args.checkpoint or str(_REPO / "outputs" / "biochem" / "biochem_teacher_last.pth")
    rows = _load_run_jsonl(args.run_note)
    fi_series = _series(rows, "val_species_fi_log_mae")
    species_ok = bool(fi_series) and min(v for _, v in fi_series) <= fi_max

    eval_cmd = [
        sys.executable,
        str(_REPO / "scripts" / "eval_neighbor_band_trigger.py"),
        "--checkpoint",
        ckpt,
        "--split",
        "val",
        "--min-band-f1",
        str(f1_min),
    ]
    trigger_rc = subprocess.call(eval_cmd, cwd=str(_REPO))

    ok = species_ok and trigger_rc == 0
    print(
        json.dumps(
            {
                "gate": "neighbor_band_species",
                "pass": ok,
                "species_ok": species_ok,
                "trigger_ok": trigger_rc == 0,
                "fi_max": fi_max,
                "band_f1_min": f1_min,
                "best_val_fi": min((v for _, v in fi_series), default=float("nan")),
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
