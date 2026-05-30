"""Rank I.1 X block legs from run.jsonl / x_block_log.jsonl; optionally pick best ckpt path.

Usage:
  python scripts/summarize_passive_x_block.py
  python scripts/summarize_passive_x_block.py --pick-best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_REPORTS = _REPO / "outputs" / "reports" / "training" / "biochem"
_OUT = _REPO / "outputs" / "biochem" / "x_block"
_NOTES = (
    "x_block_X3_mask_global",
    "x_block_X4_data_bio",
    "x_block_X5_fi2mat2",
    "x_block_X_m3_union",
    "x_block_X6_confirm",
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_run_dir(run_note: str) -> Path | None:
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


def _last_val_fi(run_dir: Path) -> float | None:
    rows = _load_jsonl(run_dir / "run.jsonl")
    fi_vals = []
    for r in rows:
        if r.get("event") == "val" and r.get("stage") == "teacher":
            v = r.get("val_species_fi_log_mae")
            if v is not None:
                fi_vals.append((int(r.get("epoch", 0)), float(v)))
    if not fi_vals:
        return None
    fi_vals.sort(key=lambda x: x[0])
    return fi_vals[-1][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick-best", action="store_true")
    ap.add_argument("--out", default=str(_OUT / "x_block_summary.json"))
    args = ap.parse_args()

    ranked: list[dict] = []
    for note in _NOTES:
        rd = _find_run_dir(note)
        fi = _last_val_fi(rd) if rd else None
        ckpt = _OUT / f"{note}_last.pth"
        ranked.append(
            {
                "run_note": note,
                "val_species_fi_log_mae": fi,
                "ckpt": str(ckpt) if ckpt.is_file() else None,
                "run_dir": str(rd) if rd else None,
            }
        )

    ranked.sort(
        key=lambda r: (r["val_species_fi_log_mae"] is None, r["val_species_fi_log_mae"] or 999.0),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"legs": ranked}, indent=2), encoding="utf-8")

    print("[i]  X block legs (lowest val FI first):")
    for row in ranked:
        fi_s = f"{row['val_species_fi_log_mae']:.4f}" if row["val_species_fi_log_mae"] is not None else "n/a"
        print(f"     {row['run_note']:28s}  FI={fi_s}  ckpt={row['ckpt']}")

    if args.pick_best:
        best = next((r for r in ranked if r.get("ckpt")), None)
        if not best:
            locked = _REPO / "outputs/biochem/biochem_teacher_passive_align_locked.pth"
            pick = locked if locked.is_file() else None
        else:
            pick = Path(best["ckpt"])
        if pick is None or not pick.is_file():
            print("[ERR] no ckpt to pick", file=sys.stderr)
            return 2
        pick_txt = _OUT / "best_teacher_for_dump.txt"
        pick_txt.write_text(str(pick.resolve()), encoding="utf-8")
        print(f"[OK] pick -> {pick_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
