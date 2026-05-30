"""Gate passive teacher runs from run.jsonl (flow health + L_Back descent).

Usage:
  python scripts/check_passive_teacher_gate.py --run-note ladder_smoke_cb
  python scripts/check_passive_teacher_gate.py --run-dir outputs/reports/training/biochem/20260528T175241Z
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
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _find_run_dir(*, run_note: str | None, run_dir: Path | None) -> Path | None:
    if run_dir is not None:
        return run_dir if run_dir.is_dir() else None
    if not run_note:
        return None
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
    candidates = sorted(_REPORTS.glob("*/run.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        rows = _load_jsonl(p)
        if rows and rows[0].get("run_note") == run_note:
            return p.parent
    return None


def evaluate_run_dir(run_dir: Path, *, min_speed: float, max_l_back_ratio: float) -> dict:
    rows = _load_jsonl(run_dir / "run.jsonl")
    vals = [r for r in rows if r.get("event") == "val" and r.get("stage") == "teacher"]
    if not vals:
        return {"ok": False, "reason": "no teacher val rows in run.jsonl"}

    speeds = [float(v["val_viz_t0_speed_mean"]) for v in vals if v.get("val_viz_t0_speed_mean") is not None]
    backs = [float(v["train_L_back_avg"]) for v in vals if v.get("train_L_back_avg") is not None]
    if not speeds or not backs:
        return {"ok": False, "reason": "missing speed or L_Back in val rows"}

    speed_ok = min(speeds) >= min_speed
    l0, l1 = backs[0], backs[-1]
    back_ok = l1 < l0 * max_l_back_ratio if l0 > 0 else True

    return {
        "ok": bool(speed_ok and back_ok),
        "run_note": rows[0].get("run_note"),
        "n_val": len(vals),
        "t0_speed_min": min(speeds),
        "t0_speed_last": speeds[-1],
        "L_back_first": l0,
        "L_back_last": l1,
        "speed_ok": speed_ok,
        "L_back_ok": back_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--min-speed", type=float, default=0.5)
    ap.add_argument("--max-l-back-ratio", type=float, default=0.95, help="last/first must be below this")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    rd = _find_run_dir(run_note=(args.run_note or None), run_dir=run_dir)
    if rd is None:
        print("[ERR] run not found", file=sys.stderr)
        return 2

    out = evaluate_run_dir(rd, min_speed=args.min_speed, max_l_back_ratio=args.max_l_back_ratio)
    out["run_dir"] = str(rd)
    tag = "[OK]" if out["ok"] else "[FAIL]"
    print(f"{tag} passive teacher gate: {json.dumps(out, sort_keys=True)}")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
