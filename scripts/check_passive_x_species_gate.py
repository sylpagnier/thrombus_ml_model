"""Species-only gate for I.1 X legs (no L_bio / m3_align descent requirement).

Pass (default):
  - val_species_fi_log_mae <= species_fi_max (0.05)
  - train_species_fi_log_mae_mean <= train_fi_max when logged (0.04)
  - val_viz_t0_speed_mean >= 0.5 when logged

Usage:
  python scripts/check_passive_x_species_gate.py --run-note x_block_X6_confirm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def _teacher_val_epochs(run_dir: Path) -> list[dict]:
    rows = _load_jsonl(run_dir / "run.jsonl")
    out = [r for r in rows if r.get("event") == "val" and r.get("stage") == "teacher" and r.get("epoch") is not None]
    out.sort(key=lambda x: int(x["epoch"]))
    return out


def _series(epochs: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in epochs if r.get(key) is not None]


def eval_species_gate(
    run_dir: Path,
    *,
    species_fi_max: float,
    train_fi_max: float,
    min_speed: float,
    probe: bool = False,
    probe_fi_max: float = 0.08,
    probe_improve_ratio: float = 0.98,
    probe_regress_ratio: float = 1.05,
) -> dict:
    eps = _teacher_val_epochs(run_dir)
    fi = _series(eps, "val_species_fi_log_mae")
    train_fi = _series(eps, "train_species_fi_log_mae_mean")
    speeds = _series(eps, "val_viz_t0_speed_mean")

    speed_ok = True
    if speeds:
        speed_ok = min(speeds) >= min_speed

    if probe:
        # Short legs: absolute cap OR mild improvement; hard fail on clear regression vs ep0.
        abs_ok = bool(fi) and fi[-1] <= probe_fi_max
        trend_ok = False
        regress_ok = True
        if len(fi) >= 2:
            trend_ok = fi[-1] <= fi[0] * probe_improve_ratio
            regress_ok = fi[-1] <= fi[0] * probe_regress_ratio
        val_ok = bool(fi) and (abs_ok or trend_ok) and regress_ok
        train_ok = True
        ok = bool(val_ok and speed_ok and len(fi) >= 1)
        mode = "probe"
    else:
        val_ok = bool(fi) and fi[-1] <= species_fi_max
        train_ok = True
        if train_fi:
            train_ok = train_fi[-1] <= train_fi_max
        ok = bool(val_ok and train_ok and speed_ok and len(eps) >= 1)
        mode = "promote"

    return {
        "ok": ok,
        "mode": mode,
        "n_epochs": len(eps),
        "val_species_fi_log_mae": fi,
        "train_species_fi_log_mae_mean": train_fi,
        "val_ok": val_ok,
        "train_ok": train_ok,
        "speed_ok": speed_ok,
        "t0_speed_min": min(speeds) if speeds else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--species-fi-max", type=float, default=0.05)
    ap.add_argument("--train-fi-max", type=float, default=0.04)
    ap.add_argument("--min-speed", type=float, default=0.5)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Relaxed 2-4ep leg: FI cap 0.08 or mild improve; fail if >5%% regression vs ep0.",
    )
    ap.add_argument("--probe-fi-max", type=float, default=0.08)
    ap.add_argument("--probe-improve-ratio", type=float, default=0.98)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    rd = None
    for attempt in range(8):
        rd = _find_run_dir(run_note=args.run_note)
        if rd is not None and _teacher_val_epochs(rd):
            break
        time.sleep(1.5 if attempt < 7 else 0.0)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2
    if not _teacher_val_epochs(rd):
        print(f"[ERR] run has no teacher val rows yet: {args.run_note}", file=sys.stderr)
        return 2

    out = eval_species_gate(
        rd,
        species_fi_max=args.species_fi_max,
        train_fi_max=args.train_fi_max,
        min_speed=args.min_speed,
        probe=args.probe,
        probe_fi_max=args.probe_fi_max,
        probe_improve_ratio=args.probe_improve_ratio,
    )
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    if not args.quiet:
        print(f"{tag} x_species: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
