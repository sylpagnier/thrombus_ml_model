"""Phase A pass/fail from teacher run.jsonl.

X pass: train_L_bio_avg falls (last/first <= ratio) over >=3 epochs; flow not trivial.
Y pass: isolated train loss for target term falls monotonically epoch-over-epoch
        (allow one small rebound on last epoch if overall last/first <= ratio).

Usage:
  python scripts/check_phase_a_gate.py --mode x --run-note phaseA_X_cb_lr1e3_seed101
  python scripts/check_phase_a_gate.py --mode y --run-note phaseA_Y_ADR_F_clip10 --term ADR_F
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_REPORTS = _REPO / "outputs" / "reports" / "training" / "biochem"

_TERM_KEYS = {
    "ADR_F": "train_L_ADR_F_avg",
    "ADR_S": "train_L_ADR_S_avg",
    "W_PHY": "train_L_W_Phy_avg",
    "W_BIO": "train_L_W_Bio_avg",
    "BIO_IO": "train_L_B_IO_avg",
    "DATA_BIO": "train_L_bio_avg",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
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


def _teacher_epochs(run_dir: Path) -> list[dict]:
    rows = _load_jsonl(run_dir / "run.jsonl")
    out = []
    for r in rows:
        if r.get("event") != "val" or r.get("stage") != "teacher":
            continue
        if r.get("epoch") is None:
            continue
        out.append(r)
    out.sort(key=lambda x: int(x["epoch"]))
    return out


def _series(epochs: list[dict], key: str) -> list[float]:
    vals = []
    for r in epochs:
        v = r.get(key)
        if v is not None:
            vals.append(float(v))
    return vals


def _descent_ok(vals: list[float], *, max_ratio: float, monotone: bool) -> tuple[bool, str]:
    if len(vals) < 2:
        return False, "need>=2_epochs"
    first, last = vals[0], vals[-1]
    if first <= 0:
        return False, "first<=0"
    ratio = last / first
    if ratio > max_ratio:
        return False, f"ratio={ratio:.4f}>{max_ratio}"
    if monotone:
        for a, b in zip(vals, vals[1:]):
            if b > a * 1.02:
                return False, "non_monotone"
    return True, f"ratio={ratio:.4f}"


def eval_x(run_dir: Path, *, max_ratio: float) -> dict:
    eps = _teacher_epochs(run_dir)
    bio = _series(eps, "train_L_bio_avg")
    ok_bio, note_bio = _descent_ok(bio, max_ratio=max_ratio, monotone=False)
    speeds = [float(r["val_viz_t0_speed_mean"]) for r in eps if r.get("val_viz_t0_speed_mean") is not None]
    speed_ok = bool(speeds) and min(speeds) >= 0.5
    return {
        "ok": bool(ok_bio and speed_ok and len(bio) >= 3),
        "n_epochs": len(bio),
        "L_bio": bio,
        "bio_ok": ok_bio,
        "bio_note": note_bio,
        "speed_ok": speed_ok,
        "t0_speed_min": min(speeds) if speeds else None,
    }


def eval_y(run_dir: Path, *, term: str, max_ratio: float) -> dict:
    key = _TERM_KEYS.get(term.upper())
    if not key:
        return {"ok": False, "reason": f"unknown term {term}"}
    eps = _teacher_epochs(run_dir)
    vals = _series(eps, key)
    if len(vals) < 2:
        # Isolate runs often log only train_L_tot / train_L_back_avg in run.jsonl val rows.
        vals = _series(eps, "train_L_tot")
        key = "train_L_tot"
    ok, note = _descent_ok(vals, max_ratio=max_ratio, monotone=True)
    return {
        "ok": bool(ok and len(vals) >= 3),
        "term": term.upper(),
        "key": key,
        "n_epochs": len(vals),
        "values": vals,
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("x", "y"), required=True)
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--term", default="ADR_F", help="Y mode: ADR_F|ADR_S|W_PHY|W_BIO|BIO_IO")
    ap.add_argument("--max-ratio", type=float, default=0.92, help="last/first must be below this")
    args = ap.parse_args()

    rd = _find_run_dir(run_note=args.run_note)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2

    if args.mode == "x":
        out = eval_x(rd, max_ratio=args.max_ratio)
    else:
        out = eval_y(rd, term=args.term, max_ratio=args.max_ratio)
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    print(f"{tag} phaseA_{args.mode}: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
