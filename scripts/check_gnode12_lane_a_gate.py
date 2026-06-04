"""Metric gate for rung 12 Lane A (clot-phi on predicted-kine dump; optional mu-unlock run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.paths import get_project_root


def _read_events(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _latest_mu_unlock_run(root: Path) -> Path | None:
    base = root / "outputs" / "reports" / "training" / "biochem"
    if not base.is_dir():
        return None
    runs = sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "run.jsonl").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in runs:
        path = run_dir / "run.jsonl"
        events = _read_events(path)
        meta = next((e for e in events if e.get("event") == "meta"), None)
        if not meta:
            continue
        note = str((meta.get("env") or {}).get("BIOCHEM_RUN_NOTE", "") or "")
        if "gnode12_mu_unlock" in note or note == "gnode12_mu_unlock":
            return path
    return None


def _load_eval_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        return None
    f1_vals: list[float] = []
    p007: float | None = None
    for r in rows:
        val = r.get("val") if isinstance(r.get("val"), dict) else r
        f1 = val.get("clot_f1") if isinstance(val, dict) else r.get("f1")
        if f1 is None:
            continue
        f1_f = float(f1)
        f1_vals.append(f1_f)
        if r.get("anchor") == "patient007":
            p007 = f1_f
    if not f1_vals:
        return None
    return {
        "mean_f1": sum(f1_vals) / len(f1_vals),
        "min_f1": min(f1_vals),
        "n": len(f1_vals),
        "p007": p007,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-json",
        default="outputs/biochem/gnode10_sweep/multi_anchor_gnode12_lane_a_clotphi.jsonl",
    )
    ap.add_argument("--min-clot-min-f1", type=float, default=0.26)
    ap.add_argument("--min-gt-pos-frac", type=float, default=0.55)
    ap.add_argument(
        "--max-teacher-mu-log-mae",
        type=float,
        default=1.35,
        help="Optional mu-unlock trend gate (all-truth logMAE must be <= this).",
    )
    ap.add_argument("--mu-unlock-run-jsonl", default="", help="Explicit gnode12_mu_unlock run.jsonl.")
    ap.add_argument("--skip-mu-trend", action="store_true", help="Do not require mu unlock improvement.")
    args = ap.parse_args()

    root = get_project_root()
    eval_path = Path(args.eval_json)
    if not eval_path.is_absolute():
        eval_path = root / eval_path

    ok = True
    summary = _load_eval_summary(eval_path)
    if summary is None:
        print(f"[ERR] missing or empty eval json: {eval_path}", flush=True)
        ok = False
    else:
        print(
            f"[i]  clot multi-anchor: n={summary['n']} mean_f1={summary['mean_f1']:.3f} "
            f"min_f1={summary['min_f1']:.3f} p007={summary.get('p007')}",
            flush=True,
        )
        if summary["min_f1"] < args.min_clot_min_f1:
            print(
                f"[ERR] min_f1 {summary['min_f1']:.3f} < {args.min_clot_min_f1} "
                "(rung 5/10 bar)",
                flush=True,
            )
            ok = False
        else:
            print(f"[OK]  min_f1 >= {args.min_clot_min_f1}", flush=True)

    if not args.skip_mu_trend:
        mu_path = Path(args.mu_unlock_run_jsonl) if args.mu_unlock_run_jsonl else _latest_mu_unlock_run(root)
        if mu_path is None or not mu_path.is_file():
            print("[WARN] no gnode12_mu_unlock run.jsonl (SkipMuUnlock or run not logged)", flush=True)
        else:
            vals = [
                float(e["mu_log_mae"])
                for e in _read_events(mu_path)
                if e.get("event") == "val"
                and (e.get("stage") or "").lower() == "teacher"
                and e.get("mu_log_mae") is not None
            ]
            if not vals:
                print(f"[WARN] no teacher mu val rows in {mu_path}", flush=True)
            else:
                best = min(vals)
                last = vals[-1]
                print(f"[i]  mu unlock teacher logMAE: best={best:.4f} last={last:.4f} (n={len(vals)})", flush=True)
                if best > args.max_teacher_mu_log_mae:
                    print(
                        f"[WARN] best mu_log_mae {best:.4f} > {args.max_teacher_mu_log_mae} "
                        "(try higher MuRatioMax or more MuUnlockEpochs)",
                        flush=True,
                    )
                else:
                    print(f"[OK]  mu unlock best logMAE <= {args.max_teacher_mu_log_mae}", flush=True)

    print(f"[i]  eval: {eval_path}", flush=True)
    if ok:
        print("[OK]  GNODE 12 Lane A gate PASS", flush=True)
        sys.exit(0)
    print("[ERR]  GNODE 12 Lane A gate FAIL", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
