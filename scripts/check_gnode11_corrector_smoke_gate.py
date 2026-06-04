"""Plumbing gate for rung 11a/11b corrector smoke (not a metric gate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.paths import get_project_root


def _latest_run_jsonl(root: Path) -> Path | None:
    base = root / "outputs" / "reports" / "training" / "biochem"
    if not base.is_dir():
        return None
    runs = sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "run.jsonl").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return (runs[0] / "run.jsonl") if runs else None


def _read_events(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-jsonl", default="", help="Explicit run.jsonl path.")
    ap.add_argument("--archive-dir", default="", help="Optional; only used for log hint.")
    ap.add_argument("--max-species-fi", type=float, default=0.15, help="Species FI sanity ceiling.")
    ap.add_argument(
        "--step2",
        action="store_true",
        help="Require step-2 bridge (COMPLEXITY_STEP=2, LOSS_DATA_ONLY=1).",
    )
    ap.add_argument(
        "--step3",
        action="store_true",
        help="Require step-3 multitask (COMPLEXITY_STEP=3, LOSS_DATA_ONLY=0).",
    )
    args = ap.parse_args()
    if args.step2 and args.step3:
        print("[ERR] use only one of --step2 or --step3", flush=True)
        sys.exit(1)

    root = get_project_root()
    run_path = Path(args.run_jsonl) if args.run_jsonl else _latest_run_jsonl(root)
    if not run_path or not run_path.is_file():
        print("[ERR] No run.jsonl found under outputs/reports/training/biochem/", flush=True)
        sys.exit(1)

    events = _read_events(run_path)
    meta = next((e for e in events if e.get("event") == "meta"), None)
    end = next((e for e in events if e.get("event") == "end"), None)
    val_rows = [e for e in events if e.get("event") == "val"]
    corrector_vals = [e for e in val_rows if (e.get("stage") or "").lower() == "corrector"]
    teacher_vals = [e for e in val_rows if (e.get("stage") or "").lower() == "teacher"]

    ok = True
    if meta is None:
        print("[ERR] missing meta event", flush=True)
        ok = False
    else:
        stop_teacher = meta.get("stop_after_teacher")
        env_stop = (meta.get("env") or {}).get("BIOCHEM_STOP_AFTER_TEACHER", "")
        if stop_teacher in (True, 1, "1"):
            print("[ERR] stop_after_teacher=1 (corrector did not run)", flush=True)
            ok = False
        else:
            print(f"[OK]  meta: stop_after_teacher=0 (env={env_stop})", flush=True)
        env = meta.get("env") or {}
        step = str(env.get("BIOCHEM_COMPLEXITY_STEP", "") or "").strip().lower()
        data_only = str(env.get("BIOCHEM_LOSS_DATA_ONLY", "") or "").strip().lower()
        if args.step3:
            if step not in ("3", "3.0", "phase3", "full_multitask", "corrector_full"):
                print(f"[ERR] expected COMPLEXITY_STEP=3, got {step!r}", flush=True)
                ok = False
            elif data_only in ("1", "true", "yes", "on"):
                print(f"[ERR] expected LOSS_DATA_ONLY=0 for step-3, got {data_only!r}", flush=True)
                ok = False
            else:
                print("[OK]  step-3 env: COMPLEXITY_STEP=3, LOSS_DATA_ONLY=0", flush=True)
        elif args.step2:
            if step not in ("2", "2.0", "phase2", "teacher_v2", ""):
                print(f"[WARN] expected COMPLEXITY_STEP=2, got {step!r}", flush=True)
            if data_only not in ("1", "true", "yes", "on"):
                print(f"[WARN] expected LOSS_DATA_ONLY=1 for step-2, got {data_only!r}", flush=True)
            else:
                print("[OK]  step-2 env: LOSS_DATA_ONLY=1", flush=True)

    if len(teacher_vals) < 1:
        print("[WARN] no teacher val rows (short run?)", flush=True)
    else:
        print(f"[OK]  teacher val rows: {len(teacher_vals)}", flush=True)

    if len(corrector_vals) < 1:
        print("[ERR] no corrector val rows (Phase 3 did not run)", flush=True)
        ok = False
    else:
        print(f"[OK]  corrector val rows: {len(corrector_vals)}", flush=True)

    if end is None:
        print("[WARN] no end event (run may have been interrupted)", flush=True)
    else:
        print(f"[OK]  end: teacher_best_epoch={end.get('teacher_best_epoch')} last={end.get('last_epoch_completed')}", flush=True)

    fi_vals = []
    for v in val_rows:
        fi = v.get("val_species_fi_log_mae")
        if fi is not None:
            fi_vals.append(float(fi))
    if fi_vals:
        last_fi = fi_vals[-1]
        print(f"[i]  val species FI (last): {last_fi:.4f}", flush=True)
        if last_fi > args.max_species_fi:
            print(f"[WARN] species FI > {args.max_species_fi} (unstable)", flush=True)
            ok = False
        else:
            print(f"[OK]  species FI <= {args.max_species_fi}", flush=True)

    print(f"[i]  run log: {run_path}", flush=True)
    if args.archive_dir:
        print(f"[i]  archive: {args.archive_dir}", flush=True)

    label = "11b step-3" if args.step3 else ("11a step-2" if args.step2 else "11a/11b")
    if ok:
        print(f"[OK]  GNODE {label} plumbing gate PASS", flush=True)
        sys.exit(0)
    print(f"[ERR]  GNODE {label} plumbing gate FAIL", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
