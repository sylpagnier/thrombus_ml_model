"""Plumbing gate for rung 11 finish (Phase II.0 pseudo supervision; not a metric gate)."""

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
    ap.add_argument("--min-corrector-val", type=int, default=3, help="Minimum corrector val rows.")
    ap.add_argument("--min-pseudo-w", type=float, default=0.01, help="Minimum end-event pseudo_w.")
    ap.add_argument(
        "--min-pseudo-coverage",
        type=float,
        default=0.5,
        help="Minimum pseudo_label_coverage on end event (0-1).",
    )
    args = ap.parse_args()

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
        if meta.get("stop_after_teacher") in (True, 1, "1"):
            print("[ERR] stop_after_teacher=1 (corrector did not run)", flush=True)
            ok = False
        else:
            print("[OK]  meta: stop_after_teacher=0", flush=True)
        env = meta.get("env") or {}
        step = str(env.get("BIOCHEM_COMPLEXITY_STEP", "") or "").strip().lower()
        data_only = str(env.get("BIOCHEM_LOSS_DATA_ONLY", "") or "").strip().lower()
        if step not in ("2", "2.0", "phase2", "teacher_v2", ""):
            print(f"[WARN] expected step-2 COMPLEXITY_STEP, got {step!r}", flush=True)
        if data_only not in ("1", "true", "yes", "on"):
            print(f"[WARN] expected LOSS_DATA_ONLY=1 for step-2 finish, got {data_only!r}", flush=True)
        else:
            print("[OK]  step-2 bridge env (LOSS_DATA_ONLY=1)", flush=True)
        pseudo_min = env.get("BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE")
        if pseudo_min is not None:
            print(f"[i]  PSEUDO_MIN_TEACHER_MU_SCORE={pseudo_min}", flush=True)

    if len(teacher_vals) < 1:
        print("[WARN] no teacher val rows", flush=True)
    else:
        print(f"[OK]  teacher val rows: {len(teacher_vals)}", flush=True)

    if len(corrector_vals) < args.min_corrector_val:
        print(
            f"[ERR] corrector val rows {len(corrector_vals)} < {args.min_corrector_val}",
            flush=True,
        )
        ok = False
    else:
        print(f"[OK]  corrector val rows: {len(corrector_vals)}", flush=True)

    if end is None:
        print("[ERR] no end event (run interrupted?)", flush=True)
        ok = False
    else:
        print(
            f"[OK]  end: teacher_best_epoch={end.get('teacher_best_epoch')} "
            f"last={end.get('last_epoch_completed')}",
            flush=True,
        )
        pw = end.get("pseudo_w")
        if pw is None:
            print(
                "[ERR] end event missing pseudo_w (re-run with updated train_biochem_corrector)",
                flush=True,
            )
            ok = False
        else:
            pw_f = float(pw)
            print(f"[i]  pseudo_w={pw_f:.4f}", flush=True)
            if pw_f < args.min_pseudo_w:
                print(
                    f"[ERR] pseudo_w {pw_f:.4f} < {args.min_pseudo_w} "
                    "(lower PSEUDO_MIN_TEACHER_MU_SCORE or improve teacher mu_score)",
                    flush=True,
                )
                ok = False
            else:
                print(f"[OK]  pseudo_w >= {args.min_pseudo_w}", flush=True)
        cov = end.get("pseudo_label_coverage")
        if cov is not None:
            cov_f = float(cov)
            print(f"[i]  pseudo_label_coverage={cov_f:.3f}", flush=True)
            if cov_f < args.min_pseudo_coverage:
                print(
                    f"[ERR] pseudo_label_coverage {cov_f:.3f} < {args.min_pseudo_coverage}",
                    flush=True,
                )
                ok = False
            else:
                print(f"[OK]  pseudo_label_coverage >= {args.min_pseudo_coverage}", flush=True)
        else:
            print("[WARN] end missing pseudo_label_coverage (older run log?)", flush=True)

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

    if ok:
        print("[OK]  GNODE 11 finish (Phase II.0) gate PASS", flush=True)
        sys.exit(0)
    print("[ERR]  GNODE 11 finish gate FAIL", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
