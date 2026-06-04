"""Shared metric gates for GNODE rung 12 Lane A / Lane B."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.paths import get_project_root


def read_events(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def latest_mu_unlock_run(root: Path) -> Path | None:
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
        events = read_events(path)
        meta = next((e for e in events if e.get("event") == "meta"), None)
        if not meta:
            continue
        note = str((meta.get("env") or {}).get("BIOCHEM_RUN_NOTE", "") or "")
        if "gnode12_mu_unlock" in note or note == "gnode12_mu_unlock":
            return path
    return None


def load_eval_summary(path: Path) -> dict | None:
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


def run_lane_gate(
    *,
    lane_label: str,
    eval_path: Path,
    min_clot_min_f1: float,
    skip_mu_trend: bool,
    max_teacher_mu_log_mae: float,
    mu_unlock_run_jsonl: str,
    compare_lane_a_json: str = "",
    warn_if_below_lane_a_p007: bool = False,
) -> int:
    root = get_project_root()
    ok = True
    summary = load_eval_summary(eval_path)
    if summary is None:
        print(f"[ERR] missing or empty eval json: {eval_path}", flush=True)
        ok = False
    else:
        print(
            f"[i]  clot multi-anchor: n={summary['n']} mean_f1={summary['mean_f1']:.3f} "
            f"min_f1={summary['min_f1']:.3f} p007={summary.get('p007')}",
            flush=True,
        )
        if summary["min_f1"] < min_clot_min_f1:
            print(
                f"[ERR] min_f1 {summary['min_f1']:.3f} < {min_clot_min_f1} (rung 5/10 bar)",
                flush=True,
            )
            ok = False
        else:
            print(f"[OK]  min_f1 >= {min_clot_min_f1}", flush=True)

    if warn_if_below_lane_a_p007 and summary is not None:
        lane_a_path = Path(compare_lane_a_json) if compare_lane_a_json else (
            root / "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/multi_anchor.jsonl"
        )
        if not lane_a_path.is_absolute():
            lane_a_path = root / lane_a_path
        lane_a = load_eval_summary(lane_a_path)
        if lane_a is None:
            print(f"[WARN] no Lane A eval for compare: {lane_a_path}", flush=True)
        else:
            p007_b = summary.get("p007")
            p007_a = lane_a.get("p007")
            if p007_b is not None and p007_a is not None:
                if float(p007_b) + 1e-6 < float(p007_a):
                    print(
                        f"[WARN] p007 F1 {p007_b:.3f} < Lane A {p007_a:.3f} "
                        "(corrector dump did not beat teacher-unlock path)",
                        flush=True,
                    )
                else:
                    print(f"[OK]  p007 F1 >= Lane A ({p007_a:.3f})", flush=True)

    if not skip_mu_trend:
        mu_path = Path(mu_unlock_run_jsonl) if mu_unlock_run_jsonl else latest_mu_unlock_run(root)
        if mu_path is None or not mu_path.is_file():
            print("[WARN] no gnode12_mu_unlock run.jsonl (SkipMuUnlock or run not logged)", flush=True)
        else:
            vals = [
                float(e["mu_log_mae"])
                for e in read_events(mu_path)
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
                if best > max_teacher_mu_log_mae:
                    print(
                        f"[WARN] best mu_log_mae {best:.4f} > {max_teacher_mu_log_mae} "
                        "(try higher MuRatioMax or more MuUnlockEpochs)",
                        flush=True,
                    )
                else:
                    print(f"[OK]  mu unlock best logMAE <= {max_teacher_mu_log_mae}", flush=True)

    print(f"[i]  eval: {eval_path}", flush=True)
    if ok:
        print(f"[OK]  GNODE 12 Lane {lane_label} gate PASS", flush=True)
        return 0
    print(f"[ERR]  GNODE 12 Lane {lane_label} gate FAIL", flush=True)
    return 1
