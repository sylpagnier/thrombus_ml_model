"""Rank GNODE rung-10 sweep legs from manifest + run.jsonl val rows.

Lower ``composite_score`` is better. Writes ``leaderboard.json`` and optional ``winners.json``.

Usage:
  python scripts/score_gnode10_sweep.py --sweep-dir outputs/biochem/gnode10_sweep
  python scripts/score_gnode10_sweep.py --sweep-dir outputs/biochem/gnode10_sweep --phase probe --top 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_REPORTS = _REPO / "outputs" / "reports" / "training" / "biochem"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _find_run_dir(run_id: str) -> Path | None:
    p = _REPORTS / run_id
    return p if p.is_dir() else None


def _best_val_row(run_id: str) -> dict[str, Any] | None:
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        return None
    val_rows = [r for r in _load_jsonl(run_dir / "run.jsonl") if r.get("event") == "val"]
    if not val_rows:
        return None

    def _key(r: dict[str, Any]) -> tuple[float, float]:
        fi = float(r.get("val_species_fi_log_mae", float("inf")))
        health = float(r.get("val_viz_health_score", float("inf")))
        return (fi, health)

    return min(val_rows, key=_key)


def _score_val(row: dict[str, Any], train_row: dict[str, Any] | None) -> dict[str, float]:
    fi = float(row.get("val_species_fi_log_mae", float("nan")))
    mat = float(row.get("val_species_mat_log_mae", float("nan")))
    speed = float(row.get("val_viz_t0_speed_mean", float("nan")))
    health = float(row.get("val_viz_health_score", float("nan")))
    mu_all = float(row.get("val_mu_log_mae", float("nan")))

    fi_term = fi if math.isfinite(fi) else 9.0
    mat_term = mat if math.isfinite(mat) else 9.0

    flow_pen = 0.0
    if math.isfinite(speed):
        if speed < 0.15:
            flow_pen += 2.0 * (0.15 - speed)
        if speed > 1.35:
            flow_pen += 1.0 * (speed - 1.35)

    health_term = health / 10.0 if math.isfinite(health) else 1.0
    mu_term = 0.05 * mu_all if math.isfinite(mu_all) else 0.5

    kine_pen = 0.0
    if train_row is not None:
        lk = float(train_row.get("train_L_kine_avg", float("nan")))
        lb0 = float(train_row.get("train_L_bio_avg", float("nan")))
        if math.isfinite(lk) and lk > 2.0:
            kine_pen += 0.15 * (lk - 2.0)

    composite = (
        3.0 * fi_term
        + 0.5 * mat_term
        + flow_pen
        + 0.35 * health_term
        + mu_term
        + kine_pen
    )
    return {
        "composite_score": composite,
        "val_species_fi_log_mae": fi,
        "val_species_mat_log_mae": mat,
        "val_viz_t0_speed_mean": speed,
        "val_viz_health_score": health,
        "val_mu_log_mae": mu_all,
        "flow_penalty": flow_pen,
    }


def _train_last_row(run_id: str) -> dict[str, Any] | None:
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        return None
    val_rows = [r for r in _load_jsonl(run_dir / "run.jsonl") if r.get("event") == "val"]
    if not val_rows:
        return None
    return max(val_rows, key=lambda r: int(r.get("epoch", -1)))


def score_manifest(manifest: Path, phase: str | None, top: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(manifest)
    scored: list[dict[str, Any]] = []
    for row in rows:
        if row.get("event") != "leg":
            continue
        if phase and str(row.get("phase", "")) != phase:
            continue
        if str(row.get("status", "")).upper() not in ("OK", "WARN"):
            continue
        run_id = str(row.get("run_id", "") or "")
        if not run_id:
            continue
        best = _best_val_row(run_id)
        if best is None:
            continue
        train_last = _train_last_row(run_id)
        metrics = _score_val(best, train_last)
        scored.append(
            {
                "leg_id": row.get("leg_id"),
                "phase": row.get("phase"),
                "run_id": run_id,
                "run_note": row.get("run_note"),
                "epoch": best.get("epoch"),
                "ckpt_dir": row.get("ckpt_dir"),
                **metrics,
            }
        )
    scored.sort(key=lambda r: float(r["composite_score"]))
    if top > 0:
        scored = scored[:top]
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, default=_REPO / "outputs" / "biochem" / "gnode10_sweep")
    ap.add_argument("--phase", type=str, default="", help="probe | semi | final (empty = all)")
    ap.add_argument("--top", type=int, default=0, help="Keep top N (0 = all)")
    ap.add_argument("--write-winners", action="store_true", help="Write winners.json for requested phase")
    args = ap.parse_args()

    manifest = args.sweep_dir / "manifest.jsonl"
    if not manifest.is_file():
        print(f"[ERR] missing manifest: {manifest}", file=sys.stderr)
        return 1

    phase = args.phase.strip() or None
    ranked = score_manifest(manifest, phase, args.top if args.top > 0 else 0)

    out_lb = args.sweep_dir / "leaderboard.json"
    out_lb.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    print(f"[i]  leaderboard ({len(ranked)} legs) -> {out_lb}")
    for i, row in enumerate(ranked[: min(10, len(ranked))], start=1):
        fi = row.get("val_species_fi_log_mae")
        fi_s = f"{fi:.4f}" if isinstance(fi, (int, float)) and math.isfinite(fi) else "nan"
        spd = row.get("val_viz_t0_speed_mean")
        spd_s = f"{spd:.3f}" if isinstance(spd, (int, float)) and math.isfinite(spd) else "nan"
        print(
            f"  {i:2d}. {row.get('leg_id')} ({row.get('phase')}) "
            f"score={row['composite_score']:.3f} FI={fi_s} t0|u|={spd_s} run={row.get('run_id')}"
        )

    if args.write_winners and phase and ranked:
        winners_path = args.sweep_dir / f"winners_{phase}.json"
        payload = {
            "phase": phase,
            "top": ranked[: max(1, args.top)] if args.top > 0 else ranked[:3],
            "winner": ranked[0],
        }
        winners_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK]  winners -> {winners_path} ({payload['winner'].get('leg_id')})")

    return 0 if ranked else 2


if __name__ == "__main__":
    raise SystemExit(main())
