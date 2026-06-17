"""Summarize species GNN s34 arch sweep vs locked baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.paths import get_project_root  # noqa: E402

SWEEP_DIR = "outputs/biochem/sweep_species_gnn_s34_arch"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=SWEEP_DIR)
    args = ap.parse_args()

    root = get_project_root()
    sweep = root / args.sweep_dir
    results_path = sweep / "sweep_results.json"
    if not results_path.is_file():
        print(f"[ERR] missing {results_path}", file=sys.stderr)
        return 1

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    baseline = float(payload.get("baseline_composite_score", 0.0))
    rows = list(payload.get("results") or [])
    rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    print(f"[i] baseline composite={baseline:.3f}  legs={len(rows)}", flush=True)
    print(f"{'leg':<18} {'score':>7} {'d_base':>7} {'p007':>6} {'p004':>6} {'p006':>6} {'mean_h':>6}", flush=True)
    print("-" * 62, flush=True)
    for r in rows:
        print(
            f"{r.get('leg','?'):<18} "
            f"{r.get('score', 0.0):7.3f} "
            f"{r.get('delta_vs_baseline', 0.0):+7.3f} "
            f"{r.get('patient007_f1', 0.0):6.3f} "
            f"{r.get('patient004_f1', 0.0):6.3f} "
            f"{r.get('patient006_f1', 0.0):6.3f} "
            f"{r.get('mean_holdout_f1', 0.0):6.3f}",
            flush=True,
        )

    winner = payload.get("winner") or (rows[0] if rows else None)
    summary = {
        "baseline_composite_score": baseline,
        "winner": winner,
        "ranked": rows,
    }
    out = sweep / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if winner:
        beat = winner.get("delta_vs_baseline", 0.0) >= 0.0
        print(
            f"\n[OK] winner={winner.get('leg')} score={winner.get('score', 0):.3f} "
            f"beats_baseline={beat}",
            flush=True,
        )
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
