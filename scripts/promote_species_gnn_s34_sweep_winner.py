"""Promote best leg from s34 arch sweep into deploy baseline (if it beats baseline)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.paths import get_project_root  # noqa: E402

SWEEP_DIR = "outputs/biochem/sweep_species_gnn_s34_arch"
MIN_DELTA = 0.005


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=SWEEP_DIR)
    ap.add_argument("--min-delta", type=float, default=MIN_DELTA)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = get_project_root()
    sweep = root / args.sweep_dir
    results_path = sweep / "sweep_results.json"
    if not results_path.is_file():
        print(f"[ERR] missing {results_path}", file=sys.stderr)
        return 1

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    winner = payload.get("winner")
    if not winner or winner.get("leg") == "ref_baseline":
        print("[i] no sweep winner to promote", flush=True)
        return 0

    delta = float(winner.get("delta_vs_baseline", 0.0))
    if delta < float(args.min_delta) and not args.force:
        print(
            f"[i] winner {winner.get('leg')} delta={delta:+.3f} < {args.min_delta}; skip promote",
            flush=True,
        )
        return 0

    src = Path(winner["ckpt"])
    if not src.is_file():
        print(f"[ERR] missing winner ckpt: {src}", file=sys.stderr)
        return 1

    dst = root / "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    meta_src = src.with_suffix(".json")
    if meta_src.is_file():
        shutil.copy2(meta_src, dst.with_suffix(".json"))

    note = {
        "promoted_from_sweep_leg": winner.get("leg"),
        "sweep_score": winner.get("score"),
        "patient007_f1": winner.get("patient007_f1"),
        "delta_vs_baseline": delta,
    }
    note_path = root / "outputs/biochem/species_gnn_deploy_baseline/sweep_winner.json"
    note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"[OK] promoted {winner.get('leg')} -> {dst}", flush=True)
    print("[i] re-run: scripts/promote_species_gnn_baseline.py --skip-copy", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
