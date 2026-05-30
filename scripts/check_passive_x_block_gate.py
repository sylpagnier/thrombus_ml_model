"""I.1 X block completion gate: species teacher + optional dump directory.

Usage:
  python scripts/check_passive_x_block_gate.py --checkpoint outputs/biochem/biochem_teacher_passive_species_locked.pth
  python scripts/check_passive_x_block_gate.py --checkpoint ... --anchor-dir outputs/biochem/x_block/anchors_stride36_m6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _anchor_graph_count(anchor_dir: Path) -> int:
    if not anchor_dir.is_dir():
        return 0
    return len(list(anchor_dir.glob("*.pt")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="outputs/biochem/biochem_teacher_passive_species_locked.pth",
    )
    ap.add_argument("--anchor-dir", default="")
    ap.add_argument("--min-anchors", type=int, default=3)
    ap.add_argument("--max-fi-mean", type=float, default=0.05)
    ap.add_argument("--max-fi-train-anchor", type=float, default=0.04)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Trust calibration ckpt; only verify anchor dump + manifest.",
    )
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        ckpt = _REPO / args.checkpoint
    if not ckpt.is_file():
        print(f"[ERR] missing checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2

    if not args.skip_eval:
        rc = subprocess.call(
            [
                sys.executable,
                str(_REPO / "scripts" / "eval_passive_species_anchors.py"),
                "--checkpoint",
                str(ckpt),
                "--device",
                args.device,
                "--split",
                "train",
                "--max-fi-mean",
                str(args.max_fi_mean),
                "--max-fi-per-anchor",
                str(args.max_fi_train_anchor),
            ],
            cwd=str(_REPO),
        )
        if rc != 0:
            return rc
    else:
        print("[i]  skip-eval: promote gate uses anchor dump only (calibration ckpt assumed OK)")

    if args.anchor_dir:
        ad = Path(args.anchor_dir)
        if not ad.is_dir():
            ad = _REPO / args.anchor_dir
        n = _anchor_graph_count(ad)
        if n < args.min_anchors:
            print(
                f"[FAIL] anchor dump sparse: {n} graphs in {ad} (need>={args.min_anchors})",
                file=sys.stderr,
            )
            return 1
        print(f"[OK] anchor dump: {n} graphs in {ad}")

    manifest = _REPO / "outputs" / "biochem" / "passive_species_locked_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        print(f"[OK] manifest: {json.dumps(data, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
