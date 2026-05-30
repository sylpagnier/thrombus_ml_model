"""M3 block pass: calibration align gate + locked teacher ckpt.

Usage:
  python scripts/check_m3_block_pass.py
  python scripts/check_m3_block_pass.py --run-note m3_align_transport_union
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_NOTE = "m3_align_transport_union"
_DEFAULT_LOCKED = _REPO / "outputs/biochem/biochem_teacher_passive_m3_locked.pth"


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 analytical ADR block pass gate")
    ap.add_argument("--run-note", default=_DEFAULT_NOTE)
    ap.add_argument("--locked-ckpt", type=Path, default=_DEFAULT_LOCKED)
    ap.add_argument("--min-mask-n", type=float, default=32.0)
    args = ap.parse_args()

    align = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "check_m3_align_gate.py"),
            "--run-note",
            args.run_note,
            "--min-mask-n",
            str(args.min_mask_n),
        ],
        cwd=_REPO,
    )
    if align.returncode != 0:
        print("[ERR] M3 align calibration gate failed (masked L_bio + L_ADR_S co-descent)")
        return align.returncode

    locked = args.locked_ckpt
    if not locked.is_file():
        print(f"[ERR] Missing locked ckpt: {locked}")
        return 2

    manifest = _REPO / "outputs/biochem/passive_m3_locked_manifest.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8-sig"))
        print(f"[OK] manifest: {json.dumps(meta, separators=(',', ':'))}")
    else:
        print(f"[WARN] manifest missing: {manifest}")

    print("[OK] M3 block pass (calibration gate + locked ckpt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
