"""I.3 XY viability: step-2 bridge keeps species + bridge flag; mu must not blow up.

Pass (default):
  - check_passive_step2_bridge_gate with relaxed bio/adr ratios when --saturated
    (init already calibrated, e.g. passive_m3_locked)
  - OR full bridge gate when not saturated

Usage:
  python scripts/check_passive_xy_viability_pass.py --run-note passive_step2_bridge_m3_hold
  python scripts/check_passive_xy_viability_pass.py --run-note passive_step2_bridge_align_6ep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_BRIDGE = _REPO / "scripts" / "check_passive_step2_bridge_gate.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="I.3 XY step-2 bridge viability gate")
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--saturated", action="store_true", help="Skip L_bio/L_ADR descent (M3-cal init)")
    ap.add_argument("--bio-max-ratio", type=float, default=None)
    ap.add_argument("--adr-max-ratio", type=float, default=None)
    args = ap.parse_args()

    bio_r = args.bio_max_ratio if args.bio_max_ratio is not None else (1.02 if args.saturated else 0.92)
    adr_r = args.adr_max_ratio if args.adr_max_ratio is not None else (1.02 if args.saturated else 0.88)

    rc = subprocess.run(
        [
            sys.executable,
            str(_BRIDGE),
            "--run-note",
            args.run_note,
            "--min-epochs",
            str(args.min_epochs),
            "--bio-max-ratio",
            str(bio_r),
            "--adr-max-ratio",
            str(adr_r),
        ],
        cwd=_REPO,
    )
    if rc.returncode == 0:
        mode = "saturated-hold" if args.saturated else "learn"
        print(f"[OK] I.3 XY bridge viability pass ({mode}, run_note={args.run_note})")
        return 0
    print("[ERR] XY bridge viability failed", file=sys.stderr)
    return rc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
