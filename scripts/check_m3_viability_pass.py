"""M3 viability gate: masked ADR + species co-train works (optimize later).

Pass if any calibration run passes check_m3_align_gate (default min_epochs=6).
Known-good run notes are tried first.

Usage:
  python scripts/check_m3_viability_pass.py
  python scripts/check_m3_viability_pass.py --run-note m3_align_transport_union_12ep --min-epochs 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_NOTES = (
    "m3_align_transport_union_12ep",
    "m3_align_transport_union",
    "passive_align_20ep",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 viability (components work; full M3 optimize later)")
    ap.add_argument("--run-note", action="append", dest="run_notes", default=[])
    ap.add_argument("--min-epochs", type=int, default=6)
    ap.add_argument("--min-mask-n", type=float, default=32.0)
    args = ap.parse_args()

    notes = args.run_notes or list(_DEFAULT_NOTES)
    for note in notes:
        rc = subprocess.run(
            [
                sys.executable,
                str(_REPO / "scripts" / "check_m3_align_gate.py"),
                "--run-note",
                note,
                "--min-epochs",
                str(args.min_epochs),
                "--min-mask-n",
                str(args.min_mask_n),
            ],
            cwd=_REPO,
        )
        if rc.returncode == 0:
            print(f"[OK] M3 viability pass (run_note={note})")
            print(
                "[i]  Deferred (optimize later): global ramp2 raw ADR, full narrowing/sweep, "
                "passive_m3_locked promote, ADR weight / epoch tuning"
            )
            return 0

    print("[ERR] No run passed m3_align gate; run go_m3_align_probe.ps1 -Epochs 12 from phaseB_ramp1", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
