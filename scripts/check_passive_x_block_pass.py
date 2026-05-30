"""I.1 X block pass: probe legs logged + promote gate (optional).

Usage:
  python scripts/check_passive_x_block_pass.py
  python scripts/check_passive_x_block_pass.py --require-promote
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LOG = _REPO / "outputs" / "biochem" / "x_block" / "x_block_log.jsonl"
_PROBE_NOTES = (
    "x_block_X3_mask_global",
    "x_block_X4_data_bio",
    "x_block_X5_fi2mat2",
    "x_block_X_m3_union",
)
_MANIFEST = _REPO / "outputs" / "biochem" / "passive_species_locked_manifest.json"
_LOCKED = _REPO / "outputs" / "biochem" / "biochem_teacher_passive_species_locked.pth"


def _last_status(note: str) -> str | None:
    if not _LOG.is_file():
        return None
    last = None
    for line in _LOG.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("step") == note:
            last = str(row.get("status", ""))
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-promote", action="store_true")
    ap.add_argument("--min-probe-ok", type=int, default=3)
    args = ap.parse_args()

    ok_probe = 0
    for note in _PROBE_NOTES:
        st = _last_status(note)
        if st in ("OK", "WARN"):
            ok_probe += 1
        print(f"[i]  probe {note}: {st or 'missing'}")

    if ok_probe < args.min_probe_ok:
        print(f"[FAIL] probe legs ok={ok_probe} need>={args.min_probe_ok}", file=sys.stderr)
        return 1

    if not args.require_promote:
        print(f"[OK] I.1 X probe block ({ok_probe}/{len(_PROBE_NOTES)} legs logged)")
        return 0

    if not _LOCKED.is_file():
        print("[FAIL] missing biochem_teacher_passive_species_locked.pth", file=sys.stderr)
        return 1

    anchor_dir = ""
    if _MANIFEST.is_file():
        anchor_dir = json.loads(_MANIFEST.read_text(encoding="utf-8-sig")).get("anchor_dir", "")

    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "check_passive_x_block_gate.py"),
        "--checkpoint",
        str(_LOCKED),
        "--skip-eval",
    ]
    if anchor_dir:
        cmd.extend(["--anchor-dir", anchor_dir])

    rc = subprocess.call(cmd, cwd=str(_REPO))
    if rc == 0:
        print("[OK] I.1 X block pass (probe + promote)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
