"""Mark probe legs OK in x_block_log when run.jsonl exists and species probe gate passes."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LOG = _REPO / "outputs" / "biochem" / "x_block" / "x_block_log.jsonl"
_NOTES = (
    "x_block_X3_mask_global",
    "x_block_X4_data_bio",
    "x_block_X5_fi2mat2",
    "x_block_X_m3_union",
)


def main() -> int:
    updated = 0
    for note in _NOTES:
        rc = subprocess.call(
            [sys.executable, str(_REPO / "scripts" / "check_passive_x_species_gate.py"), "--run-note", note, "--probe", "-q"],
            cwd=str(_REPO),
        )
        if rc != 0:
            print(f"[i]  skip {note} (gate rc={rc})")
            continue
        row = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "step": note,
            "status": "OK",
            "backfill": True,
        }
        with _LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"[OK] backfill {note}")
        updated += 1
    print(f"[OK] backfilled {updated} leg(s) -> {_LOG}")
    return 0 if updated >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
