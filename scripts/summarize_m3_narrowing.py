"""Print ranked table of m3n_* narrowing legs from runs_index + run.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "outputs" / "reports" / "training" / "biochem" / "runs_index.jsonl"


def main() -> int:
    prefix = (sys.argv[1] if len(sys.argv) > 1 else "m3n_").strip()
    rows = []
    if not _INDEX.is_file():
        print("[ERR] runs_index missing")
        return 2
    for line in _INDEX.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        note = r.get("run_note") or ""
        if not note.startswith(prefix):
            continue
        rd = Path(r["run_dir"])
        vals = []
        for vl in (rd / "run.jsonl").read_text(encoding="utf-8").splitlines():
            if not vl.strip():
                continue
            v = json.loads(vl)
            if v.get("event") == "val" and v.get("stage") == "teacher":
                vals.append(v)
        vals.sort(key=lambda x: int(x["epoch"]))
        if len(vals) < 2:
            continue
        e0, el = vals[0], vals[-1]
        bio0 = float(e0.get("train_L_bio_avg") or e0.get("train_L_back_avg") or 0)
        biol = float(el.get("train_L_bio_avg") or el.get("train_L_back_avg") or 0)
        adr0 = float(e0.get("train_L_ADR_S_avg") or e0.get("train_L_ADR_S_global_avg") or 0)
        adrl = float(el.get("train_L_ADR_S_avg") or el.get("train_L_ADR_S_global_avg") or 0)
        rows.append(
            {
                "note": note,
                "bio_ratio": biol / bio0 if bio0 > 0 else float("nan"),
                "adr_ratio": adrl / adr0 if adr0 > 0 else float("nan"),
                "L_back_last": biol,
                "ADR_mode": el.get("ADR_residual_mode", "-"),
                "ADR_mask": el.get("ADR_mask_mode", "-"),
                "adr_w": el.get("passive_adr_weight", "-"),
            }
        )

    rows.sort(key=lambda x: (x["L_back_last"], x["bio_ratio"]))
    print(f"[i] M3 narrowing summary (prefix={prefix!r}, n={len(rows)})")
    print(f"{'note':28s} {'L_bio_last':>10s} {'bio_r':>7s} {'adr_r':>7s} {'form':>14s} {'mask':>16s}")
    for r in rows:
        print(
            f"{r['note']:28s} {r['L_back_last']:10.1f} {r['bio_ratio']:7.3f} "
            f"{r['adr_ratio']:7.3f} {str(r['ADR_mode']):>14s} {str(r['ADR_mask']):>16s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
