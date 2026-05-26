"""Parse physics-oracle logs under outputs/biochem/clot_phi_ladder/."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_VAL_RE = re.compile(
    r"physics val dice=([\d.]+) f1=([\d.]+) prec=([\d.]+) rec=([\d.]+) "
    r"logMAE=([\d.]+)(?:\s+bio_mse=[\d.]+)?\s+pred\+=([\d.]+)\s+score=([-\d.]+)"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-dir", default="outputs/biochem/clot_phi_ladder")
    args = ap.parse_args()
    root = (_REPO / args.ladder_dir).resolve()
    rows: list[dict] = []
    for path in sorted(root.glob("physics_*.log")):
        text = ""
        for enc in ("utf-8", "utf-16", "utf-16-le", "cp1252"):
            try:
                text = path.read_text(encoding=enc)
                break
            except (UnicodeError, UnicodeDecodeError):
                continue
        if not text:
            text = path.read_bytes().decode("utf-8", errors="replace")
        m = _VAL_RE.search(text)
        if not m:
            continue
        name = path.stem.replace("physics_", "")
        rows.append(
            {
                "name": name,
                "dice": float(m.group(1)),
                "f1": float(m.group(2)),
                "prec": float(m.group(3)),
                "rec": float(m.group(4)),
                "log_mae": float(m.group(5)),
                "pred_pos_frac": float(m.group(6)),
                "score": float(m.group(7)),
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    print(f"[i]  physics logs={len(rows)}")
    for r in rows:
        print(
            f"{r['name']:<28} score={r['score']:6.3f} f1={r['f1']:5.3f} rec={r['rec']:5.3f} "
            f"logMAE={r['log_mae']:6.3f} pred+={r['pred_pos_frac']:5.3f}"
        )
    if rows:
        print(f"[OK]  best={rows[0]['name']} score={rows[0]['score']:.3f}")


if __name__ == "__main__":
    main()
