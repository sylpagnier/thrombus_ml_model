"""Print T0 physics sweep ranking (PowerShell-safe; no inline -c)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description="Print T0 physics sweep ranking")
    ap.add_argument(
        "--index",
        default="outputs/biochem/clot_trigger/t0_physics_sweep/sweep_index.json",
    )
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--include-oracle", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = REPO / index_path
    if not index_path.is_file():
        print(f"[ERR] missing {index_path}", file=sys.stderr)
        return 2

    data = json.loads(index_path.read_text(encoding="utf-8"))
    rows = list(data.get("ranking") or [])
    if not args.include_oracle:
        rows = [r for r in rows if not r.get("gt_mu_oracle")]

    print(f"[i] best_physics={data.get('best_leg_id', '')} elapsed={data.get('elapsed_s', 0):.0f}s")
    for i, r in enumerate(rows[: max(1, int(args.top))]):
        print(
            f"  {i + 1}. {r.get('leg_id', '?')}: "
            f"score={float(r.get('score', 0)):.3f} "
            f"full_F1={float(r.get('mean_full_mesh_f1', 0)):.3f} "
            f"p007={float(r.get('val_full_mesh_f1', 0)):.3f} "
            f"lumen_fp={float(r.get('mean_lumen_fp_deploy', 0)):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
