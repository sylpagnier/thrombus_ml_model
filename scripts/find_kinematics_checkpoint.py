"""Scan kinematics .pth files for epoch / val rel_L2 (recover pruned production ep-80)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _meta(path: Path) -> dict:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}
    if not isinstance(raw, dict):
        return {"path": str(path), "error": "not a dict checkpoint"}
    ep = raw.get("epoch", raw.get("best_epoch"))
    rel = float(raw.get("rel_l2", float("nan")))
    comp = float(raw.get("composite", float("nan")))
    role = raw.get("checkpoint_role", "")
    return {
        "path": str(path),
        "epoch": int(ep) if ep is not None else None,
        "rel_l2": rel if math.isfinite(rel) else None,
        "composite": comp if math.isfinite(comp) else None,
        "role": str(role),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=str,
        default="outputs",
        help="Search root (default: outputs).",
    )
    p.add_argument(
        "--target-rel-l2",
        type=float,
        default=0.1263,
        help="Highlight checkpoints near this rel_L2 (production ep-80).",
    )
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERR] not a directory: {root}")
        return 1

    rows = []
    for path in sorted(root.rglob("*.pth")):
        if "kinematics" not in path.name.lower():
            continue
        m = _meta(path)
        if "error" in m:
            continue
        if m.get("rel_l2") is not None:
            m["rel_gap"] = abs(float(m["rel_l2"]) - float(args.target_rel_l2))
        rows.append(m)

    if not rows:
        print(f"[i] no kinematics checkpoints under {root.resolve()}")
        return 1

    by_rel = sorted(
        [r for r in rows if r.get("rel_l2") is not None],
        key=lambda r: (r["rel_gap"], r["rel_l2"]),
    )
    by_comp = sorted(
        [r for r in rows if r.get("composite") is not None],
        key=lambda r: r["composite"],
    )

    print(f"[i] found {len(rows)} kinematics .pth under {root.resolve()}")
    print(f"[i] closest to rel_L2={args.target_rel_l2} (production ep-80):")
    for r in by_rel[: args.top]:
        print(
            f"  rel_L2={r['rel_l2']:.4f} gap={r['rel_gap']:.4f} "
            f"epoch={r['epoch']} composite={r.get('composite')} "
            f"{r['size_mb']}MB {r['path']}"
        )
    if by_comp:
        print("[i] best composite:")
        r = by_comp[0]
        print(
            f"  composite={r['composite']:.4f} rel_L2={r.get('rel_l2')} "
            f"epoch={r['epoch']} {r['path']}"
        )
    best = by_rel[0]
    print(f"[i] RECOMMENDED_RESUME={best['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
