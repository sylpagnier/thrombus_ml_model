"""Build clot-regime-balanced train/val splits for fast generalization experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.utils.paths import get_project_root


def peak_clot(anchor_path: Path, phys: PhysicsConfig) -> int:
    data = torch.load(anchor_path, map_location="cpu", weights_only=False)
    n_t = int(data.y.shape[0])
    idx = np.unique(np.linspace(0, n_t - 1, num=min(24, n_t), dtype=int))
    peak = 0
    for t in idx:
        phi = gt_clot_phi_at_time(data, int(t), phys, torch.device("cpu")).detach().cpu().numpy().reshape(-1)
        peak = max(peak, int((phi >= 0.5).sum()))
    return int(peak)


def bucket(peak: int) -> str:
    if peak >= 180:
        return "high"
    if peak >= 70:
        return "mid"
    if peak > 0:
        return "low"
    return "zero"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfolds", type=int, default=4)
    ap.add_argument("--out", default="outputs/biochem/offwall_model/generalization_fast/splits.json")
    args = ap.parse_args()

    root = get_project_root()
    anchor_dir = root / "data/processed/graphs_biochem_anchors"
    phys = PhysicsConfig(phase="biochem")

    anchors = sorted(anchor_dir.glob("*.pt"))
    rows = []
    for p in anchors:
        pk = peak_clot(p, phys)
        rows.append({"anchor": p.stem, "peak_clot": pk, "bucket": bucket(pk)})

    nonzero = [r for r in rows if r["bucket"] != "zero"]
    by_bucket = {
        "high": [r["anchor"] for r in nonzero if r["bucket"] == "high"],
        "mid": [r["anchor"] for r in nonzero if r["bucket"] == "mid"],
        "low": [r["anchor"] for r in nonzero if r["bucket"] == "low"],
    }
    for k in by_bucket:
        by_bucket[k] = sorted(by_bucket[k], key=lambda a: next(r["peak_clot"] for r in rows if r["anchor"] == a), reverse=True)

    kfolds = max(int(args.kfolds), 2)
    folds = []
    for i in range(kfolds):
        val = []
        for arr in by_bucket.values():
            if arr:
                val.extend([a for j, a in enumerate(arr) if j % kfolds == i])
        train = [r["anchor"] for r in nonzero if r["anchor"] not in set(val)]
        val = sorted(set(val))
        train = sorted(set(train))
        val_peak = {a: next(r["peak_clot"] for r in rows if r["anchor"] == a) for a in val}
        primary_val = max(val_peak, key=val_peak.get) if val_peak else (train[0] if train else "")
        folds.append({
            "fold": i,
            "train_anchors": train,
            "val_anchors": val,
            "primary_val_anchor": primary_val,
        })

    out = {
        "kfolds": kfolds,
        "anchors": rows,
        "folds": folds,
        "recommended_challenge": ["patient009", "patient032", "patient013", "patient015"],
        "notes": "Balanced by peak clot regime (high/mid/low), zeros excluded from train/val.",
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[save] {out_path}")
    print("[i] folds built:")
    for f in folds:
        print(f"  fold{f['fold']}: train={len(f['train_anchors'])} val={len(f['val_anchors'])} primary={f['primary_val_anchor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
