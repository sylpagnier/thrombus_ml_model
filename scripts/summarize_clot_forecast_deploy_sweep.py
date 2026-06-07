"""Rank clot-forecast deploy sweep legs by clot_shape (north-star) then band F1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.paths import get_project_root


def _best_from_log(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    best: dict | None = None
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            va = row.get("val") or {}
            shape = float(va.get("clot_shape", 0.0))
            score = float(row.get("val_score", -99.0))
            rank = shape * 0.65 + float(va.get("clot_f1", 0.0)) * 0.35
            if score < 0 and shape <= 0.0:
                rank = -1.0
            row["_rank"] = rank
            if best is None or rank > float(best.get("_rank", -99.0)):
                best = row
    return best


def _mean_multi_anchor(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}
    keys = ("clot_shape", "clot_f1", "clot_shape_rec", "mu_log_mae")
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r.get("val", {}).get(k, 0.0)) for r in rows if "val" in r]
        if not vals:
            vals = [float(r.get(k, 0.0)) for r in rows if k in r]
        if vals:
            out[f"mean_{k}"] = sum(vals) / len(vals)
    p007 = next((r for r in rows if "patient007" in str(r.get("anchor", ""))), None)
    if p007:
        va = p007.get("val") or p007
        out["p007_shape"] = float(va.get("clot_shape", 0.0))
        out["p007_f1"] = float(va.get("clot_f1", 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="outputs/biochem/sweep_clot_forecast_deploy_30m")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    root = get_project_root()
    sweep = (root / args.sweep_dir).resolve()
    rows: list[dict] = []

    for leg_dir in sorted(p for p in sweep.iterdir() if p.is_dir()):
        log_path = leg_dir / "clot_phi_train_log.jsonl"
        ckpt_path = leg_dir / "clot_phi_best.pth"
        best = _best_from_log(log_path)
        if best is None:
            continue
        va = best.get("val") or {}
        cfg: dict = {}
        if ckpt_path.is_file():
            import torch

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = dict(ckpt.get("config") or {})
        ma = _mean_multi_anchor(leg_dir / "multi_anchor.jsonl")
        shape = float(va.get("clot_shape", 0.0))
        f1 = float(va.get("clot_f1", 0.0))
        rank = shape * 0.65 + f1 * 0.35
        rows.append(
            {
                "leg": leg_dir.name,
                "rank_score": rank,
                "epoch": int(best.get("epoch", -1)),
                "val_score": float(best.get("val_score", 0.0)),
                "val_f1": f1,
                "val_shape": shape,
                "val_shape_rec": float(va.get("clot_shape_rec", 0.0)),
                "val_shape_pred_frac": float(va.get("clot_shape_pred_frac", 0.0)),
                "val_shape_gt_frac": float(va.get("clot_shape_gt_frac", 0.0)),
                "val_prec": float(va.get("clot_prec", 0.0)),
                "val_rec": float(va.get("clot_rec", 0.0)),
                "pred_pos_frac": float(va.get("pred_pos_frac", 0.0)),
                "gt_pos_frac": float(va.get("gt_pos_frac", 0.0)),
                "forecast_mask": cfg.get("forecast_mask"),
                "mesh_aux_lambda": cfg.get("mesh_aux_lambda"),
                "mesh_bulk_lambda": cfg.get("mesh_bulk_lambda"),
                **ma,
            }
        )

    rows.sort(key=lambda r: (r.get("rank_score") or -1.0), reverse=True)
    summary_path = sweep / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[OK]  ranked {len(rows)} legs -> {summary_path}", flush=True)
    print("[i]  rank = 0.65*val_clot_shape + 0.35*val_band_f1 (p007 val epoch)", flush=True)
    for r in rows[: args.top]:
        print(
            f"  {r['leg']:16s} rank={r.get('rank_score', 0):.3f} "
            f"shape={r.get('val_shape', 0):.3f} f1={r.get('val_f1', 0):.3f} "
            f"shape_pred={r.get('val_shape_pred_frac', 0):.3f} "
            f"mask={r.get('forecast_mask')} mesh_aux={r.get('mesh_aux_lambda')} "
            f"ep={r.get('epoch')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
