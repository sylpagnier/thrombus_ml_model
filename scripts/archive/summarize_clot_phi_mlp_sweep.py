"""Aggregate clot-phi MLP sweep legs by best val checkpoint score."""

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
            score = float(row.get("val_score", -99.0))
            if best is None or score > float(best.get("val_score", -99.0)):
                best = row
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="outputs/biochem/sweep_clot_phi_mlp")
    ap.add_argument("--top", type=int, default=8)
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
        rows.append(
            {
                "leg": leg_dir.name,
                "epoch": int(best.get("epoch", -1)),
                "val_score": float(best.get("val_score", 0.0)),
                "val_f1": float(va.get("clot_f1", 0.0)),
                "val_prec": float(va.get("clot_prec", 0.0)),
                "val_rec": float(va.get("clot_rec", 0.0)),
                "val_log_mae": float(va.get("mu_log_mae", 0.0)),
                "val_dice": float(va.get("dice", 0.0)),
                "pred_pos_frac": float(va.get("pred_pos_frac", 0.0)),
                "gt_pos_frac": float(va.get("gt_pos_frac", 0.0)),
                "hidden": cfg.get("hidden"),
                "mlp_depth": cfg.get("mlp_depth"),
                "dropout": cfg.get("dropout"),
                "lr": cfg.get("lr"),
                "weight_decay": cfg.get("weight_decay"),
                "mu_log_lambda": cfg.get("mu_log_lambda"),
            }
        )

    rows.sort(key=lambda r: r["val_score"], reverse=True)
    summary_path = sweep / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[i]  sweep={sweep} legs={len(rows)}")
    print(f"{'leg':<14} {'score':>6} {'f1':>5} {'prec':>5} {'rec':>5} {'logMAE':>7} {'pred+':>6}  config")
    for r in rows[: max(args.top, 1)]:
        cfg = (
            f"h={r.get('hidden')} d={r.get('mlp_depth')} drop={r.get('dropout')} "
            f"lr={r.get('lr')} mu={r.get('mu_log_lambda')}"
        )
        print(
            f"{r['leg']:<14} {r['val_score']:6.3f} {r['val_f1']:5.3f} {r['val_prec']:5.3f} "
            f"{r['val_rec']:5.3f} {r['val_log_mae']:7.3f} {r['pred_pos_frac']:6.3f}  {cfg}"
        )
    if rows:
        print(f"[OK]  best={rows[0]['leg']} score={rows[0]['val_score']:.3f} -> {summary_path}")
    else:
        print("[WARN] no leg logs found")


if __name__ == "__main__":
    main()
