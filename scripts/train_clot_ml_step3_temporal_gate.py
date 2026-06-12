"""Step 3 ladder: train per-vessel learned onset gate.

Usage:
  python scripts/train_clot_ml_step3_temporal_gate.py
  python scripts/train_clot_ml_step3_temporal_gate.py --epochs 40 --val patient007
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_training_device  # noqa: E402
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.training.clot_ml_step3_temporal_gate import (  # noqa: E402
    ClotTemporalGateModel,
    Step3TrainConfig,
    default_onset_bounds,
    default_step3_out_dir,
    eval_step3_on_anchor,
    load_step3_checkpoint,
    resolve_step3_rule_cfg,
    save_step3_checkpoint,
    step2_feature_dim,
    train_one_graph,
)
from src.training.train_clot_phi_simple import _split_train_val  # noqa: E402


def _apply_deploy_env() -> None:
    os.environ["BIOCHEM_PRIOR_COMSOL_ALIGNED"] = "1"
    os.environ["BIOCHEM_PRIOR_NORM_MASK"] = "adjacent"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "1"
    os.environ["CLOT_PHI_CEILING_HOPS"] = "2"
    os.environ["CLOT_FORECAST_MASK"] = "ceiling_growth"
    os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1"
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    os.environ.setdefault("CLOT_PHI_KINE_CKPT", "outputs/kinematics/kinematics_best.pth")


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 3: learned temporal onset gate")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()

    _apply_deploy_env()
    onset_min, onset_max = default_onset_bounds()
    cfg_train = Step3TrainConfig(
        step0_json=args.step0_json,
        hidden=int(args.hidden),
        lr=float(args.lr),
        epochs=int(args.epochs),
        onset_min=onset_min,
        onset_max=onset_max,
    )

    anchor_dir = REPO / args.anchor_dir
    paths = [str(p) for p in discover_anchor_paths(anchor_dir)]
    train_paths, _val_paths = _split_train_val(paths, args.val)
    rule_cfg = resolve_step3_rule_cfg(REPO / cfg_train.step0_json)

    device = resolve_clot_ml_training_device()
    model = ClotTemporalGateModel(
        in_dim=step2_feature_dim(),
        hidden=cfg_train.hidden,
        onset_min=cfg_train.onset_min,
        onset_max=cfg_train.onset_max,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    out_dir = Path(args.out_dir) if args.out_dir else default_step3_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "clot_ml_step3_best.pth"
    log_path = out_dir / "train_log.jsonl"

    best_mean_deploy = -1.0
    print(
        f"[i] Step3 train epochs={cfg_train.epochs} onset=[{cfg_train.onset_min:.2f},"
        f"{cfg_train.onset_max:.2f}] val_holdout={args.val} "
        f"ckpt=mean_deploy_all_anchors device={device}",
        flush=True,
    )

    for ep in range(1, cfg_train.epochs + 1):
        model.train()
        train_loss = 0.0
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            opt.zero_grad(set_to_none=True)
            loss = train_one_graph(
                model,
                data,
                rule_cfg,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                gt_onset_threshold=cfg_train.gt_onset_threshold,
                early_onset_weight=cfg_train.early_onset_weight,
                late_onset_weight=cfg_train.late_onset_weight,
            )
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_paths), 1)

        model.eval()
        all_rows = [
            eval_step3_on_anchor(
                model,
                rule_cfg,
                graph_path=Path(p),
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
            )
            for p in paths
        ]
        mean_deploy = sum(r["deploy_score"] for r in all_rows) / len(all_rows)
        row_log = {
            "epoch": ep,
            "train_loss": train_loss,
            "mean_deploy_all": mean_deploy,
            "per_anchor": all_rows,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(row_log)) + "\n")
        print(f"[i] ep={ep} loss={train_loss:.4f} mean_deploy={mean_deploy:.3f}", flush=True)

        if mean_deploy > best_mean_deploy:
            best_mean_deploy = mean_deploy
            meta = {
                "step": 3,
                "step0_json": str(cfg_train.step0_json),
                "hidden": cfg_train.hidden,
                "onset_min": cfg_train.onset_min,
                "onset_max": cfg_train.onset_max,
                "val_holdout": args.val,
                "best_mean_deploy": best_mean_deploy,
            }
            save_step3_checkpoint(ckpt_path, model=model, meta=meta)

    model, meta = load_step3_checkpoint(ckpt_path, device=device)
    per_anchor = [
        eval_step3_on_anchor(
            model,
            rule_cfg,
            graph_path=Path(p),
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        for p in paths
    ]
    mean_deploy = sum(r["deploy_score"] for r in per_anchor) / len(per_anchor)
    summary = {
        "step": 3,
        "checkpoint": str(ckpt_path),
        "step0_json": cfg_train.step0_json,
        "mean_deploy": mean_deploy,
        "per_anchor": per_anchor,
        "meta": meta,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_sanitize(summary), indent=2), encoding="utf-8")
    print(f"[OK] mean_deploy={mean_deploy:.3f} best_mean={best_mean_deploy:.3f}", flush=True)
    print(f"[save] {ckpt_path}", flush=True)
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
