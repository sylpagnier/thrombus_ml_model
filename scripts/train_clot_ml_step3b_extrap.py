"""Step 3b: finetune step1 with continuous-time extrap prior.

Usage:
  python scripts/train_clot_ml_step3b_extrap.py
  python scripts/train_clot_ml_step3b_extrap.py --epochs 24 --sim-end-scale 2.0 --val patient007
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_training_device  # noqa: E402
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.training.clot_ml_step1_residual import (  # noqa: E402
    ClotRuleResidualMLP,
    resolve_step1_rule_cfg,
    save_step1_checkpoint,
    step1_feature_dim,
)
from src.training.clot_ml_step3b_extrap import (  # noqa: E402
    Step3bTrainConfig,
    apply_step3b_train_env,
    default_step3b_out_dir,
    eval_step3b_on_anchor,
    load_step3b_checkpoint,
    train_one_graph_step3b,
)
from src.training.train_clot_phi_simple import _split_train_val  # noqa: E402


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 3b: step1 finetune + extrap growth prior")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--init-step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--sim-end-scale", type=float, default=2.0)
    ap.add_argument("--extrap-weight", type=float, default=0.15)
    args = ap.parse_args()

    cfg = Step3bTrainConfig(
        step0_json=args.step0_json,
        init_step1_ckpt=args.init_step1_ckpt,
        alpha=float(args.alpha),
        hidden=int(args.hidden),
        lr=float(args.lr),
        epochs=int(args.epochs),
        sim_end_scale=float(args.sim_end_scale),
        extrap_weight=float(args.extrap_weight),
    )
    apply_step3b_train_env(sim_end_scale=cfg.sim_end_scale)

    anchor_dir = REPO / args.anchor_dir
    paths = [str(p) for p in discover_anchor_paths(anchor_dir)]
    train_paths, val_paths = _split_train_val(paths, args.val)
    rule_cfg = resolve_step1_rule_cfg(REPO / cfg.step0_json)

    device = resolve_clot_ml_training_device()
    init_path = REPO / cfg.init_step1_ckpt
    model, meta = load_step3b_checkpoint(init_path, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    out_dir = Path(args.out_dir) if args.out_dir else default_step3b_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "clot_ml_step3b_best.pth"
    log_path = out_dir / "train_log.jsonl"

    best_val_deploy = -1.0
    print(
        f"[i] Step3b train epochs={cfg.epochs} sim_end_scale={cfg.sim_end_scale} "
        f"extrap_w={cfg.extrap_weight} val={args.val} device={device}",
        flush=True,
    )

    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            loss = train_one_graph_step3b(
                model,
                data,
                rule_cfg,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                alpha=cfg.alpha,
                cfg=cfg,
                early_paint_weight=0.25,
                final_bce_weight=2.0,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_paths), 1)

        model.eval()
        val_rows = []
        for p in val_paths:
            row = eval_step3b_on_anchor(
                model,
                rule_cfg,
                graph_path=Path(p),
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                alpha=cfg.alpha,
                sim_end_scale=cfg.sim_end_scale,
            )
            val_rows.append(row)
        mean_deploy = sum(r.get("deploy_score", 0.0) for r in val_rows) / max(len(val_rows), 1)
        mean_extrap = sum(r.get("extrap_growth_delta", 0.0) for r in val_rows) / max(len(val_rows), 1)
        print(
            f"[i] ep={ep:02d} train_loss={train_loss:.4f} val_deploy={mean_deploy:.3f} "
            f"extrap_delta={mean_extrap:.4f}",
            flush=True,
        )
        log_path.open("a", encoding="utf-8").write(
            json.dumps(
                _sanitize(
                    {
                        "epoch": ep,
                        "train_loss": train_loss,
                        "val_deploy": mean_deploy,
                        "mean_extrap_delta": mean_extrap,
                        "val_rows": val_rows,
                    }
                )
            )
            + "\n"
        )
        if mean_deploy > best_val_deploy:
            best_val_deploy = mean_deploy
            save_step1_checkpoint(
                ckpt_path,
                model=model,
                meta={
                    "step": "3b_extrap",
                    "alpha": cfg.alpha,
                    "sim_end_scale": cfg.sim_end_scale,
                    "init_step1": str(init_path),
                    "val_deploy": mean_deploy,
                    "mean_extrap_delta": mean_extrap,
                },
            )
            print(f"[save] {ckpt_path} val_deploy={mean_deploy:.3f}", flush=True)

    recipe_path = REPO / "data/reference/clot_ml_deploy_v1_extrap.json"
    if recipe_path.is_file() and ckpt_path.is_file():
        raw = json.loads(recipe_path.read_text(encoding="utf-8"))
        raw["step1_ckpt"] = str(ckpt_path.relative_to(REPO)).replace("\\", "/")
        recipe_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"[save] updated recipe -> {recipe_path}", flush=True)

    print(f"[OK] best val deploy={best_val_deploy:.3f} -> {ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
