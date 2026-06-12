"""Phase 6 (s35): freeze s34 species GNN; finetune global Mat boost beta for mu @ T=53.

Usage::

    python scripts/train_clot_phi_calibration.py --gnn-ckpt outputs/biochem/species_snapshot_s34/best.pth
    python scripts/train_clot_phi_calibration.py --all-anchors --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.species_gnn_clot_rollout import (
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_species_series,
)
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6
from src.core_physics.species_viscosity_calibration import (
    DEFAULT_S34_GNN_CKPT,
    MatViscosityCalibrator,
    predict_mu_at_time_with_beta,
    save_viscosity_calibration,
)
from src.core_physics.t0_mu_physics import _mu_log_mae, _pearson, _region_masks
from src.utils.paths import get_project_root
from src.core_physics.t0_device import require_cuda_device


def _parse_anchors(raw: str, *, all_anchors: bool) -> list[str]:
    if all_anchors:
        return list(BIOCHEM_ANCHORS_6)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or ["patient007"]


@torch.no_grad()
def _cache_species_rollouts(
    anchors: list[str],
    *,
    gnn_ckpt: Path,
    device: torch.device,
    root: Path,
) -> dict[str, dict]:
    bundle = load_species_gnn_rollout_bundle(gnn_ckpt, device=device)
    if bundle is None:
        raise FileNotFoundError(f"missing GNN checkpoint: {gnn_ckpt}")
    model = bundle.continuous.model if bundle.continuous is not None else None
    if model is None:
        raise RuntimeError("s35 expects continuous species GNN checkpoint")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    out: dict[str, dict] = {}
    for anc in anchors:
        graph = root / "data/processed/graphs_biochem_anchors" / f"{anc}.pt"
        if not graph.is_file():
            raise FileNotFoundError(f"missing graph: {graph}")
        data = torch.load(graph, map_location=device, weights_only=False)
        static = prepare_species_gnn_rollout_static(data, device=device)
        series = rollout_species_gnn_species_series(data, bundle, static, device=device)
        t_last = int(series.shape[0]) - 1
        out[anc] = {"data": data, "series": series.detach(), "t_eval": t_last}
        print(f"[OK] cached rollout {anc} T_last={t_last}", flush=True)
    return out


def _anchor_t_eval(pack: dict, default_t: int) -> int:
    t_last = int(pack.get("t_eval", default_t))
    return min(int(default_t), t_last)


def _eval_beta(
    calibrator: MatViscosityCalibrator,
    cache: dict[str, dict],
    *,
    time_index: int,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
) -> dict[str, float]:
    beta = calibrator.beta
    logs: list[float] = []
    pears: list[float] = []
    growth_logs: list[float] = []
    for anc, pack in cache.items():
        t_i = _anchor_t_eval(pack, time_index)
        mu_pred, mu_gt = predict_mu_at_time_with_beta(
            pack["data"],
            pack["series"],
            beta,
            t_i,
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
            anchor=anc,
            soft_gelation=False,
        )
        masks = _region_masks(pack["data"], t_i, phys, device, mu_gt)
        growth = masks["growth"]
        logs.append(_mu_log_mae(mu_pred, mu_gt))
        pears.append(_pearson(mu_pred, mu_gt))
        if bool(growth.any().item()):
            growth_logs.append(_mu_log_mae(mu_pred, mu_gt, growth))
    return {
        "mu_log_mae_all": float(sum(logs) / max(len(logs), 1)),
        "pearson_all": float(sum(pears) / max(pears and len(pears) or 1, 1)),
        "mu_log_mae_growth": float(sum(growth_logs) / max(len(growth_logs), 1))
        if growth_logs
        else float("nan"),
    }


def _weighted_mu_loss(
    mu_pred: torch.Tensor,
    mu_gt: torch.Tensor,
    growth_mask: torch.Tensor,
    *,
    growth_weight: float = 4.0,
) -> torch.Tensor:
    p = mu_pred.reshape(-1).clamp(min=1e-8)
    t = mu_gt.reshape(-1).clamp(min=1e-8)
    err = (torch.log(p) - torch.log(t)).pow(2)
    w = torch.ones_like(err)
    g = growth_mask.reshape(-1).bool()
    if bool(g.any().item()):
        w = w + (float(growth_weight) - 1.0) * g.to(dtype=err.dtype)
    return (err * w).mean()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6: finetune Mat beta for T=53 mu calibration")
    ap.add_argument("--gnn-ckpt", default=DEFAULT_S34_GNN_CKPT)
    ap.add_argument("--anchors", default="")
    ap.add_argument("--all-anchors", action="store_true")
    ap.add_argument("--time-index", type=int, default=53)
    ap.add_argument("--beta-init", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="outputs/biochem/species_snapshot_s35/beta.pth")
    ap.add_argument("--log-space", action="store_true", default=True)
    ap.add_argument("--linear-mse", action="store_true", help="Use linear MSE instead of log-MSE")
    ap.add_argument("--growth-weight", type=float, default=4.0)
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    anchors = _parse_anchors(args.anchors, all_anchors=bool(args.all_anchors))
    gnn_ckpt = Path(args.gnn_ckpt)
    if not gnn_ckpt.is_absolute():
        gnn_ckpt = root / gnn_ckpt

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    t_eval = int(args.time_index)

    print(
        f"[i] s35 viscosity calibration anchors={anchors} gnn={gnn_ckpt} t={t_eval} "
        f"beta_init={args.beta_init:.2f} epochs={args.epochs} lr={args.lr:.3f}",
        flush=True,
    )
    t0 = time.perf_counter()
    cache = _cache_species_rollouts(anchors, gnn_ckpt=gnn_ckpt, device=device, root=root)

    calibrator = MatViscosityCalibrator(beta_init=float(args.beta_init)).to(device)
    opt = optim.Adam(calibrator.parameters(), lr=float(args.lr))
    log_space = not bool(args.linear_mse)

    best_loss = float("inf")
    best_beta = float(args.beta_init)
    log_path = Path(args.out).parent / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, int(args.epochs) + 1):
        opt.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for pack in cache.values():
            t_i = _anchor_t_eval(pack, t_eval)
            mu_pred, mu_gt = predict_mu_at_time_with_beta(
                pack["data"],
                pack["series"],
                calibrator.beta,
                t_i,
                phys_cfg=phys,
                bio_cfg=bio,
                device=device,
                soft_gelation=False,
            )
            masks = _region_masks(pack["data"], t_i, phys, device, mu_gt)
            if log_space:
                losses.append(_weighted_mu_loss(mu_pred, mu_gt, masks["growth"], growth_weight=float(args.growth_weight)))
            else:
                import torch.nn.functional as F

                losses.append(F.mse_loss(mu_pred, mu_gt))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        beta_val = float(calibrator.beta.detach().cpu().item())  # bounded [0.5, 1.5]
        metrics = _eval_beta(
            calibrator, cache, time_index=t_eval, phys=phys, bio=bio, device=device
        )
        row = {
            "epoch": ep,
            "loss": float(loss.detach().cpu().item()),
            "beta": beta_val,
            **metrics,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_beta = beta_val
            out_path = Path(args.out)
            if not out_path.is_absolute():
                out_path = root / out_path
            save_viscosity_calibration(
                out_path,
                calibrator,
                gnn_ckpt=str(gnn_ckpt),
                time_index=t_eval,
                meta={
                    "anchors": anchors,
                    "best_loss": best_loss,
                    "mu_log_mae_all": metrics["mu_log_mae_all"],
                    "pearson_all": metrics["pearson_all"],
                    "mu_log_mae_growth": metrics["mu_log_mae_growth"],
                },
            )

        if ep == 1 or ep % 20 == 0 or ep == int(args.epochs):
            print(
                f"[ep {ep:03d}] loss={row['loss']:.5f} beta={beta_val:.4f} "
                f"logMAE={metrics['mu_log_mae_all']:.4f} r={metrics['pearson_all']:.3f} "
                f"logMAE_g={metrics['mu_log_mae_growth']:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"[OK] best_beta={best_beta:.4f} best_loss={best_loss:.5f} elapsed={elapsed:.1f}s "
        f"out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
