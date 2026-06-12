"""Train Rung 4 s1: residual MLP on physics phi inside E(t).

Usage::

    python -m src.training.train_t0_r4_s1_mlp_phi
    python -m src.training.train_t0_r4_s1_mlp_phi --val-anchor patient007 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_r4_s1_mlp_phi import (
    DEFAULT_S1_CKPT,
    T0R4S1PhiMLP,
    build_s1_phi_features,
    feature_dim,
    s1_residual_alpha,
    s1_train_target_phi,
    save_s1_checkpoint,
)
from src.core_physics.t0_rung4_ladder import _build_s0_deploy_species, rung4_use_dgamma_wall_seed
from src.core_physics.t0_r4_s1_mlp_phi import _s0_gate_from_species
from src.core_physics.clot_nucleation_mask import project_phi_with_nucleation, resolve_nucleation_eligibility
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def _list_anchors(root: Path) -> list[Path]:
    paths = sorted(root.glob("patient*.pt"))
    if not paths:
        raise FileNotFoundError(f"No anchors in {root}")
    return paths


def _coupled_loss_on_anchor(
    data,
    model: T0R4S1PhiMLP,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    alpha: float,
    time_stride: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    n_steps = int(data.y.shape[0])
    stride = max(int(time_stride), 1)
    times = list(range(0, n_steps, stride))
    if times[-1] != n_steps - 1:
        times.append(n_steps - 1)

    phi_prev = None
    commits_prev = None
    pred_series = data.y.clone().to(device=device)
    losses: list[torch.Tensor] = []
    f1_last = 0.0

    with t0_rung2_env():
        for t in times:
            elig = resolve_nucleation_eligibility(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                commits_prev=commits_prev,
                growth_seed="pred",
                nucleation_hops=1,
                use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
            ).reshape(-1).bool()
            s0_sp = _build_s0_deploy_species(
                data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev
            )
            pred_series[t, :, 4:16] = s0_sp
            phi_phys, _ = predict_clot_phi_at_time(
                data,
                t,
                phys_cfg,
                bio_cfg,
                device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="gt",
                pred_species_series=pred_series,
            )
            gate = _s0_gate_from_species(s0_sp, data, t, device, bio_cfg, elig)
            feats = build_s1_phi_features(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                elig=elig,
                phi_physics=phi_phys,
                s0_species=s0_sp,
                s0_gate=gate,
            )
            delta = alpha * model(feats)
            phi_raw = (phi_phys.reshape(-1) + delta * elig.float()).clamp(0.0, 1.0)
            phi = project_phi_with_nucleation(phi_raw, phi_prev, elig)
            phi_gt = s1_train_target_phi(data, t, phys_cfg, device)
            m = elig
            if bool(m.any().item()):
                losses.append(F.binary_cross_entropy(phi[m], phi_gt[m]))
            commits_prev = phi.detach().reshape(-1).ge(0.5)
            phi_prev = phi.detach()
            if t == n_steps - 1:
                met = _clot_metrics(phi.detach(), phi_gt, torch.ones_like(phi_gt, dtype=torch.bool))
                f1_last = float(met["clot_f1"])

    if not losses:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, {"clot_f1": 0.0}
    return torch.stack(losses).mean(), {"clot_f1": f1_last}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train T0 Rung4 s1 residual phi MLP")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=-1.0, help="residual scale; default from env")
    ap.add_argument("--time-stride", type=int, default=3)
    ap.add_argument("--out", default=DEFAULT_S1_CKPT)
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    graph_dir = root / args.graph_dir
    anchors = _list_anchors(graph_dir)
    val_stem = args.val_anchor.strip().lower()
    train_paths = [p for p in anchors if p.stem.lower() != val_stem]
    val_paths = [p for p in anchors if p.stem.lower() == val_stem]
    if not val_paths:
        val_paths = [anchors[-1]]
        train_paths = anchors[:-1]
    if not train_paths:
        train_paths = val_paths

    alpha = s1_residual_alpha() if args.alpha < 0 else float(args.alpha)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    model = T0R4S1PhiMLP(in_dim=feature_dim(), hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = root / args.out
    best_f1 = -1.0
    log_path = out_path.parent / "train_log.jsonl"

    print(f"[i] train={[p.stem for p in train_paths]} val={[p.stem for p in val_paths]}", flush=True)
    print(f"[i] in_dim={feature_dim()} hidden={args.hidden} alpha={alpha}", flush=True)

    for ep in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        train_losses: list[torch.Tensor] = []
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            loss, _ = _coupled_loss_on_anchor(
                data,
                model,
                phys_cfg=phys,
                bio_cfg=bio,
                device=device,
                alpha=alpha,
                time_stride=args.time_stride,
            )
            train_losses.append(loss)
        loss = torch.stack(train_losses).mean()
        loss.backward()
        opt.step()

        model.eval()
        val_f1s: list[float] = []
        with torch.no_grad():
            for p in val_paths:
                data = torch.load(p, map_location=device, weights_only=False)
                _, met = _coupled_loss_on_anchor(
                    data,
                    model,
                    phys_cfg=phys,
                    bio_cfg=bio,
                    device=device,
                    alpha=alpha,
                    time_stride=1,
                )
                val_f1s.append(float(met["clot_f1"]))
        val_f1 = sum(val_f1s) / max(len(val_f1s), 1)
        row = {
            "epoch": ep,
            "train_loss": float(loss.item()),
            "val_f1": val_f1,
            "sec": round(time.perf_counter() - t0, 2),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[ep {ep:3d}] loss={row['train_loss']:.4f} val_f1={val_f1:.3f} ({row['sec']}s)", flush=True)
        if val_f1 >= best_f1:
            best_f1 = val_f1
            save_s1_checkpoint(
                out_path,
                model,
                alpha=alpha,
                hidden=args.hidden,
                meta={"val_anchor": val_stem, "val_f1": val_f1, "epoch": ep},
            )

    print(f"[OK] best val_f1={best_f1:.3f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
