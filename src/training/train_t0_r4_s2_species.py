"""Train Rung 4 s2 species step inside E(t).

Modes:
  loc (default) -- risk reweight before s0 top-frac hotspot gate
  delta         -- deprecated FI/Mat residual

Usage::

    python -m src.training.train_t0_r4_s2_species --mode loc
    python -m src.training.train_t0_r4_s2_species --val-anchor patient007 --epochs 80
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time
from src.core_physics.t0_r4_s2_species import (
    DEFAULT_S2_CKPT,
    DEFAULT_S2_DELTA_CKPT,
    S2_MODE_DELTA,
    S2_MODE_LOC,
    T0R4S2LocMLP,
    T0R4S2SpeciesMLP,
    _apply_loc_risk_adjustment,
    _risk_n_at_time,
    apply_s2_species_delta,
    build_s2_features,
    feature_dim,
    rollout_s2_species_series,
    s2_delta_scale,
    s2_loc_scale,
    save_s2_checkpoint,
)
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    _s0_onset_factor,
    resting_species_log_nd,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health
from src.core_physics.t0_r4_s2_species import _s0_gate_from_species
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def _list_anchors(root: Path) -> list[Path]:
    paths = sorted(root.glob("patient*.pt"))
    if not paths:
        raise FileNotFoundError(f"No anchors in {root}")
    return paths


def _fn_fp_masks(
    s0_sp: torch.Tensor,
    sp_gt: torch.Tensor,
    phi_gt: torch.Tensor,
    elig: torch.Tensor,
    gate: torch.Tensor,
    *,
    hot_gate: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FN = GT clot missed by s0; FP = s0 hotspot outside GT clot (in E(t))."""
    gt_clot = phi_gt.reshape(-1) >= 0.5
    e = elig.reshape(-1).bool()
    s0_hot = gate.reshape(-1) > float(hot_gate)
    fn = gt_clot & e & ~s0_hot
    fp = s0_hot & e & ~gt_clot
    return fn, fp


def _coupled_loc_loss(
    data,
    model: T0R4S2LocMLP,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    loc_scale: float,
    time_stride: int,
    w_fn: float = 3.0,
    w_fp: float = 2.0,
    fn_target: float = 0.85,
    fp_target: float = -0.85,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise tanh(logit): boost FN clot nodes, suppress s0 FP hotspots."""
    n_steps = int(data.y.shape[0])
    stride = max(int(time_stride), 1)
    times = list(range(0, n_steps, stride))
    if times[-1] != n_steps - 1:
        times.append(n_steps - 1)

    commits_prev = None
    pred_series = data.y.clone().to(device=device)
    losses: list[torch.Tensor] = []
    n_fn = 0
    n_fp = 0

    for t in times:
        elig = resolve_nucleation_eligibility(
            data, t, device, phys_cfg, bio_cfg, commits_prev=commits_prev,
            growth_seed="pred", nucleation_hops=1, use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
        ).reshape(-1).bool()
        s0_sp = _build_s0_deploy_species(
            data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev
        )
        gate = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)
        feats = build_s2_features(
            data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=gate
        )
        tau = float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))
        onset = float(_s0_onset_factor(tau))
        logit = model(feats) * onset
        tnh = torch.tanh(logit.reshape(-1))

        phi_gt = gt_clot_phi_at_time(data, t, phys_cfg, device).reshape(-1)
        fn, fp = _fn_fp_masks(s0_sp, data.y[t, :, 4:16], phi_gt, elig, gate)
        if bool(fn.any().item()):
            losses.append(w_fn * F.mse_loss(tnh[fn], torch.full_like(tnh[fn], fn_target)))
            n_fn += 1
        if bool(fp.any().item()):
            losses.append(w_fp * F.mse_loss(tnh[fp], torch.full_like(tnh[fp], fp_target)))
            n_fp += 1

        with torch.no_grad():
            risk_n = _risk_n_at_time(data, t, device, bio_cfg, elig=elig)
            risk_adj = _apply_loc_risk_adjustment(
                risk_n, logit.detach(), elig, onset=onset, loc_scale=loc_scale
            )
            sp = _build_s0_deploy_species(
                data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev,
                risk_n_override=risk_adj,
            )
        pred_series[t, :, 4:16] = sp

        with torch.no_grad():
            with t0_rung2_env():
                phi_raw, _ = predict_clot_phi_at_time(
                    data, t, phys_cfg, bio_cfg, device,
                    gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_series,
                )
            commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()

    if not losses:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, {"n_fn_steps": 0, "n_fp_steps": 0}
    return torch.stack(losses).mean(), {"n_fn_steps": n_fn, "n_fp_steps": n_fp}


def _coupled_delta_loss(
    data,
    model: T0R4S2SpeciesMLP,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    delta_scale: float,
    time_stride: int,
    w_fn: float = 3.0,
    w_fp: float = 2.0,
    loss_scale: float = 1.0e4,
) -> tuple[torch.Tensor, dict[str, float]]:
    n_steps = int(data.y.shape[0])
    stride = max(int(time_stride), 1)
    times = list(range(0, n_steps, stride))
    if times[-1] != n_steps - 1:
        times.append(n_steps - 1)

    commits_prev = None
    pred_series = data.y.clone().to(device=device)
    rest = resting_species_log_nd(data, device)
    losses: list[torch.Tensor] = []
    fi_mae_acc = 0.0
    mat_mae_acc = 0.0
    n_fn = 0
    n_fp = 0
    scale = float(loss_scale)

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
        gate = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)
        feats = build_s2_features(
            data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=gate
        )
        tau = float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))
        onset = float(_s0_onset_factor(tau))
        delta = model(feats) * onset
        sp = apply_s2_species_delta(s0_sp, delta, elig, delta_scale=delta_scale)
        pred_series[t, :, 4:16] = sp.detach()

        sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
        phi_gt = gt_clot_phi_at_time(data, t, phys_cfg, device).reshape(-1)
        fn, fp = _fn_fp_masks(s0_sp, sp_gt, phi_gt, elig, gate)
        target = torch.zeros_like(delta)
        if bool(fn.any().item()):
            target[fn, 0] = (sp_gt[fn, FI_SLICE_IDX] - s0_sp[fn, FI_SLICE_IDX]) / max(delta_scale, 1e-6)
            target[fn, 1] = (sp_gt[fn, MAT_SLICE_IDX] - s0_sp[fn, MAT_SLICE_IDX]) / max(delta_scale, 1e-6)
            losses.append(
                w_fn
                * F.mse_loss(scale * delta[fn], scale * target[fn])
            )
            fi_mae_acc += float((sp[fn, FI_SLICE_IDX] - sp_gt[fn, FI_SLICE_IDX]).abs().mean().item())
            mat_mae_acc += float((sp[fn, MAT_SLICE_IDX] - sp_gt[fn, MAT_SLICE_IDX]).abs().mean().item())
            n_fn += 1
        if bool(fp.any().item()):
            target[fp, 0] = (rest[fp, FI_SLICE_IDX] - s0_sp[fp, FI_SLICE_IDX]) / max(delta_scale, 1e-6)
            target[fp, 1] = (rest[fp, MAT_SLICE_IDX] - s0_sp[fp, MAT_SLICE_IDX]) / max(delta_scale, 1e-6)
            losses.append(
                w_fp
                * F.mse_loss(scale * delta[fp], scale * target[fp])
            )
            n_fp += 1

        with torch.no_grad():
            with t0_rung2_env():
                phi_raw, _ = predict_clot_phi_at_time(
                    data,
                    t,
                    phys_cfg,
                    bio_cfg,
                    device,
                    gamma_mode=RUNG2_GAMMA_MODE,
                    flow_source="gt",
                    pred_species_series=pred_series,
                )
            commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()

    if not losses:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, {"fi_log_mae": 0.0, "mat_log_mae": 0.0, "clot_f1": 0.0}

    loss = torch.stack(losses).mean()
    stats = {
        "fi_log_mae": fi_mae_acc / max(n_fn, 1),
        "mat_log_mae": mat_mae_acc / max(n_fn, 1),
        "n_fn_steps": n_fn,
        "n_fp_steps": n_fp,
        "clot_f1": 0.0,
    }
    return loss, stats


@torch.no_grad()
def _val_rollout_metrics(
    data,
    model: T0R4S2LocMLP | T0R4S2SpeciesMLP,
    *,
    mode: str,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    delta_scale: float,
    loc_scale: float,
    hidden: int,
) -> dict[str, float]:
    from src.core_physics.t0_mu_physics import rollout_t0_clot_phi
    from src.core_physics.t0_r4_s2_species import T0R4S2Bundle

    bundle = T0R4S2Bundle(
        model=model,
        mode=mode,
        delta_scale=delta_scale,
        loc_scale=loc_scale,
        in_dim=feature_dim(),
        hidden=hidden,
        device=device,
    )
    pred = rollout_s2_species_series(data, phys_cfg, bio_cfg, device, bundle)
    t_last = int(data.y.shape[0]) - 1
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data,
            phys_cfg,
            bio_cfg,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            flow_source="gt",
            pred_species_series=pred,
            nucleation=True,
            nucleation_hops=1,
        )
    phi_traj = {int(t): v["phi"] for t, v in traj.items()}
    health = compute_rung4_rollout_health(phi_traj, data, phys_cfg, bio_cfg, device)
    sp_gt = data.y[t_last, :, 4:16].to(device=device)
    sp_p = pred[t_last, :, 4:16]
    commits_prev = None
    for t in range(t_last):
        with t0_rung2_env():
            phi_t, _ = predict_clot_phi_at_time(
                data,
                t,
                phys_cfg,
                bio_cfg,
                device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="gt",
                pred_species_series=pred,
            )
        commits_prev = (phi_t.reshape(-1) >= 0.5).bool()
    elig = resolve_nucleation_eligibility(
        data,
        t_last,
        device,
        phys_cfg,
        bio_cfg,
        commits_prev=commits_prev,
        growth_seed="pred",
    )
    m = elig.reshape(-1).bool()
    fi_mae = float((sp_p[m, FI_SLICE_IDX] - sp_gt[m, FI_SLICE_IDX]).abs().mean().item()) if m.any() else 0.0
    mat_mae = float((sp_p[m, MAT_SLICE_IDX] - sp_gt[m, MAT_SLICE_IDX]).abs().mean().item()) if m.any() else 0.0
    return {
        "val_f1": float(health["final_f1"]),
        "val_fi_log_mae": fi_mae,
        "val_mat_log_mae": mat_mae,
        "health_score": float(health["health_score"]),
        "health_pass": float(health["health_pass"]),
        "wall_ring_t0": float(health["wall_ring_t0"]),
        "frozen_wall_ring": float(health["frozen_wall_ring"]),
        "wall_carpet": float(health["wall_carpet"]),
        "early_phi_wall_max": float(health["early_phi_wall_max"]),
        "min_f1": float(health["min_f1"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train T0 Rung4 s2 species (loc or delta)")
    ap.add_argument("--mode", choices=["loc", "delta"], default="loc")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--delta-scale", type=float, default=-1.0)
    ap.add_argument("--loc-scale", type=float, default=-1.0)
    ap.add_argument("--time-stride", type=int, default=2)
    ap.add_argument("--w-fn", type=float, default=3.0)
    ap.add_argument("--w-fp", type=float, default=2.0)
    ap.add_argument("--loss-scale", type=float, default=1.0e4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    mode = S2_MODE_LOC if args.mode == "loc" else S2_MODE_DELTA
    if not args.out.strip():
        args.out = DEFAULT_S2_CKPT if mode == S2_MODE_LOC else DEFAULT_S2_DELTA_CKPT

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

    delta_scale = s2_delta_scale() if args.delta_scale < 0 else float(args.delta_scale)
    loc_scale = s2_loc_scale() if args.loc_scale < 0 else float(args.loc_scale)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    if mode == S2_MODE_LOC:
        model = T0R4S2LocMLP(in_dim=feature_dim(), hidden=args.hidden).to(device)
    else:
        model = T0R4S2SpeciesMLP(in_dim=feature_dim(), hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = root / args.out
    best_score = -1e9
    log_path = out_path.parent / "train_log.jsonl"

    print(f"[i] train={[p.stem for p in train_paths]} val={[p.stem for p in val_paths]}", flush=True)
    print(
        f"[i] mode={mode} in_dim={feature_dim()} hidden={args.hidden} "
        f"loc_scale={loc_scale} delta_scale={delta_scale} "
        f"w_fn={args.w_fn} w_fp={args.w_fp}",
        flush=True,
    )

    for ep in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        train_losses: list[torch.Tensor] = []
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            if mode == S2_MODE_LOC:
                loss, _ = _coupled_loc_loss(
                    data, model, phys_cfg=phys, bio_cfg=bio, device=device,
                    loc_scale=loc_scale, time_stride=args.time_stride,
                    w_fn=args.w_fn, w_fp=args.w_fp,
                )
            else:
                loss, _ = _coupled_delta_loss(
                    data, model, phys_cfg=phys, bio_cfg=bio, device=device,
                    delta_scale=delta_scale, time_stride=args.time_stride,
                    w_fn=args.w_fn, w_fp=args.w_fp, loss_scale=args.loss_scale,
                )
            train_losses.append(loss)
        loss = torch.stack(train_losses).mean()
        loss.backward()
        opt.step()

        model.eval()
        val_rows: list[dict[str, float]] = []
        with torch.no_grad():
            for p in val_paths:
                data = torch.load(p, map_location=device, weights_only=False)
                val_rows.append(
                    _val_rollout_metrics(
                        data, model, mode=mode, phys_cfg=phys, bio_cfg=bio,
                        device=device, delta_scale=delta_scale, loc_scale=loc_scale,
                        hidden=args.hidden,
                    )
                )

        def _mean(key: str) -> float:
            return sum(r[key] for r in val_rows) / max(len(val_rows), 1)

        val_f1 = _mean("val_f1")
        val_fi_mae = _mean("val_fi_log_mae")
        val_mat_mae = _mean("val_mat_log_mae")
        health_score = _mean("health_score")
        wall_ring_t0 = _mean("wall_ring_t0")
        frozen = _mean("frozen_wall_ring") >= 0.5
        wall_carpet = _mean("wall_carpet") >= 0.5
        early_phi_wall = _mean("early_phi_wall_max")
        row = {
            "epoch": ep,
            "train_loss": float(loss.item()),
            "val_f1": val_f1,
            "val_fi_log_mae": val_fi_mae,
            "val_mat_log_mae": val_mat_mae,
            "health_score": health_score,
            "wall_ring_t0": wall_ring_t0,
            "frozen_wall_ring": frozen,
            "sec": round(time.perf_counter() - t0, 2),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        fail_tag = ""
        if frozen:
            fail_tag = " [FAIL frozen wall]"
        elif wall_carpet:
            fail_tag = " [FAIL wall carpet]"
        print(
            f"[ep {ep:3d}] loss={row['train_loss']:.6f} "
            f"val_f1={val_f1:.3f} health={health_score:.3f} early_wall={early_phi_wall:.3f}"
            f"{fail_tag} ({row['sec']}s)",
            flush=True,
        )
        score = health_score
        if score >= best_score:
            best_score = score
            save_s2_checkpoint(
                out_path,
                model,
                mode=mode,
                hidden=args.hidden,
                delta_scale=delta_scale,
                loc_scale=loc_scale,
                meta={
                    "val_anchor": val_stem,
                    "val_f1": val_f1,
                    "health_score": health_score,
                    "wall_ring_t0": wall_ring_t0,
                    "frozen_wall_ring": frozen,
                    "val_fi_log_mae": val_fi_mae,
                    "val_mat_log_mae": val_mat_mae,
                    "epoch": ep,
                },
            )

    print(f"[OK] best health_score={best_score:.3f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
