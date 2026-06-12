"""Train Rung 4 s3: wall-band GRU + s2_loc residual on hotspot logits in E(t).

Usage::

    python -m src.training.train_t0_r4_s3_temporal
    python -m src.training.train_t0_r4_s3_temporal --val-anchor patient007 --epochs 60 --init-s2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import (
    gt_clot_phi_at_time,
    predict_clot_phi_at_time,
    rollout_t0_clot_phi,
)
from src.core_physics.t0_r4_s2_species import (
    DEFAULT_S2_CKPT,
    _apply_loc_gate_residual,
    _s0_gate_from_species,
    build_s3_features,
    s3_feature_dim,
)
from src.core_physics.t0_r4_s3_temporal import (
    DEFAULT_S3_CKPT,
    S3_ACTUATOR_GATE,
    T0R4S3TemporalModel,
    _species_from_logit,
    load_s3_residual_from_s2,
    rollout_s3_species_series,
    s3_actuator,
    save_s3_checkpoint,
    s3_loc_scale,
    s3_res_scale,
)
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    _s0_onset_factor,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health
from src.training.train_t0_r4_s2_species import _fn_fp_masks
from src.utils.paths import get_project_root


def _list_anchors(root: Path) -> list[Path]:
    paths = sorted(root.glob("patient*.pt"))
    if not paths:
        raise FileNotFoundError(f"No anchors in {root}")
    return paths


_SKIP_GPU_KEYS = frozenset({"y", "y_valid_mask"})


def _anchor_to_device(data, device: torch.device):
    """Keep ``y`` on CPU; move other graph tensors to GPU once per anchor."""
    if getattr(data, "_t0_r4_gpu_ready", False):
        return data
    for key in data.keys():
        if key in _SKIP_GPU_KEYS:
            continue
        val = data[key]
        if isinstance(val, torch.Tensor):
            data[key] = val.to(device=device)
    data._t0_r4_gpu_ready = True
    return data


def _coupled_gru_loss(
    data,
    model: T0R4S3TemporalModel,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    loc_scale: float,
    res_scale: float,
    time_stride: int,
    tbptt_len: int,
    w_fn: float,
    w_fp: float,
    fn_target: float,
    fp_target: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    n_steps = int(data.y.shape[0])
    n = int(data.num_nodes)
    stride = max(int(time_stride), 1)
    tbptt = max(int(tbptt_len), 1)
    loss_times = set(range(0, n_steps, stride))
    if (n_steps - 1) not in loss_times:
        loss_times.add(n_steps - 1)

    data = _anchor_to_device(data, device)
    commits_prev = None
    pred_series = data.y.to(device=device)
    h = model.init_hidden(n, device, torch.float32)
    losses: list[torch.Tensor] = []
    tbptt_i = 0
    phi_prev: torch.Tensor | None = None

    for t in range(n_steps):
        elig = resolve_nucleation_eligibility(
            data, t, device, phys_cfg, bio_cfg, commits_prev=commits_prev,
            growth_seed="pred", nucleation_hops=1, use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
        ).reshape(-1).bool()
        s0_sp = _build_s0_deploy_species(
            data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev
        )
        gate = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)
        feats = build_s3_features(
            data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=gate,
            commits_prev=commits_prev, phi_prev=phi_prev,
        )
        if t in loss_times:
            logit, h = model.forward_step(feats, h, res_scale=res_scale)
            onset = float(_s0_onset_factor(float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))))
            tnh = torch.tanh(logit.reshape(-1) * onset)
            sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
            phi_gt = gt_clot_phi_at_time(data, t, phys_cfg, device).reshape(-1)
            fn, fp = _fn_fp_masks(s0_sp, sp_gt, phi_gt, elig, gate)
            if bool(fn.any().item()):
                losses.append(w_fn * F.mse_loss(tnh[fn], torch.full_like(tnh[fn], fn_target)))
            if bool(fp.any().item()):
                losses.append(w_fp * F.mse_loss(tnh[fp], torch.full_like(tnh[fp], fp_target)))
            if s3_actuator() == S3_ACTUATOR_GATE:
                gate_pred = _apply_loc_gate_residual(
                    gate, logit, elig, onset=onset, loc_scale=loc_scale
                )
                gate_fn_tgt = min(0.5 + 0.5 * float(loc_scale), 1.0)
                gate_fp_tgt = max(0.5 - 0.5 * float(loc_scale), 0.0)
                if bool(fn.any().item()):
                    losses.append(w_fn * F.mse_loss(gate_pred[fn], torch.full_like(gate_pred[fn], gate_fn_tgt)))
                if bool(fp.any().item()):
                    losses.append(w_fp * F.mse_loss(gate_pred[fp], torch.full_like(gate_pred[fp], gate_fp_tgt)))
            tbptt_i += 1
            if tbptt_i >= tbptt:
                h = h.detach()
                tbptt_i = 0
            logit_fwd = logit.detach()
        else:
            with torch.no_grad():
                logit, h = model.forward_step(feats, h, res_scale=res_scale)
            logit_fwd = logit

        with torch.no_grad():
            sp = _species_from_logit(
                data, t, device, bio_cfg,
                elig=elig, commits_prev=commits_prev,
                logit=logit_fwd, loc_scale=loc_scale,
                s0_sp=s0_sp, s0_gate=gate,
            )
            pred_series[t, :, 4:16] = sp
            with t0_rung2_env():
                phi_raw, _ = predict_clot_phi_at_time(
                    data, t, phys_cfg, bio_cfg, device,
                    gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_series,
                )
            phi_prev = phi_raw.reshape(-1).clamp(0.0, 1.0)
            commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()

    if not losses:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, {}
    return torch.stack(losses).mean(), {}


@torch.no_grad()
def _val_rollout_metrics(
    data,
    model: T0R4S3TemporalModel,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    loc_scale: float,
    res_scale: float,
    gru_hidden: int,
    res_hidden: int,
) -> dict[str, float]:
    from src.core_physics.t0_r4_s3_temporal import T0R4S3Bundle

    bundle = T0R4S3Bundle(
        model=model,
        loc_scale=loc_scale,
        res_scale=res_scale,
        in_dim=s3_feature_dim(),
        gru_hidden=gru_hidden,
        res_hidden=res_hidden,
        device=device,
        actuator=s3_actuator(),
    )
    pred = rollout_s3_species_series(data, phys_cfg, bio_cfg, device, bundle)
    t_last = int(data.y.shape[0]) - 1
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys_cfg, bio_cfg, device,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
            pred_species_series=pred, nucleation=True, nucleation_hops=1,
        )
    phi_traj = {int(t): v["phi"] for t, v in traj.items()}
    health = compute_rung4_rollout_health(phi_traj, data, phys_cfg, bio_cfg, device)
    sp_gt = data.y[t_last, :, 4:16].to(device=device)
    sp_p = pred[t_last, :, 4:16]
    commits_prev = None
    for t in range(t_last):
        with t0_rung2_env():
            phi_t, _ = predict_clot_phi_at_time(
                data, t, phys_cfg, bio_cfg, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
            )
        commits_prev = (phi_t.reshape(-1) >= 0.5).bool()
    elig = resolve_nucleation_eligibility(
        data, t_last, device, phys_cfg, bio_cfg, commits_prev=commits_prev, growth_seed="pred",
    ).reshape(-1).bool()
    fi_mae = float((sp_p[elig, FI_SLICE_IDX] - sp_gt[elig, FI_SLICE_IDX]).abs().mean().item()) if elig.any() else 0.0
    mat_mae = float((sp_p[elig, MAT_SLICE_IDX] - sp_gt[elig, MAT_SLICE_IDX]).abs().mean().item()) if elig.any() else 0.0
    return {
        "val_f1": float(health["final_f1"]),
        "health_score": float(health["health_score"]),
        "health_pass": float(health["health_pass"]),
        "wall_carpet": float(health["wall_carpet"]),
        "early_phi_wall_max": float(health["early_phi_wall_max"]),
        "val_fi_log_mae": fi_mae,
        "val_mat_log_mae": mat_mae,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train T0 Rung4 s3 temporal GRU")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gru-hidden", type=int, default=32)
    ap.add_argument("--res-hidden", type=int, default=48)
    ap.add_argument("--loc-scale", type=float, default=-1.0)
    ap.add_argument("--res-scale", type=float, default=-1.0)
    ap.add_argument("--time-stride", type=int, default=2, help="Macro-step subsample for train loss (2 fits 4GB GPU)")
    ap.add_argument("--tbptt-len", type=int, default=1, help="Detach GRU state every N loss steps (1=minimal VRAM)")
    ap.add_argument("--w-fn", type=float, default=3.0)
    ap.add_argument("--w-fp", type=float, default=2.0)
    ap.add_argument("--fn-target", type=float, default=0.85)
    ap.add_argument("--fp-target", type=float, default=-0.85)
    ap.add_argument("--early-stop", type=int, default=12, help="Stop if val_f1 does not improve")
    ap.add_argument("--actuator", default=S3_ACTUATOR_GATE, choices=("gate", "risk"))
    ap.add_argument("--with-residual", action="store_true", help="Add frozen s2_loc MLP path (legacy risk stack)")
    ap.add_argument("--init-s2", action="store_true")
    ap.add_argument("--init-s2-ckpt", default=DEFAULT_S2_CKPT)
    ap.add_argument("--out", default=DEFAULT_S3_CKPT)
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

    loc_scale = s3_loc_scale() if args.loc_scale < 0 else float(args.loc_scale)
    res_scale = s3_res_scale() if args.res_scale < 0 else float(args.res_scale)
    use_residual = bool(args.with_residual)
    os.environ["T0_R4_S3_ACTUATOR"] = str(args.actuator)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    model = T0R4S3TemporalModel(
        in_dim=s3_feature_dim(),
        gru_hidden=args.gru_hidden,
        res_hidden=args.res_hidden,
        use_residual=use_residual,
    ).to(device)

    if args.init_s2 and use_residual and s3_actuator() == "risk":
        s2_path = root / args.init_s2_ckpt
        if load_s3_residual_from_s2(model, s2_path, device=device):
            print(f"[i] warm-start res_mlp from {s2_path}", flush=True)
        else:
            print(f"[WARN] could not load s2 residual from {s2_path}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_path = root / args.out
    best_f1 = -1e9
    stale_epochs = 0
    log_path = out_path.parent / "train_log.jsonl"

    print(f"[i] train={[p.stem for p in train_paths]} val={[p.stem for p in val_paths]}", flush=True)
    print(
        f"[i] actuator={s3_actuator()} in_dim={s3_feature_dim()} gru_hidden={args.gru_hidden} "
        f"res_hidden={args.res_hidden} residual={use_residual} "
        f"loc_scale={loc_scale} res_scale={res_scale} "
        f"time_stride={args.time_stride} tbptt_len={args.tbptt_len} "
        f"w_fn={args.w_fn} w_fp={args.w_fp} early_stop={args.early_stop}",
        flush=True,
    )

    for ep in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        train_loss_acc = 0.0
        n_train = 0
        for p in train_paths:
            data = torch.load(p, map_location="cpu", weights_only=False)
            loss, _ = _coupled_gru_loss(
                data, model, phys_cfg=phys, bio_cfg=bio, device=device,
                loc_scale=loc_scale, res_scale=res_scale,
                time_stride=args.time_stride, tbptt_len=args.tbptt_len,
                w_fn=args.w_fn, w_fp=args.w_fp,
                fn_target=args.fn_target, fp_target=args.fp_target,
            )
            loss.backward()
            train_loss_acc += float(loss.detach().item())
            n_train += 1
            del data, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        opt.step()
        loss_mean = train_loss_acc / max(n_train, 1)

        model.eval()
        val_rows: list[dict[str, float]] = []
        with torch.no_grad():
            for p in val_paths:
                data = torch.load(p, map_location="cpu", weights_only=False)
                data = _anchor_to_device(data, device)
                val_rows.append(
                    _val_rollout_metrics(
                        data, model, phys_cfg=phys, bio_cfg=bio, device=device,
                        loc_scale=loc_scale, res_scale=res_scale,
                        gru_hidden=args.gru_hidden, res_hidden=args.res_hidden,
                    )
                )

        def _mean(key: str) -> float:
            return sum(r[key] for r in val_rows) / max(len(val_rows), 1)

        row = {
            "epoch": ep,
            "train_loss": float(loss_mean),
            "val_f1": _mean("val_f1"),
            "health_score": _mean("health_score"),
            "health_pass": _mean("health_pass") >= 0.5,
            "wall_carpet": _mean("wall_carpet") >= 0.5,
            "early_phi_wall_max": _mean("early_phi_wall_max"),
            "sec": round(time.perf_counter() - t0, 2),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        fail = ""
        if row["wall_carpet"]:
            fail = " [FAIL wall carpet]"
        elif not row["health_pass"]:
            fail = " [FAIL health]"
        print(
            f"[ep {ep:3d}] loss={row['train_loss']:.4f} "
            f"val_f1={row['val_f1']:.3f} health={row['health_score']:.3f} "
            f"early_wall={row['early_phi_wall_max']:.3f}{fail} ({row['sec']}s)",
            flush=True,
        )
        eligible = not row["wall_carpet"]
        if eligible and row["val_f1"] > best_f1 + 1e-5:
            best_f1 = row["val_f1"]
            stale_epochs = 0
            save_s3_checkpoint(
                out_path,
                model,
                loc_scale=loc_scale,
                res_scale=res_scale,
                gru_hidden=args.gru_hidden,
                res_hidden=args.res_hidden,
                use_residual=use_residual,
                actuator=s3_actuator(),
                meta={
                    "val_anchor": val_stem,
                    "val_f1": row["val_f1"],
                    "health_score": row["health_score"],
                    "epoch": ep,
                    "actuator": s3_actuator(),
                },
            )
        else:
            stale_epochs += 1

        if stale_epochs >= int(args.early_stop):
            print(f"[OK] early stop: no val_f1 gain for {stale_epochs} epochs", flush=True)
            break

    print(f"[OK] best val_f1={best_f1:.3f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
