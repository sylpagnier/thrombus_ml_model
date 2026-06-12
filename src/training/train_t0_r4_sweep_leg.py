"""Train one T0 Rung4 sweep leg (s4 / s5 / S-star recipe).

Usage::

    python -m src.training.train_t0_r4_sweep_leg --recipe s4_delta_gnn
    python -m src.training.train_t0_r4_sweep_leg --recipe s_star_full --out outputs/biochem/sweep_t0_r4_arch_6h/s_star_full/best.pth
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
from src.core_physics.t0_mu_physics import (
    gt_clot_phi_at_time,
    predict_clot_phi_at_time,
    rollout_t0_clot_phi,
)
from src.core_physics.t0_r4_s2_species import _apply_loc_gate_residual, _s0_gate_from_species
from src.core_physics.t0_r4_sweep import (
    DEFAULT_SWEEP_CKPT,
    T0R4SweepBundle,
    _build_models,
    apply_sweep_species_at_time,
    recipe_from_id,
    rollout_sweep_species_series,
    save_sweep_checkpoint,
)
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    resting_species_log_nd,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health
from src.training.train_t0_r4_s2_species import _fn_fp_masks
from src.utils.paths import get_project_root

_SKIP_GPU_KEYS = frozenset({"y", "y_valid_mask"})


def _list_anchors(root: Path) -> list[Path]:
    paths = sorted(root.glob("patient*.pt"))
    if not paths:
        raise FileNotFoundError(f"No anchors in {root}")
    return paths


def _anchor_to_device(data, device: torch.device):
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


def _coupled_sweep_loss(
    data,
    bundle: T0R4SweepBundle,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> torch.Tensor:
    recipe = bundle.recipe
    n_steps = int(data.y.shape[0])
    stride = max(int(recipe.time_stride), 1)
    loss_times = set(range(0, n_steps, stride))
    if (n_steps - 1) not in loss_times:
        loss_times.add(n_steps - 1)

    data = _anchor_to_device(data, device)
    commits_prev = None
    phi_prev = None
    dyn_h = None
    pred_series = data.y.to(device=device)
    rest = resting_species_log_nd(data, device)
    losses: list[torch.Tensor] = []

    for t in range(n_steps):
        elig = resolve_nucleation_eligibility(
            data, t, device, phys_cfg, bio_cfg, commits_prev=commits_prev,
            growth_seed="pred", nucleation_hops=1, use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
        ).reshape(-1).bool()
        s0_sp = _build_s0_deploy_species(
            data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev,
        )
        gate_s0 = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)

        if t in loss_times:
            sp, dyn_h, aux = apply_sweep_species_at_time(
                data, t, device, phys_cfg, bio_cfg, bundle,
                commits_prev=commits_prev, phi_prev=phi_prev, dyn_h=dyn_h, train=True,
            )
            sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
            phi_gt = gt_clot_phi_at_time(data, t, phys_cfg, device).reshape(-1)
            fn, fp = _fn_fp_masks(s0_sp, sp_gt, phi_gt, elig, gate_s0)

            logit = aux.get("logit")
            if logit is not None and recipe.gate != "none":
                tnh = torch.tanh(logit.reshape(-1))
                if bool(fn.any().item()):
                    losses.append(recipe.w_fn * F.mse_loss(tnh[fn], torch.full_like(tnh[fn], recipe.fn_target)))
                if bool(fp.any().item()):
                    losses.append(recipe.w_fp * F.mse_loss(tnh[fp], torch.full_like(tnh[fp], recipe.fp_target)))
                gate_pred = _apply_loc_gate_residual(
                    gate_s0, logit, elig, onset=1.0, loc_scale=recipe.loc_scale,
                )
                gate_fn_tgt = min(0.5 + 0.5 * float(recipe.loc_scale), 1.0)
                gate_fp_tgt = max(0.5 - 0.5 * float(recipe.loc_scale), 0.0)
                if bool(fn.any().item()):
                    losses.append(recipe.w_fn * F.mse_loss(gate_pred[fn], torch.full_like(gate_pred[fn], gate_fn_tgt)))
                if bool(fp.any().item()):
                    losses.append(recipe.w_fp * F.mse_loss(gate_pred[fp], torch.full_like(gate_pred[fp], gate_fp_tgt)))

            if recipe.w_species > 0.0:
                delta = aux.get("delta")
                ds = max(float(recipe.delta_scale), 1e-6)
                if delta is not None and recipe.species in ("gnn_delta", "mlp_delta"):
                    # FN-only delta targets (s2 delta mode). FP->rest species MSE kills s0 hotspots.
                    if bool(fn.any().item()):
                        tgt = torch.zeros_like(delta)
                        tgt[fn, 0] = (sp_gt[fn, FI_SLICE_IDX] - s0_sp[fn, FI_SLICE_IDX]) / ds
                        tgt[fn, 1] = (sp_gt[fn, MAT_SLICE_IDX] - s0_sp[fn, MAT_SLICE_IDX]) / ds
                        losses.append(recipe.w_species * F.mse_loss(delta[fn], tgt[fn]))
                    if recipe.w_fp > 0.0 and bool(fp.any().item()):
                        losses.append(recipe.w_fp * (delta[fp] ** 2).mean())
                else:
                    if bool(fn.any().item()):
                        losses.append(
                            recipe.w_species * F.mse_loss(sp[fn, FI_SLICE_IDX], sp_gt[fn, FI_SLICE_IDX])
                            + recipe.w_species * F.mse_loss(sp[fn, MAT_SLICE_IDX], sp_gt[fn, MAT_SLICE_IDX])
                        )
                    if bool(fp.any().item()):
                        losses.append(
                            recipe.w_species * F.mse_loss(sp[fp, FI_SLICE_IDX], rest[fp, FI_SLICE_IDX])
                            + recipe.w_species * F.mse_loss(sp[fp, MAT_SLICE_IDX], rest[fp, MAT_SLICE_IDX])
                        )

            if recipe.w_commit > 0.0 and bool(elig.any().item()):
                pred_series[t, :, 4:16] = sp
                with t0_rung2_env():
                    phi_raw, _ = predict_clot_phi_at_time(
                        data, t, phys_cfg, bio_cfg, device,
                        gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_series,
                    )
                e = elig.reshape(-1).bool()
                gt_clot = (phi_gt >= 0.5).float()
                pred_phi = phi_raw.reshape(-1).clamp(1e-4, 1 - 1e-4)
                mask = e & (fn | fp)
                if bool(mask.any().item()):
                    losses.append(recipe.w_commit * F.binary_cross_entropy(pred_phi[mask], gt_clot[mask]))
            sp_fwd = sp.detach()
            dyn_h = dyn_h.detach() if dyn_h is not None else None
        else:
            with torch.no_grad():
                sp_fwd, dyn_h, _ = apply_sweep_species_at_time(
                    data, t, device, phys_cfg, bio_cfg, bundle,
                    commits_prev=commits_prev, phi_prev=phi_prev, dyn_h=dyn_h,
                )

        with torch.no_grad():
            pred_series[t, :, 4:16] = sp_fwd
            with t0_rung2_env():
                phi_raw, _ = predict_clot_phi_at_time(
                    data, t, phys_cfg, bio_cfg, device,
                    gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_series,
                )
            phi_prev = phi_raw.reshape(-1).clamp(0.0, 1.0)
            commits_prev = (phi_prev >= 0.5).bool()

    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


def _val_metrics(data, bundle: T0R4SweepBundle, *, phys_cfg, bio_cfg, device) -> dict[str, float]:
    pred = rollout_sweep_species_series(data, phys_cfg, bio_cfg, device, bundle)
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
    ap = argparse.ArgumentParser(description="Train T0 Rung4 sweep leg")
    ap.add_argument("--recipe", required=True, help="Recipe id from t0_r4_sweep.RECIPES")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--epochs", type=int, default=0, help="0 = use recipe default")
    ap.add_argument("--early-stop", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_SWEEP_CKPT)
    args = ap.parse_args()

    recipe = recipe_from_id(args.recipe)
    if recipe.family == "ref":
        print(f"[skip] recipe {recipe.id} is eval-only")
        return 0

    epochs = int(args.epochs) if args.epochs > 0 else int(recipe.epochs)
    early_stop = int(args.early_stop) if args.early_stop > 0 else int(recipe.early_stop)

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

    bundle = _build_models(recipe, device)
    params = []
    for m in (bundle.gate_model, bundle.species_model, bundle.dyn_model):
        if m is not None:
            params.extend(m.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)

    out_path = root / args.out
    best_f1 = -1e9
    stale = 0
    log_path = out_path.parent / "train_log.jsonl"
    phys0 = PhysicsConfig(phase="biochem")
    bio0 = BiochemConfig(phase="biochem")

    # Zero-init equals s0 for delta heads; seed ckpt so ep1 cannot wipe clot preds.
    for m in (bundle.gate_model, bundle.species_model, bundle.dyn_model):
        if m is not None:
            m.eval()
    with torch.no_grad():
        init_rows = []
        for p in val_paths:
            data0 = torch.load(p, map_location="cpu", weights_only=False)
            data0 = _anchor_to_device(data0, device)
            init_rows.append(_val_metrics(data0, bundle, phys_cfg=phys0, bio_cfg=bio0, device=device))
        init_f1 = sum(r["val_f1"] for r in init_rows) / max(len(init_rows), 1)
    best_f1 = float(init_f1)
    save_sweep_checkpoint(
        out_path, bundle,
        meta={"val_anchor": val_stem, "val_f1": init_f1, "epoch": 0, "recipe": recipe.id, "note": "zero_init"},
    )
    print(f"[i] zero_init val_f1={init_f1:.3f} -> {out_path}", flush=True)

    print(
        f"[i] recipe={recipe.id} family={recipe.family} gate={recipe.gate} "
        f"species={recipe.species} dyn={recipe.dyn}",
        flush=True,
    )
    print(
        f"[i] train={[p.stem for p in train_paths]} val={[p.stem for p in val_paths]} "
        f"epochs={epochs} early_stop={early_stop}",
        flush=True,
    )
    print(
        f"[i] loc_scale={recipe.loc_scale} delta_scale={recipe.delta_scale} "
        f"w_fn={recipe.w_fn} w_fp={recipe.w_fp} w_commit={recipe.w_commit} w_species={recipe.w_species}",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        for m in (bundle.gate_model, bundle.species_model, bundle.dyn_model):
            if m is not None:
                m.train()
        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        n_train = 0
        for p in train_paths:
            data = torch.load(p, map_location="cpu", weights_only=False)
            loss = _coupled_sweep_loss(data, bundle, phys_cfg=PhysicsConfig(phase="biochem"),
                                       bio_cfg=BiochemConfig(phase="biochem"), device=device)
            loss.backward()
            train_loss += float(loss.detach().item())
            n_train += 1
            del data, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        opt.step()
        loss_mean = train_loss / max(n_train, 1)

        for m in (bundle.gate_model, bundle.species_model, bundle.dyn_model):
            if m is not None:
                m.eval()
        phys = PhysicsConfig(phase="biochem")
        bio = BiochemConfig(phase="biochem")
        val_rows = []
        with torch.no_grad():
            for p in val_paths:
                data = torch.load(p, map_location="cpu", weights_only=False)
                data = _anchor_to_device(data, device)
                val_rows.append(_val_metrics(data, bundle, phys_cfg=phys, bio_cfg=bio, device=device))

        def _mean(k: str) -> float:
            return sum(r[k] for r in val_rows) / max(len(val_rows), 1)

        row = {
            "epoch": ep,
            "recipe": recipe.id,
            "train_loss": loss_mean,
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
            f"[ep {ep:3d}] loss={row['train_loss']:.4f} val_f1={row['val_f1']:.3f} "
            f"health={row['health_score']:.3f}{fail} ({row['sec']}s)",
            flush=True,
        )
        eligible = not row["wall_carpet"]
        if eligible and row["val_f1"] > best_f1 + 1e-5:
            best_f1 = row["val_f1"]
            stale = 0
            save_sweep_checkpoint(
                out_path, bundle,
                meta={"val_anchor": val_stem, "val_f1": row["val_f1"], "epoch": ep, "recipe": recipe.id},
            )
        else:
            stale += 1
        if stale >= early_stop:
            print(f"[OK] early stop: no val_f1 gain for {stale} epochs", flush=True)
            break

    print(f"[OK] best val_f1={best_f1:.3f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
