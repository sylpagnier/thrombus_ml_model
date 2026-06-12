"""Rung 4 mini-ladder eval (CUDA): Step s0, s0_phi_rule, oracle audits, vs R2/R4 teacher."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_t0_rung4 import _clot_timeline, _mu_timeline_rung4  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import eval_anchor_t0_mu  # noqa: E402
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_rung4_ladder import (
    describe_rung4_step,
    rollout_rung4_phi_trajectory,
    rollout_rung4_species_series,
    rung4_step_uses_coupled_species_rollout,
    rung4_step_uses_gt_species,
    rung4_use_dgamma_wall_seed,
    species_log_mae_in_mask,
)
from src.core_physics.t0_rung_config import (
    DEFAULT_SPECIES_DUMP_DIR,
    RUNG2_GAMMA_MODE,
    resolve_default_teacher_ckpt,
    rollout_t0_pred_species_series,
    t0_rung2_env,
    t0_rung4_env,
)
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time, metrics_for_step, predict_mu_si_at_time
from src.core_physics.t0_rung_config import t0_rung4_step_env
from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def _mu_timeline_rung4_step(
    data, phys, bio, device, times, *, pred_species_series, step: str
) -> list[dict]:
    rows = []
    with t0_rung4_step_env(step=step):
        for t in times:
            step_out = predict_mu_si_at_time(
                data, t, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
                pred_species_series=pred_species_series,
            )
            m = metrics_for_step(step_out, data, phys, device)
            rows.append({"time": int(t), **{k: v for k, v in m.__dict__.items() if k != "time_index"}})
    return rows


def _clot_from_phi_traj(phi_traj, data, phys, device, times) -> list[dict]:
    mask = torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    rows = []
    for t in times:
        phi_gt = gt_clot_phi_at_time(data, t, phys, device)
        phi_pred = phi_traj[int(t)]
        m = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)
        rows.append({"time": int(t), **m})
    return rows


def _elig_from_pred_series(data, t, phys, bio, device, pred_series) -> torch.Tensor:
    """E(t) mask from coupled pred commits (for steps without per-t species builder)."""
    commits_prev = None
    for ti in range(int(t)):
        with t0_rung2_env():
            phi_raw, _ = predict_clot_phi_at_time(
                data, ti, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_series,
            )
        commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()
    return resolve_nucleation_eligibility(
        data, int(t), device, phys, bio,
        commits_prev=commits_prev, growth_seed="pred", nucleation_hops=1,
        use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
    ).reshape(-1).bool()


def main() -> int:
    ap = argparse.ArgumentParser(description="Rung 4 mini-ladder eval")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,27,53")
    ap.add_argument("--step", default="s0")
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    if not graph.is_file():
        print(f"[ERR] missing {graph}", file=sys.stderr)
        return 1

    step_info = describe_rung4_step(args.step)
    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(graph, map_location=device, weights_only=False)

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] step={step_info.step} deploy={step_info.deploy} gt_species={step_info.uses_gt_species}", flush=True)
    print(f"[i] {step_info.description}", flush=True)

    deploy_ctx = contextlib.nullcontext()
    if step_info.step in ("species_gnn", "s34_gnn"):
        from src.inference.species_gnn_deploy_env import load_deploy_manifest, species_gnn_deploy_env

        deploy_ctx = species_gnn_deploy_env(
            load_deploy_manifest(), anchor=args.anchor, prefer_loao=True,
        )

    t0 = time.perf_counter()
    with deploy_ctx:
        phi_traj = rollout_rung4_phi_trajectory(data, phys, bio, device, step=args.step)
        pred_series = None
        if args.step not in ("s0_phi_rule", "s1_mlp_phi"):
            pred_series = rollout_rung4_species_series(data, phys, bio, device, step=args.step)
        elif args.step == "s1_mlp_phi":
            pred_series = rollout_rung4_species_series(data, phys, bio, device, step="s0")
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    with t0_rung2_env():
        rung2_mu = eval_anchor_t0_mu(graph, times=times, gamma_mode=RUNG2_GAMMA_MODE)
    clot2 = _clot_timeline(data, phys, bio, device, times, gamma_mode=RUNG2_GAMMA_MODE, env_ctx=t0_rung2_env())
    clot_step = _clot_from_phi_traj(phi_traj, data, phys, device, times)

    teacher = Path(args.teacher_ckpt) if args.teacher_ckpt.strip() else Path(resolve_default_teacher_ckpt())
    if not teacher.is_absolute():
        teacher = root / teacher
    species_dump = root / DEFAULT_SPECIES_DUMP_DIR / f"{args.anchor}.pt"
    clot4 = None
    if teacher.is_file() and pred_series is not None:
        pred_teacher = rollout_t0_pred_species_series(
            data, str(teacher), device, bio_cfg=bio,
            dumped_graph=str(species_dump) if species_dump.is_file() else None,
            time_stride=6,
        )
        clot4 = _clot_timeline(
            data, phys, bio, device, times, gamma_mode=RUNG2_GAMMA_MODE,
            pred_species_series=pred_teacher, env_ctx=t0_rung4_env(teacher_ckpt=str(teacher)),
        )

    species_mae = []
    if pred_series is not None:
        from src.core_physics.t0_rung4_ladder import build_rung4_species_log_nd_at_time

        commits_prev = None
        for t in sorted(set(times)):
            if rung4_step_uses_coupled_species_rollout(args.step):
                elig = _elig_from_pred_series(data, t, phys, bio, device, pred_series)
            else:
                _, elig = build_rung4_species_log_nd_at_time(
                    data, t, device, phys, bio, commits_prev=commits_prev, step=args.step
                )
            species_mae.append({
                "time": int(t),
                "nuc": species_log_mae_in_mask(pred_series, data, t, elig, device),
            })
            if args.step != "s3_temporal":
                with t0_rung2_env():
                    phi_raw, _ = predict_clot_phi_at_time(
                        data, t, phys, bio, device, gamma_mode=RUNG2_GAMMA_MODE,
                        flow_source="gt", pred_species_series=pred_series,
                    )
                commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()

    t_last = times[-1]
    c2 = next(r for r in clot2 if r["time"] == t_last)
    cs = next(r for r in clot_step if r["time"] == t_last)

    health = compute_rung4_rollout_health(phi_traj, data, phys, bio, device, times=times)

    payload = {
        "anchor": args.anchor,
        "device": "cuda",
        "step": step_info.step,
        "step_deploy": step_info.deploy,
        "step_uses_gt_species": step_info.uses_gt_species,
        "step_description": step_info.description,
        "nucleation_mask": "deploy_pred_commits",
        "rollout_health": {k: v for k, v in health.items() if k != "timeline"},
        "rung2": {"mu": rung2_mu.to_dict(), "clot_nucleation": clot2},
        "rung4_step": {"clot_nucleation": clot_step, "species_mae": species_mae},
    }
    if clot4 is not None:
        payload["rung4_teacher"] = {"clot_nucleation": clot4}

    out = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/clot_trigger/t0_rung4_{args.step}_{args.anchor}.json"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] {out}")
    print(f"[i] R2 F1={c2['clot_f1']:.3f} R4.{args.step} F1={cs['clot_f1']:.3f}")
    print(
        f"[i] health_score={health['health_score']:.3f} "
        f"early_phi_wall={health['early_phi_wall_max']:.3f} "
        f"wall_carpet={health['wall_carpet']} health_pass={health['health_pass']}"
    )
    if not health["health_pass"]:
        why = []
        if health["frozen_wall_ring"]:
            why.append("frozen wall-ring at t~0")
        if health["wall_carpet"]:
            why.append("wall carpet (high rec / low prec)")
        if health["early_false_commit_max"] >= 0.002:
            why.append("commits before GT clot")
        print(f"[WARN] Rollout failed health gate ({', '.join(why) or 'see rollout_health'}); "
              "do not trust final F1 alone.")
    if clot4 is not None:
        c4 = next(r for r in clot4 if r["time"] == t_last)
        print(f"[i] R4 teacher F1={c4['clot_f1']:.3f}")
    if species_mae:
        sp = next(r for r in species_mae if r["time"] == t_last)
        print(
            f"[i] nuc FI mae={sp['nuc']['fi_log_mae']:.4f} "
            f"Mat mae={sp['nuc']['mat_log_mae']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
