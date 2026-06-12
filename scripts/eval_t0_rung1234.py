"""Rung 1-4 T0 isolation eval (mu + clot nucleation).

Rung 4: GT flow + pred GNODE species + proxy gamma.

Usage::

    python scripts/eval_t0_rung1234.py --anchor patient007
    python scripts/eval_t0_rung1234.py --anchor patient007 --teacher-ckpt outputs/biochem/biochem_teacher_last.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_mu_physics import (  # noqa: E402
    eval_anchor_t0_mu,
    gt_clot_phi_at_time,
    metrics_for_step,
    predict_clot_phi_at_time,
    predict_mu_si_at_time,
    resolve_t0_species_log_nd,
    rollout_t0_clot_phi,
    t0_physics_env,
)
from src.core_physics.t0_rung_config import (  # noqa: E402
    DEFAULT_KINE_CKPT,
    DEFAULT_SPECIES_DUMP_DIR,
    RUNG2_GAMMA_MODE,
    RUNG2_GAMMA_SCALE,
    RUNG2_POISEUILLE_SCALE,
    resolve_default_teacher_ckpt,
    rollout_t0_pred_species_series,
    t0_rung2_env,
    t0_rung3_env,
    t0_rung4_env,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _species_log_mae(pred_series: torch.Tensor, data, times: list[int], device: torch.device) -> list[dict]:
    rows: list[dict] = []
    for t in times:
        pred = resolve_t0_species_log_nd(data, t, device, pred_species_series=pred_series)
        gt = data.y[int(t), :, 4:16].to(device=device, dtype=torch.float32)
        mae = float((pred - gt).abs().mean().item())
        rows.append({"time": int(t), "species_log_mae": mae})
    return rows


def _clot_timeline(
    data,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    times: list[int],
    *,
    gamma_mode: str,
    flow_source: str = "gt",
    pred_species_series: torch.Tensor | None = None,
    env_ctx,
    nucleation: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    mask = torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)
    with env_ctx:
        traj = rollout_t0_clot_phi(
            data,
            phys,
            bio,
            device,
            gamma_mode=gamma_mode,
            flow_source=flow_source,
            pred_species_series=pred_species_series,
            nucleation=nucleation,
            nucleation_hops=1,
        )
        for t in times:
            phi_gt = gt_clot_phi_at_time(data, t, phys, device)
            phi_pred = traj[t]["phi"]
            m = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)
            rows.append({"time": int(t), **m})
    return rows


def _mu_timeline_rung3(
    data,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    times: list[int],
    *,
    kine_ckpt: str,
) -> list[dict]:
    rows: list[dict] = []
    with t0_rung3_env(kine_ckpt=kine_ckpt):
        for t in times:
            step = predict_mu_si_at_time(
                data,
                t,
                phys,
                bio,
                device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="kinematics",
            )
            m = metrics_for_step(step, data, phys, device)
            rows.append(
                {
                    "time": int(t),
                    "mu_gt_median": float(step.mu_gt_si.median().item()),
                    "mu_pred_median": float(step.mu_pred_si.median().item()),
                    **{k: v for k, v in m.__dict__.items() if k != "time_index"},
                }
            )
    return rows


def _mu_timeline_rung4(
    data,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    times: list[int],
    *,
    pred_species_series: torch.Tensor,
    teacher_ckpt: str,
) -> list[dict]:
    rows: list[dict] = []
    with t0_rung4_env(teacher_ckpt=teacher_ckpt):
        for t in times:
            step = predict_mu_si_at_time(
                data,
                t,
                phys,
                bio,
                device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="gt",
                pred_species_series=pred_species_series,
            )
            m = metrics_for_step(step, data, phys, device)
            rows.append(
                {
                    "time": int(t),
                    "mu_gt_median": float(step.mu_gt_si.median().item()),
                    "mu_pred_median": float(step.mu_pred_si.median().item()),
                    **{k: v for k, v in m.__dict__.items() if k != "time_index"},
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Rung 1/2/3/4 T0 eval")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,27,53")
    ap.add_argument("--kine-ckpt", default=DEFAULT_KINE_CKPT)
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--species-dump", default="", help="Cached teacher-species anchor .pt (fast path).")
    ap.add_argument("--teacher-time-stride", type=int, default=6, help="Subsample macro steps for live rollout.")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_rung1234_eval.json")
    args = ap.parse_args()

    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    kine = root / args.kine_ckpt
    teacher = Path(args.teacher_ckpt) if args.teacher_ckpt.strip() else Path(resolve_default_teacher_ckpt())
    if not teacher.is_absolute():
        teacher = root / teacher
    if not graph.is_file():
        print(f"[ERR] missing {graph}", file=sys.stderr)
        return 1
    if not kine.is_file():
        print(f"[ERR] missing kinematics ckpt {kine}", file=sys.stderr)
        return 1
    if not teacher.is_file():
        print(f"[ERR] missing teacher ckpt {teacher}", file=sys.stderr)
        return 1

    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(graph, map_location=device, weights_only=False)

    species_dump = Path(args.species_dump) if args.species_dump.strip() else (
        root / DEFAULT_SPECIES_DUMP_DIR / f"{args.anchor}.pt"
    )
    if not species_dump.is_absolute():
        species_dump = root / species_dump

    print(f"[i] teacher ckpt: {teacher}", flush=True)
    if species_dump.is_file():
        print(f"[i] using cached species dump: {species_dump}", flush=True)
    else:
        print(
            f"[i] rolling pred species on GT flow (stride={int(args.teacher_time_stride)})...",
            flush=True,
        )
    t_roll = time.perf_counter()
    pred_species = rollout_t0_pred_species_series(
        data,
        str(teacher),
        device,
        bio_cfg=bio,
        dumped_graph=str(species_dump) if species_dump.is_file() else None,
        time_stride=max(int(args.teacher_time_stride), 1),
    )
    print(
        f"[i] teacher rollout done in {time.perf_counter() - t_roll:.1f}s "
        f"shape={tuple(pred_species.shape)}",
        flush=True,
    )
    species_mae = _species_log_mae(pred_species, data, times, device)

    with t0_physics_env(args.anchor, gamma_mode="comsol_sr") as env1:
        rung1_mu = eval_anchor_t0_mu(graph, times=times, gamma_mode="comsol_sr")
    with t0_rung2_env():
        rung2_mu = eval_anchor_t0_mu(graph, times=times, gamma_mode=RUNG2_GAMMA_MODE)
    rung3_mu_rows = _mu_timeline_rung3(
        data, phys, bio, device, times, kine_ckpt=str(kine)
    )
    rung4_mu_rows = _mu_timeline_rung4(
        data,
        phys,
        bio,
        device,
        times,
        pred_species_series=pred_species,
        teacher_ckpt=str(teacher),
    )

    clot1 = _clot_timeline(
        data,
        phys,
        bio,
        device,
        times,
        gamma_mode="comsol_sr",
        flow_source="gt",
        env_ctx=t0_physics_env(args.anchor, gamma_mode="comsol_sr"),
    )
    clot2 = _clot_timeline(
        data,
        phys,
        bio,
        device,
        times,
        gamma_mode=RUNG2_GAMMA_MODE,
        flow_source="gt",
        env_ctx=t0_rung2_env(),
    )
    clot3 = _clot_timeline(
        data,
        phys,
        bio,
        device,
        times,
        gamma_mode=RUNG2_GAMMA_MODE,
        flow_source="kinematics",
        env_ctx=t0_rung3_env(kine_ckpt=str(kine)),
    )
    clot4 = _clot_timeline(
        data,
        phys,
        bio,
        device,
        times,
        gamma_mode=RUNG2_GAMMA_MODE,
        flow_source="gt",
        pred_species_series=pred_species,
        env_ctx=t0_rung4_env(teacher_ckpt=str(teacher)),
    )

    t_last = times[-1]
    t0_2 = next(r for r in rung2_mu.times if r["time"] == 0)
    t53_2 = next(r for r in rung2_mu.times if r["time"] == t_last)
    t0_4 = next(r for r in rung4_mu_rows if r["time"] == 0)
    t53_4 = next(r for r in rung4_mu_rows if r["time"] == t_last)
    c53_2 = next(r for r in clot2 if r["time"] == t_last)
    c53_4 = next(r for r in clot4 if r["time"] == t_last)
    sp53 = next(r for r in species_mae if r["time"] == t_last)

    gates = {
        "rung2_bulk_t0": 0.95 <= float(t0_2["ratio_median_bulk"]) <= 1.05,
        "rung2_clot_f1_nuc_t_last": float(c53_2["clot_f1"]) >= 0.85,
        "rung4_species_log_mae_t_last": float(sp53["species_log_mae"]) < 2.0,
        "rung4_clot_f1_nuc_t_last": float(c53_4["clot_f1"]) >= 0.35,
        "rung4_clot_f1_within_0p25_of_rung2": abs(float(c53_2["clot_f1"]) - float(c53_4["clot_f1"])) <= 0.25,
    }

    payload = {
        "anchor": args.anchor,
        "times": times,
        "kine_ckpt": str(kine.relative_to(root)) if kine.is_relative_to(root) else str(kine),
        "teacher_ckpt": str(teacher.relative_to(root)) if teacher.is_relative_to(root) else str(teacher),
        "rung2": {
            "label": "GT flow + GT species + proxy gamma",
            "gamma_mode": RUNG2_GAMMA_MODE,
            "mu": rung2_mu.to_dict(),
            "clot_nucleation": clot2,
        },
        "rung4": {
            "label": "GT flow + pred teacher species + proxy gamma",
            "gamma_mode": RUNG2_GAMMA_MODE,
            "species_log_mae": species_mae,
            "mu": {"times": rung4_mu_rows},
            "clot_nucleation": clot4,
        },
        "rung1": {"mu": rung1_mu.to_dict(), "clot_nucleation": clot1, "physics_env": env1},
        "rung3": {
            "mu": {"times": rung3_mu_rows},
            "clot_nucleation": clot3,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] {args.anchor} -> {out}")
    print(f"[i] rung2 t={t_last} clot F1={c53_2['clot_f1']:.3f}")
    print(
        f"[i] rung4 t={t_last} species_log_mae={sp53['species_log_mae']:.4f} "
        f"growth_ratio={t53_4['ratio_median_growth']:.4f} clot F1={c53_4['clot_f1']:.3f}"
    )
    print(f"[i] gates={gates} all_pass={payload['all_gates_pass']}")
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
