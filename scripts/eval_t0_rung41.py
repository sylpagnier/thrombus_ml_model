"""Rung 4.1: GT flow + rules-based species in nucleation band (CUDA).

Usage::

    python scripts/eval_t0_rung41.py --anchor patient007
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

from scripts.eval_t0_rung4 import _clot_timeline  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import eval_anchor_t0_mu, metrics_for_step, predict_mu_si_at_time  # noqa: E402
from src.core_physics.t0_rules_species import (  # noqa: E402
    build_rules_species_log_nd_at_time,
    rollout_t0_rules_species_series,
    rules_species_is_oracle,
    species_log_mae_in_mask,
)
from src.core_physics.t0_rung_config import (  # noqa: E402
    DEFAULT_SPECIES_DUMP_DIR,
    RUNG2_GAMMA_MODE,
    resolve_default_teacher_ckpt,
    rollout_t0_pred_species_series,
    t0_rung2_env,
    t0_rung41_env,
    t0_rung4_env,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _mu_timeline_rung41(
    data,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    times: list[int],
    *,
    pred_species_series: torch.Tensor,
    rules_mode: str,
) -> list[dict]:
    rows: list[dict] = []
    with t0_rung41_env(rules_mode=rules_mode):
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


def _species_mae_timeline(
    pred_series: torch.Tensor,
    data,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    times: list[int],
    *,
    rules_mode: str,
) -> list[dict]:
    from src.core_physics.t0_mu_physics import predict_clot_phi_at_time

    rows: list[dict] = []
    commits_prev = None
    with t0_rung41_env(rules_mode=rules_mode):
        for t in times:
            _, elig = build_rules_species_log_nd_at_time(
                data, t, device, phys, bio, commits_prev=commits_prev, mode=rules_mode
            )
            m_nuc = species_log_mae_in_mask(pred_series, data, t, elig, device)
            m_all = species_log_mae_in_mask(
                pred_series,
                data,
                t,
                torch.ones(int(data.num_nodes), device=device, dtype=torch.bool),
                device,
            )
            rows.append({"time": int(t), "nuc": m_nuc, "all": m_all})
            phi_raw, _ = predict_clot_phi_at_time(
                data,
                t,
                phys,
                bio,
                device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="gt",
                pred_species_series=pred_series,
            )
            commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Rung 4.1 rules-species eval (CUDA)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,27,53")
    ap.add_argument("--rules-mode", default="s0", help="Rung4 step (s0 deploy default; s0_oracle_* audit)")
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_rung41_eval.json")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    if not graph.is_file():
        print(f"[ERR] missing {graph}", file=sys.stderr)
        return 1

    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(graph, map_location=device, weights_only=False)

    teacher = Path(args.teacher_ckpt) if args.teacher_ckpt.strip() else Path(resolve_default_teacher_ckpt())
    if not teacher.is_absolute():
        teacher = root / teacher
    species_dump = root / DEFAULT_SPECIES_DUMP_DIR / f"{args.anchor}.pt"

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] rules mode: {args.rules_mode}", flush=True)

    t0 = time.perf_counter()
    with t0_rung41_env(rules_mode=args.rules_mode):
        pred_rules = rollout_t0_rules_species_series(
            data, phys, bio, device, mode=args.rules_mode
        )
    print(f"[i] rules species rollout {time.perf_counter() - t0:.1f}s", flush=True)

    pred_teacher = None
    if teacher.is_file():
        t1 = time.perf_counter()
        pred_teacher = rollout_t0_pred_species_series(
            data,
            str(teacher),
            device,
            bio_cfg=bio,
            dumped_graph=str(species_dump) if species_dump.is_file() else None,
            time_stride=6,
        )
        print(f"[i] teacher species cache {time.perf_counter() - t1:.1f}s", flush=True)

    with t0_rung2_env():
        rung2_mu = eval_anchor_t0_mu(graph, times=times, gamma_mode=RUNG2_GAMMA_MODE)
    clot2 = _clot_timeline(
        data, phys, bio, device, times, gamma_mode=RUNG2_GAMMA_MODE, env_ctx=t0_rung2_env()
    )

    clot41 = _clot_timeline(
        data,
        phys,
        bio,
        device,
        times,
        gamma_mode=RUNG2_GAMMA_MODE,
        pred_species_series=pred_rules,
        env_ctx=t0_rung41_env(rules_mode=args.rules_mode),
    )
    rung41_mu = _mu_timeline_rung41(
        data,
        phys,
        bio,
        device,
        times,
        pred_species_series=pred_rules,
        rules_mode=args.rules_mode,
    )

    clot4 = None
    rung4_mu = None
    if pred_teacher is not None:
        from scripts.eval_t0_rung4 import _mu_timeline_rung4
        clot4 = _clot_timeline(
            data,
            phys,
            bio,
            device,
            times,
            gamma_mode=RUNG2_GAMMA_MODE,
            pred_species_series=pred_teacher,
            env_ctx=t0_rung4_env(teacher_ckpt=str(teacher)),
        )
        rung4_mu = _mu_timeline_rung4(
            data, phys, bio, device, times, pred_species_series=pred_teacher, teacher_ckpt=str(teacher)
        )

    species41 = _species_mae_timeline(
        pred_rules, data, phys, bio, device, times, rules_mode=args.rules_mode
    )
    species_r4 = None
    if pred_teacher is not None:
        species_r4 = []
        for t in times:
            m_all = species_log_mae_in_mask(
                pred_teacher,
                data,
                t,
                torch.ones(int(data.num_nodes), device=device, dtype=torch.bool),
                device,
            )
            species_r4.append({"time": int(t), "all": m_all})

    t_last = times[-1]
    c2 = next(r for r in clot2 if r["time"] == t_last)
    c41 = next(r for r in clot41 if r["time"] == t_last)
    sp41 = next(r for r in species41 if r["time"] == t_last)

    gates = {
        "rung2_clot_f1_nuc_t_last": float(c2["clot_f1"]) >= 0.85,
        "rung41_nuc_fi_log_mae_t_last": float(sp41["nuc"]["fi_log_mae"]) < 0.05,
        "rung41_clot_f1_nuc_t_last": float(c41["clot_f1"]) >= 0.70,
        "rung41_beats_rung4_clot": True,
    }
    if clot4 is not None:
        c4 = next(r for r in clot4 if r["time"] == t_last)
        gates["rung41_beats_rung4_clot"] = float(c41["clot_f1"]) >= float(c4["clot_f1"])

    payload = {
        "anchor": args.anchor,
        "device": "cuda",
        "rules_mode": args.rules_mode,
        "rules_oracle_species": rules_species_is_oracle(args.rules_mode),
        "nucleation_mask": "deploy_pred_commits",
        "rung2": {"mu": rung2_mu.to_dict(), "clot_nucleation": clot2},
        "rung41": {
            "mu": {"times": rung41_mu},
            "clot_nucleation": clot41,
            "species_mae": species41,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    if clot4 is not None and rung4_mu is not None:
        payload["rung4"] = {
            "teacher_ckpt": str(teacher),
            "mu": {"times": rung4_mu},
            "clot_nucleation": clot4,
            "species_mae": species_r4,
        }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {out}")
    print(f"[i] R2 F1={c2['clot_f1']:.3f} R4.1 F1={c41['clot_f1']:.3f}")
    print(
        f"[i] R4.1 nuc FI mae={sp41['nuc']['fi_log_mae']:.4f} "
        f"Mat mae={sp41['nuc']['mat_log_mae']:.4f} n_nuc={sp41['nuc']['n_mask']}"
    )
    if clot4 is not None:
        print(f"[i] R4 F1={c4['clot_f1']:.3f}")
    print(f"[i] gates={gates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
