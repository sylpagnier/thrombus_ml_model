"""Eval one T0 Rung4 sweep leg vs R2 / s0 / teacher."""

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

from scripts.eval_t0_rung4_step import (  # noqa: E402
    _clot_from_phi_traj,
    _elig_from_pred_series,
    _mu_timeline_rung4_step,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import eval_anchor_t0_mu, rollout_t0_clot_phi  # noqa: E402
from src.core_physics.t0_r4_sweep import load_sweep_bundle, recipe_from_id, rollout_sweep_species_series  # noqa: E402
from src.core_physics.t0_r4_sweep import RECIPES  # noqa: E402
from src.core_physics.t0_rung4_ladder import (  # noqa: E402
    rollout_rung4_species_series,
    species_log_mae_in_mask,
)
from src.core_physics.t0_rung_config import (  # noqa: E402
    DEFAULT_SPECIES_DUMP_DIR,
    RUNG2_GAMMA_MODE,
    resolve_default_teacher_ckpt,
    rollout_t0_pred_species_series,
    t0_rung2_env,
)
from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _parse_times(raw: str, n_steps: int) -> list[int]:
    if not raw.strip():
        return list(range(n_steps))
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return sorted({max(0, min(n_steps - 1, t)) for t in out})


def _phi_traj_from_species(data, phys, bio, device, pred_series) -> dict[int, torch.Tensor]:
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, device,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
            pred_species_series=pred_series, nucleation=True, nucleation_hops=1,
        )
    return {int(t): v["phi"] for t, v in traj.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval T0 Rung4 sweep leg")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,7,15,22,27,40,53")
    ap.add_argument("--leg-dir", default="", help="Sweep leg dir with best.pth")
    ap.add_argument("--ckpt", default="", help="Override checkpoint path")
    ap.add_argument("--recipe", default="", help="Override recipe id (else from ckpt json)")
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    if args.ckpt.strip():
        ckpt = Path(args.ckpt)
    elif args.leg_dir.strip():
        ckpt = Path(args.leg_dir) / "best.pth"
    else:
        raise SystemExit("[ERR] pass --leg-dir or --ckpt")
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    bundle = load_sweep_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] missing checkpoint: {ckpt}")

    recipe_id = args.recipe.strip() or bundle.recipe.id
    recipe = recipe_from_id(recipe_id) if recipe_id in RECIPES else bundle.recipe

    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    times = _parse_times(args.times, int(data.y.shape[0]))

    teacher = Path(args.teacher_ckpt) if args.teacher_ckpt.strip() else Path(resolve_default_teacher_ckpt())
    if not teacher.is_absolute():
        teacher = root / teacher
    species_dump = root / DEFAULT_SPECIES_DUMP_DIR / f"{args.anchor}.pt"

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] recipe={recipe.id} family={recipe.family} hypothesis={recipe.hypothesis}", flush=True)

    t0 = time.perf_counter()
    pred = rollout_sweep_species_series(data, phys, bio, device, bundle)
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    phi_traj = _phi_traj_from_species(data, phys, bio, device, pred)
    health = compute_rung4_rollout_health(phi_traj, data, phys, bio, device)

    with t0_rung2_env():
        pred_r2 = torch.load(species_dump, map_location=device, weights_only=False)
        if not isinstance(pred_r2, torch.Tensor):
            pred_r2 = pred_r2.get("y", pred_r2)
        mu_r2 = eval_anchor_t0_mu(graph_path, times=times, gamma_mode=RUNG2_GAMMA_MODE, device=device)

    mu_step = _mu_timeline_rung4_step(
        data, phys, bio, device, times, pred_species_series=pred, step="s0",
    )
    clot_step = _clot_from_phi_traj(phi_traj, data, phys, device, times)
    species_rows = []
    for t in times:
        elig = _elig_from_pred_series(data, t, phys, bio, device, pred)
        mae = species_log_mae_in_mask(pred, data, t, elig, device)
        species_rows.append({"time": int(t), "nuc": mae})

    pred_s0 = rollout_rung4_species_series(data, phys, bio, device, step="s0")
    phi_s0 = _phi_traj_from_species(data, phys, bio, device, pred_s0)
    c_s0 = _clot_from_phi_traj(phi_s0, data, phys, device, [times[-1]])[0]

    with t0_rung2_env():
        pred_teacher = rollout_t0_pred_species_series(
            data, str(teacher), device, bio_cfg=bio,
            dumped_graph=str(species_dump) if species_dump.is_file() else None,
        )
    phi_teacher = _phi_traj_from_species(data, phys, bio, device, pred_teacher)
    c_teacher = _clot_from_phi_traj(phi_teacher, data, phys, device, [times[-1]])[0]
    phi_r2 = _phi_traj_from_species(data, phys, bio, device, pred_r2)
    c_r2 = _clot_from_phi_traj(phi_r2, data, phys, device, [times[-1]])[0]

    cs = clot_step[-1]
    out_path = Path(args.out) if args.out.strip() else (
        root / "outputs/biochem/sweep_t0_r4_arch_6h" / recipe.id / f"eval_{args.anchor}.json"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "anchor": args.anchor,
        "device": str(device),
        "recipe_id": recipe.id,
        "recipe_family": recipe.family,
        "hypothesis": recipe.hypothesis,
        "checkpoint": str(ckpt),
        "rollout_health": health,
        "rung2": {"mu": mu_r2.to_dict(), "clot_nucleation": _clot_from_phi_traj(phi_r2, data, phys, device, times)},
        "sweep_leg": {"clot_nucleation": clot_step, "species_mae": species_rows},
        "rung4_s0": {"clot_nucleation": [c_s0]},
        "rung4_teacher": {"clot_nucleation": [c_teacher]},
        "delta_vs_s0": {
            "f1": float(cs["clot_f1"]) - float(c_s0["clot_f1"]),
            "pred_pos_frac": float(cs["pred_pos_frac"]) - float(c_s0["pred_pos_frac"]),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {out_path}", flush=True)
    print(
        f"[i] R2 F1={c_r2['clot_f1']:.3f} s0 F1={c_s0['clot_f1']:.3f} "
        f"{recipe.id} F1={cs['clot_f1']:.3f} (delta={payload['delta_vs_s0']['f1']:+.3f})",
        flush=True,
    )
    print(
        f"[i] health_score={health['health_score']:.3f} early_phi_wall={health['early_phi_wall_max']:.3f} "
        f"wall_carpet={health['wall_carpet']} health_pass={health['health_pass']}",
        flush=True,
    )
    print(f"[i] R4 teacher F1={c_teacher['clot_f1']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
