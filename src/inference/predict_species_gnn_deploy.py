"""Deploy inference: pred kinematics + species GNN rollout + clot phi (new vessels).

Usage::

    python -m src.inference.predict_species_gnn_deploy --graph data/processed/graphs_biochem_anchors/patient004.pt
    python -m src.inference.predict_species_gnn_deploy --graph new_vessel.pt --flow kinematics --loao
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.species_gnn_clot_rollout import (
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
    rollout_species_gnn_species_series,
    species_gnn_rollout_ckpt,
)
from src.core_physics.species_snapshot_gnn import wall_band_mask
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_mu_si_at_time
try:  # optional species log-MAE diagnostic (helper removed in a prior reorg)
    from src.core_physics.t0_rung4_ladder import species_log_mae_in_mask
except ImportError:  # pragma: no cover
    species_log_mae_in_mask = None
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.inference.species_gnn_deploy_env import (
    load_deploy_manifest,
    species_ckpt_for_anchor,
    species_gnn_deploy_env,
)
from src.core_physics.species_pushforward_continuous import default_deploy_metric_times
from src.evaluation.clot_relaxed_metrics import legacy_clot_f1_metrics as _clot_metrics


def _resolve_ckpt(
    graph_stem: str,
    *,
    manifest: dict,
    loao: bool,
    explicit: str,
) -> Path:
    root = Path(__file__).resolve().parents[2]
    if explicit.strip():
        p = Path(explicit)
        return p if p.is_absolute() else root / p
    if loao:
        return species_ckpt_for_anchor(graph_stem, manifest, prefer_loao=True)
    return species_ckpt_for_anchor(graph_stem, manifest, prefer_loao=False)


@torch.no_grad()
def predict_species_gnn_deploy(
    graph_path: Path | str,
    *,
    device: torch.device | None = None,
    flow_source: str = "kinematics",
    manifest: dict[str, str] | None = None,
    loao: bool = False,
    species_ckpt: str = "",
    times: list[int] | None = None,
    eval_gt: bool = True,
) -> dict:
    """Run full deploy stack on one vessel graph."""
    dev = device or require_cuda_device()
    path = Path(graph_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    m = dict(manifest or load_deploy_manifest())
    stem = path.stem
    explicit = species_ckpt.strip()
    env_overrides: dict[str, str] = {"T0_R4_FLOW_SOURCE": flow_source}
    if explicit:
        ckpt = _resolve_ckpt(stem, manifest=m, loao=loao, explicit=explicit)
        env_overrides["SPECIES_GNN_CLOUT_CKPT"] = str(ckpt)
        env_overrides["T0_R4_SPECIES_GNN_CKPT"] = str(ckpt)
        anchor_for_env = stem
    else:
        ckpt = _resolve_ckpt(stem, manifest=m, loao=loao, explicit="")
        anchor_for_env = stem

    with species_gnn_deploy_env(
        m,
        overrides=env_overrides,
        anchor=anchor_for_env,
        prefer_loao=loao,
    ):
        ckpt = Path(species_gnn_rollout_ckpt())
        data = torch.load(path, map_location=dev, weights_only=False)
        phys = PhysicsConfig(phase="biochem")
        bio = BiochemConfig(phase="biochem")
        n_steps = int(data.y.shape[0])
        if times is None:
            times = default_deploy_metric_times(n_steps)
        times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})

        bundle = load_species_gnn_rollout_bundle(ckpt, device=dev)
        if bundle is None:
            raise FileNotFoundError(f"missing species GNN ckpt: {ckpt}")
        static = prepare_species_gnn_rollout_static(data, device=dev)
        t_roll = time.perf_counter()
        species_series = rollout_species_gnn_species_series(
            data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=dev,
        )
        phi_traj = rollout_species_gnn_phi_trajectory(
            data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=dev,
            flow_source=flow_source,
        )
        roll_s = time.perf_counter() - t_roll

        band = wall_band_mask(data, dev, wall_hops=1).reshape(-1).bool()
        mask = torch.ones(int(data.num_nodes), device=dev, dtype=torch.bool)
        t_last = times[-1]
        sp_mae = (
            species_log_mae_in_mask(species_series, data, t_last, band, dev)
            if species_log_mae_in_mask is not None
            else None
        )

        clot_rows: list[dict] = []
        mu_rows: list[dict] = []
        with t0_rung2_env():
            for t in times:
                phi_gt = gt_clot_phi_at_time(data, t, phys, dev) if eval_gt else None
                phi_pred = phi_traj[int(t)]
                cm = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask) if eval_gt else {}
                step = predict_mu_si_at_time(
                    data,
                    t,
                    phys,
                    bio,
                    dev,
                    gamma_mode=RUNG2_GAMMA_MODE,
                    flow_source=flow_source,
                    pred_species_series=species_series,
                )
                mu_rows.append(
                    {
                        "time": int(t),
                        "mu_pred_median": float(step.mu_pred_si.median().item()),
                        "mu_gt_median": float(step.mu_gt_si.median().item()) if eval_gt else None,
                    }
                )
                clot_rows.append({"time": int(t), **cm})

        from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health

        health = compute_rung4_rollout_health(
            phi_traj, data, phys, bio, dev, times=times,
        )

    return {
        "anchor": stem,
        "graph": str(path),
        "species_ckpt": str(ckpt),
        "flow_source": flow_source,
        "loao": bool(loao),
        "rollout_s": roll_s,
        "species_band_t_last": sp_mae,
        "clot": clot_rows,
        "mu": mu_rows,
        "clot_f1_t_last": float(clot_rows[-1].get("clot_f1", 0.0)) if clot_rows else 0.0,
        "rollout_health": {k: v for k, v in health.items() if k != "timeline"},
        "health_pass": bool(health.get("health_pass", False)),
    }


def _cli() -> int:
    ap = argparse.ArgumentParser(description="Species GNN deploy predict (new vessel)")
    ap.add_argument("--graph", required=True, help="Vessel .pt graph (biochem anchor format)")
    ap.add_argument("--flow", default="kinematics", choices=("gt", "kinematics"))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--species-ckpt", default="")
    ap.add_argument("--loao", action="store_true", help="Use LOAO fold ckpt for this vessel stem")
    ap.add_argument(
        "--times",
        default="",
        help="comma macro indices; empty = per-graph default (0, 27, legacy mid, last)",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--no-gt-eval", action="store_true")
    args = ap.parse_args()

    times_arg = [int(x.strip()) for x in args.times.split(",") if x.strip()] if args.times.strip() else None
    manifest = load_deploy_manifest(args.manifest.strip() or None)
    result = predict_species_gnn_deploy(
        args.graph,
        flow_source=args.flow,
        manifest=manifest,
        loao=bool(args.loao),
        species_ckpt=args.species_ckpt.strip(),
        times=times_arg,
        eval_gt=not bool(args.no_gt_eval),
    )
    out = Path(args.out) if args.out.strip() else None
    if out is None:
        stem = Path(args.graph).stem
        out = Path("outputs/biochem/species_gnn_deploy/predict") / f"{stem}_deploy.json"
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[OK] {result['anchor']} F1@t_last={result['clot_f1_t_last']:.3f} "
        f"health={result['health_pass']} ckpt={result['species_ckpt']}",
        flush=True,
    )
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
