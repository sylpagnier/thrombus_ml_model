"""Orig10 cohort: vel_decay ON (baseline probe) vs OFF (re-eval).

Re-pins SPECIES_CONTINUOUS_VEL_DECAY after load_continuous_bundle meta restore.
Clot metrics only (no timeline) for speed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    load_continuous_bundle,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.species_snapshot_gnn import snapshot_wall_hops  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.inference.corrector_coupling import resolve_kinematics_checkpoint  # noqa: E402
from src.training.train_offwall_growth import _band_static_to_device  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
)
from src.utils.paths import get_project_root  # noqa: E402

ORIG10 = [
    "patient001",
    "patient002",
    "patient003",
    "patient004",
    "patient005",
    "patient006",
    "patient007",
    "patient008",
    "patient010",
    "patient011",
]

KEYS = (
    "deploy_clot_f1",
    "deploy_clot_score",
    "deploy_clot_relaxed_prec",
    "deploy_clot_offwall_relaxed_f1",
    "deploy_clot_offwall_strict_f1",
    "deploy_clot_offwall_n_pred",
    "deploy_clot_offwall_n_pred_hop_ge2",
    "deploy_clot_offwall_n_gt_hop_ge2",
    "deploy_clot_offwall_strict_f1_hop_ge2",
)


def _summarize(m: dict) -> dict[str, float]:
    return {k: float(m.get(k, 0.0) or 0.0) for k in KEYS}


def _mean(per: dict[str, dict]) -> dict[str, float]:
    if not per:
        return {k: 0.0 for k in KEYS}
    n = float(len(per))
    return {k: sum(float(r.get(k, 0.0) or 0.0) for r in per.values()) / n for k in KEYS}


@torch.no_grad()
def eval_mode(
    *,
    wall_ckpt: Path,
    growth_ckpt: Path,
    anchors: list[str],
    graph_dir: Path,
    device,
    phys,
    bio,
    kine,
    vel_decay: str,
    wall_only: str = "1",
    route: str = "wall",
) -> dict:
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = vel_decay
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = wall_only
    clear_offwall_model_cache()
    os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
    os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(growth_ckpt).replace("\\", "/")
    os.environ["SPECIES_TWO_MODEL_ROUTE"] = route

    wall_b = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
    assert wall_b is not None
    # Meta restore may rewrite decay knobs — re-pin ablation contract.
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = vel_decay
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = wall_only
    flow = train_deploy_eval_flow_source()
    apply_deploy_env(
        overrides={
            "T0_R4_FLOW_SOURCE": os.environ.get("T0_R4_FLOW_SOURCE", flow),
            "SPECIES_CONTINUOUS_VEL_DECAY": vel_decay,
            "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": wall_only,
        }
    )
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = vel_decay
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = wall_only
    flow = train_deploy_eval_flow_source()
    print(
        f"[i] mode vel_decay={os.environ.get('SPECIES_CONTINUOUS_VEL_DECAY')} "
        f"wall_only={os.environ.get('SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY')} "
        f"flow={flow} route={route}",
        flush=True,
    )

    per: dict[str, dict] = {}
    for anc in anchors:
        reset_species_rollout_flow_cache()
        data = torch.load(graph_dir / f"{anc}.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            uv, z = predict_kinematics_and_latent(kine, data)
        data.u0_pred = uv[:, 0].detach().cpu()
        data.v0_pred = uv[:, 1].detach().cpu()
        band = build_band_base_features(
            data, kine, device, wall_hops=snapshot_wall_hops(), z_kin_override=z
        )
        static = _band_static_to_device(
            {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in band.items()},
            device,
        )
        os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = vel_decay
        os.environ["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = wall_only
        m = _summarize(
            eval_deploy_clot_f1(
                wall_b.model,
                data.to(device),
                static,
                phys,
                bio,
                device,
                time_index=deploy_eval_time_index(int(data.y.shape[0])),
                flow_source=flow,
            )
        )
        per[anc] = m
        tag = f"decay={vel_decay},wall_only={wall_only}"
        print(
            f"  [{tag}] {anc}: f1={m['deploy_clot_f1']:.3f} "
            f"score={m['deploy_clot_score']:.3f} "
            f"ge2={m['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
            f"{m['deploy_clot_offwall_n_gt_hop_ge2']:.0f} "
            f"off={m['deploy_clot_offwall_n_pred']:.0f}",
            flush=True,
        )
    return {
        "vel_decay": vel_decay,
        "wall_only": wall_only,
        "per_anchor": per,
        "mean": _mean(per),
        "sum_ge2_pred": float(sum(r["deploy_clot_offwall_n_pred_hop_ge2"] for r in per.values())),
        "sum_ge2_gt": float(sum(r["deploy_clot_offwall_n_gt_hop_ge2"] for r in per.values())),
        "sum_offwall_pred": float(sum(r["deploy_clot_offwall_n_pred"] for r in per.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-ckpt", default="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
    ap.add_argument(
        "--growth-ckpt",
        default="outputs/biochem/offwall_model/wc_v7_open001_6h/growth_D_Orig10_Band/best.pth",
    )
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument("--route", default="wall")
    ap.add_argument(
        "--modes",
        default="wall_only",
        help="Comma list: 0 (decay off), 1 (legacy full-band), wall_only (decay on, wall-only)",
    )
    ap.add_argument(
        "--baseline-probe",
        default="outputs/biochem/offwall_model/wc_v7_open001_6h/probe_D_Orig10_Band.json",
        help="Existing legacy full-band decay-ON probe to compare against",
    )
    ap.add_argument(
        "--out",
        default="outputs/biochem/offwall_model/wc_v7_open001_6h/compare_vel_decay_wall_only.json",
    )
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    wall = Path(args.wall_ckpt)
    if not wall.is_absolute():
        wall = root / wall
    growth = Path(args.growth_ckpt)
    if not growth.is_absolute():
        growth = root / growth
    apply_mat_growth_leg_env(args.mat_leg, force=True)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    graph_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )

    report: dict = {
        "growth_ckpt": str(growth),
        "wall_ckpt": str(wall),
        "route": args.route,
        "anchors": ORIG10,
        "modes": {},
    }

    base_path = Path(args.baseline_probe)
    if not base_path.is_absolute():
        base_path = root / base_path
    if base_path.is_file():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        simple = base.get("simple") or base
        pa = simple.get("per_anchor") or {}
        per = {a: _summarize(pa[a]) for a in ORIG10 if a in pa}
        report["modes"]["legacy_fullband"] = {
            "vel_decay": "1",
            "wall_only": "0",
            "source": str(base_path),
            "per_anchor": per,
            "mean": _mean(per),
            "sum_ge2_pred": float(sum(r["deploy_clot_offwall_n_pred_hop_ge2"] for r in per.values())),
            "sum_ge2_gt": float(sum(r["deploy_clot_offwall_n_gt_hop_ge2"] for r in per.values())),
            "sum_offwall_pred": float(sum(r["deploy_clot_offwall_n_pred"] for r in per.values())),
        }
        m = report["modes"]["legacy_fullband"]["mean"]
        print(
            f"[i] baseline(legacy full-band decay): mean_f1={m['deploy_clot_f1']:.3f} "
            f"mean_score={m['deploy_clot_score']:.3f} "
            f"sum_ge2={report['modes']['legacy_fullband']['sum_ge2_pred']:.0f}/"
            f"{report['modes']['legacy_fullband']['sum_ge2_gt']:.0f}",
            flush=True,
        )

    def _parse_mode(token: str) -> tuple[str, str, str]:
        t = token.strip().lower()
        if t in ("wall_only", "wall-only"):
            return "wall_only", "1", "1"
        if t in ("0", "off"):
            return "decay_off", "0", "1"
        if t in ("1", "fullband", "legacy"):
            return "legacy_rerun", "1", "0"
        raise ValueError(f"unknown mode {token!r}")

    for token in [x.strip() for x in args.modes.split(",") if x.strip()]:
        name, decay, wall_only = _parse_mode(token)
        print(f"\n=== mode={name} (vel_decay={decay}, wall_only={wall_only}) ===", flush=True)
        report["modes"][name] = eval_mode(
            wall_ckpt=wall,
            growth_ckpt=growth,
            anchors=ORIG10,
            graph_dir=graph_dir,
            device=device,
            phys=phys,
            bio=bio,
            kine=kine,
            vel_decay=decay,
            wall_only=wall_only,
            route=args.route,
        )

    def _print_delta(base_key: str, new_key: str, label: str) -> None:
        if base_key not in report["modes"] or new_key not in report["modes"]:
            return
        a = report["modes"][base_key]["mean"]
        b = report["modes"][new_key]["mean"]
        d = {k: float(b.get(k, 0.0) - a.get(k, 0.0)) for k in KEYS}
        report[f"delta_{new_key}_minus_{base_key}"] = d
        report[f"delta_sum_ge2_{new_key}_minus_{base_key}"] = float(
            report["modes"][new_key]["sum_ge2_pred"] - report["modes"][base_key]["sum_ge2_pred"]
        )
        print(f"\n=== DELTA ({label}) ===", flush=True)
        print(
            f"  mean clot_f1: {a['deploy_clot_f1']:.3f} -> {b['deploy_clot_f1']:.3f} "
            f"({d['deploy_clot_f1']:+.3f})",
            flush=True,
        )
        print(
            f"  mean clot_score: {a['deploy_clot_score']:.3f} -> {b['deploy_clot_score']:.3f} "
            f"({d['deploy_clot_score']:+.3f})",
            flush=True,
        )
        print(
            f"  sum hop_ge2: {report['modes'][base_key]['sum_ge2_pred']:.0f} -> "
            f"{report['modes'][new_key]['sum_ge2_pred']:.0f} "
            f"({report[f'delta_sum_ge2_{new_key}_minus_{base_key}']:+.0f})",
            flush=True,
        )
        print(
            f"  mean offwall_strict: {a['deploy_clot_offwall_strict_f1']:.3f} -> "
            f"{b['deploy_clot_offwall_strict_f1']:.3f}",
            flush=True,
        )

    _print_delta("legacy_fullband", "wall_only", "wall_only - legacy full-band")
    _print_delta("legacy_fullband", "decay_off", "decay_off - legacy full-band")

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
