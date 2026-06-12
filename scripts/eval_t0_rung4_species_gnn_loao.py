"""LOAO-style deploy eval: species GNN (s34+s35) on all biochem anchors.

Compares Rung 4 mini-ladder step ``species_gnn`` vs deploy baseline ``s0`` under:
  - ``gt`` flow (Rung 4 isolation)
  - ``kinematics`` flow (pred GINO-DEQ + pred species -- full deploy stack)

Usage::

    python scripts/eval_t0_rung4_species_gnn_loao.py
    python scripts/eval_t0_rung4_species_gnn_loao.py --flow kinematics
    python scripts/promote_species_gnn_deploy.py   # write manifest first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_species_series,
    species_gnn_rollout_ckpt,
    load_species_gnn_rollout_bundle,
)
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.core_physics.species_snapshot_gnn import wall_band_mask  # noqa: E402
from src.core_physics.species_viscosity_calibration import (  # noqa: E402
    load_viscosity_calibration,
    predict_mu_at_time_with_beta,
    viscosity_calibration_dir,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_rung4_ladder import (  # noqa: E402
    eval_rung4_step_clot,
    species_log_mae_in_mask,
)
from src.inference.species_gnn_deploy_env import (  # noqa: E402
    load_deploy_manifest,
    resolve_loao_ckpt_for_anchor,
    species_gnn_deploy_env,
)
from src.utils.paths import get_project_root  # noqa: E402


def _eval_anchor(
    anchor: str,
    *,
    device: torch.device,
    flow_source: str,
    times: list[int],
    manifest: dict[str, str],
    compare_s0: bool,
    prefer_loao: bool,
) -> dict:
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
    data = torch.load(graph, map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    n_steps = int(data.y.shape[0])
    times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})
    t_last = times[-1]

    band = wall_band_mask(data, device, wall_hops=1).reshape(-1).bool()
    val_anchor = str(manifest.get("train_val_anchor", "patient007"))
    is_val = anchor == val_anchor

    loao_ckpt = resolve_loao_ckpt_for_anchor(anchor, manifest.get("loao_dir", ""))
    use_loao = prefer_loao and loao_ckpt.is_file()
    with species_gnn_deploy_env(
        manifest,
        overrides={"T0_R4_FLOW_SOURCE": flow_source},
        anchor=anchor if use_loao else None,
        prefer_loao=use_loao,
    ):
        gnn_report = eval_rung4_step_clot(
            data, phys, bio, device, step="species_gnn", times=times,
        )
        ckpt = species_gnn_rollout_ckpt()
        bundle = load_species_gnn_rollout_bundle(ckpt, device=device, quiet=True)
        static = prepare_species_gnn_rollout_static(data, device=device)
        species_series = rollout_species_gnn_species_series(
            data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device,
        )
        sp_mae = species_log_mae_in_mask(species_series, data, t_last, band, device)

        cal_path = Path(
            manifest.get("viscosity_beta")
            or os.environ.get("SPECIES_VISCOSITY_CALIB_PATH")
            or str(viscosity_calibration_dir() / "beta.pth")
        )
        if not cal_path.is_absolute():
            cal_path = root / cal_path
        mu_row: dict[str, float] = {}
        if cal_path.is_file():
            cal, _ = load_viscosity_calibration(cal_path, device=device)
            mu_pred, mu_gt = predict_mu_at_time_with_beta(
                data,
                species_series,
                cal.beta,
                t_last,
                phys_cfg=phys,
                bio_cfg=bio,
                device=device,
                anchor=anchor,
                soft_gelation=False,
            )
            from src.core_physics.t0_mu_physics import _mu_log_mae, _pearson

            mu_row = {
                "mu_log_mae_t_last": _mu_log_mae(mu_pred, mu_gt),
                "mu_pearson_t_last": _pearson(mu_pred, mu_gt),
                "beta": float(cal.beta.detach().cpu().item()),
            }

        s0_report = None
        if compare_s0:
            s0_report = eval_rung4_step_clot(
                data, phys, bio, device, step="s0", times=times,
            )

    gnn_clot = {int(r["time"]): r for r in gnn_report["clot"]}
    row = {
        "anchor": anchor,
        "flow_source": flow_source,
        "is_train_val_anchor": is_val,
        "loao_ckpt": use_loao,
        "ckpt": str(ckpt),
        "species_gnn": {
            "clot": gnn_report["clot"],
            "clot_f1_t_last": float(gnn_clot[t_last].get("clot_f1", 0.0)),
            "species_band": sp_mae,
            "mu": mu_row,
            "rollout_health": gnn_report.get("rollout_health", {}),
            "health_pass": bool(gnn_report.get("rollout_health", {}).get("health_pass", False)),
        },
    }
    if s0_report is not None:
        s0_clot = {int(r["time"]): r for r in s0_report["clot"]}
        row["s0_baseline"] = {
            "clot_f1_t_last": float(s0_clot[t_last].get("clot_f1", 0.0)),
            "health_pass": bool(s0_report.get("rollout_health", {}).get("health_pass", False)),
        }
        row["delta_f1_vs_s0"] = (
            row["species_gnn"]["clot_f1_t_last"] - row["s0_baseline"]["clot_f1_t_last"]
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="LOAO deploy eval for species GNN Rung 4")
    ap.add_argument("--anchors", default=",".join(BIOCHEM_ANCHORS_6))
    ap.add_argument("--times", default="0,27,53")
    ap.add_argument(
        "--flow",
        default="both",
        choices=("gt", "kinematics", "both"),
        help="gt=Rung4 isolation; kinematics=full deploy (pred kine flow)",
    )
    ap.add_argument("--manifest", default="")
    ap.add_argument("--no-s0", action="store_true", help="Skip s0 baseline per anchor")
    ap.add_argument("--global-ckpt", action="store_true", help="Force single global s34 ckpt (no LOAO fold)")
    ap.add_argument("--out", default="outputs/biochem/species_gnn_deploy/loao_eval.json")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    manifest = load_deploy_manifest(args.manifest.strip() or None)
    prefer_loao = not bool(args.global_ckpt)
    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    flows = ["gt", "kinematics"] if args.flow == "both" else [args.flow]

    print(f"[i] manifest val_anchor={manifest.get('train_val_anchor')} ckpt={manifest.get('species_gnn_ckpt')}", flush=True)
    t0 = time.perf_counter()
    rows: list[dict] = []
    for flow in flows:
        print(f"[i] flow_source={flow}", flush=True)
        for anc in anchors:
            print(f"[i] eval {anc} ({flow}) ...", flush=True)
            rows.append(
                _eval_anchor(
                    anc,
                    device=device,
                    flow_source=flow,
                    times=times,
                    manifest=manifest,
                    compare_s0=not bool(args.no_s0),
                    prefer_loao=prefer_loao,
                )
            )
            r = rows[-1]
            print(
                f"  gnn F1@t53={r['species_gnn']['clot_f1_t_last']:.3f} "
                f"mat_mae={r['species_gnn']['species_band'].get('mat_log_mae', float('nan')):.5f} "
                f"health={r['species_gnn']['health_pass']}",
                flush=True,
            )

    # Summary aggregates (exclude train-val anchor for LOAO holdout view)
    holdout = [r for r in rows if not r.get("is_train_val_anchor")]
    for flow in flows:
        subset = [r for r in holdout if r["flow_source"] == flow]
        if not subset:
            continue
        mean_f1 = sum(r["species_gnn"]["clot_f1_t_last"] for r in subset) / len(subset)
        mean_delta = sum(r.get("delta_f1_vs_s0", 0.0) for r in subset) / len(subset)
        print(
            f"[i] LOAO holdout ({flow}) mean clot_f1@t53={mean_f1:.3f} delta_vs_s0={mean_delta:+.3f}",
            flush=True,
        )

    payload = {
        "manifest": manifest,
        "times": times,
        "anchors": anchors,
        "flow_modes": flows,
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out} ({payload['elapsed_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
