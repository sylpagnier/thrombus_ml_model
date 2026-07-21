"""Targeted Mat-tuning sweep for species snapshot Phase 1.

Usage::

    python scripts/sweep_species_snapshot_mat_tune.py
    python scripts/sweep_species_snapshot_mat_tune.py --epochs 60 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.clot_phi_simple import sdf_nd_from_data  # noqa: E402
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    SpeciesSnapshotGNN,
    build_snapshot_features,
    fi_mat_active_labels,
    fi_mat_log_targets,
    induced_subgraph,
    load_snapshot_bundle,
    logits_to_probs,
    resolve_time_index,
    snapshot_feature_dim,
    trigger_metrics,
    wall_band_mask,
)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

SWEEP_DIR = "outputs/biochem/sweep_species_snapshot_mat"


def _leg_env(leg: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for k, v in leg.items():
        if k.startswith("SPECIES_") or k.startswith("CUDA_"):
            env[k] = v
    return env


@torch.no_grad()
def _eval_ckpt(
    ckpt: Path,
    *,
    anchor: str,
    time_s: float,
    mat_thresh: float | None = None,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_snapshot_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        return {}
    root = get_project_root()
    data = torch.load(
        root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    t_idx = resolve_time_index(data, time_s=time_s)
    kine = load_kinematics_predictor(
        resolve_kinematics_checkpoint(), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=bundle.wall_hops)
    node_idx, edge_sub, _ = induced_subgraph(band, data.edge_index)
    z_kin = predict_kinematics_latent(kine, data.to(device))
    sdf = sdf_nd_from_data(data, device, n)
    feats = build_snapshot_features(z_kin, sdf)[node_idx]
    tgt_act = fi_mat_active_labels(
        fi_mat_log_targets(data, t_idx, device)[node_idx],
        thresh_log_nd=bundle.active_log_nd,
    )
    logits = bundle.model(feats, edge_sub)
    pred = logits_to_probs(
        logits, loss_mode=bundle.loss_mode, mat_thresh=mat_thresh, fi_thresh=0.5
    )
    m = torch.ones(pred.shape[0], device=device, dtype=torch.bool)
    return trigger_metrics(pred, tgt_act, m)


def _train_leg(
    leg_id: str,
    leg_env: dict[str, str],
    *,
    anchor: str,
    time_s: float,
    epochs: int,
    out_dir: Path,
) -> Path:
    ckpt = out_dir / leg_id / "best.pth"
    if ckpt.is_file():
        print(f"[skip] {leg_id} ckpt exists", flush=True)
        return ckpt
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _leg_env(leg_env)
    cmd = [
        sys.executable,
        "-m",
        "src.training.train_species_snapshot_gnn",
        "--anchor",
        anchor,
        "--time-s",
        str(time_s),
        "--epochs",
        str(epochs),
        "--loss",
        "focal",
        "--early-stop",
        "15",
        "--out",
        str(ckpt),
    ]
    print(f"[NEW] train {leg_id}", flush=True)
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True)
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description="Mat-focused species snapshot sweep")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--time-s", type=float, default=5000.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--quick", action="store_true", help="3 legs only, 40 epochs")
    ap.add_argument("--out-dir", default=SWEEP_DIR)
    args = ap.parse_args()

    epochs = 40 if args.quick else int(args.epochs)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = get_project_root() / out_dir

    legs: list[tuple[str, dict[str, str], str]] = [
        (
            "baseline_a90",
            {"SPECIES_SNAPSHOT_FOCAL_ALPHA": "0.90"},
            "reference: uniform alpha=0.90",
        ),
        (
            "fi90_mat75",
            {
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_FI": "0.90",
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT": "0.75",
            },
            "lower Mat alpha -> penalize FP halo",
        ),
        (
            "fi90_mat70",
            {
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_FI": "0.90",
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT": "0.70",
            },
            "aggressive Mat FP penalty",
        ),
        (
            "fi90_mat80_g25",
            {
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_FI": "0.90",
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT": "0.80",
                "SPECIES_SNAPSHOT_FOCAL_GAMMA_MAT": "2.5",
            },
            "sharper Mat focal gamma",
        ),
        (
            "fi90_mat78_w15",
            {
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_FI": "0.90",
                "SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT": "0.78",
                "SPECIES_SNAPSHOT_CHANNEL_WEIGHT_MAT": "1.5",
            },
            "more Mat loss weight, moderate alpha",
        ),
    ]
    if args.quick:
        legs = legs[:3]

    rows: list[dict] = []

    # Baseline eval from existing best if present
    existing = get_project_root() / "outputs/biochem/species_snapshot_s1/best.pth"
    if existing.is_file():
        base_m = _eval_ckpt(existing, anchor=args.anchor, time_s=args.time_s)
        for mt in (None, 0.55, 0.60, 0.65):
            m = _eval_ckpt(existing, anchor=args.anchor, time_s=args.time_s, mat_thresh=mt)
            rows.append({
                "leg": "promoted_best",
                "note": f"threshold sweep mat_thresh={mt}",
                "mat_thresh": mt,
                **m,
            })

    for leg_id, leg_env, note in legs:
        ckpt = _train_leg(
            leg_id, leg_env, anchor=args.anchor, time_s=args.time_s, epochs=epochs, out_dir=out_dir
        )
        for mt in (None, 0.55, 0.60):
            m = _eval_ckpt(ckpt, anchor=args.anchor, time_s=args.time_s, mat_thresh=mt)
            rows.append({
                "leg": leg_id,
                "note": note,
                "mat_thresh": mt,
                "ckpt": str(ckpt),
                **m,
            })

    def _score(r: dict) -> float:
        # Prefer high Mat F1 without destroying FI
        return float(r.get("mat_f1", 0.0)) + 0.35 * float(r.get("fi_f1", 0.0))

    rows_sorted = sorted(rows, key=_score, reverse=True)
    summary = {
        "anchor": args.anchor,
        "time_s": args.time_s,
        "epochs": epochs,
        "results": rows_sorted,
        "winner": rows_sorted[0] if rows_sorted else None,
    }
    out_json = out_dir / "sweep_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_json}", flush=True)
    if rows_sorted:
        w = rows_sorted[0]
        print(
            f"[i] top: leg={w.get('leg')} mat_thresh={w.get('mat_thresh')} "
            f"mat_f1={w.get('mat_f1', 0):.3f} fi_f1={w.get('fi_f1', 0):.3f} "
            f"trigger_f1={w.get('trigger_f1', 0):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
