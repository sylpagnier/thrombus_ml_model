"""Hyperparameter sweep for species pushforward phase 2.

Usage::

    python scripts/sweep_species_pushforward_tune.py --quick
    python scripts/sweep_species_pushforward_tune.py --epochs 60
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
from src.core_physics.species_pushforward_gnn import (  # noqa: E402
    active_series_on_band,
    eval_pushforward_window,
    load_pushforward_bundle,
)
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    build_snapshot_features,
    induced_subgraph,
    wall_band_mask,
)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

SWEEP_DIR = "outputs/biochem/sweep_species_pushforward"


def _leg_env(leg: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for k, v in leg.items():
        if k.startswith("SPECIES_") or k.startswith("CUDA_"):
            env[k] = v
    return env


@torch.no_grad()
def _eval_ckpt(ckpt: Path, *, anchor: str, t0: int, unroll: int) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_pushforward_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        return {}
    root = get_project_root()
    data = torch.load(
        root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    window = [t0 + i for i in range(unroll + 1) if t0 + i < int(data.y.shape[0])]
    kine = load_kinematics_predictor(
        resolve_kinematics_checkpoint(), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=2)
    node_idx, edge_sub, _ = induced_subgraph(band, data.edge_index)
    z_kin = predict_kinematics_latent(kine, data)
    sdf = sdf_nd_from_data(data, device, n)
    base_feats = build_snapshot_features(z_kin, sdf)[node_idx]
    series = active_series_on_band(data, window, device, node_idx)
    mask = torch.ones(len(node_idx), device=device, dtype=torch.bool)
    m = eval_pushforward_window(
        bundle.model,
        base_feats=base_feats,
        edge_index=edge_sub,
        active_series=series,
        mask=mask,
        state0=series[0],
    )
    return {
        "mean_growth_f1": float(m["mean_growth_f1"]),
        "mean_growth_mat_f1": float(m["mean_growth_mat_f1"]),
        "final_state_f1": float(m["final_state_f1"]),
        "final_state_mat_f1": float(m["final_state_mat_f1"]),
        "score": 0.75 * float(m["mean_growth_f1"]) + 0.25 * float(m["final_state_f1"]),
    }


def _train_leg(
    leg_id: str,
    leg_env: dict[str, str],
    *,
    anchor: str,
    epochs: int,
    early_stop: int,
    unroll: int,
) -> Path:
    root = get_project_root()
    out = root / SWEEP_DIR / leg_id / "best.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
    cmd = [
        sys.executable,
        "-m",
        "src.training.train_species_snapshot_pushforward",
        "--anchor",
        anchor,
        "--epochs",
        str(epochs),
        "--unroll",
        str(unroll),
        "--early-stop",
        str(early_stop),
        "--out",
        str(out.relative_to(root)).replace("\\", "/"),
    ]
    print(f"[NEW] leg={leg_id} env={ {k: leg_env[k] for k in leg_env if k.startswith('SPECIES_')} }", flush=True)
    subprocess.run(cmd, cwd=root, env=_leg_env(leg_env), check=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep species pushforward hyperparameters")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--early-stop", type=int, default=15)
    ap.add_argument("--unroll", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="Fewer epochs + 4 legs only")
    args = ap.parse_args()

    epochs = 30 if args.quick else int(args.epochs)
    early_stop = 12 if args.quick else int(args.early_stop)

    legs: list[tuple[str, dict[str, str]]] = [
        (
            "base",
            {
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "0.95",
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.92",
                "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.05",
            },
        ),
        (
            "hi_alpha",
            {
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "0.98",
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.97",
                "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.05",
            },
        ),
        (
            "growth_focus",
            {
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "0.98",
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.97",
                "SPECIES_PUSHFORWARD_TRAIN_T0_MAX": "22",
                "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.03",
            },
        ),
        (
            "mat_ch2_thresh70",
            {
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "0.98",
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.98",
                "SPECIES_PUSHFORWARD_CHANNEL_WEIGHT_MAT": "2.0",
                "SPECIES_PUSHFORWARD_GROWTH_THRESH_MAT": "0.70",
                "SPECIES_PUSHFORWARD_TRAIN_T0_MAX": "22",
                "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.02",
            },
        ),
    ]
    if not args.quick:
        legs.extend(
            [
                (
                    "low_noise",
                    {
                        "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "0.95",
                        "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.95",
                        "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.02",
                    },
                ),
                (
                    "gamma3",
                    {
                        "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "0.98",
                        "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.97",
                        "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT": "3.0",
                        "SPECIES_PUSHFORWARD_TRAIN_T0_MAX": "22",
                    },
                ),
                (
                    "combo_best",
                    {
                        "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "0.98",
                        "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.98",
                        "SPECIES_PUSHFORWARD_CHANNEL_WEIGHT_MAT": "2.5",
                        "SPECIES_PUSHFORWARD_GROWTH_THRESH_MAT": "0.68",
                        "SPECIES_PUSHFORWARD_TRAIN_T0_MAX": "22",
                        "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.02",
                        "SPECIES_PUSHFORWARD_STEP_LOSS": "linear",
                    },
                ),
            ]
        )

    results: list[dict] = []
    for leg_id, leg_env in legs:
        ckpt = _train_leg(
            leg_id,
            leg_env,
            anchor=args.anchor,
            epochs=epochs,
            early_stop=early_stop,
            unroll=args.unroll,
        )
        ev_active = _eval_ckpt(ckpt, anchor=args.anchor, t0=10, unroll=args.unroll)
        ev_plateau = _eval_ckpt(ckpt, anchor=args.anchor, t0=28, unroll=args.unroll)
        row = {
            "leg": leg_id,
            "ckpt": str(ckpt),
            "env": leg_env,
            "active_t10": ev_active,
            "plateau_t28": ev_plateau,
            "score": 0.7 * ev_active.get("score", 0.0) + 0.3 * ev_plateau.get("score", 0.0),
        }
        results.append(row)
        print(
            f"[leg {leg_id}] active_growth={ev_active.get('mean_growth_f1', 0):.3f} "
            f"active_state={ev_active.get('final_state_mat_f1', 0):.3f} "
            f"plateau_state={ev_plateau.get('final_state_mat_f1', 0):.3f} score={row['score']:.3f}",
            flush=True,
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    winner = results[0]
    out_path = get_project_root() / SWEEP_DIR / "sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"winner": winner, "results": results}, indent=2), encoding="utf-8")

    # Promote winner to canonical s2 ckpt
    promote = get_project_root() / "outputs/biochem/species_snapshot_s2/best.pth"
    import shutil

    shutil.copy2(winner["ckpt"], promote)
    meta_src = Path(winner["ckpt"]).with_suffix(".json")
    if meta_src.is_file():
        shutil.copy2(meta_src, promote.with_suffix(".json"))

    print(f"[OK] winner={winner['leg']} score={winner['score']:.3f}", flush=True)
    print(f"[OK] promoted -> {promote}", flush=True)
    print(f"[OK] sweep_results={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
