"""Evaluate neighbor-band species + physics trigger clot metrics on anchor graphs.

Rollout: GT [u,v,p] each step; species from teacher; clot from explicit
``physics_mu_eff_si`` (Mat/FI sigmoids, mu_ratio_max=4 by default).

Usage:
  python scripts/eval_neighbor_band_trigger.py --checkpoint outputs/biochem/biochem_teacher_best_high_mu.pth --final-only --compare-gt-species
  python scripts/eval_neighbor_band_trigger.py --checkpoint ... --split val --final-only --out outputs/biochem/neighbor_band_species/trigger_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
from src.core_physics.neighbor_band_trigger import (
    apply_neighbor_band_clot_phi_env,
    apply_neighbor_band_species_train_env,
    apply_physics_trigger_baseline_env,
    eval_neighbor_band_rollout,
)
from src.core_physics.physics_kernels import PhysicsKernels
from src.inference.biochem_teacher_loader import build_biochem_teacher, resolve_rollout_mu_ratio_max
from src.training.train_biochem_corrector import (
    PatientDataset,
    _biochem_dataloader_kw,
    _biochem_anchor_basename,
    _validation_eval_times,
)


def _apply_eval_env() -> None:
    apply_neighbor_band_species_train_env()
    apply_physics_trigger_baseline_env()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/biochem/biochem_teacher_last.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=("all", "train", "val"), default="all")
    ap.add_argument("--compare-gt-species", action="store_true", help="Also eval physics trigger with GT FI/Mat.")
    ap.add_argument("--out", default="", help="Optional JSON output path.")
    ap.add_argument("--max-fi-final", type=float, default=None, help="Exit 1 if any anchor final FI logMAE exceeds.")
    ap.add_argument("--min-band-f1", type=float, default=None, help="Exit 1 if mean final band F1 below.")
    ap.add_argument(
        "--final-only",
        action="store_true",
        help="Score physics trigger at final rollout time only (faster).",
    )
    args = ap.parse_args()

    _apply_eval_env()
    os.environ.setdefault("BIOCHEM_DATALOADER_WORKERS", "0")

    device = torch.device(
        args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    )
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    kernels = BiochemPhysicsKernels(bio_cfg, PhysicsKernels(phys_cfg=phys_cfg))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = _REPO / args.checkpoint
    if not ckpt_path.is_file():
        print(f"[ERR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mu_ratio = resolve_rollout_mu_ratio_max(bio_cfg, cli_value=None)
    model = build_biochem_teacher(
        ckpt,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        mu_ratio_max=mu_ratio,
    )
    model.eval()

    anchor_dir = _REPO / VesselConfig(phase="biochem_anchors").graph_output_dir
    paths = sorted(str(p) for p in anchor_dir.glob("*.pt"))
    val_stem = "patient007"
    if args.split == "val":
        paths = [p for p in paths if Path(p).stem.lower() == val_stem]
    elif args.split == "train":
        paths = [p for p in paths if Path(p).stem.lower() != val_stem]

    ds = PatientDataset(root="", file_list=paths)
    loader = DataLoader(ds, batch_size=1, shuffle=False, **_biochem_dataloader_kw(device))

    rows: list[dict] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            eval_times = _validation_eval_times(data, bio_cfg, device)
            pred = model(data, eval_times)
            if isinstance(pred, tuple):
                pred = pred[0]
            bundle = eval_neighbor_band_rollout(
                data=data,
                pred_series=pred,
                bio_cfg=bio_cfg,
                phys_cfg=phys_cfg,
                kernels=kernels,
                device=device,
                eval_times=eval_times,
                trigger_mode="physics",
                final_only=bool(args.final_only),
            )
            row = {
                "anchor": _biochem_anchor_basename(data),
                **{k: v for k, v in bundle.items() if k != "per_step"},
            }
            if args.compare_gt_species:
                gt_bundle = eval_neighbor_band_rollout(
                    data=data,
                    pred_series=pred,
                    bio_cfg=bio_cfg,
                    phys_cfg=phys_cfg,
                    kernels=kernels,
                    device=device,
                    eval_times=eval_times,
                    trigger_mode="gt_species",
                    final_only=bool(args.final_only),
                )
                row["gt_species_final_band_f1"] = gt_bundle["final_band_f1"]
                row["gt_species_final_clot_shape"] = gt_bundle["final_clot_shape"]
            rows.append(row)
            print(
                f"[OK] {_biochem_anchor_basename(data)}: "
                f"FI={row['final_species_fi_log_mae']:.4f} "
                f"band_F1={row['final_band_f1']:.3f} "
                f"clot_shape={row['final_clot_shape']:.3f}"
            )

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "checkpoint": args.checkpoint,
        "n_anchors": len(rows),
        "mean_final_species_fi_log_mae": _mean("final_species_fi_log_mae"),
        "mean_final_band_f1": _mean("final_band_f1"),
        "mean_final_clot_shape": _mean("final_clot_shape"),
        "per_anchor": rows,
    }
    print(
        f"[summary] mean FI={summary['mean_final_species_fi_log_mae']:.4f} "
        f"band_F1={summary['mean_final_band_f1']:.3f} "
        f"clot_shape={summary['mean_final_clot_shape']:.3f}"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[save] {out_path}")

    fail = False
    if args.max_fi_final is not None:
        for r in rows:
            if float(r["final_species_fi_log_mae"]) > float(args.max_fi_final):
                print(f"[FAIL] {r['anchor']} FI {r['final_species_fi_log_mae']:.4f} > {args.max_fi_final}")
                fail = True
    if args.min_band_f1 is not None:
        if _mean("final_band_f1") < float(args.min_band_f1):
            print(f"[FAIL] mean band F1 {_mean('final_band_f1'):.3f} < {args.min_band_f1}")
            fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
