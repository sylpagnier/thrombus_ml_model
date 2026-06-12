"""Evaluate FI/Mat logMAE on all biochem anchor graphs (train+val split shown).

Uses the same clot-band supervision mask as the passive-align recipe (union TBPTT by default).

Usage:
  python scripts/eval_passive_species_anchors.py --checkpoint outputs/biochem/biochem_teacher_passive_align_locked.pth
  python scripts/eval_passive_species_anchors.py --checkpoint outputs/biochem/biochem_teacher_last.pth --device cpu
"""

from __future__ import annotations

import argparse
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
from src.core_physics.physics_kernels import PhysicsKernels
from src.training.train_biochem_corrector import (
    PatientDataset,
    _biochem_dataloader_kw,
    _compute_passive_species_on_loader,
)
from src.inference.biochem_teacher_loader import build_biochem_teacher, resolve_rollout_mu_ratio_max


def _apply_align_eval_env(*, predicted_kine: bool = False) -> None:
    if predicted_kine:
        os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
        os.environ.pop("BIOCHEM_GT_KINE_SKIP_DEQ", None)
    else:
        os.environ.setdefault("BIOCHEM_GT_KINE_VEL", "1")
        os.environ.setdefault("BIOCHEM_GT_KINE_SKIP_DEQ", "1")
    os.environ.setdefault("BIOCHEM_TEACHER_MU_RATIO_MAX", "1.0")
    os.environ.setdefault("BIOCHEM_DATA_BIO_MASK_MODE", "clot_band")
    os.environ.setdefault("BIOCHEM_ADR_MASK_MODE", "match_data_bio")
    os.environ.setdefault("BIOCHEM_ADR_EXCLUDE_WALL", "1")
    os.environ.setdefault("BIOCHEM_SUPERVISION_MASK_TIMES", "union")
    os.environ.setdefault("BIOCHEM_ADR_RESIDUAL_MODE", "transport_only")


def _list_anchor_files() -> list[Path]:
    anchor_dir = _REPO / VesselConfig(phase="biochem_anchors").graph_output_dir
    return sorted(anchor_dir.glob("*.pt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/biochem/biochem_teacher_last.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=("all", "train", "val"), default="all")
    ap.add_argument(
        "--max-fi-mean",
        type=float,
        default=None,
        help="Exit 1 if mean FI logMAE exceeds this (gate use).",
    )
    ap.add_argument(
        "--max-fi-per-anchor",
        type=float,
        default=None,
        help="Exit 1 if any anchor FI logMAE exceeds this.",
    )
    ap.add_argument(
        "--predicted-kine",
        action="store_true",
        help="Evaluate with Stage-A DEQ (BIOCHEM_GT_KINE_VEL=0), for rung 10 teachers.",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated anchor stems (e.g. patient007) for quick diagnostics.",
    )
    args = ap.parse_args()

    _apply_align_eval_env(predicted_kine=bool(args.predicted_kine))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = _REPO / args.checkpoint
    if not ckpt_path.is_file():
        print(f"[ERR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = torch.device(
        args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    )
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    kernels = BiochemPhysicsKernels(bio_cfg, PhysicsKernels(phys_cfg=phys_cfg))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mu_ratio_max = resolve_rollout_mu_ratio_max(bio_cfg, cli_value=None)
    teacher = build_biochem_teacher(
        ckpt,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        mu_ratio_max=mu_ratio_max,
    )

    files = _list_anchor_files()
    only = [s.strip().lower() for s in (args.only or "").split(",") if s.strip()]
    if only:
        files = [p for p in files if p.stem.lower() in set(only)]
    if not files:
        print("[ERR] no anchor graphs found", file=sys.stderr)
        return 2

    # Match train_biochem_corrector 90/10 anchor split when reporting train vs val.
    n = len(files)
    split_idx = max(1, min(int(0.9 * n), n - 1)) if n > 1 else 1
    train_files = [str(p) for p in files[:split_idx]]
    val_files = [str(p) for p in files[split_idx:]] or train_files

    if args.split == "train":
        use_files = train_files
        label = "train"
    elif args.split == "val":
        use_files = val_files
        label = "val"
    else:
        use_files = [str(p) for p in files]
        label = "all"

    dataset = PatientDataset(root="", file_list=use_files)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, **_biochem_dataloader_kw(device))
    bundle = _compute_passive_species_on_loader(
        teacher, loader, kernels, bio_cfg, device, non_blocking=False
    )

    print(f"[i]  checkpoint={ckpt_path.name} split={label} n_anchors={len(use_files)}")
    print(
        f"[OK] mean FI logMAE={bundle['species_fi_log_mae_mean']:.4f} "
        f"Mat logMAE={bundle['species_mat_log_mae_mean']:.4f} "
        f"mask_n~{bundle['species_mask_n_mean']:.0f}"
    )
    per_max = 0.0
    for row in bundle.get("per_anchor") or []:
        fi_v = float(row["species_fi_log_mae"])
        per_max = max(per_max, fi_v)
        print(
            f"     {row['anchor']:28s}  FI={fi_v:.4f}  "
            f"Mat={row['species_mat_log_mae']:.4f}  n={int(row['species_mask_n'])}"
        )

    mean_fi = float(bundle["species_fi_log_mae_mean"])
    if args.max_fi_mean is not None and mean_fi > args.max_fi_mean:
        print(
            f"[FAIL] mean FI {mean_fi:.4f} > max {args.max_fi_mean:.4f}",
            file=sys.stderr,
        )
        return 1
    if args.max_fi_per_anchor is not None and per_max > args.max_fi_per_anchor:
        print(
            f"[FAIL] max per-anchor FI {per_max:.4f} > max {args.max_fi_per_anchor:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
