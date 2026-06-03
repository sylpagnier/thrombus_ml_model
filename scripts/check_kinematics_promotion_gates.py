"""Promotion gates for Stage-A checkpoints (synthetic + clinical holdout).

Example:
    python scripts/check_kinematics_promotion_gates.py --checkpoint outputs/kinematics/clinical_anchor_finetune/kinematics_best.pth
    python scripts/check_kinematics_promotion_gates.py --checkpoint outputs/kinematics/production_allfix/kinematics_best.pth --holdout patient007,patient003
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.architecture.kinematics_model_config import (
    build_gino_deq_from_ctor,
    kinematics_checkpoint_tensors,
    resolve_gino_deq_ctor_kwargs,
)
from src.config import PhysicsConfig
from src.core_physics.physics_kernels import PhysicsKernels
from src.training.train_kinematics_predictor import load_dataset
from src.utils.kinematics_geometry import graph_geometry_level, split_clinical_anchor_train_val
from src.utils.metrics import quantify_performance


def _eval_rel_l2(model, graphs, kernels, device) -> float:
    if not graphs:
        return float("nan")
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    scores = quantify_performance(model, loader, kernels, device, phase="kinematics")
    return float(scores.get("rel_l2", float("nan")))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--holdout", type=str, default="patient007", help="Comma-separated patient stems for val.")
    p.add_argument("--synthetic-cap", type=int, default=200)
    p.add_argument("--max-patient-rel-l2", type=float, default=0.25)
    p.add_argument("--max-synthetic-rel-l2", type=float, default=0.20)
    p.add_argument(
        "--max-synthetic-l2-rel-l2",
        type=float,
        default=0.22,
        help="Mean rel_L2 on synthetic val graphs with geometry_level=2.",
    )
    p.add_argument("--baseline-composite", type=float, default=float("nan"))
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"[ERR] missing checkpoint: {ckpt_path}")
        return 1

    os.environ["KINEMATICS_INCLUDE_PATIENT_ANCHORS"] = "1"
    os.environ["KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS"] = args.holdout.strip()
    os.environ["KINEMATICS_GRAPH_CAP"] = str(int(args.synthetic_cap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta, state = kinematics_checkpoint_tensors(raw)
    ctor = resolve_gino_deq_ctor_kwargs(meta, state)
    phys = PhysicsConfig(phase="kinematics")
    model = build_gino_deq_from_ctor(phys, ctor).to(device)
    model.load_state_dict(state)
    model.eval()
    kernels = PhysicsKernels(phys_cfg=phys)

    dataset = load_dataset("kinematics", "carreau", shuffle_graphs=False)
    holdout = {s.strip() for s in args.holdout.split(",") if s.strip()}
    splits = split_clinical_anchor_train_val(dataset, holdout_stems=sorted(holdout))
    val = splits["val"]
    val_patients = [d for d in val if getattr(d, "is_clinical_anchor", False)]
    val_synth = [d for d in val if not getattr(d, "is_clinical_anchor", False)]

    patient_rel = _eval_rel_l2(model, val_patients, kernels, device)
    synth_rel = _eval_rel_l2(model, val_synth, kernels, device)
    val_synth_l2 = [d for d in val_synth if graph_geometry_level(d, default=-1) == 2]
    synth_l2_rel = _eval_rel_l2(model, val_synth_l2, kernels, device)

    patient_ok = math.isfinite(patient_rel) and patient_rel <= float(args.max_patient_rel_l2)
    synth_ok = math.isfinite(synth_rel) and synth_rel <= float(args.max_synthetic_rel_l2)
    synth_l2_ok = (
        len(val_synth_l2) > 0
        and math.isfinite(synth_l2_rel)
        and synth_l2_rel <= float(args.max_synthetic_l2_rel_l2)
    )

    print(f"[gates] checkpoint={ckpt_path}")
    print(f"[gates] holdout patients ({len(val_patients)}): rel_L2={patient_rel:.4f}  gate<={args.max_patient_rel_l2}  -> {'PASS' if patient_ok else 'FAIL'}")
    print(f"[gates] synthetic val ({len(val_synth)}): rel_L2={synth_rel:.4f}  gate<={args.max_synthetic_rel_l2}  -> {'PASS' if synth_ok else 'FAIL'}")
    print(
        f"[gates] synthetic L2 val ({len(val_synth_l2)}): rel_L2={synth_l2_rel:.4f}  "
        f"gate<={args.max_synthetic_l2_rel_l2}  -> {'PASS' if synth_l2_ok else 'FAIL'}"
    )

    baseline = float(args.baseline_composite)
    if not math.isfinite(baseline) and isinstance(raw, dict):
        baseline = float(raw.get("composite", float("nan")))
    if math.isfinite(baseline):
        print(f"[gates] checkpoint composite (train val metric): {baseline:.4f}")

    if patient_ok and synth_ok and synth_l2_ok:
        print("[gates] PROMOTE OK")
        return 0
    print("[gates] PROMOTE BLOCKED")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
