"""Evaluate a clot_phi checkpoint on every biochem anchor (leave-one-out style)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import (
    ClotPhiSpeciesHead,
    build_clot_phi_model,
    clot_phi_feature_dim,
    clot_phi_joint_bio_enabled,
)
from src.training.train_clot_phi_simple import (
    _checkpoint_score,
    _list_anchor_paths,
    _run_epoch,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.utils.paths import get_project_root


def _load_models(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    apply_clot_phi_config_from_checkpoint(cfg)
    in_dim = int(cfg.get("in_dim", clot_phi_feature_dim()))
    hidden = int(cfg.get("hidden", 32))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    species_head = None
    if clot_phi_joint_bio_enabled() or "species_head_state_dict" in ckpt:
        species_head = ClotPhiSpeciesHead(
            in_dim=in_dim, hidden=int(os.environ.get("CLOT_PHI_SPECIES_HIDDEN", "32"))
        ).to(device)
        if "species_head_state_dict" in ckpt:
            species_head.load_state_dict(ckpt["species_head_state_dict"])
    return model, species_head, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/biochem/clot_phi_finalize/multi_anchor_eval.jsonl")
    ap.add_argument(
        "--anchor-dir",
        default="",
        help="Dumped anchor cache (overrides ckpt config anchor_dir after load).",
    )
    args = ap.parse_args()

    root = get_project_root()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir

    cli_anchor = (args.anchor_dir or "").strip()
    if cli_anchor:
        anchor_dir = Path(cli_anchor)
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
        os.environ["CLOT_PHI_ANCHOR_DIR"] = str(anchor_dir.resolve())

    model, species_head, cfg = _load_models(ckpt_path, device)
    apply_clot_phi_eval_defaults()

    raw_anchor = (os.environ.get("CLOT_PHI_ANCHOR_DIR") or "").strip()
    if raw_anchor:
        anchor_dir = Path(raw_anchor)
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
        paths = _list_anchor_paths(anchor_dir.resolve())
    else:
        paths = _list_anchor_paths(anchor_dir)

    rows: list[dict] = []
    for val_path in paths:
        stem = Path(val_path).stem
        train_paths = [p for p in paths if Path(p).stem != stem]
        va = _run_epoch(
            model,
            [val_path],
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
            train=False,
            time_stride=1,
            pos_weight=1.0,
            balanced=False,
            species_head=species_head,
        )
        tr = _run_epoch(
            model,
            train_paths,
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
            train=False,
            time_stride=2,
            pos_weight=1.0,
            balanced=False,
            species_head=species_head,
        )
        row = {
            "anchor": stem,
            "val": va,
            "train_loo": tr,
            "val_score": _checkpoint_score(va),
        }
        rows.append(row)
        print(
            f"{stem}: val f1={va['clot_f1']:.3f} rec={va['clot_rec']:.3f} "
            f"logMAE={va['mu_log_mae']:.3f} pred+={va['pred_pos_frac']:.3f} score={row['val_score']:.3f}"
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    f1s = [r["val"]["clot_f1"] for r in rows]
    maes = [r["val"]["mu_log_mae"] for r in rows]
    scores = [r["val_score"] for r in rows]
    mean_f1 = sum(f1s) / max(len(f1s), 1)
    min_f1 = min(f1s) if f1s else 0.0
    mean_mae = sum(maes) / max(len(maes), 1)
    min_mae = min(maes) if maes else 0.0
    mean_score = sum(scores) / max(len(scores), 1)
    print(
        f"[OK]  n={len(rows)} mean_f1={mean_f1:.3f} min_f1={min_f1:.3f} "
        f"mean_logMAE={mean_mae:.3f} min_logMAE={min_mae:.3f} mean_score={mean_score:.3f} -> {out_path}"
    )


if __name__ == "__main__":
    main()
