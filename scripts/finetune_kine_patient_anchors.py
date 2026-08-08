"""Short Stage-A finetune on clinical patient kine anchors (+ synthetic L2 regularization).

Clinical COMSOL graphs live under ``graphs_kinematics_anchors/carreau/`` and are NOT
in the default ``load_dataset`` corpus. This script resumes a healthy Stage-A checkpoint,
merges patient anchors, and finetunes with heavy patient sampling.

Env (set by ``go_kinematics_clinical_anchor_finetune.ps1``):
  KINEMATICS_INCLUDE_PATIENT_ANCHORS=1
  KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS=patient007,...
  KINEMATICS_CLINICAL_ANCHOR_BOOST=10
  KINEMATICS_GRAPH_CAP=<synthetic cap>

Example:
    python scripts/finetune_kine_patient_anchors.py
    python scripts/finetune_kine_patient_anchors.py --epochs 25 --lr 5e-6 --resume outputs/kinematics/production_allfix/kinematics_best.pth
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Must be set before train_kinematics_predictor import side effects.
os.environ.setdefault("KINEMATICS_INCLUDE_PATIENT_ANCHORS", "1")
os.environ.setdefault("KINEMATICS_SKIP_LBFGS", "1")
os.environ.setdefault("KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS", "patient007")
os.environ.setdefault("KINEMATICS_DUAL_PROMOTION_GATES", "1")
os.environ.setdefault("KINEMATICS_SYNTHETIC_VAL_RATIO", "0.15")
os.environ.setdefault("KINEMATICS_SYNTHETIC_VAL_MIN", "20")
os.environ.setdefault("KINEMATICS_SYNTHETIC_VAL_MIN_L2", "6")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--synthetic-cap", type=int, default=120, help="Max Carreau synthetic graphs.")
    p.add_argument(
        "--resume",
        type=str,
        default="outputs/kinematics/production_allfix/kinematics_best.pth",
        help="Checkpoint path (production ep-80 best recommended).",
    )
    p.add_argument("--out-dir", type=str, default="outputs/kinematics/clinical_anchor_finetune")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume = Path(args.resume.strip())
    if not resume.is_file():
        print(f"[ERR] resume checkpoint missing: {resume}")
        return 1

    import torch

    ckpt_raw = torch.load(resume, map_location="cpu", weights_only=False)
    if isinstance(ckpt_raw, dict) and "model_state_dict" in ckpt_raw:
        resume_start = int(ckpt_raw.get("epoch", ckpt_raw.get("best_epoch", -1))) + 1
        comp = float(ckpt_raw.get("composite", float("nan")))
        if math.isfinite(comp):
            os.environ["KINEMATICS_BASELINE_COMPOSITE"] = str(comp)
    else:
        resume_start = 0

    total_epochs = resume_start + int(args.epochs)

    os.environ["KINEMATICS_OUTPUT_DIR"] = str(out_dir)
    os.environ["KINEMATICS_GRAPH_CAP"] = str(max(0, int(args.synthetic_cap)))
    if "KINEMATICS_CLINICAL_ANCHOR_BOOST" not in os.environ:
        os.environ["KINEMATICS_CLINICAL_ANCHOR_BOOST"] = "10.0"

    stage1 = 0
    stage2 = 0
    adam_epochs = total_epochs

    import subprocess

    argv = [
        sys.executable,
        "-m",
        "src.training.train_kinematics_predictor",
        "--epochs",
        str(total_epochs),
        "--adam-epochs",
        str(adam_epochs),
        "--stage1-end-epoch",
        str(stage1),
        "--stage2-end-epoch",
        str(stage2),
        "--finetune-lr",
        str(float(args.lr)),
        "--resume",
        str(resume.resolve()),
        "--no-prompt",
        "--geometry-phase",
        "l2_heavy",
        "--weight-data",
        "500.0",
        "--shuffle-graphs",
        "--graph-load-seed",
        "42",
        "--quiet",
    ]
    print(f"[i] patient kine finetune -> {out_dir}")
    print(
        f"[i] resume={resume} finetune_epochs={args.epochs} "
        f"(total_epochs={total_epochs}, resume_start={resume_start}) lr={args.lr} "
        f"synthetic_cap={args.synthetic_cap}"
    )
    # subprocess (not os.execv): Windows Store / embeddable Pythons often break overlay exec.
    return int(subprocess.call(argv))


if __name__ == "__main__":
    raise SystemExit(main())
