r"""Does the s10.4 regime router survive on DEPLOY flow? (docs/WALL_MODEL_PLAN.md s10.4)

s10.4 calibrated `band_speed_q25 >= 0.060 -> inverted regime` on **GT** t=0 flow
(scripts/eda_clot_physics.py reads y[0,:,0:2] directly). The deployed gate cannot use GT --
`apply_pocket_gate` resolves t=0 flow through `resolve_species_rollout_uv`, which in deploy
mode returns the **kinematics-predicted** field `data.u0_pred/v0_pred`.

If the RGP-DEQ predictor biases speed (docs/WALL_MODEL_PLAN.md s7.2 measured |u| 1.058 vs
0.991, ~7% over-prediction), the calibrated threshold does not transfer unchanged. This
probe measures the shift and re-scores routing accuracy under the flow the gate actually
sees, so the threshold can be recalibrated rather than assumed.

No rollout, no clot grading -- just the kinematics predictor + one statistic per vessel.

    python scripts/probe_regime_route.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _load_static  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.evaluation.pocket_gate import DEFAULT_REGIME_BAND_SPEED_THRESH, band_speed_q25  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
EDA_JSON = "outputs/biochem/eda/clot_physics/eda.json"


def acc_at(vals: np.ndarray, lab: np.ndarray, thr: float) -> float:
    return float(((vals >= thr).astype(int) == lab).mean())


def best_threshold(vals: np.ndarray, lab: np.ndarray) -> tuple[float, float]:
    cands = np.unique(vals)
    best = max(((t, acc_at(vals, lab, t)) for t in cands), key=lambda x: x[1])
    return float(best[0]), float(best[1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Regime router under deploy (predicted) flow")
    ap.add_argument("--eda", default=EDA_JSON)
    ap.add_argument("--min-clot", type=int, default=20)
    ap.add_argument("--out", default="outputs/biochem/eda/clot_physics/regime_route.json")
    args = ap.parse_args()

    root = get_project_root()
    eda_p = Path(args.eda)
    if not eda_p.is_absolute():
        eda_p = root / eda_p
    rows = json.load(open(eda_p, encoding="utf-8"))["anchors"]
    rich = [r for r in rows if r["n_clot_wall"] >= args.min_clot and r.get("where_auc") and r.get("agg")]
    print(f"[i] {len(rich)} clot-rich vessels from {eda_p.name}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] device={dev}")
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()), dev, phys_cfg=PhysicsConfig(phase="kinematics")
    )

    out = []
    print(f"\n  {'vessel':>12} {'GT q25':>9} {'PRED q25':>9} {'ratio':>7} {'regime':>9}")
    for r in rich:
        anc = r["anchor"]
        d = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=dev, weights_only=False)
        wall = d.mask_wall.bool()
        gt_q = band_speed_q25(d, dev, d.y[0, :, 0], d.y[0, :, 1], wall)
        # _load_static attaches u0_pred/v0_pred (the field the deployed gate resolves)
        _ = _load_static(d, dev, kine, 3, anc)
        if getattr(d, "u0_pred", None) is None:
            print(f"  {anc:>12}  (no u0_pred attached -- skipped)")
            continue
        pr_q = band_speed_q25(d, dev, d.u0_pred, d.v0_pred, wall)
        inverted = bool(r["where_auc"]["speed_h2"] < 0.5)
        out.append({"anchor": anc, "gt_q25": gt_q, "pred_q25": pr_q, "inverted": inverted})
        print(f"  {anc:>12} {gt_q:9.5f} {pr_q:9.5f} {pr_q/max(gt_q,1e-9):7.2f} "
              f"{'INVERTED' if inverted else 'normal':>9}")
        del d
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    gt = np.array([o["gt_q25"] for o in out])
    pr = np.array([o["pred_q25"] for o in out])
    lab = np.array([int(o["inverted"]) for o in out])

    print(f"\n{'='*74}\n=== Does the router survive on deploy flow? ===\n{'='*74}")
    print(f"  predicted/GT band_speed_q25 ratio: median={np.median(pr/np.maximum(gt,1e-9)):.3f}"
          f"  (1.0 = predictor preserves the statistic)")
    print(f"  Spearman(GT, PRED) rank agreement: "
          f"{np.corrcoef(np.argsort(np.argsort(gt)), np.argsort(np.argsort(pr)))[0,1]:+.3f}")

    thr0 = DEFAULT_REGIME_BAND_SPEED_THRESH
    print(f"\n  {'flow source':>14} {'thr':>8} {'accuracy':>9}   note")
    print(f"  {'GT':>14} {thr0:8.4f} {acc_at(gt, lab, thr0):9.3f}   s10.4 calibrated threshold")
    print(f"  {'PREDICTED':>14} {thr0:8.4f} {acc_at(pr, lab, thr0):9.3f}   <- what the deployed gate would do")
    t_gt, a_gt = best_threshold(gt, lab)
    t_pr, a_pr = best_threshold(pr, lab)
    print(f"  {'GT':>14} {t_gt:8.4f} {a_gt:9.3f}   refit on GT")
    print(f"  {'PREDICTED':>14} {t_pr:8.4f} {a_pr:9.3f}   refit on PREDICTED  <- use this threshold")

    # LOO on predicted flow -- the number that matters for deployment
    correct = 0
    for i in range(len(pr)):
        tr = np.array([j for j in range(len(pr)) if j != i])
        t_best, _ = best_threshold(pr[tr], lab[tr])
        correct += int((pr[i] >= t_best) == lab[i])
    print(f"\n  LEAVE-ONE-VESSEL-OUT on PREDICTED flow: {correct}/{len(pr)} = {correct/len(pr):.3f}")

    p = Path(args.out)
    if not p.is_absolute():
        p = root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "rows": out,
        "thresh_default": thr0,
        "acc_gt_default": acc_at(gt, lab, thr0),
        "acc_pred_default": acc_at(pr, lab, thr0),
        "best_thresh_gt": t_gt, "best_acc_gt": a_gt,
        "best_thresh_pred": t_pr, "best_acc_pred": a_pr,
        "loo_acc_pred": correct / max(len(pr), 1),
    }, indent=2), encoding="utf-8")
    print(f"\n[save] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
