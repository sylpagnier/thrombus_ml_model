"""V0: audit nucleation eligibility vs GT new commits and legacy ceiling."""

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

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_nucleation_mask import (  # noqa: E402
    gt_new_commit_mask,
    resolve_catalytic_hood,
    resolve_commits_at_time,
    resolve_nucleation_eligibility,
    snapshot_nucleation_config,
)
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache  # noqa: E402
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.training.clot_ml_step1_residual import (  # noqa: E402
    apply_step1_eval_env,
    load_step1_checkpoint,
    resolve_step1_rule_cfg,
    rollout_step1_phi,
)
from src.utils.paths import get_project_root  # noqa: E402


def _mask_frac(mask: torch.Tensor) -> float:
    return float(mask.reshape(-1).float().mean().item())


def _recall_inside(new_nodes: torch.Tensor, region: torch.Tensor) -> float:
    new_nodes = new_nodes.reshape(-1).bool()
    region = region.reshape(-1).bool()
    n_new = int(new_nodes.sum().item())
    if n_new <= 0:
        return float("nan")
    hit = int((new_nodes & region).sum().item())
    return hit / n_new


def audit_anchor(
    graph_path: Path,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    seed_modes: list[str],
    phi_pred_by_time: dict[int, torch.Tensor] | None,
) -> dict:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    ceiling_frac = _mask_frac(ceiling)

    out: dict = {
        "anchor": graph_path.stem,
        "n_comsol_steps": n_steps,
        "ceiling_frac": ceiling_frac,
        "modes": {},
    }

    for mode in seed_modes:
        per_t: list[dict] = []
        recalls: list[float] = []
        elig_fracs: list[float] = []
        catalytic_fracs: list[float] = []

        for t in range(n_steps):
            elig = resolve_nucleation_eligibility(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                growth_seed=mode,
                phi_pred_by_time=phi_pred_by_time if mode == "pred" else None,
            )
            new_gt = gt_new_commit_mask(data, t, phys_cfg, device)
            commits_prev = resolve_commits_at_time(
                data,
                max(t - 1, 0),
                device=device,
                phys_cfg=phys_cfg,
                growth_seed=mode,
                phi_pred_by_time=phi_pred_by_time if mode == "pred" else None,
            )
            hood = resolve_catalytic_hood(commits_prev, data.edge_index.to(device))
            rec = _recall_inside(new_gt, elig)
            per_t.append(
                {
                    "t": t,
                    "tau": float(macro_tau_at_index(data, t, bio_cfg=bio_cfg)),
                    "n_new_gt": int(new_gt.sum().item()),
                    "recall_in_elig": rec,
                    "elig_frac": _mask_frac(elig),
                    "catalytic_frac": _mask_frac(hood),
                    "ceiling_covers_elig": bool((~elig | ceiling).all().item()),
                }
            )
            if new_gt.any():
                recalls.append(rec)
            elig_fracs.append(_mask_frac(elig))

        t_final = n_steps - 1
        elig_final = resolve_nucleation_eligibility(
            data,
            t_final,
            device,
            phys_cfg,
            bio_cfg,
            growth_seed=mode,
            phi_pred_by_time=phi_pred_by_time if mode == "pred" else None,
        )
        out["modes"][mode] = {
            "mean_recall_new_gt": float(sum(recalls) / max(len(recalls), 1)) if recalls else float("nan"),
            "min_recall_new_gt": float(min(recalls)) if recalls else float("nan"),
            "elig_frac_tfinal": _mask_frac(elig_final),
            "elig_frac_mean": float(sum(elig_fracs) / max(len(elig_fracs), 1)),
            "elig_vs_ceiling_frac_ratio": _mask_frac(elig_final) / max(ceiling_frac, 1e-8),
            "per_t": per_t,
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="V0 nucleation mask audit")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--out", default="outputs/biochem/clot_ml_ladder_v2/v0_nucleation/audit.json")
    ap.add_argument("--compare-pred", action="store_true", help="also audit pred seed from V1 step1 phi")
    args = ap.parse_args()

    apply_step1_eval_env()
    os.environ.setdefault("CLOT_ML_USE_MACRO_TAU", "1")
    os.environ.setdefault("CLOT_PHI_GROWTH_SEED", "gt")

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    phi_by_anchor: dict[str, dict[int, torch.Tensor]] = {}
    if args.compare_pred:
        rule_cfg = resolve_step1_rule_cfg(root / args.step0_json)
        model, _ = load_step1_checkpoint(root / args.step1_ckpt, device=device)
        for p in discover_anchor_paths(root / args.anchor_dir):
            reset_temporal_kinematics_cache()
            data = torch.load(p, map_location=device, weights_only=False)
            phi_by_anchor[p.stem] = rollout_step1_phi(
                data,
                rule_cfg,
                model,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                alpha=0.35,
                sim_end_scale=1.0,
            )

    modes = ["gt"]
    if args.compare_pred:
        modes.append("pred")

    rows = []
    for p in discover_anchor_paths(root / args.anchor_dir):
        rows.append(
            audit_anchor(
                p,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                seed_modes=modes,
                phi_pred_by_time=phi_by_anchor.get(p.stem),
            )
        )

    gt_recalls = [r["modes"]["gt"]["mean_recall_new_gt"] for r in rows if r["modes"].get("gt")]
    pass_recall = sum(1 for x in gt_recalls if x == x and x >= 0.95)
    pass_elig = sum(
        1
        for r in rows
        if r["modes"]["gt"]["elig_frac_tfinal"] <= r["ceiling_frac"] + 1e-6
        or r["modes"]["gt"]["elig_vs_ceiling_frac_ratio"] <= 1.0
    )

    payload = {
        "config": snapshot_nucleation_config(),
        "pass_gate": {
            "recall_ge_095_on_4_anchors": pass_recall >= 4,
            "pass_recall_count": pass_recall,
            "elig_not_wider_than_ceiling_count": pass_elig,
        },
        "anchors": rows,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[i] anchors={len(rows)} nucleation_hops={payload['config']['nucleation_hops']}")
    for r in rows:
        gt = r["modes"]["gt"]
        print(
            f"  {r['anchor']:12s} recall={gt['mean_recall_new_gt']:.3f} "
            f"elig@tfinal={gt['elig_frac_tfinal']:.4f} ceiling={r['ceiling_frac']:.4f}",
            flush=True,
        )
        if "pred" in r["modes"]:
            pr = r["modes"]["pred"]
            print(
                f"    pred seed   recall={pr['mean_recall_new_gt']:.3f} "
                f"elig@tfinal={pr['elig_frac_tfinal']:.4f}",
                flush=True,
            )
    print(f"[i] pass recall>=0.95 on >=4 anchors: {payload['pass_gate']['recall_ge_095_on_4_anchors']}")
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
