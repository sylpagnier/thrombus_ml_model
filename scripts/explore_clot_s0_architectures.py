"""Fast curated S0 rule architecture compare (no full Cartesian sweep)."""

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
from src.core_physics.clot_phi_simple import ClotPriorRuleConfig, predict_prior_rule_deploy  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics, _list_anchor_paths  # noqa: E402
from src.utils.paths import get_project_root


def _apply_deploy_env() -> None:
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
    os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
    os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
    os.environ.setdefault("CLOT_PHI_HYBRID", "0")
    os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
    os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")


def s0_architecture_grid() -> list[ClotPriorRuleConfig]:
    """Hand-picked S0 recipes: prior_score top-k inside ceiling + optional legs/gates."""
    cfgs: list[ClotPriorRuleConfig] = []

    def add(name: str, **kwargs) -> None:
        cfgs.append(ClotPriorRuleConfig(name=name, **kwargs))

    # Baselines
    add("baseline_p80_ceiling", prior_p=0.80, use_t0_strip=False)
    add("old_winner_p95_or_st10", prior_p=0.95, flux_stag_top_frac=0.10, use_t0_strip=False)

    # prior_score only (t0 probe best feature, AUC ~0.83 @ t0)
    for p in (0.85, 0.90, 0.95):
        add(f"prior_p{p:.2f}_ceiling", prior_p=p, use_t0_strip=False)

    # Seed-local post-gates (t0 dgamma strip is growth seed, not ceiling)
    for p in (0.80, 0.85, 0.90, 0.95):
        add(f"p{p:.2f}_gate_t0strip", prior_p=p, use_t0_strip=False, post_gate="t0_strip")
        add(f"p{p:.2f}_gate_t0d1", prior_p=p, use_t0_strip=False, post_gate="t0_d1")
        add(f"p{p:.2f}_gate_t0d2", prior_p=p, use_t0_strip=False, post_gate="t0_d2")

    # AND legs (tighter than OR union)
    add("p85_and_st10_ceiling", prior_p=0.85, flux_stag_top_frac=0.10, use_t0_strip=False, combine_legs="and")
    add("p90_and_st10_ceiling", prior_p=0.90, flux_stag_top_frac=0.10, use_t0_strip=False, combine_legs="and")
    add("p90_and_st10_gate_t0d1", prior_p=0.90, flux_stag_top_frac=0.10, use_t0_strip=False, combine_legs="and", post_gate="t0_d1")

    # OR with t0 strip leg (seed union)
    add("p85_or_t0strip", prior_p=0.85, use_t0_strip=True)
    add("p90_or_t0strip", prior_p=0.90, use_t0_strip=True)

    # Wall proximity filters inside ceiling
    add("p85_hop_wall1", prior_p=0.85, use_t0_strip=False, max_hop_from_wall=1)
    add("p90_hop_wall1", prior_p=0.90, use_t0_strip=False, max_hop_from_wall=1)

    return cfgs


def eval_rule(path: Path, rule: ClotPriorRuleConfig, *, phys, bio, device) -> dict | None:
    data = torch.load(path, map_location=device, weights_only=False)
    t_out = int(data.y.shape[0]) - 1
    step, phi, _mu, meta = predict_prior_rule_deploy(
        data, t_out, phys_cfg=phys, bio_cfg=bio, device=device, t_in=0, rule=rule
    )
    loss_m = step.loss_mask.reshape(-1).bool()
    if not bool(loss_m.any()):
        return None
    band = _clot_metrics(phi, step.phi_gt, loss_m)
    if float(band["gt_pos_frac"]) < 0.02:
        return None
    return {
        "anchor": path.stem,
        "rule": rule.name,
        "describe": rule.describe(),
        "n_flag": int(meta["n_flag"]),
        "n_ceiling": int(meta["n_ceiling"]),
        "n_post_gate": int(meta.get("n_post_gate", meta["n_ceiling"])),
        "band_f1": float(band["clot_f1"]),
        "band_prec": float(band["clot_prec"]),
        "band_rec": float(band["clot_rec"]),
        "band_pred_frac": float(band["pred_pos_frac"]),
        "band_gt_frac": float(band["gt_pos_frac"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Explore curated S0 prior-rule architectures")
    p.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    p.add_argument("--out-json", default="outputs/biochem/diagnostics/clot_s0_architecture_explore.json")
    p.add_argument("--top", type=int, default=12, help="Print top N by mean band F1")
    args = p.parse_args()
    _apply_deploy_env()

    root = get_project_root()
    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir

    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    paths = [Path(x) for x in _list_anchor_paths(anchor_dir.resolve())]
    rules = s0_architecture_grid()

    per_rule: dict[str, list[dict]] = {r.name: [] for r in rules}
    for path in paths:
        if not path.is_file():
            continue
        for rule in rules:
            row = eval_rule(path, rule, phys=phys, bio=bio, device=device)
            if row:
                per_rule[rule.name].append(row)

    summaries = []
    for rule in rules:
        rows = per_rule[rule.name]
        if not rows:
            continue
        n = len(rows)
        mean_f1 = sum(r["band_f1"] for r in rows) / n
        mean_prec = sum(r["band_prec"] for r in rows) / n
        mean_rec = sum(r["band_rec"] for r in rows) / n
        mean_pred = sum(r["band_pred_frac"] for r in rows) / n
        mean_gt = sum(r["band_gt_frac"] for r in rows) / n
        mean_flag = sum(r["n_flag"] for r in rows) / n
        summaries.append(
            {
                "rule": rule.name,
                "describe": rule.describe(),
                "n_anchors": n,
                "mean_band_f1": mean_f1,
                "mean_band_prec": mean_prec,
                "mean_band_rec": mean_rec,
                "mean_pred_frac": mean_pred,
                "mean_gt_frac": mean_gt,
                "mean_n_flag": mean_flag,
                "per_anchor": rows,
            }
        )

    summaries.sort(key=lambda x: (-x["mean_band_f1"], -x["mean_band_prec"]))
    top_n = max(int(args.top), 1)
    print(f"[i] evaluated {len(rules)} configs on {len(paths)} anchors")
    print(f"{'rule':<28} mean_F1  prec   rec   pred+  gt+   flag")
    for s in summaries[:top_n]:
        print(
            f"{s['rule']:<28} {s['mean_band_f1']:.3f}   "
            f"{s['mean_band_prec']:.3f}  {s['mean_band_rec']:.3f}  "
            f"{s['mean_pred_frac']:.3f}  {s['mean_gt_frac']:.3f}  {s['mean_n_flag']:.0f}"
        )

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summaries": summaries, "winner": summaries[0] if summaries else None}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {out_path}")
    if summaries:
        w = summaries[0]
        print(f"[OK] best: {w['rule']} ({w['describe']}) mean F1={w['mean_band_f1']:.3f}")


if __name__ == "__main__":
    main()
