"""S0 sweep: refined rule ranking (off-wall stag, raw dx, tie-break dx+hop)."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
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


def build_refined_rule_grid(*, fast: bool = False) -> list[ClotPriorRuleConfig]:
    """Curated grid over refined ranking knobs + prior / stag / raw-dx legs."""
    prior_ps = [0.80, 0.85, 0.90] if fast else [0.80, 0.85, 0.90, 0.95]
    stag_fracs = [None, 0.10, 0.15] if fast else [None, 0.10, 0.15, 0.20]
    dx_raw_fracs = [None, 0.10] if fast else [None, 0.10, 0.15]
    offwall_opts = [False, True]
    tie_opts = [False, True]

    rules: list[ClotPriorRuleConfig] = []
    seen: set[str] = set()
    for pp, fst, dxr, offwall, tie in itertools.product(
        prior_ps, stag_fracs, dx_raw_fracs, offwall_opts, tie_opts
    ):
        n_legs = sum(x is not None for x in (pp, fst, dxr))
        if n_legs == 0:
            continue
        if offwall and fst is None:
            continue
        if tie and pp is None and fst is None and dxr is None:
            continue
        cfg = ClotPriorRuleConfig(
            prior_p=pp,
            use_t0_strip=False,
            flux_stag_top_frac=fst,
            flux_dx_raw_top_frac=dxr,
            stag_off_wall_adjacent=offwall,
            rank_tie_break=tie,
            combine_legs="or",
        )
        key = cfg.describe()
        if key in seen:
            continue
        seen.add(key)
        rules.append(
            ClotPriorRuleConfig(
                name=key,
                prior_p=cfg.prior_p,
                use_t0_strip=False,
                flux_stag_top_frac=cfg.flux_stag_top_frac,
                flux_dx_raw_top_frac=cfg.flux_dx_raw_top_frac,
                stag_off_wall_adjacent=cfg.stag_off_wall_adjacent,
                rank_tie_break=cfg.rank_tie_break,
                combine_legs="or",
            )
        )
    return rules


def _rule_score(mean_f1: float, mean_prec: float, mean_rec: float, mean_pred: float, mean_gt: float) -> float:
    score = mean_f1
    if mean_gt >= 0.05 and mean_pred > 2.5 * mean_gt:
        score -= 0.15
    if mean_prec < 0.12 and mean_rec > 0.85:
        score -= 0.10
    return score


def eval_rule_on_anchor(path: Path, rule: ClotPriorRuleConfig, *, phys, bio, device) -> dict | None:
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
        "n_flag": meta.get("n_flag", 0),
        "band_f1": band["clot_f1"],
        "band_prec": band["clot_prec"],
        "band_rec": band["clot_rec"],
        "band_pred_pos_frac": band["pred_pos_frac"],
        "band_gt_pos_frac": band["gt_pos_frac"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep refined S0 prior rules")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--fast", action="store_true", help="Smaller grid (~36 rules)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--out-json",
        default="outputs/biochem/diagnostics/clot_prior_rule_sweep_refined.json",
    )
    args = ap.parse_args()
    _apply_deploy_env()

    root = get_project_root()
    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    paths = [Path(p) for p in _list_anchor_paths(anchor_dir.resolve()) if Path(p).is_file()]
    rules = build_refined_rule_grid(fast=args.fast)

    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    per_anchor_rows: list[dict] = []
    t0 = time.time()
    for i, rule in enumerate(rules, start=1):
        for path in paths:
            row = eval_rule_on_anchor(path, rule, phys=phys, bio=bio, device=device)
            if row:
                per_anchor_rows.append(row)
        if i == 1 or i % 10 == 0 or i == len(rules):
            elapsed = time.time() - t0
            print(f"[i] {i}/{len(rules)} rules ({elapsed:.0f}s) last={rule.name[:60]}")

    pool: dict[str, list[dict]] = {}
    for row in per_anchor_rows:
        pool.setdefault(row["rule"], []).append(row)

    summary: list[dict] = []
    for rule_name, rows in pool.items():
        if len(rows) < 2:
            continue
        mf1 = sum(r["band_f1"] for r in rows) / len(rows)
        mprec = sum(r["band_prec"] for r in rows) / len(rows)
        mrec = sum(r["band_rec"] for r in rows) / len(rows)
        mpp = sum(r["band_pred_pos_frac"] for r in rows) / len(rows)
        mgt = sum(r["band_gt_pos_frac"] for r in rows) / len(rows)
        summary.append(
            {
                "rule": rule_name,
                "n_anchors": len(rows),
                "mean_band_f1": mf1,
                "mean_band_prec": mprec,
                "mean_band_rec": mrec,
                "mean_pred_pos_frac": mpp,
                "mean_gt_pos_frac": mgt,
                "score": _rule_score(mf1, mprec, mrec, mpp, mgt),
            }
        )
    summary.sort(key=lambda x: (-x["score"], -x["mean_band_f1"]))

    print()
    print(f"Refined rule sweep: {len(rules)} configs x {len(paths)} anchors")
    print(f"{'rank':>4} {'score':>6} {'F1':>6} {'prec':>6} {'rec':>6} {'pred+':>6}  rule")
    print("-" * 100)
    for i, row in enumerate(summary[: args.top], start=1):
        print(
            f"{i:>4} {row['score']:>6.3f} {row['mean_band_f1']:>6.3f} {row['mean_band_prec']:>6.3f} "
            f"{row['mean_band_rec']:>6.3f} {row['mean_pred_pos_frac']:>6.3f}  {row['rule']}"
        )

    baseline = next((s for s in summary if s["rule"].startswith("prior_p0.80") and "|" not in s["rule"]), None)
    if summary:
        best = summary[0]
        print()
        print(f"[OK] best: {best['rule']}  mean F1={best['mean_band_f1']:.3f}")
        if baseline and baseline["rule"] != best["rule"]:
            print(
                f"[i]  baseline prior_p0.80 F1={baseline['mean_band_f1']:.3f} "
                f"delta={best['mean_band_f1'] - baseline['mean_band_f1']:+.3f}"
            )

    out_path = root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "per_anchor": per_anchor_rows, "n_rules": len(rules)}, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
