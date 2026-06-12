"""Comprehensive clot rule architecture sweep with clot_shape north-star metric.

Usage:
  python scripts/sweep_clot_rule_architectures.py
  python scripts/sweep_clot_rule_architectures.py --fast --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import (  # noqa: E402
    TemporalGrowthRuleConfig,
    aggregate_architecture_sweep,
    comprehensive_rule_architecture_grid,
    curated_p007_rule_grid,
    eval_temporal_rule_on_anchor,
    hybrid_teacher_species_rule_grid,
    ideas_rule_grid,
    incubation_rule_grid,
    offset_ramp_rule_grid,
    threshold_accum_rule_grid,
    shear_risk_rule_grid,
    pick_architecture_winner,
    pick_architecture_winner_balanced,
    pick_architecture_winner_deploy,
    pick_architecture_winner_incubation,
    pick_architecture_winner_shape,
)
from src.core_physics.clot_t0_pattern_probe import discover_anchor_paths  # noqa: E402
from src.utils.paths import get_project_root


def _sanitize_for_json(obj: object) -> object:
    """Recursively replace non-finite floats (incl. nested NaN) for strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        try:
            return _sanitize_for_json(obj.item())
        except (TypeError, ValueError):
            pass
    return obj


def _write_json(path: Path, payload: dict) -> None:
    clean = _sanitize_for_json(payload)
    path.write_text(json.dumps(clean, indent=2, allow_nan=False), encoding="utf-8")


def _apply_deploy_env() -> None:
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "from_t0")
    os.environ.setdefault("CLOT_FORECAST_PAIR_STRIDE", "1")
    os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
    os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
    os.environ.setdefault("CLOT_PHI_HYBRID", "0")
    os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
    os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_P", "0.80")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_T0_STRIP", "0")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_FLUX_STAG_TOP", "0.20")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_TIE_BREAK", "1")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_SKIP_INLET_Q", "0.25")
    os.environ.setdefault("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    # Bulk Carreau mu ~0.056 Pa*s off ceiling; evaluate shape within wall band only.
    os.environ.setdefault("CLOT_SHAPE_EVAL_MASK", "ceiling")
    # Timeline mean clot_shape over all forecast intervals (tfinal kept separately).
    os.environ.setdefault("CLOT_ARCH_SWEEP_TFINAL_SHAPE_ONLY", "0")


def _fast_rule_subset(rules: list[TemporalGrowthRuleConfig]) -> list[TemporalGrowthRuleConfig]:
    keep_prefixes = (
        "static_global",
        "ranked_onset_global",
        "prog_global",
        "hop_growth",
        "neighbor_ac",
        "loc_rank_both_t25_s15",
        "loc_rank_both_t25_s0",
        "loc_rank_lower_t25_s15",
        "loc_prog_both_t25_s15",
        "loc_prog_lower_t25_s15",
        "loc_rank_arc4",
    )
    out = [r for r in rules if any(r.name.startswith(p) for p in keep_prefixes)]
    return out or rules[:12]


def _format_table(agg: list[dict], *, incubation: bool = False) -> str:
    lines = [
        "",
        "Rule sweep (deploy_score = tfinal shape + early timing + anti-paint):",
        f"{'rule':<36} {'deploy':>6} {'p007tf':>7} {'early+':>6} {'p007tl':>7} {'pred+':>6}",
        "-" * 78,
    ]
    for row in agg[:25]:
        lines.append(
            f"{row['rule']:<36} {row.get('deploy_score', float('nan')):>6.3f} "
            f"{row.get('p007_tfinal_clot_shape', float('nan')):>7.3f} "
            f"{row.get('p007_early_pred_frac', float('nan')):>6.3f} "
            f"{row.get('p007_timeline_clot_shape', row.get('p007_clot_shape', float('nan'))):>7.3f} "
            f"{row['tfinal_mean_pred_frac']:>6.3f}"
        )
    if agg:
        w_deploy = pick_architecture_winner_deploy(agg)
        w_shape = pick_architecture_winner_shape(agg)
        w_bal = pick_architecture_winner_balanced(agg)
        lines.append("")
        if w_deploy:
            lines.append(
                f"[OK] deploy winner: {w_deploy['rule']} "
                f"deploy={w_deploy.get('deploy_score', float('nan')):.3f} "
                f"tfinal={w_deploy.get('p007_tfinal_clot_shape', float('nan')):.3f}"
            )
        if w_shape:
            lines.append(
                f"[OK] tfinal shape: {w_shape['rule']} "
                f"p007_tfinal={w_shape.get('p007_tfinal_clot_shape', float('nan')):.3f}"
            )
        w_inc = pick_architecture_winner_incubation(agg) if incubation else None
        if w_inc:
            lines.append(
                f"[OK] incubation: {w_inc['rule']} deploy={w_inc.get('deploy_score', float('nan')):.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Comprehensive clot rule architecture sweep")
    p.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    p.add_argument("--anchor", default="", help="Single anchor stem for fast mode")
    p.add_argument("--fast", action="store_true", help="Smaller rule subset")
    p.add_argument(
        "--curated",
        action="store_true",
        help="~35-rule p007-focused grid (30m sweep budget)",
    )
    p.add_argument(
        "--hybrid-species",
        action="store_true",
        help="Hybrid grid: kinematic localized rules + teacher Fi/Mat risk blend (use baked anchor dir)",
    )
    p.add_argument(
        "--incubation",
        action="store_true",
        help="Incubation quick grid: top localized templates + t_frac gate (no early clots)",
    )
    p.add_argument(
        "--ideas",
        action="store_true",
        help="Offset-ramp + threshold-accumulation probe grid",
    )
    p.add_argument("--offset-ramp", action="store_true", help="Offset ramp grid only")
    p.add_argument("--threshold-accum", action="store_true", help="Threshold accumulation grid only")
    p.add_argument(
        "--shear-risk",
        action="store_true",
        help="Shear-gradient risk blend grid on inc40 template (~15m)",
    )
    p.add_argument("--include-oracle", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip (anchor,rule) pairs already in out-json")
    p.add_argument(
        "--out-json",
        default="",
        help="Output JSON (default: architecture or hybrid-specific path)",
    )
    p.add_argument(
        "--out-txt",
        default="",
        help="Output table txt (default: matches out-json stem)",
    )
    args = p.parse_args()

    _apply_deploy_env()
    root = get_project_root()
    if not args.out_json.strip():
        if args.shear_risk:
            args.out_json = "outputs/biochem/diagnostics/clot_rule_shear_risk_sweep.json"
        elif args.ideas:
            args.out_json = "outputs/biochem/diagnostics/clot_rule_ideas_sweep.json"
        elif args.incubation:
            args.out_json = "outputs/biochem/diagnostics/clot_rule_incubation_sweep.json"
        elif args.curated:
            args.out_json = "outputs/biochem/diagnostics/clot_rule_curated_sweep.json"
        elif args.hybrid_species:
            args.out_json = "outputs/biochem/diagnostics/clot_hybrid_species_sweep.json"
        else:
            args.out_json = "outputs/biochem/diagnostics/clot_rule_architecture_sweep.json"
    if not args.out_txt.strip():
        args.out_txt = str(Path(args.out_json).with_suffix(".txt"))

    anchor_rel = args.anchor_dir
    if args.hybrid_species and anchor_rel == "data/processed/graphs_biochem_anchors":
        anchor_rel = "outputs/biochem/anchors_teacher_species"
    anchor_dir = root / anchor_rel
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    if args.shear_risk:
        rules = shear_risk_rule_grid()
    elif args.ideas:
        rules = ideas_rule_grid()
    elif args.offset_ramp:
        rules = offset_ramp_rule_grid()
    elif args.threshold_accum:
        rules = threshold_accum_rule_grid()
    elif args.incubation:
        rules = incubation_rule_grid()
    elif args.curated:
        rules = curated_p007_rule_grid(include_hybrid_species=bool(args.hybrid_species))
    elif args.hybrid_species:
        rules = hybrid_teacher_species_rule_grid()
    else:
        rules = comprehensive_rule_architecture_grid(include_oracle=args.include_oracle)
        if args.fast:
            rules = _fast_rule_subset(rules)

    paths = discover_anchor_paths(anchor_dir)
    if args.anchor.strip():
        paths = [p for p in paths if p.stem == args.anchor.strip()]
    if not paths:
        print(f"[ERR] no anchors under {anchor_dir}")
        sys.exit(1)

    print(f"[i] sweep: {len(rules)} rules x {len(paths)} anchors = {len(rules) * len(paths)} evals")
    print("[i] clot_shape eval mask: CLOT_SHAPE_EVAL_MASK=ceiling (bulk Carreau ~0.056 off-wall)")
    results: list[dict] = []
    out_json = root / args.out_json
    out_txt = root / args.out_txt
    out_json.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str]] = set()
    if args.resume and out_json.is_file():
        try:
            prev = json.loads(out_json.read_text(encoding="utf-8"))
            for row in prev.get("per_anchor") or []:
                if row.get("n_pairs", 0) >= 1:
                    done_keys.add((str(row.get("anchor", "")), str(row.get("rule", ""))))
            results = list(prev.get("per_anchor") or [])
            print(f"[i] resume: {len(done_keys)} evals already done", flush=True)
        except (json.JSONDecodeError, OSError, TypeError):
            done_keys = set()

    sweep_tag = "architecture"
    if args.shear_risk:
        sweep_tag = "shear_risk"
    elif args.ideas:
        sweep_tag = "ideas"
    elif args.offset_ramp:
        sweep_tag = "offset_ramp"
    elif args.threshold_accum:
        sweep_tag = "threshold_accum"
    elif args.incubation:
        sweep_tag = "incubation"
    elif args.hybrid_species:
        sweep_tag = "hybrid_species"

    def _write_partial() -> None:
        partial_agg = aggregate_architecture_sweep(results)
        w_deploy = pick_architecture_winner_deploy(partial_agg)
        w_inc = pick_architecture_winner_incubation(partial_agg) if args.incubation else None
        payload = {
            "sweep_kind": sweep_tag,
            "n_rules": len(rules),
            "n_anchors": len(paths),
            "n_done": len(results),
            "winner": w_deploy or pick_architecture_winner(partial_agg),
            "winner_deploy": w_deploy,
            "winner_shape": pick_architecture_winner_shape(partial_agg),
            "winner_balanced": pick_architecture_winner_balanced(partial_agg),
            "winner_incubation": w_inc,
            "aggregated": partial_agg,
            "per_anchor": results,
        }
        _write_json(out_json, payload)

    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        for rule in rules:
            if (path.stem, rule.name) in done_keys:
                prev_row = next(
                    (r for r in results if r.get("anchor") == path.stem and r.get("rule") == rule.name),
                    None,
                )
                if prev_row is not None and "early_mean_pred_frac" in prev_row:
                    continue
            row = eval_temporal_rule_on_anchor(
                data,
                rule,
                stem=path.stem,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                pair_stride=1,
            )
            results = [r for r in results if not (r.get("anchor") == path.stem and r.get("rule") == rule.name)]
            results.append(row)
            print(
                f"[i] {path.stem} {rule.name:<32} shape={row.get('tfinal_clot_shape', float('nan')):.3f} "
                f"bandF1={row.get('tfinal_band_f1', float('nan')):.3f} pred+={row.get('tfinal_band_pred_frac', float('nan')):.3f}",
                flush=True,
            )
        _write_partial()
        print(f"[save] partial {out_json} ({len(results)} evals)", flush=True)

    agg = aggregate_architecture_sweep(results)
    table = _format_table(agg, incubation=bool(args.incubation))
    print(table)

    w_deploy = pick_architecture_winner_deploy(agg)
    w_inc = pick_architecture_winner_incubation(agg) if args.incubation else None
    payload = {
        "sweep_kind": sweep_tag,
        "n_rules": len(rules),
        "n_anchors": len(paths),
        "n_done": len(results),
        "winner": w_deploy or pick_architecture_winner(agg),
        "winner_deploy": w_deploy,
        "winner_shape": pick_architecture_winner_shape(agg),
        "winner_balanced": pick_architecture_winner_balanced(agg),
        "winner_incubation": w_inc,
        "aggregated": agg,
        "per_anchor": results,
    }
    _write_json(out_json, payload)
    out_txt.write_text(table, encoding="utf-8")
    print(f"[save] {out_json}")
    print(f"[save] {out_txt}")


if __name__ == "__main__":
    main()
