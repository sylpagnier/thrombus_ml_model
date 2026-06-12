"""Temporal clot dynamics probe + growing rule sweep (all anchors).

Usage:
  python scripts/sweep_clot_temporal_growth_rules.py
  python scripts/sweep_clot_temporal_growth_rules.py --probe-only
  python scripts/sweep_clot_temporal_growth_rules.py --rules-only --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.core_physics.clot_temporal_dynamics_probe import (  # noqa: E402
    aggregate_temporal_features,
    format_temporal_probe_report,
    probe_all_temporal,
    probe_anchor_temporal_dynamics,
    write_temporal_probe_json,
)
from src.core_physics.clot_localized_spatial import probe_species_oracle_auc  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import (  # noqa: E402
    TemporalGrowthRuleConfig,
    aggregate_temporal_rule_sweep,
    default_localized_rule_grid,
    default_temporal_rule_grid,
    eval_temporal_rule_on_anchor,
    pick_growing_winner,
    pick_localized_deploy_winner,
)
from src.core_physics.clot_t0_pattern_probe import discover_anchor_paths  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.utils.paths import get_project_root


def _apply_sweep_env() -> None:
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "from_t0")
    os.environ.setdefault("CLOT_FORECAST_PAIR_STRIDE", "1")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_P", "0.80")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_T0_STRIP", "0")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_FLUX_STAG_TOP", "0.20")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_TIE_BREAK", "1")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_SKIP_INLET_Q", "0.25")


def _format_rule_table(agg: list[dict]) -> str:
    lines = [
        "",
        "Temporal growing rule sweep (mean band F1 over from_t0 pairs):",
        f"{'rule':<28} {'bal':>6} {'meanF1':>7} {'early':>6} {'late':>6} "
        f"{'tfinal':>7} {'pred+':>6} {'p007':>6}",
        "-" * 88,
    ]
    for row in agg:
        lines.append(
            f"{row['rule']:<28} {row['balance_score']:>6.3f} {row['mean_band_f1']:>7.3f} "
            f"{row['early_mean_f1']:>6.3f} {row['late_mean_f1']:>6.3f} "
            f"{row['tfinal_mean_f1']:>7.3f} {row['tfinal_mean_pred_frac']:>6.3f} "
            f"{row['p007_tfinal_f1']:>6.3f}"
        )
    if agg:
        best = agg[0]
        lines.append("")
        lines.append(
            f"[OK] winner: {best['rule']} ({best['rule_desc']}) "
            f"balance={best['balance_score']:.3f} tfinal_mean_F1={best['tfinal_mean_f1']:.3f}"
        )
        grow = pick_growing_winner(agg)
        if grow:
            lines.append(
                f"[OK] growing winner: {grow['rule']} ({grow['rule_desc']}) "
                f"balance={grow['balance_score']:.3f} p007_tfinal={grow['p007_tfinal_f1']:.3f}"
            )
        loc = pick_localized_deploy_winner(agg)
        if loc:
            lines.append(
                f"[OK] localized deploy winner: {loc['rule']} ({loc['rule_desc']}) "
                f"p007_tfinal={loc['p007_tfinal_f1']:.3f} pred+={loc['tfinal_mean_pred_frac']:.3f}"
            )
    return "\n".join(lines)


def run_probe(anchor_dir: Path, anchor: str) -> list:
    if anchor.strip():
        path = anchor_dir / f"{anchor.strip()}.pt"
        if not path.is_file():
            print(f"[ERR] missing {path}")
            sys.exit(1)
        data = torch.load(path, map_location="cpu", weights_only=False)
        return [probe_anchor_temporal_dynamics(data, stem=path.stem)]
    return probe_all_temporal(anchor_dir)


def run_rule_sweep(
    anchor_dir: Path,
    anchor: str,
    *,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    rules: list[TemporalGrowthRuleConfig],
) -> list[dict]:
    paths = discover_anchor_paths(anchor_dir)
    if anchor.strip():
        paths = [p for p in paths if p.stem == anchor.strip()]
    if not paths:
        print(f"[ERR] no patient*.pt under {anchor_dir}")
        sys.exit(1)

    results: list[dict] = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        for rule in rules:
            row = eval_temporal_rule_on_anchor(
                data,
                rule,
                stem=path.stem,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                pair_stride=1,
            )
            results.append(row)
            print(
                f"[i] {path.stem} {rule.name:<22} meanF1={row.get('mean_band_f1', float('nan')):.3f} "
                f"early={row.get('early_mean_f1', float('nan')):.3f} "
                f"tfinal={row.get('tfinal_band_f1', float('nan')):.3f} "
                f"pred+={row.get('tfinal_band_pred_frac', float('nan')):.3f}",
                flush=True,
            )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal clot probe + growing rule sweep")
    p.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    p.add_argument("--anchor", default="")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--rules-only", action="store_true")
    p.add_argument(
        "--localized",
        action="store_true",
        help="Sweep localized segment rules (wall-half/arc/recess/skip15/species)",
    )
    p.add_argument("--species-probe", action="store_true", help="Report GT species AUC in localized pool")
    p.add_argument(
        "--out-probe-json",
        default="outputs/biochem/diagnostics/clot_temporal_dynamics_probe.json",
    )
    p.add_argument(
        "--out-probe-txt",
        default="outputs/biochem/diagnostics/clot_temporal_dynamics_probe.txt",
    )
    p.add_argument(
        "--out-sweep-json",
        default="outputs/biochem/diagnostics/clot_temporal_growth_rule_sweep.json",
    )
    p.add_argument(
        "--out-sweep-txt",
        default="outputs/biochem/diagnostics/clot_temporal_growth_rule_sweep.txt",
    )
    args = p.parse_args()
    if args.localized:
        if args.out_sweep_json.endswith("clot_temporal_growth_rule_sweep.json"):
            args.out_sweep_json = "outputs/biochem/diagnostics/clot_localized_growth_rule_sweep.json"
        if args.out_sweep_txt.endswith("clot_temporal_growth_rule_sweep.txt"):
            args.out_sweep_txt = "outputs/biochem/diagnostics/clot_localized_growth_rule_sweep.txt"

    _apply_sweep_env()
    root = get_project_root()
    anchor_dir = root / args.anchor_dir

    probe_reports = None
    if not args.rules_only:
        probe_reports = run_probe(anchor_dir, args.anchor)
        report = format_temporal_probe_report(probe_reports)
        print(report)
        agg_feat = aggregate_temporal_features(probe_reports)
        if agg_feat:
            print()
            print("--- Top NEW-clot predictors (aggregated) ---")
            for row in agg_feat[:12]:
                print(
                    f"  {row.feature:<22} AUC={row.mean_auc:.3f} delta={row.mean_delta:.3g} "
                    f"({row.group}, n={row.n_event_anchors})"
                )

        probe_json = root / args.out_probe_json
        probe_txt = root / args.out_probe_txt
        write_temporal_probe_json(probe_reports, probe_json)
        probe_txt.parent.mkdir(parents=True, exist_ok=True)
        probe_txt.write_text(report, encoding="utf-8")
        print(f"[save] {probe_json}")
        print(f"[save] {probe_txt}")

    if args.probe_only:
        return

    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    species_rows: list[dict] = []
    if args.species_probe or args.localized:
        paths = discover_anchor_paths(anchor_dir)
        if args.anchor.strip():
            paths = [p for p in paths if p.stem == args.anchor.strip()]
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            t_out = min(37, int(data.y.shape[0]) - 1)
            species_rows.extend(
                probe_species_oracle_auc(data, stem=path.stem, t_out=t_out, device=device, phys=phys, bio_cfg=bio)
            )
        if species_rows:
            print()
            print("--- GT species oracle AUC (localized pool, skip arc 15%, t_out~37) ---")
            print(f"{'anchor':<12} {'species':<16} {'AUC':>6} {'clot':>10} {'non':>10}")
            for row in sorted(species_rows, key=lambda r: (-r["auc"], r["anchor"])):
                print(
                    f"{row['anchor']:<12} {row['species']:<16} {row['auc']:>6.3f} "
                    f"{row['clot_mean']:>10.3g} {row['non_mean']:>10.3g}"
                )

    rules = default_localized_rule_grid() if args.localized else default_temporal_rule_grid()
    sweep_rows = run_rule_sweep(anchor_dir, args.anchor, device=device, phys=phys, bio=bio, rules=rules)
    agg = aggregate_temporal_rule_sweep(sweep_rows)
    table = _format_rule_table(agg)
    print(table)

    payload = {
        "env": {
            "pair_schedule": os.environ.get("CLOT_FORECAST_PAIR_SCHEDULE"),
            "prior_rule_p": os.environ.get("CLOT_PHI_PRIOR_RULE_P"),
            "skip_inlet_q": os.environ.get("CLOT_PHI_PRIOR_RULE_SKIP_INLET_Q"),
        },
        "winner": agg[0] if agg else None,
        "growing_winner": pick_growing_winner(agg),
        "localized_deploy_winner": pick_localized_deploy_winner(agg),
        "aggregated": agg,
        "per_anchor": sweep_rows,
    }
    if species_rows:
        payload["species_oracle_probe"] = species_rows
    if probe_reports is not None:
        payload["probe_summary"] = {
            "n_anchors": len(probe_reports),
            "mean_adjacent_pct": sum(r.pct_new_adjacent_to_existing for r in probe_reports)
            / max(len(probe_reports), 1),
            "top_features": [f.feature for f in aggregate_temporal_features(probe_reports)[:5]],
        }

    sweep_json = root / args.out_sweep_json
    sweep_txt = root / args.out_sweep_txt
    sweep_json.parent.mkdir(parents=True, exist_ok=True)
    sweep_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sweep_txt.write_text(table, encoding="utf-8")
    print(f"[save] {sweep_json}")
    print(f"[save] {sweep_txt}")


if __name__ == "__main__":
    main()
