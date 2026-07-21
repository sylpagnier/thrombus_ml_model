"""Rank fi_mat + X screen legs and summarize cumulative ladder results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training.biochem_species_scope import BULK_CHANNEL_NAMES, bulk_channel_name


def _metrics(eval_path: Path) -> dict[str, float]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = [
        r
        for r in data.get("rows", [])
        if r.get("mode") == "deploy_frozen" and not r.get("error")
    ]
    if not rows:
        return {}
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), rows[0])
    hold = [r for r in rows if r.get("anchor") != "patient007"]
    out = {
        "p007_f1": float(p007.get("clot_f1_main", 0.0)),
        "p007_guiding": float(p007.get("clot_guiding_main", p007.get("clot_score_main", 0.0))),
        "p007_f05": float(p007.get("clot_relaxed_f05_main", 0.0)),
    }
    if hold:
        out["holdout_mean_f1"] = sum(float(r.get("clot_f1_main", 0.0)) for r in hold) / len(hold)
        out["holdout_mean_guiding"] = sum(
            float(r.get("clot_guiding_main", r.get("clot_score_main", 0.0))) for r in hold
        ) / len(hold)
    out["rank_score"] = 0.55 * out["p007_guiding"] + 0.45 * float(out.get("holdout_mean_guiding", out["p007_guiding"]))
    return out


def _addon_label(ch: int) -> str:
    return bulk_channel_name(ch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-root", required=True, help="species_rank_2h/screen directory")
    ap.add_argument("--ladder-root", default="", help="species_rank_2h/ladder directory")
    ap.add_argument("--baseline-eval", default="", help="optional fi_mat baseline eval json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--top-n", type=int, default=4, help="how many addons to keep for ladder report")
    args = ap.parse_args()

    screen_root = Path(args.screen_root)
    if not screen_root.is_absolute():
        screen_root = REPO / screen_root

    manifest_path = screen_root / "screen_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing screen manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legs = list(manifest.get("legs") or [])

    baseline_m: dict[str, float] = {}
    if args.baseline_eval:
        bp = Path(args.baseline_eval)
        if not bp.is_absolute():
            bp = REPO / bp
        if bp.is_file():
            baseline_m = _metrics(bp)

    ranked: list[dict] = []
    for leg in legs:
        label = str(leg.get("label") or "")
        if label == "fi_mat":
            continue
        eval_path = screen_root / label / "eval" / "deploy_ab_eval.json"
        if not eval_path.is_file():
            continue
        m = _metrics(eval_path)
        addon_ch = leg.get("addon_channel")
        ranked.append(
            {
                "label": label,
                "addon_channel": addon_ch,
                "addon_name": _addon_label(int(addon_ch)) if addon_ch is not None else label,
                "channels": leg.get("channels"),
                "metrics": m,
                "delta_p007_guiding": m.get("p007_guiding", 0.0) - baseline_m.get("p007_guiding", 0.0)
                if baseline_m
                else None,
                "delta_holdout_guiding": m.get("holdout_mean_guiding", 0.0)
                - baseline_m.get("holdout_mean_guiding", 0.0)
                if baseline_m
                else None,
            }
        )

    ranked.sort(key=lambda r: float(r["metrics"].get("rank_score", 0.0)), reverse=True)
    top_addons = [int(r["addon_channel"]) for r in ranked[: max(1, args.top_n)] if r.get("addon_channel") is not None]

    ladder_rows: list[dict] = []
    ladder_root = Path(args.ladder_root) if args.ladder_root else None
    if ladder_root and not ladder_root.is_absolute():
        ladder_root = REPO / ladder_root
    if ladder_root and ladder_root.is_dir():
        ladder_manifest = ladder_root / "ladder_manifest.json"
        if ladder_manifest.is_file():
            for leg in json.loads(ladder_manifest.read_text(encoding="utf-8")).get("legs") or []:
                label = str(leg.get("label") or "")
                eval_path = ladder_root / label / "eval" / "deploy_ab_eval.json"
                if not eval_path.is_file():
                    continue
                ladder_rows.append(
                    {
                        "label": label,
                        "channels": leg.get("channels"),
                        "metrics": _metrics(eval_path),
                    }
                )

    summary = {
        "baseline_fi_mat": baseline_m,
        "screen_ranked": ranked,
        "top_addon_channels": top_addons,
        "ladder": ladder_rows,
    }

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = REPO / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Species rank screen + ladder (~2h)",
        "",
        "Rank key: `0.55 * p007_guiding + 0.45 * holdout_mean_guiding`.",
        "",
    ]
    if baseline_m:
        lines += [
            f"Baseline fi_mat: p007 guiding {baseline_m.get('p007_guiding', 0.0):.3f}, "
            f"holdout {baseline_m.get('holdout_mean_guiding', 0.0):.3f}",
            "",
        ]
    lines += [
        "## Screen (fi_mat + single addon)",
        "",
        "| Rank | Addon | p007 F1 | p007 guiding | holdout guiding | rank score | d guiding (p007) |",
        "|------|-------|---------|--------------|-----------------|------------|------------------|",
    ]
    for i, row in enumerate(ranked, start=1):
        m = row["metrics"]
        d = row.get("delta_p007_guiding")
        d_s = f"{d:+.3f}" if d is not None else "-"
        lines.append(
            f"| {i} | {row['addon_name']} | {m.get('p007_f1', 0.0):.3f} | "
            f"{m.get('p007_guiding', 0.0):.3f} | {m.get('holdout_mean_guiding', 0.0):.3f} | "
            f"{m.get('rank_score', 0.0):.3f} | {d_s} |"
        )

    if ladder_rows:
        lines += [
            "",
            "## Cumulative ladder",
            "",
            "| Step | Scope | p007 F1 | p007 guiding | holdout guiding | rank score |",
            "|------|-------|---------|--------------|-----------------|------------|",
        ]
        for row in ladder_rows:
            m = row["metrics"]
            lines.append(
                f"| {row['label']} | {row.get('channels')} | {m.get('p007_f1', 0.0):.3f} | "
                f"{m.get('p007_guiding', 0.0):.3f} | {m.get('holdout_mean_guiding', 0.0):.3f} | "
                f"{m.get('rank_score', 0.0):.3f} |"
            )

    lines.append("")
    lines.append(f"Top addon channels for ladder: {top_addons}")
    lines.append(f"Names: {', '.join(_addon_label(c) for c in top_addons)}")

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = REPO / out_md
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"top_addons": top_addons, "best_screen": ranked[0] if ranked else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
