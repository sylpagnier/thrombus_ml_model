"""Summarize the FI species-channel ablation (mat / mat_fg / fi_mat / fi_mat_fg).

Ranks legs by holdout-mean guiding (generalization) and prints p007 + holdout F1/guiding.
Run: python scripts/summarize_species_fi_ablation.py --run-root outputs/biochem/biochem_gnn/fi_ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LEGS = ["mat", "mat_fg", "fi_mat", "fi_mat_fg"]
LABELS = {"mat": "Mat", "mat_fg": "Mat+FG", "fi_mat": "FI+Mat", "fi_mat_fg": "FI+Mat+FG"}


def _metrics(eval_path: Path) -> dict[str, float]:
    if not eval_path.is_file():
        return {}
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = [r for r in data.get("rows", []) if r.get("mode") == "deploy_frozen" and not r.get("error")]
    if not rows:
        return {}
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), None)
    hold = [r for r in rows if r.get("anchor") != "patient007"]
    out: dict[str, float] = {}
    if p007 is not None:
        out["p007_f1"] = float(p007.get("clot_f1_main", 0.0))
        out["p007_guiding"] = float(p007.get("clot_guiding_main", p007.get("clot_score_main", 0.0)))
    if hold:
        out["holdout_f1"] = sum(float(r.get("clot_f1_main", 0.0)) for r in hold) / len(hold)
        out["holdout_guiding"] = sum(
            float(r.get("clot_guiding_main", r.get("clot_score_main", 0.0))) for r in hold
        ) / len(hold)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/biochem/biochem_gnn/fi_ablation")
    args = ap.parse_args()
    root = Path(args.run_root)
    if not root.is_absolute():
        root = REPO / root

    res = {leg: _metrics(root / leg / "eval" / "deploy_ab_eval.json") for leg in LEGS}
    ranked = sorted(
        [leg for leg in LEGS if res[leg]],
        key=lambda l: res[l].get("holdout_guiding", -1.0),
        reverse=True,
    )

    hdr = f"{'leg':<12}{'p007_f1':>9}{'p007_guid':>10}{'hold_f1':>9}{'hold_guid':>10}"
    print("\n[i] FI ablation (deploy_frozen, kine flow), ranked by holdout guiding\n")
    print(hdr)
    print("-" * len(hdr))
    lines = ["# Species FI ablation (deploy_frozen, kine flow)", "",
             "Ranked by holdout-mean guiding (generalization).", "",
             "| Leg | p007 F1 | p007 guiding | holdout F1 | holdout guiding |",
             "|-----|---------|--------------|------------|-----------------|"]
    for leg in ranked:
        m = res[leg]
        print(f"{LABELS[leg]:<12}{m.get('p007_f1', 0):>9.3f}{m.get('p007_guiding', 0):>10.3f}"
              f"{m.get('holdout_f1', 0):>9.3f}{m.get('holdout_guiding', 0):>10.3f}")
        lines.append(f"| {LABELS[leg]} | {m.get('p007_f1', 0):.3f} | {m.get('p007_guiding', 0):.3f} "
                     f"| {m.get('holdout_f1', 0):.3f} | {m.get('holdout_guiding', 0):.3f} |")

    if ranked:
        best = ranked[0]
        fi_mat = res.get("fi_mat", {})
        d = res[best].get("holdout_guiding", 0) - fi_mat.get("holdout_guiding", 0)
        verdict = (f"Best holdout: {LABELS[best]} (holdout guiding {res[best].get('holdout_guiding',0):.3f}, "
                   f"{d:+.3f} vs FI+Mat baseline).")
        print(f"\n[verdict] {verdict}")
        lines += ["", f"**Verdict:** {verdict}"]

    out_json = root / "fi_ablation_summary.json"
    out_md = root / "fi_ablation_report.md"
    out_json.write_text(json.dumps({"results": res, "ranked": ranked}, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[save] {out_json}\n[save] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
