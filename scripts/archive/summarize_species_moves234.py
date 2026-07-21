"""Summarize the moves 2+3+4 A/B (baseline_matfg vs moves234_matfg).

Reports p007 + holdout-mean F1 / guiding / precision / recall (deploy_frozen, kine flow) so we can
see whether the precision lever (move 2) lifts precision without crushing recall, and whether the
lumen term (move 3) holds recall. Run:
  python scripts/summarize_species_moves234.py --run-root outputs/biochem/biochem_gnn/moves234
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LEGS = ["baseline_matfg", "moves234_matfg"]
LABELS = {"baseline_matfg": "Mat+FG (control)", "moves234_matfg": "Mat+FG +moves234"}
FIELDS = ["f1", "guiding", "prec", "rec"]


def _metrics(eval_path: Path) -> dict[str, float]:
    if not eval_path.is_file():
        return {}
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = [r for r in data.get("rows", []) if r.get("mode") == "deploy_frozen" and not r.get("error")]
    if not rows:
        return {}
    p007 = next((r for r in rows if r.get("anchor") == "patient007"), None)
    hold = [r for r in rows if r.get("anchor") != "patient007"]

    def pick(r, key):
        return float(r.get(f"clot_{key}_main", 0.0)) if r else 0.0

    out: dict[str, float] = {}
    if p007 is not None:
        out["p007_f1"] = pick(p007, "f1")
        out["p007_guiding"] = float(p007.get("clot_guiding_main", p007.get("clot_score_main", 0.0)))
        out["p007_prec"] = pick(p007, "prec")
        out["p007_rec"] = pick(p007, "rec")
    if hold:
        n = len(hold)
        out["holdout_f1"] = sum(pick(r, "f1") for r in hold) / n
        out["holdout_guiding"] = sum(
            float(r.get("clot_guiding_main", r.get("clot_score_main", 0.0))) for r in hold
        ) / n
        out["holdout_prec"] = sum(pick(r, "prec") for r in hold) / n
        out["holdout_rec"] = sum(pick(r, "rec") for r in hold) / n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/biochem/biochem_gnn/moves234")
    args = ap.parse_args()
    root = Path(args.run_root)
    if not root.is_absolute():
        root = REPO / root

    res = {leg: _metrics(root / leg / "eval" / "deploy_ab_eval.json") for leg in LEGS}

    hdr = (f"{'leg':<20}{'p007_f1':>9}{'p007_prec':>10}{'p007_rec':>9}"
           f"{'hold_f1':>9}{'hold_prec':>10}{'hold_rec':>9}{'hold_guid':>10}")
    print("\n[i] moves234 A/B (deploy_frozen, kine flow)\n")
    print(hdr)
    print("-" * len(hdr))
    lines = ["# Moves 2+3+4 A/B (deploy_frozen, kine flow)", "",
             "| Leg | p007 F1 | p007 prec | p007 rec | holdout F1 | holdout prec | holdout rec | holdout guiding |",
             "|-----|---------|-----------|----------|------------|--------------|-------------|-----------------|"]
    for leg in LEGS:
        m = res.get(leg) or {}
        if not m:
            continue
        print(f"{LABELS[leg]:<20}{m.get('p007_f1', 0):>9.3f}{m.get('p007_prec', 0):>10.3f}"
              f"{m.get('p007_rec', 0):>9.3f}{m.get('holdout_f1', 0):>9.3f}{m.get('holdout_prec', 0):>10.3f}"
              f"{m.get('holdout_rec', 0):>9.3f}{m.get('holdout_guiding', 0):>10.3f}")
        lines.append(f"| {LABELS[leg]} | {m.get('p007_f1', 0):.3f} | {m.get('p007_prec', 0):.3f} "
                     f"| {m.get('p007_rec', 0):.3f} | {m.get('holdout_f1', 0):.3f} "
                     f"| {m.get('holdout_prec', 0):.3f} | {m.get('holdout_rec', 0):.3f} "
                     f"| {m.get('holdout_guiding', 0):.3f} |")

    b = res.get("baseline_matfg") or {}
    mv = res.get("moves234_matfg") or {}
    if b and mv:
        df1 = mv.get("holdout_f1", 0) - b.get("holdout_f1", 0)
        dpr = mv.get("holdout_prec", 0) - b.get("holdout_prec", 0)
        drc = mv.get("holdout_rec", 0) - b.get("holdout_rec", 0)
        verdict = (f"moves234 vs control (holdout): F1 {df1:+.3f}, precision {dpr:+.3f}, recall {drc:+.3f}. "
                   f"Goal: precision up (move 2) while recall holds (move 3).")
        print(f"\n[verdict] {verdict}")
        lines += ["", f"**Verdict:** {verdict}"]

    out_json = root / "moves234_summary.json"
    out_md = root / "moves234_report.md"
    out_json.write_text(json.dumps({"results": res}, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[save] {out_json}\n[save] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
