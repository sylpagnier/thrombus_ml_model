"""Summarize fi_mat vs fi_mat_thrombin species-scope A/B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-a", required=True, help="fi_mat deploy_ab_eval.json")
    ap.add_argument("--eval-b", required=True, help="fi_mat_thrombin deploy_ab_eval.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    root = REPO
    ma = _metrics(Path(args.eval_a) if Path(args.eval_a).is_absolute() else root / args.eval_a)
    mb = _metrics(Path(args.eval_b) if Path(args.eval_b).is_absolute() else root / args.eval_b)

    def _winner(key: str) -> str:
        a = float(ma.get(key, 0.0))
        b = float(mb.get(key, 0.0))
        if abs(a - b) < 1e-6:
            return "tie"
        return "B (+thrombin)" if b > a else "A (fi_mat)"

    summary = {
        "A_fi_mat": ma,
        "B_fi_mat_thrombin": mb,
        "winner_p007_f1": _winner("p007_f1"),
        "winner_p007_guiding": _winner("p007_guiding"),
        "winner_holdout_mean_f1": _winner("holdout_mean_f1"),
    }
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = root / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Species scope A/B (fi_mat vs fi_mat+thrombin)",
        "",
        "| Metric | A fi_mat | B +thrombin | Winner |",
        "|--------|----------|-------------|--------|",
    ]
    for key, label in (
        ("p007_f1", "p007 F1@t53"),
        ("p007_guiding", "p007 guiding"),
        ("p007_f05", "p007 F0.5"),
        ("holdout_mean_f1", "holdout mean F1"),
        ("holdout_mean_guiding", "holdout mean guiding"),
    ):
        a = ma.get(key)
        b = mb.get(key)
        if a is None and b is None:
            continue
        w = _winner(key) if key in ma and key in mb else "-"
        lines.append(f"| {label} | {a:.3f} | {b:.3f} | {w} |")

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = root / out_md
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
