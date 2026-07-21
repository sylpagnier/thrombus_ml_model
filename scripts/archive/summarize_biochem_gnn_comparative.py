"""Summarise gate + species-scope comparative eval (comparative_6h launcher)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.summarize_biochem_gnn_gate_ab import GUIDING_KEYS  # noqa: E402


def _load_leg_metrics(eval_dir: str) -> dict[str, float]:
    """Load deploy_ab_eval.json (preferred) or legacy per-anchor JSON dir."""
    p = pathlib.Path(eval_dir)
    ab = p / "deploy_ab_eval.json"
    if ab.is_file():
        data = json.loads(ab.read_text(encoding="utf-8"))
        rows = [
            r
            for r in data.get("rows", [])
            if r.get("mode") == "deploy_frozen" and not r.get("error")
        ]
        if not rows:
            return {}
        p007 = next((r for r in rows if r.get("anchor") == "patient007"), rows[0])
        hold = [r for r in rows if r.get("anchor") != "patient007"]
        out: dict[str, float] = {}
        for key, val in p007.items():
            if isinstance(val, (int, float)) and not key.startswith("clot_f1"):
                out[key] = float(val)
        out["deploy_clot_score"] = float(
            p007.get("clot_score_main", p007.get("clot_guiding_main", p007.get("clot_f1_main", 0.0)))
        )
        out["clot_f1_main"] = float(p007.get("clot_f1_main", 0.0))
        out["clot_guiding_main"] = float(p007.get("clot_guiding_main", 0.0))
        if hold:
            for key in ("clot_f1_main", "clot_guiding_main", "clot_score_main", "clot_relaxed_f05_main"):
                out[f"holdout_mean_{key}"] = sum(float(r.get(key, 0.0)) for r in hold) / len(hold)
        return out

    results: dict[str, list[float]] = {}
    for fp in sorted(p.glob("*.json")):
        if fp.name == "deploy_ab_eval.json":
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)):
                results.setdefault(k, []).append(float(v))
    if not results:
        return {}
    return {k: sum(vs) / len(vs) for k, vs in results.items()}


def _load_json_arg(raw: str | None, file_path: str | None) -> dict:
    if file_path:
        return json.loads(pathlib.Path(file_path).read_text(encoding="utf-8-sig"))
    if raw:
        return json.loads(raw)
    raise SystemExit("need --eval-dirs or --eval-dirs-file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dirs", default="")
    ap.add_argument("--eval-dirs-file", default="")
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--meta-file", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    eval_dirs: dict[str, str] = _load_json_arg(
        args.eval_dirs or None,
        args.eval_dirs_file or None,
    )
    if args.meta_file:
        meta: dict = json.loads(pathlib.Path(args.meta_file).read_text(encoding="utf-8-sig"))
    else:
        meta = json.loads(args.meta)

    leg_metrics: dict[str, dict[str, float]] = {}
    for leg, d in eval_dirs.items():
        if isinstance(d, list):
            d = next((x for x in d if isinstance(x, str)), str(d[-1]))
        leg_metrics[leg] = _load_leg_metrics(str(d))

    gate_winner = None
    species_winner = None
    best_gate = -1e9
    best_species = -1e9
    winner_key = None
    for gk in GUIDING_KEYS:
        if not any(m.get(gk) is not None for m in leg_metrics.values()):
            continue
        winner_key = gk
        for leg, m in leg_metrics.items():
            score = float(m.get(gk, -1e9))
            if leg.startswith("G") and score > best_gate:
                best_gate = score
                gate_winner = leg
            if leg.startswith("S") and score > best_species:
                best_species = score
                species_winner = leg
        break

    summary = {
        "meta": meta,
        "legs": leg_metrics,
        "gate_winner": gate_winner,
        "species_winner": species_winner,
        "winner_key": winner_key,
        "gate_winner_score": best_gate,
        "species_winner_score": best_species,
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    if gate_winner:
        print(f"[i] gate winner: {gate_winner} ({winner_key}={best_gate:.4f})")
    if species_winner:
        print(f"[i] species winner: {species_winner} ({winner_key}={best_species:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
