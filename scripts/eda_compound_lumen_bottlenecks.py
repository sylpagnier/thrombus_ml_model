"""EDA: hop-stratified clot prevalence + compound failure modes (orig10).

Answers: how rare is hop>=2 GT? which anchors can teach lumen? where does
Arm S overpredict vs under-recall vs spray on zero-GT vessels?

Usage:
  python scripts/eda_compound_lumen_bottlenecks.py
  python scripts/eda_compound_lumen_bottlenecks.py --out outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/eda_lumen_bottlenecks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.species_pushforward_continuous import compute_hop_distances  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ORIG10 = [
    "patient001",
    "patient002",
    "patient003",
    "patient004",
    "patient005",
    "patient006",
    "patient007",
    "patient008",
    "patient010",
    "patient011",
]


def _load_eval_per(path: Path) -> dict:
    if not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return dict((report.get("simple") or {}).get("per_anchor") or {})


def _anchor_gt_hop_profile(anchor: str, root: Path, phys: PhysicsConfig) -> dict:
    path = root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    n_nodes = int(data.num_nodes)
    n_t = int(data.y.shape[0])
    wall = _wall_mask_from_data(data, torch.device("cpu"), n_nodes)
    hops = compute_hop_distances(data.edge_index, wall, n_nodes).cpu().numpy()

    # Sample up to 24 frames across the timeline (EDA budget).
    idx = np.unique(np.linspace(0, n_t - 1, num=min(24, n_t), dtype=int))
    rows = []
    for t in idx:
        phi = gt_clot_phi_at_time(data, int(t), phys, torch.device("cpu")).detach().cpu().numpy().reshape(-1)
        clot = phi >= 0.5
        counts = {
            "t": int(t),
            "n_clot": int(clot.sum()),
            "n_wall": int((clot & (hops == 0)).sum()),
            "n_hop1": int((clot & (hops == 1)).sum()),
            "n_hop_ge2": int((clot & (hops >= 2)).sum()),
            "n_hop2": int((clot & (hops == 2)).sum()),
            "n_hop3": int((clot & (hops == 3)).sum()),
            "n_hop_ge4": int((clot & (hops >= 4)).sum()),
        }
        rows.append(counts)

    n_clot = np.array([r["n_clot"] for r in rows], dtype=float)
    n_ge2 = np.array([r["n_hop_ge2"] for r in rows], dtype=float)
    n_wall = np.array([r["n_wall"] for r in rows], dtype=float)
    n_h1 = np.array([r["n_hop1"] for r in rows], dtype=float)
    peak_i = int(np.argmax(n_clot)) if len(rows) else 0
    peak = rows[peak_i] if rows else {}
    any_ge2 = float((n_ge2 > 0).mean()) if len(rows) else 0.0
    # Mass share at peak clot frame
    peak_clot = max(float(peak.get("n_clot", 0)), 1.0)
    return {
        "anchor": anchor,
        "n_nodes": n_nodes,
        "n_times": n_t,
        "n_frames_sampled": len(rows),
        "frac_frames_any_clot": float((n_clot > 0).mean()) if len(rows) else 0.0,
        "frac_frames_any_hop_ge2": any_ge2,
        "mean_n_clot": float(n_clot.mean()) if len(rows) else 0.0,
        "mean_n_hop_ge2": float(n_ge2.mean()) if len(rows) else 0.0,
        "peak_n_clot": float(peak.get("n_clot", 0)),
        "peak_n_wall": float(peak.get("n_wall", 0)),
        "peak_n_hop1": float(peak.get("n_hop1", 0)),
        "peak_n_hop_ge2": float(peak.get("n_hop_ge2", 0)),
        "peak_share_wall": float(peak.get("n_wall", 0)) / peak_clot,
        "peak_share_hop1": float(peak.get("n_hop1", 0)) / peak_clot,
        "peak_share_hop_ge2": float(peak.get("n_hop_ge2", 0)) / peak_clot,
        "mean_share_hop_ge2_when_clot": float(
            np.mean([r["n_hop_ge2"] / max(r["n_clot"], 1) for r in rows if r["n_clot"] > 0] or [0.0])
        ),
        "timeline": rows,
    }


def _classify_failure(a: dict, s: dict) -> str:
    gt = float(s.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0)
    pred = float(s.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0)
    strict = float(s.get("deploy_clot_offwall_strict_f1_hop_ge2", 0) or 0)
    d_clot = float(s.get("deploy_clot_f1", 0) or 0) - float(a.get("deploy_clot_f1", 0) or 0)

    if gt <= 0 and pred <= 0.5:
        return "no_lumen_gt_idle"
    if gt <= 0 and pred > 0.5:
        return "zero_gt_spray"
    if gt > 0 and pred <= 0.5:
        return "miss_all_lumen"
    if gt > 0 and pred > 1.5 * gt and strict < 0.05:
        return "overvolume_low_prec"
    if gt > 0 and pred > 0 and strict >= 0.10:
        return "signal_localized" if d_clot >= -0.03 else "signal_but_wall_bleed"
    if gt > 0 and pred > 0 and strict < 0.05:
        return "weak_overlap"
    return "mixed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchors",
        default=",".join(ORIG10),
        help="Comma-separated anchors (default orig10)",
    )
    ap.add_argument(
        "--eval-a",
        default="outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/eval_A_canonical.json",
    )
    ap.add_argument(
        "--eval-s",
        default="outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/eval_S_frontier_ge2_prec.json",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    print("=" * 78, flush=True)
    print("EDA: lumen / hop>=2 clot bottlenecks (compound push)", flush=True)
    print("=" * 78, flush=True)

    profiles = []
    for anc in anchors:
        print(f"[i] GT hop profile {anc}...", flush=True)
        profiles.append(_anchor_gt_hop_profile(anc, root, phys))

    # Prevalence summary
    print("\n--- GT prevalence (sampled timeline) ---", flush=True)
    print(
        f"{'anchor':<12} {'%fr_ge2':>8} {'mean_ge2':>9} {'peak_ge2':>9} "
        f"{'peak_share_ge2':>14} {'peak_clot':>9}",
        flush=True,
    )
    for p in profiles:
        print(
            f"{p['anchor']:<12} {100*p['frac_frames_any_hop_ge2']:7.1f}% "
            f"{p['mean_n_hop_ge2']:9.1f} {p['peak_n_hop_ge2']:9.0f} "
            f"{100*p['peak_share_hop_ge2']:13.1f}% {p['peak_n_clot']:9.0f}",
            flush=True,
        )

    n_with_ge2 = sum(1 for p in profiles if p["peak_n_hop_ge2"] > 0)
    n_strong = sum(1 for p in profiles if p["peak_n_hop_ge2"] >= 20)
    mean_frac_frames = float(np.mean([p["frac_frames_any_hop_ge2"] for p in profiles]))
    # Clot mass at peak: wall+hop1 vs ge2 across cohort
    tot_peak_clot = sum(p["peak_n_clot"] for p in profiles) or 1.0
    tot_peak_ge2 = sum(p["peak_n_hop_ge2"] for p in profiles)
    tot_peak_wall = sum(p["peak_n_wall"] for p in profiles)
    tot_peak_h1 = sum(p["peak_n_hop1"] for p in profiles)

    prevalence = {
        "n_anchors": len(profiles),
        "n_anchors_any_peak_hop_ge2": n_with_ge2,
        "n_anchors_peak_hop_ge2_ge20": n_strong,
        "mean_frac_frames_any_hop_ge2": mean_frac_frames,
        "cohort_peak_share_wall": tot_peak_wall / tot_peak_clot,
        "cohort_peak_share_hop1": tot_peak_h1 / tot_peak_clot,
        "cohort_peak_share_hop_ge2": tot_peak_ge2 / tot_peak_clot,
    }
    print("\n[i] Cohort peak-clot mass share: "
          f"wall={100*prevalence['cohort_peak_share_wall']:.1f}% "
          f"hop1={100*prevalence['cohort_peak_share_hop1']:.1f}% "
          f"hop_ge2={100*prevalence['cohort_peak_share_hop_ge2']:.1f}%", flush=True)
    print(
        f"[i] Anchors with any peak hop_ge2: {n_with_ge2}/{len(profiles)}; "
        f"peak>=20 nodes: {n_strong}/{len(profiles)}; "
        f"mean %frames with hop_ge2: {100*mean_frac_frames:.1f}%",
        flush=True,
    )

    # Failure modes from A/S eval
    eval_a_path = Path(args.eval_a)
    if not eval_a_path.is_absolute():
        eval_a_path = root / eval_a_path
    eval_s_path = Path(args.eval_s)
    if not eval_s_path.is_absolute():
        eval_s_path = root / eval_s_path
    per_a = _load_eval_per(eval_a_path)
    per_s = _load_eval_per(eval_s_path)

    print("\n--- Arm S failure taxonomy (vs A, deploy hop_ge2) ---", flush=True)
    print(
        f"{'anchor':<12} {'class':<22} {'ge2_gt':>7} {'ge2_pr':>7} {'strict':>7} {'dclot':>7}",
        flush=True,
    )
    classes: dict[str, list[str]] = {}
    fail_rows = []
    for anc in anchors:
        a = per_a.get(anc) or {}
        s = per_s.get(anc) or {}
        if not s:
            continue
        cls = _classify_failure(a, s)
        classes.setdefault(cls, []).append(anc)
        gt = float(s.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0)
        pr = float(s.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0)
        st = float(s.get("deploy_clot_offwall_strict_f1_hop_ge2", 0) or 0)
        dcl = float(s.get("deploy_clot_f1", 0) or 0) - float(a.get("deploy_clot_f1", 0) or 0)
        print(f"{anc:<12} {cls:<22} {gt:7.1f} {pr:7.1f} {st:7.3f} {dcl:+7.3f}", flush=True)
        fail_rows.append(
            {
                "anchor": anc,
                "class": cls,
                "ge2_gt": gt,
                "ge2_pred": pr,
                "ge2_strict": st,
                "d_clot_f1": dcl,
                "clot_fp_p90": float(s.get("clot_fp_p90", 0) or 0),
                "clot_fn_p90": float(s.get("clot_fn_p90", 0) or 0),
            }
        )

    print("\n[i] Class counts:", flush=True)
    for k, v in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k}: {len(v)} -> {', '.join(v)}", flush=True)

    # Bottleneck hypotheses ranked
    bottlenecks = []
    if prevalence["n_anchors_peak_hop_ge2_ge20"] <= 3:
        bottlenecks.append(
            {
                "id": "rare_thick_clot_teachers",
                "severity": "high",
                "note": (
                    f"Only {prevalence['n_anchors_peak_hop_ge2_ge20']}/{len(profiles)} "
                    "orig10 anchors have peak hop_ge2>=20; lumen supervision is concentrated."
                ),
            }
        )
    if prevalence["cohort_peak_share_hop_ge2"] < 0.25:
        bottlenecks.append(
            {
                "id": "wall_dominated_clot_mass",
                "severity": "high",
                "note": (
                    f"At peak frames, hop_ge2 is only "
                    f"{100*prevalence['cohort_peak_share_hop_ge2']:.1f}% of clot nodes "
                    "(wall+hop1 dominate labels)."
                ),
            }
        )
    if prevalence["mean_frac_frames_any_hop_ge2"] < 0.35:
        bottlenecks.append(
            {
                "id": "sparse_lumen_time",
                "severity": "medium",
                "note": (
                    f"Hop_ge2 present in only "
                    f"{100*prevalence['mean_frac_frames_any_hop_ge2']:.1f}% of sampled frames "
                    "on average - most windows teach wall/hop1 growth."
                ),
            }
        )
    spray = classes.get("zero_gt_spray") or []
    if spray:
        bottlenecks.append(
            {
                "id": "zero_gt_spray_anchors",
                "severity": "high",
                "note": f"Arm S paints hop_ge2 on zero-GT anchors: {', '.join(spray)}.",
            }
        )
    miss = classes.get("miss_all_lumen") or []
    if miss:
        bottlenecks.append(
            {
                "id": "miss_positive_lumen",
                "severity": "high",
                "note": f"Has GT lumen but S predicts ~0: {', '.join(miss)}.",
            }
        )
    over = classes.get("overvolume_low_prec") or []
    if over:
        bottlenecks.append(
            {
                "id": "overvolume_low_prec",
                "severity": "medium",
                "note": f"Pred>>GT with low strict: {', '.join(over)}.",
            }
        )

    # Ckpt selection note from known 8h run
    bottlenecks.append(
        {
            "id": "ckpt_volume_bias",
            "severity": "medium",
            "note": (
                "8h best ckpt was ep2 under hop_ge2_balanced (high volume); later epochs "
                "had higher p007 strict but lower score - selection may lock overprediction."
            ),
        }
    )

    print("\n--- Ranked bottlenecks ---", flush=True)
    for b in bottlenecks:
        print(f"  [{b['severity']}] {b['id']}: {b['note']}", flush=True)

    report = {
        "anchors": anchors,
        "prevalence": prevalence,
        "profiles": [{k: v for k, v in p.items() if k != "timeline"} for p in profiles],
        "profiles_with_timeline": profiles,
        "failure_rows": fail_rows,
        "failure_classes": {k: v for k, v in classes.items()},
        "bottlenecks": bottlenecks,
        "eval_a": str(Path(args.eval_a)),
        "eval_s": str(Path(args.eval_s)),
    }

    out = Path(args.out.strip()) if args.out.strip() else (
        root / "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/eda_lumen_bottlenecks.json"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Compact save without full timelines in the primary file for readability;
    # keep timelines in a sibling.
    compact = dict(report)
    compact.pop("profiles_with_timeline", None)
    out.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    timeline_out = out.with_name(out.stem + "_timelines.json")
    timeline_out.write_text(
        json.dumps({"profiles": profiles}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[save] {out}", flush=True)
    print(f"[save] {timeline_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
