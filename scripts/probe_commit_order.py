"""Step 1b -- commitment-order probe (docs/WALL_MODEL_PLAN.md s2.10, s4 Step 1b).

s2.9 measured the residual failure precisely: on patient037 the checkpoint commits a
40-node TRUE and a 40-node FALSE component at min hop-2 speed 0.048 vs 0.047 -- a tie no
flow threshold can break. s2.10 proposes commitment *timing* as an orthogonal signal, on
the strength of ``mat_seed_prec = 1.000`` holding on every checkpoint examined (the first
commitment is always correct).

This measures it. One instrumented rollout per anchor, no training. For each predicted
connected component it records the first step at which any of its nodes crosses the graded
phi threshold, labels the component TP/FP by GT overlap, and computes the same
``AUC(TP < FP)`` s2.3/s2.9 used for flow -- plus the number that actually decides Step 1b:
the AUC restricted to (TP, FP) pairs flow calls a tie.

    python scripts/probe_commit_order.py --anchors patient037,patient021

Read the verdict against the s4 Step 1b decision rule:
  * tiebreak AUC clearly off 0.5, same direction on both vessels
        -> commit-time is a real second signal; combine with the flow gate and re-run the
           Step 1 percentile sweep with the combined rule BEFORE the holdout shot.
  * tiebreak AUC ~ 0.5 (or direction flips between vessels)
        -> the seed_prec guarantee is specific to the single first seed and does not extend
           to ranking concurrent pockets. Drop the direction, accept the flow-only minimax
           gate as the ceiling, and spend the one-shot holdout application on it.
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

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _load_static  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import (  # noqa: E402
    reset_species_rollout_flow_cache,
    resolve_species_rollout_uv,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_clot_phi_trajectory,
    deploy_species_rollout_series,
    load_continuous_bundle,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.evaluation.commit_time import (  # noqa: E402
    first_commit_step,
    flow_tie_pairs,
    predicted_components,
    rank_auc,
)
from src.evaluation.pocket_gate import hop2_speed_field  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_CKPT = "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth"
# s6 rule 2/5: these are the one-shot holdouts. A probe that tunes on them burns the shot.
HOLDOUTS = {"patient020", "patient043", "patient044"}


def _fmt_auc(v: float) -> str:
    return "  n/a " if not np.isfinite(v) else f"{v:6.4f}"


def _strict_f1(comps: list[dict], n_gt: int) -> tuple[float, float, float]:
    """Node-level strict ``(f1, prec, rec)`` over a set of kept components."""
    tp = sum(c["gt_overlap"] for c in comps)
    pred = sum(c["size"] for c in comps)
    prec = tp / max(pred, 1)
    rec = tp / max(n_gt, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def selection_ceiling(rows: list[dict], n_gt: int) -> dict:
    """What ORACLE pocket selection is worth on this vessel (the s2.2 construction).

    Keeping every GT-touching component is the best any selection rule -- flow, timing, or
    a perfect one -- can do. s2.2 measured 0.887 on patient020 and the plan reads it as the
    ceiling; this reports it per vessel, alongside the two quantities that cap it and which
    selection cannot touch: the purity of the kept components and how much GT the model
    grew at all.
    """
    tp_comps = [r for r in rows if r["is_tp"]]
    off_f1, off_p, off_r = _strict_f1(rows, n_gt)
    ceil_f1, ceil_p, ceil_r = _strict_f1(tp_comps, n_gt)
    kept = sum(c["size"] for c in tp_comps)
    hit = sum(c["gt_overlap"] for c in tp_comps)
    return {
        "off_gate_f1": off_f1,
        "off_gate_prec": off_p,
        "off_gate_rec": off_r,
        "oracle_f1": ceil_f1,
        "oracle_prec": ceil_p,
        "oracle_rec": ceil_r,
        "kept_purity": hit / max(kept, 1),
        "gt_coverage": hit / max(n_gt, 1),
        "headroom": ceil_f1 - off_f1,
    }


def _json_safe(obj):
    """NaN -> null, so the report stays valid JSON for any reader (not just Python's)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def probe_anchor(
    anc: str,
    *,
    model,
    kine,
    phys,
    bio,
    device,
    wall_hops: int,
    gate_pct: float,
    tie_tol: float,
) -> dict:
    print(f"\n{'=' * 74}\n=== {anc} ===\n{'=' * 74}", flush=True)
    reset_species_rollout_flow_cache()
    data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
    static = _load_static(data, device, kine, wall_hops, anc)
    static["n_times"] = int(data.y.shape[0])

    print("  rolling out (the expensive part, once)...", flush=True)
    series, data = deploy_species_rollout_series(
        model, data, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
    )
    phi_series, phi_gt, wall_mask, t_eval = deploy_clot_phi_trajectory(
        data, series, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
    )

    n_times = int(phi_series.shape[0])
    commit_t = first_commit_step(phi_series)
    pred = (phi_series[t_eval] > 0.5).cpu().numpy()
    gt = (phi_gt > 0.5).cpu().numpy()

    u0, v0 = resolve_species_rollout_uv(data, 0, device, for_training=False)
    h2 = hop2_speed_field(data, device, u0, v0).detach().cpu().numpy()
    wall = wall_mask.reshape(-1).bool().cpu().numpy() if wall_mask is not None else np.ones(pred.size, bool)
    gate_thresh = float(np.percentile(h2[wall], gate_pct))

    ei = data.edge_index.cpu().numpy()
    comps = predicted_components(pred, ei)
    print(f"  n_times={n_times}  t_eval={t_eval}  n_gt={int(gt.sum())}  n_pred={int(pred.sum())}"
          f"  ncomp={len(comps)}  flow gate pct={gate_pct:g} -> thresh={gate_thresh:.4f}")

    rows = []
    for nd in comps:
        rows.append({
            "size": int(nd.size),
            "is_tp": bool(gt[nd].any()),
            "gt_overlap": int(gt[nd].sum()),
            "h2min": float(h2[nd].min()),
            "t_first": int(commit_t[nd].min()),
            "t_median": float(np.median(commit_t[nd])),
            "survives_flow_gate": bool(float(h2[nd].min()) < gate_thresh),
        })
    rows.sort(key=lambda r: -r["size"])

    print(f"\n  {'size':>5} {'lbl':>4} {'gt_ov':>6} {'h2min':>8} {'t_first':>8} {'t_med':>7} {'gate':>6}")
    for r in rows[:20]:
        print(f"  {r['size']:5d} {'TP' if r['is_tp'] else 'FP':>4} {r['gt_overlap']:6d} "
              f"{r['h2min']:8.4f} {r['t_first']:8d} {r['t_median']:7.1f} "
              f"{'keep' if r['survives_flow_gate'] else 'drop':>6}")
    if len(rows) > 20:
        print(f"  ... {len(rows) - 20} smaller components omitted")

    ceil = selection_ceiling(rows, int(gt.sum()))
    print(f"\n  Selection ceiling (oracle = keep every GT-touching component):")
    print(f"    off-gate           F1={ceil['off_gate_f1']:.4f}  prec={ceil['off_gate_prec']:.3f} rec={ceil['off_gate_rec']:.3f}")
    print(f"    ORACLE selection   F1={ceil['oracle_f1']:.4f}  prec={ceil['oracle_prec']:.3f} rec={ceil['oracle_rec']:.3f}"
          f"   (headroom {ceil['headroom']:+.4f})")
    print(f"    kept-component purity {ceil['kept_purity']:.3f}   GT coverage {ceil['gt_coverage']:.3f}"
          f"   <- both cap the oracle; selection cannot move either")

    tp = [r for r in rows if r["is_tp"]]
    fp = [r for r in rows if not r["is_tp"]]
    out: dict = {
        "ceiling": ceil,
        "anchor": anc,
        "n_times": n_times,
        "t_eval": t_eval,
        "n_gt": int(gt.sum()),
        "n_pred": int(pred.sum()),
        "n_comp": len(rows),
        "n_tp_comp": len(tp),
        "n_fp_comp": len(fp),
        "gate_pct": gate_pct,
        "gate_thresh": gate_thresh,
        "tie_tol": tie_tol,
        "components": rows,
        "auc": {},
    }
    if not tp or not fp:
        print("\n  [skip] need at least one TP and one FP component to score an AUC.")
        return out

    tp_h2 = np.array([r["h2min"] for r in tp])
    fp_h2 = np.array([r["h2min"] for r in fp])
    tp_t = np.array([r["t_first"] for r in tp], dtype=float)
    fp_t = np.array([r["t_first"] for r in fp], dtype=float)
    tp_w = np.array([r["size"] for r in tp], dtype=float)
    fp_w = np.array([r["size"] for r in fp], dtype=float)

    ties = flow_tie_pairs(tp_h2, fp_h2, rel_tol=tie_tol)
    auc = {
        "h2min": rank_auc(tp_h2, fp_h2),
        "h2min_sizew": rank_auc(tp_h2, fp_h2, w_pos=tp_w, w_neg=fp_w),
        "commit_t": rank_auc(tp_t, fp_t),
        "commit_t_sizew": rank_auc(tp_t, fp_t, w_pos=tp_w, w_neg=fp_w),
        "commit_t_on_flow_ties": rank_auc(tp_t, fp_t, pair_mask=ties),
        "commit_t_on_flow_ties_sizew": rank_auc(tp_t, fp_t, w_pos=tp_w, w_neg=fp_w, pair_mask=ties),
        "n_tie_pairs": int(ties.sum()),
        "n_pairs": int(tp_h2.size * fp_h2.size),
    }

    # Does timing still rank once the flow gate has already removed what it can?
    stp = [r for r in tp if r["survives_flow_gate"]]
    sfp = [r for r in fp if r["survives_flow_gate"]]
    if stp and sfp:
        auc["commit_t_post_gate"] = rank_auc(
            [r["t_first"] for r in stp], [r["t_first"] for r in sfp],
        )
        auc["commit_t_post_gate_sizew"] = rank_auc(
            [r["t_first"] for r in stp], [r["t_first"] for r in sfp],
            w_pos=[r["size"] for r in stp], w_neg=[r["size"] for r in sfp],
        )
    else:
        auc["commit_t_post_gate"] = float("nan")
        auc["commit_t_post_gate_sizew"] = float("nan")
    auc["n_tp_comp_post_gate"] = len(stp)
    auc["n_fp_comp_post_gate"] = len(sfp)
    out["auc"] = auc

    print(f"\n  AUC(TP < FP)   1.0 = perfect, 0.5 = no information, <0.5 = inverted")
    print(f"    h2min (flow, the existing gate's signal)  {_fmt_auc(auc['h2min'])}"
          f"   size-weighted {_fmt_auc(auc['h2min_sizew'])}")
    print(f"    commit_t (all pairs)                      {_fmt_auc(auc['commit_t'])}"
          f"   size-weighted {_fmt_auc(auc['commit_t_sizew'])}")
    print(f"    commit_t on flow-TIED pairs  <- decides   {_fmt_auc(auc['commit_t_on_flow_ties'])}"
          f"   size-weighted {_fmt_auc(auc['commit_t_on_flow_ties_sizew'])}"
          f"   ({auc['n_tie_pairs']}/{auc['n_pairs']} pairs tied at rel_tol={tie_tol:g})")
    print(f"    commit_t among flow-gate survivors        {_fmt_auc(auc['commit_t_post_gate'])}"
          f"   size-weighted {_fmt_auc(auc['commit_t_post_gate_sizew'])}"
          f"   ({len(stp)} TP / {len(sfp)} FP survive)")

    # The specific s2.9 object: the biggest TP/FP pair flow cannot tell apart.
    if ties.any():
        best = max(
            ((i, j) for i in range(len(tp)) for j in range(len(fp)) if ties[i, j]),
            key=lambda ij: min(tp[ij[0]]["size"], fp[ij[1]]["size"]),
        )
        i, j = best
        a, b = tp[i], fp[j]
        print(f"\n  largest flow-tied pair (the s2.9 object):")
        print(f"    TRUE  size={a['size']:4d}  h2min={a['h2min']:.4f}  t_first={a['t_first']}")
        print(f"    FALSE size={b['size']:4d}  h2min={b['h2min']:.4f}  t_first={b['t_first']}")
        verdict = ("commit-time SPLITS it (true first)" if a["t_first"] < b["t_first"]
                   else "commit-time splits it BACKWARDS (false first)" if a["t_first"] > b["t_first"]
                   else "commit-time TIES it too")
        print(f"    -> {verdict}")
        out["largest_tied_pair"] = {"true": a, "false": b, "verdict": verdict}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Component-level commitment-order AUC (WALL_MODEL_PLAN s4 Step 1b)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--anchors", default="patient037,patient021",
                    help="Comma list. Default: the s2.9 pair -- 037 (flow-tied) and 021 (flow already clean).")
    ap.add_argument("--gate-pct", type=float, default=25.0,
                    help="Flow-gate percentile for the post-gate view (default 25, the s2.7 minimax)")
    ap.add_argument("--tie-tol", type=float, default=0.05,
                    help="Relative h2min gap below which a (TP, FP) pair counts as a flow tie (default 0.05)")
    ap.add_argument("--allow-holdout", action="store_true",
                    help="Permit running on patient020/043/044. Off by default -- s6 rule 2.")
    ap.add_argument("--out", default="outputs/biochem/eda/commit_order/probe.json")
    args = ap.parse_args()

    root = get_project_root()
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    blocked = [a for a in anchors if a in HOLDOUTS]
    if blocked and not args.allow_holdout:
        print(f"[ERR] {', '.join(blocked)} is a one-shot holdout (WALL_MODEL_PLAN s6 rule 2). "
              f"Probe on training vessels; pass --allow-holdout only if you mean to spend the shot.")
        return 2

    device = require_cuda_device()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {ckpt}")

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="commit_order_probe", ckpt_path=ckpt)
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        raise FileNotFoundError(f"could not load continuous bundle: {ckpt}")
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    report: dict = {"ckpt": str(ckpt), "gate_pct": args.gate_pct, "tie_tol": args.tie_tol, "per_anchor": {}}
    for anc in anchors:
        report["per_anchor"][anc] = probe_anchor(
            anc, model=model, kine=kine, phys=phys, bio=bio, device=device,
            wall_hops=wall_hops, gate_pct=args.gate_pct, tie_tol=args.tie_tol,
        )
        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'=' * 74}\n=== Selection ceiling across the cohort ===\n{'=' * 74}")
    print(f"  {'anchor':>14} {'n_gt':>5} {'off_f1':>7} {'oracle':>7} {'headrm':>7} {'purity':>7} {'gtcov':>7} {'h2AUC':>7}")
    for anc in anchors:
        d = report["per_anchor"][anc]
        c = d.get("ceiling") or {}
        a = d.get("auc") or {}
        print(f"  {anc:>14} {d.get('n_gt', 0):5d} {c.get('off_gate_f1', 0):7.4f} {c.get('oracle_f1', 0):7.4f} "
              f"{c.get('headroom', 0):+7.4f} {c.get('kept_purity', 0):7.3f} {c.get('gt_coverage', 0):7.3f} "
              f"{_fmt_auc(a.get('h2min', float('nan')))}")
    ceils = [report["per_anchor"][a].get("ceiling", {}).get("oracle_f1", 0.0) for a in anchors]
    if ceils:
        print(f"\n  oracle ceiling spread: {min(ceils):.3f} - {max(ceils):.3f}"
              f"   (patient020 measured 0.887 in s2.2 -- the plan treats that as THE ceiling)")
        print(f"  Low purity or low GT coverage means the vessel's headroom is capped by growth,")
        print(f"  not selection -- no pocket gate, however good, can reach past it.")

    print(f"\n{'=' * 74}\n=== Step 1b decision rule (s4) ===\n{'=' * 74}")
    print(f"  {'anchor':>14} {'commit_t':>9} {'on ties':>9} {'post-gate':>10} {'h2min':>8}")
    tie_aucs = []
    for anc in anchors:
        a = report["per_anchor"][anc].get("auc") or {}
        tie = a.get("commit_t_on_flow_ties", float("nan"))
        tie_aucs.append(tie)
        print(f"  {anc:>14} {_fmt_auc(a.get('commit_t', float('nan')))} {_fmt_auc(tie)} "
              f"{_fmt_auc(a.get('commit_t_post_gate', float('nan')))} {_fmt_auc(a.get('h2min', float('nan')))}")

    scored = [v for v in tie_aucs if np.isfinite(v)]
    if not scored:
        call = "INCONCLUSIVE -- no flow-tied TP/FP pairs to score. Widen --tie-tol or add anchors."
    elif all(v >= 0.70 for v in scored):
        call = ("SEPARATES (earlier commit = true) on every anchor scored -> per s4 Step 1b, combine "
                "with the flow gate (flow first, commit-time as tiebreak) and re-run the Step 1 "
                "percentile sweep on training vessels before touching the holdout.")
    elif all(v <= 0.30 for v in scored):
        call = ("SEPARATES BUT INVERTED (later commit = true) on every anchor. Consistent direction, "
                "opposite to the s2.10 hypothesis -- usable, but the direction is now fitted on train; "
                "confirm on more training vessels before trusting it.")
    else:
        call = ("NO CONSISTENT SEPARATION -> per s4 Step 1b, the mat_seed_prec=1.000 guarantee does not "
                "extend to ranking concurrent pockets. Drop this direction, accept the flow-only minimax "
                "gate as the practical ceiling, and spend the one-shot holdout application on it.")
    print(f"\n  => {call}")
    report["verdict"] = call

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
