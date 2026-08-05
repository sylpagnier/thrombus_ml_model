"""Growth-arrest probe (docs/WALL_MODEL_PLAN.md s9.10).

v2's fine-tune (WG_stenosis_subcohort_ft_v2) found the model can reach a genuinely better
state mid-rollout on patient043 -- deploy_clot_f1=0.732 at t=130/200 -- and then overshoots
it by t_final (F1 0.499 at mass 2.75). GT itself is essentially saturated by t~100-130 (89 of
95 final nodes); the model keeps depositing clot for another ~70 steps with nothing to stop
it (docs/WALL_MODEL_PLAN.md s7 bugs 1-2: the closed-loop arrest mechanism is sign-inverted
and its mask is ~97.6% phantom -- there is no working "stop growing" signal).

This asks whether that's specific to patient043 / the fine-tuned checkpoint, or a property of
the zero-shot warm-start (WG_clotrich_nplus) across the whole s9.4 cohort. One rollout per
anchor (the expensive part, same as probe_commit_order.py); GT and predicted phi at every
other grading fraction come free from data already computed by that one rollout --
deploy_clot_phi_trajectory already returns the full [T,N] predicted phi series, and GT at an
arbitrary t is a cheap lookup (gt_clot_phi_at_time), not a re-rollout. No training.

    python scripts/probe_growth_arrest.py --anchors patient039,patient040,patient041,patient042,patient043

Reads as an "arrest ratio": (model mass added in the back half of the horizon) / (GT mass
added in the back half). ~1 = the model stops growing in step with GT. >>1 = the model keeps
depositing well past where GT has stopped -- non-arrest, the same failure v2 hit on
patient043. This determines whether v3 (s9.9 -> s9.11) needs an arrest mechanism cohort-wide,
or whether patient043 was unusual.
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
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_clot_phi_trajectory,
    deploy_species_rollout_series,
    load_continuous_bundle,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.pocket_gate import apply_pocket_gate  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_CKPT = "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth"
DEFAULT_FRACS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
# One-shot holdouts (WALL_MODEL_PLAN.md s6 rule 2 / s9.2) -- this probe uses no GT-derived
# training signal (read-only diagnostics), so it does not spend the holdout shot, but keep the
# same guard as probe_commit_order.py for consistency: an explicit flag is required.
HOLDOUTS = {"patient020", "patient043", "patient044"}


def _fmt(v: float, w: int = 8) -> str:
    return f"{v:{w}.3f}" if np.isfinite(v) else f"{'n/a':>{w}}"


def probe_anchor(
    anc: str,
    *,
    model,
    kine,
    phys,
    bio,
    device,
    wall_hops: int,
    gate_pct: float | None,
    fracs: list[float],
    back_half_frac: float,
) -> dict:
    print(f"\n{'=' * 78}\n=== {anc} ===\n{'=' * 78}", flush=True)
    reset_species_rollout_flow_cache()
    data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
    static = _load_static(data, device, kine, wall_hops, anc)
    static["n_times"] = int(data.y.shape[0])

    print("  rolling out (the expensive part, once)...", flush=True)
    series, data = deploy_species_rollout_series(
        model, data, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
    )
    # One physics-trigger pass over the whole trajectory -- phi_series[t] for every t is free
    # from here on; only gt_clot_phi_at_time (a cheap lookup, no model call) is needed per point.
    phi_series, _phi_gt_final, wall_mask, t_eval = deploy_clot_phi_trajectory(
        data, series, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
    )
    wall = wall_mask.reshape(-1).bool() if wall_mask is not None else None

    rows = []
    for frac in fracs:
        t = max(0, min(int(round(frac * t_eval)), t_eval))
        phi_pred = phi_series[t].reshape(-1)
        if gate_pct is not None:
            phi_pred, _stats = apply_pocket_gate(phi_pred, data, device, percentile=gate_pct, wall_mask=wall_mask)
        pred = (phi_pred > 0.5)
        phi_gt_t = gt_clot_phi_at_time(data, t, phys, device=device)
        gt = (phi_gt_t.reshape(-1) > 0.5)
        if wall is not None:
            pred = pred & wall
            gt = gt & wall
        n_pred = int(pred.sum().item())
        n_gt = int(gt.sum().item())
        tp = int((pred & gt).sum().item())
        fp = n_pred - tp
        fn = n_gt - tp
        prec = tp / max(n_pred, 1)
        rec = tp / max(n_gt, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mass = n_pred / max(n_gt, 1)
        rows.append({
            "frac": frac, "t": t, "n_gt": n_gt, "n_pred": n_pred,
            "tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1, "mass": mass,
        })

    print(f"\n  {'frac':>5} {'t':>4} {'n_gt':>5} {'n_pred':>6} {'mass':>7} {'f1':>7} {'prec':>7} {'rec':>7} {'fp':>5} {'fn':>5}")
    for r in rows:
        print(f"  {r['frac']:5.2f} {r['t']:4d} {r['n_gt']:5d} {r['n_pred']:6d} "
              f"{r['mass']:7.3f} {r['f1']:7.4f} {r['prec']:7.4f} {r['rec']:7.4f} {r['fp']:5d} {r['fn']:5d}")

    # Arrest ratio: model growth vs GT growth over the back (1 - back_half_frac) of the horizon.
    mid_idx = max(0, int(round(len(rows) * (1.0 - back_half_frac))) - 1)
    mid_idx = min(mid_idx, len(rows) - 2)
    r_mid, r_end = rows[mid_idx], rows[-1]
    d_pred = r_end["n_pred"] - r_mid["n_pred"]
    d_gt = r_end["n_gt"] - r_mid["n_gt"]
    arrest_ratio = d_pred / max(d_gt, 1) if d_gt > 0 else (float("inf") if d_pred > 0 else 1.0)
    best_f1 = max(rows, key=lambda r: r["f1"])
    print(f"\n  back-half (t={r_mid['t']}..{r_end['t']}): GT +{d_gt} nodes, model +{d_pred} nodes"
          f"  -> arrest_ratio={arrest_ratio if np.isfinite(arrest_ratio) else float('inf'):.2f}"
          f"  ({'ARRESTS' if arrest_ratio <= 2.0 else 'DOES NOT ARREST'})")
    print(f"  best F1 anywhere on the horizon: {best_f1['f1']:.4f} at frac={best_f1['frac']:.2f} (t={best_f1['t']})"
          f"  vs at t_final: {r_end['f1']:.4f}")

    return {
        "anchor": anc, "t_eval": t_eval, "gate_pct": gate_pct,
        "curve": rows, "back_half_mid_t": r_mid["t"], "d_gt": d_gt, "d_pred": d_pred,
        "arrest_ratio": None if not np.isfinite(arrest_ratio) else arrest_ratio,
        "best_f1": best_f1, "final_f1": r_end["f1"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Growth-arrest probe across the horizon (WALL_MODEL_PLAN s9.10)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--anchors", default="patient039,patient040,patient041,patient042,patient043",
                    help="Comma list. Default: the s9.4 cohort survey vessels.")
    ap.add_argument("--gate-pct", type=float, default=25.0,
                    help="Pocket-gate percentile applied at every grading point (default 25, the s9 default). "
                         "Pass a negative number to disable the gate.")
    ap.add_argument("--fracs", default=DEFAULT_FRACS, help=f"Comma list of horizon fractions (default {DEFAULT_FRACS})")
    ap.add_argument("--back-half-frac", type=float, default=0.35,
                    help="Fraction of the horizon (from the end) used for the arrest-ratio comparison (default 0.35)")
    ap.add_argument("--allow-holdout", action="store_true", help="Permit patient020/043/044 (s6 rule 2 / s9.2)")
    ap.add_argument("--out", default="outputs/biochem/eda/growth_arrest/probe.json")
    args = ap.parse_args()

    root = get_project_root()
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    blocked = [a for a in anchors if a in HOLDOUTS]
    if blocked and not args.allow_holdout:
        print(f"[ERR] {', '.join(blocked)} is a one-shot holdout / sealed vessel. "
              f"Pass --allow-holdout if you mean to include it (read-only diagnostic, doesn't spend the shot).")
        return 2

    device = require_cuda_device()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {ckpt}")

    gate_pct = None if args.gate_pct < 0 else args.gate_pct
    fracs = [float(f) for f in args.fracs.split(",") if f.strip()]

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="growth_arrest_probe", ckpt_path=ckpt)
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

    report: dict = {"ckpt": str(ckpt), "gate_pct": gate_pct, "fracs": fracs, "per_anchor": {}}
    for anc in anchors:
        report["per_anchor"][anc] = probe_anchor(
            anc, model=model, kine=kine, phys=phys, bio=bio, device=device,
            wall_hops=wall_hops, gate_pct=gate_pct, fracs=fracs, back_half_frac=args.back_half_frac,
        )
        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\n=== Cohort summary ===\n{'=' * 78}")
    print(f"  {'anchor':>14} {'arrest_ratio':>13} {'verdict':>18} {'best_f1':>8} {'final_f1':>9} {'delta':>7}")
    ratios = []
    for anc in anchors:
        d = report["per_anchor"][anc]
        ar = d["arrest_ratio"]
        ratios.append(ar if ar is not None else float("inf"))
        verdict = "n/a" if ar is None else ("ARRESTS" if ar <= 2.0 else "DOES NOT ARREST")
        bf, ff = d["best_f1"]["f1"], d["final_f1"]
        ar_s = f"{ar:13.2f}" if ar is not None else f"{'inf':>13}"
        print(f"  {anc:>14} {ar_s} {verdict:>18} {bf:8.4f} {ff:9.4f} {bf - ff:+7.4f}")

    finite = [r for r in ratios if np.isfinite(r)]
    n_bad = sum(1 for r in ratios if r > 2.0 or not np.isfinite(r))
    print(f"\n  {n_bad}/{len(anchors)} vessels do not arrest (ratio > 2.0 or infinite).")
    if n_bad >= len(anchors) - 1:
        verdict = ("Cohort-wide: this checkpoint has no working arrest mechanism on this vessel family. "
                   "v3 needs a real stop-growing signal (differentiable mass/FP brakes), not just a "
                   "different loss ratio -- s7 bugs 1-2 explain why the physical arrest path is unavailable.")
    elif n_bad == 0:
        verdict = "Cohort-wide arrest holds -- patient043's v2 failure may be fine-tune-induced, not inherited from the warm-start."
    else:
        verdict = "Mixed -- arrest is vessel-dependent. Check which vessels fail before generalizing the v3 fix."
    print(f"\n  => {verdict}")
    report["verdict"] = verdict
    report["n_bad"] = n_bad

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def _json_safe(o):
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    out.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
