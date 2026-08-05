"""Can a global gelation beta fix wall-model precision? Answer from ONE rollout.

Motivation (docs/WALL_MODEL_PLAN.md s1). The graded clot label fires when the soft phi
readout crosses 0.5, which reduces to

    mu_c * (1 + beta * (mu_ratio_max - 1) * sigma) > sqrt(mu_solid * mu_ref)

where ``sigma = sigmoid((Mat_si - mat_crit) / temp)`` is the gelation fraction. Beta
enters *only* through the product ``beta * sigma``, so a global beta is a monotone
reparameterisation of the Mat decision boundary -- it slides one threshold along one
axis. Two things follow, and this script measures both:

1. **The beta curve is nearly free.** The GNN rollout is the expensive part; the readout
   is not. So roll out once and re-grade at every beta (``--betas``). That traces the
   exact F1 / precision / recall / mass curve with the closed loop held fixed at
   ``--rollout-beta``. Comparing one true closed-loop rollout at another beta (run this
   script twice with different ``--rollout-beta``) sizes the feedback term that this
   cheap sweep cannot see.

2. **Separability is the real question.** Beta can only help if the false positives sit
   at systematically lower ``sigma`` than the true positives. If the two distributions
   overlap, *no* global beta buys precision without shedding recall one-for-one, and the
   fix has to be conditioning or loss -- not calibration. The script reports the sigma
   distributions for TP / FP / FN separately plus the AUC of sigma as an FP-vs-TP
   discriminator. **AUC near 0.5 means the calibration hypothesis is dead**; the gain is
   not where the error is.

Deploy-faithful: reuses ``eval_mat_growth_simple``'s checkpoint recipe and the same
rollout + grading functions the scored metric uses, so numbers are comparable to
``deploy_clot_f1`` by construction.

    python scripts/diag_gelation_beta_margin.py --ckpt <ckpt> --anchors patient020
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _load_static  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    clot_phi_mu_solid_si,
    clot_phi_physics_mu_base_mode,
    clot_phi_physics_mu_ratio_max,
    mat_si_for_gelation_from_log1p,
)
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_clot_phi_fields,
    deploy_species_rollout_series,
    grade_deploy_clot_series,
    load_continuous_bundle,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils import species_channels as sc  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_BETAS = "0.20,0.30,0.40,0.50,0.60,0.69,0.80,1.00,1.30"


def _quantiles(v: torch.Tensor) -> dict[str, float]:
    if v.numel() == 0:
        return {"n": 0}
    qs = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=v.device, dtype=v.dtype)
    q = torch.quantile(v, qs)
    return {
        "n": int(v.numel()),
        "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
        "mean": float(v.mean()), "frac_saturated": float((v >= 0.99).float().mean()),
    }


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(sigma of a random TP > sigma of a random FP), ties at 0.5.

    Rank form (Mann-Whitney U), so it stays O(n log n) instead of materialising an
    |TP| x |FP| outer product on vessels with thousands of false positives.
    """
    n_p, n_n = int(pos.numel()), int(neg.numel())
    if n_p == 0 or n_n == 0:
        return float("nan")
    allv = torch.cat([pos.reshape(-1), neg.reshape(-1)]).double()
    order = torch.argsort(allv)
    ranks = torch.empty_like(allv)
    ranks[order] = torch.arange(1, allv.numel() + 1, device=allv.device, dtype=allv.dtype)
    # Average ranks within tied groups so ties score exactly 0.5.
    uniq, inv, counts = torch.unique(allv, return_inverse=True, return_counts=True)
    sums = torch.zeros_like(uniq).scatter_add_(0, inv, ranks)
    ranks = (sums / counts.to(ranks.dtype))[inv]
    r_pos = ranks[:n_p].sum()
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def _gelation_sigma(species_series: torch.Tensor, t_eval: int, bio: BiochemConfig) -> torch.Tensor:
    """Per-node gelation fraction sigma(z) that beta multiplies in the readout."""
    mat_log = species_series[t_eval][:, sc.y_index("Mat")]
    mat_si = mat_si_for_gelation_from_log1p(mat_log, bio)
    temp = max(float(bio.viscosity_gnode_temp_mat) * max(float(bio.soft_step_T_scale), 1e-5), 1e-8)
    z = ((mat_si.reshape(-1) - float(bio.viscosity_mat_crit)) / temp).clamp(-50.0, 50.0)
    return torch.sigmoid(z)


def main() -> int:
    ap = argparse.ArgumentParser(description="Beta-curve + FP/TP separability from one rollout")
    ap.add_argument("--ckpt", required=True, help="Wall-model checkpoint to diagnose")
    ap.add_argument("--anchors", required=True, help="Comma list, e.g. patient020")
    ap.add_argument("--betas", default=DEFAULT_BETAS, help=f"Re-grade grid (default {DEFAULT_BETAS})")
    ap.add_argument(
        "--rollout-beta",
        type=float,
        default=None,
        help="Beta used for the (single) closed-loop rollout itself. Default None = 1.0 = "
             "historical grading. Re-run with another value to size the feedback term.",
    )
    ap.add_argument("--out", default="outputs/biochem/eda/beta_margin/diag.json")
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {ckpt}")

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple", ckpt_path=ckpt)
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
    ratio = clot_phi_physics_mu_ratio_max(bio)

    print(
        f"[i] base_mode={clot_phi_physics_mu_base_mode()} mu_ratio_max={ratio} "
        f"mu_solid={clot_phi_mu_solid_si()} mat_crit={bio.viscosity_mat_crit:g} "
        f"temp={bio.viscosity_gnode_temp_mat:g}",
        flush=True,
    )
    print(f"[i] rollout beta = {args.rollout_beta if args.rollout_beta is not None else 1.0}", flush=True)

    report: dict = {
        "ckpt": str(ckpt),
        "rollout_beta": args.rollout_beta,
        "betas": betas,
        "mu_ratio_max": float(ratio),
        "base_mode": clot_phi_physics_mu_base_mode(),
        "per_anchor": {},
    }

    for anc in anchors:
        print(f"\n=== {anc} ===", flush=True)
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = _load_static(data, device, kine, wall_hops, anc)
        static["n_times"] = int(data.y.shape[0])

        print("  rolling out (this is the expensive part, once)...", flush=True)
        series, data = deploy_species_rollout_series(
            model, data, static, phys, bio, device,
            flow_source="kinematics", gelation_beta=args.rollout_beta,
        )

        # ---- exact beta curve, closed loop frozen at --rollout-beta ----
        curve = []
        print(f"  {'beta':>6} {'f1':>7} {'prec':>7} {'rec':>7} {'mass':>7} {'n_pred':>7} {'fp':>7} {'fn':>7}", flush=True)
        for b in betas:
            m = grade_deploy_clot_series(
                data, series, static, phys, bio, device,
                time_index=None, flow_source="kinematics", gelation_beta=b,
            )
            row = {
                "beta": b,
                "deploy_clot_f1": m.get("deploy_clot_f1", 0.0),
                "deploy_clot_prec": m.get("deploy_clot_prec", 0.0),
                "deploy_clot_rec": m.get("deploy_clot_rec", 0.0),
                "deploy_clot_mass_ratio": m.get("deploy_clot_mass_ratio", 0.0),
                "deploy_clot_fp": m.get("deploy_clot_fp", 0.0),
                "deploy_clot_fn": m.get("deploy_clot_fn", 0.0),
            }
            curve.append(row)
            print(
                f"  {b:6.2f} {row['deploy_clot_f1']:7.4f} {row['deploy_clot_prec']:7.4f} "
                f"{row['deploy_clot_rec']:7.4f} {row['deploy_clot_mass_ratio']:7.3f} "
                f"{'':>7} {row['deploy_clot_fp']:7.0f} {row['deploy_clot_fn']:7.0f}",
                flush=True,
            )

        # ---- separability: is sigma even a discriminator between FP and TP? ----
        ref_beta = args.rollout_beta
        phi_pred, phi_gt, wall_mask, t_eval = deploy_clot_phi_fields(
            data, series, static, phys, bio, device,
            time_index=None, flow_source="kinematics", gelation_beta=ref_beta,
        )
        sigma = _gelation_sigma(series, t_eval, bio).to(device=phi_pred.device)
        pred_pos = phi_pred > 0.5
        gt_pos = phi_gt > 0.5
        scope = wall_mask.bool() if wall_mask is not None else torch.ones_like(pred_pos)
        tp = pred_pos & gt_pos & scope
        fp = pred_pos & ~gt_pos & scope
        fn = ~pred_pos & gt_pos & scope

        groups = {"tp": sigma[tp], "fp": sigma[fp], "fn": sigma[fn]}
        stats = {k: _quantiles(v) for k, v in groups.items()}
        auc = _auc(groups["tp"], groups["fp"])

        print(f"\n  t_eval={t_eval}  gt={int(gt_pos.sum())}  pred={int(pred_pos.sum())}  "
              f"tp={int(tp.sum())} fp={int(fp.sum())} fn={int(fn.sum())}", flush=True)
        print(f"  {'group':>5} {'n':>6} {'p05':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7} {'sat>=.99':>9}", flush=True)
        for k in ("tp", "fp", "fn"):
            s = stats[k]
            if not s.get("n"):
                print(f"  {k:>5} {0:6d}      --", flush=True)
                continue
            print(
                f"  {k:>5} {s['n']:6d} {s['p05']:7.4f} {s['p25']:7.4f} {s['p50']:7.4f} "
                f"{s['p75']:7.4f} {s['p95']:7.4f} {s['frac_saturated']:9.3f}",
                flush=True,
            )
        print(f"\n  AUC(sigma: TP vs FP) = {auc:.4f}", flush=True)
        if auc == auc:  # not nan
            if auc < 0.60:
                verdict = ("sigma barely separates FP from TP -- a global beta cannot buy "
                           "precision without shedding recall. Calibration is NOT the bottleneck.")
            elif auc < 0.75:
                verdict = ("weak separation -- expect beta to trade recall for precision at "
                           "roughly 1:1. Check the F1 curve above for a real optimum.")
            else:
                verdict = ("sigma separates FP from TP -- a global beta can genuinely raise "
                           "precision. The calibration hypothesis survives.")
            print(f"  => {verdict}", flush=True)
        else:
            verdict = "degenerate (empty TP or FP set)"

        report["per_anchor"][anc] = {
            "t_eval": int(t_eval),
            "n_gt": int(gt_pos.sum()),
            "n_pred": int(pred_pos.sum()),
            "n_tp": int(tp.sum()), "n_fp": int(fp.sum()), "n_fn": int(fn.sum()),
            "beta_curve": curve,
            "sigma_stats": stats,
            "auc_sigma_tp_vs_fp": auc,
            "verdict": verdict,
        }

        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
