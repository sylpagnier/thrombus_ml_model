"""Per-leg T0 Rung4 sweep diagnostics: species, gate flips, commits vs s0/GT.

Usage::

    python scripts/diagnose_t0_r4_sweep_leg.py --leg-dir outputs/biochem/sweep_t0_r4_arch_6h/s4_delta_gnn
    python scripts/diagnose_t0_r4_sweep_leg.py --anchor patient007 --ckpt path/to/best.pth
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

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_r4_s2_species import _s0_gate_from_species  # noqa: E402
from src.core_physics.t0_r4_sweep import load_sweep_bundle, rollout_sweep_species_series  # noqa: E402
from src.core_physics.t0_rung4_ladder import (  # noqa: E402
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    rollout_rung4_species_series,
    species_log_mae_in_mask,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.training.train_t0_r4_s2_species import _fn_fp_masks  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _parse_times(raw: str, n_steps: int) -> list[int]:
    if not raw.strip():
        return [0, 7, 15, 22, 27, 40, n_steps - 1]
    return sorted({max(0, min(n_steps - 1, int(x.strip()))) for x in raw.split(",") if x.strip()})


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return float(inter / union) if union > 0 else 1.0


@torch.no_grad()
def diagnose_leg(
    anchor: str,
    bundle,
    *,
    times: list[int],
    device: torch.device,
) -> dict:
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
    data = torch.load(graph, map_location=device, weights_only=False)
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    t_last = int(data.y.shape[0]) - 1
    if t_last not in times:
        times = sorted(set(times + [t_last]))

    pred_s0 = rollout_rung4_species_series(data, phys, bio, device, step="s0")
    pred = rollout_sweep_species_series(data, phys, bio, device, bundle)

    max_sp_diff = float((pred[:, :, 4:16] - pred_s0[:, :, 4:16]).abs().max().item())
    mean_fi_diff_tlast = float(
        (pred[t_last, :, FI_SLICE_IDX] - pred_s0[t_last, :, FI_SLICE_IDX]).abs().mean().item()
    )

    timeline: list[dict] = []
    commits_s0_prev = None
    commits_prev = None

    for t in times:
        if t > 0:
            for ti in range(t):
                with t0_rung2_env():
                    phi_s, _ = predict_clot_phi_at_time(
                        data, ti, phys, bio, device,
                        gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_s0,
                    )
                    phi_p, _ = predict_clot_phi_at_time(
                        data, ti, phys, bio, device,
                        gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
                    )
                commits_s0_prev = (phi_s.reshape(-1) >= 0.5).bool()
                commits_prev = (phi_p.reshape(-1) >= 0.5).bool()

        elig = resolve_nucleation_eligibility(
            data, t, device, phys, bio, commits_prev=commits_prev, growth_seed="pred",
        ).reshape(-1).bool()
        sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
        phi_gt = gt_clot_phi_at_time(data, t, phys, device).reshape(-1)
        gt_clot = phi_gt >= 0.5

        s0_sp = _build_s0_deploy_species(
            data, t, device, bio, elig=elig, commits_prev=commits_s0_prev,
        )
        gate_s0 = _s0_gate_from_species(s0_sp, data, device, bio, elig)
        fn, fp = _fn_fp_masks(s0_sp, sp_gt, phi_gt, elig, gate_s0)

        mae_leg = species_log_mae_in_mask(pred, data, t, elig, device)
        mae_s0 = species_log_mae_in_mask(pred_s0, data, t, elig, device)
        sp_leg = pred[t, :, 4:16]
        sp_s0_t = pred_s0[t, :, 4:16]
        e = elig.bool()
        fi_delta_vs_s0 = float((sp_leg[e, FI_SLICE_IDX] - sp_s0_t[e, FI_SLICE_IDX]).abs().mean().item()) if e.any() else 0.0
        mat_delta_vs_s0 = float((sp_leg[e, MAT_SLICE_IDX] - sp_s0_t[e, MAT_SLICE_IDX]).abs().mean().item()) if e.any() else 0.0

        with t0_rung2_env():
            phi_leg, _ = predict_clot_phi_at_time(
                data, t, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
            )
            phi_s0_t, _ = predict_clot_phi_at_time(
                data, t, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_s0,
            )
        commit_leg = phi_leg.reshape(-1) >= 0.5
        commit_s0 = phi_s0_t.reshape(-1) >= 0.5
        cm_leg = _clot_metrics(commit_leg.float(), phi_gt, torch.ones_like(phi_gt, dtype=torch.bool))
        cm_s0 = _clot_metrics(commit_s0.float(), phi_gt, torch.ones_like(phi_gt, dtype=torch.bool))

        hot_s0 = gate_s0.reshape(-1) > 0.25
        fn_fixed = fn & ~commit_s0 & commit_leg
        fp_fixed = fp & commit_s0 & ~commit_leg
        fn_regress = fn & commit_s0 & ~commit_leg
        fp_regress = fp & ~commit_s0 & commit_leg

        timeline.append({
            "time": int(t),
            "n_elig": int(e.sum().item()),
            "n_fn": int(fn.sum().item()),
            "n_fp": int(fp.sum().item()),
            "species": {
                "fi_log_mae": float(mae_leg["fi_log_mae"]),
                "mat_log_mae": float(mae_leg["mat_log_mae"]),
                "fi_log_mae_s0": float(mae_s0["fi_log_mae"]),
                "mat_log_mae_s0": float(mae_s0["mat_log_mae"]),
                "fi_improve_vs_s0": float(mae_s0["fi_log_mae"]) - float(mae_leg["fi_log_mae"]),
                "mat_improve_vs_s0": float(mae_s0["mat_log_mae"]) - float(mae_leg["mat_log_mae"]),
                "mean_abs_fi_delta_vs_s0": fi_delta_vs_s0,
                "mean_abs_mat_delta_vs_s0": mat_delta_vs_s0,
            },
            "clot": {
                "f1": float(cm_leg["clot_f1"]),
                "f1_s0": float(cm_s0["clot_f1"]),
                "delta_f1_vs_s0": float(cm_leg["clot_f1"]) - float(cm_s0["clot_f1"]),
                "pred_pos_frac": float(cm_leg["pred_pos_frac"]),
                "pred_pos_frac_s0": float(cm_s0["pred_pos_frac"]),
                "commit_jaccard_vs_s0": _jaccard(commit_leg, commit_s0),
                "commit_jaccard_vs_gt": _jaccard(commit_leg, gt_clot),
            },
            "localization": {
                "fn_fixed": int(fn_fixed.sum().item()),
                "fp_fixed": int(fp_fixed.sum().item()),
                "fn_regress": int(fn_regress.sum().item()),
                "fp_regress": int(fp_regress.sum().item()),
                "n_s0_hotspots": int(hot_s0.sum().item()),
            },
        })

    final = timeline[-1]
    return {
        "anchor": anchor,
        "recipe_id": bundle.recipe.id,
        "recipe_family": bundle.recipe.family,
        "hypothesis": bundle.recipe.hypothesis,
        "vs_s0": {
            "max_species_abs_diff": max_sp_diff,
            "mean_fi_abs_diff_tlast": mean_fi_diff_tlast,
            "identical_to_s0": max_sp_diff < 1e-5,
            "final_f1_delta": final["clot"]["delta_f1_vs_s0"],
            "final_fi_mae_improve": final["species"]["fi_improve_vs_s0"],
            "final_mat_mae_improve": final["species"]["mat_improve_vs_s0"],
        },
        "timeline": timeline,
        "verdict": _verdict(final, max_sp_diff),
    }


def _verdict(final_row: dict, max_sp_diff: float) -> str:
    d_f1 = float(final_row["clot"]["delta_f1_vs_s0"])
    fi_imp = float(final_row["species"]["fi_improve_vs_s0"])
    fn_fix = int(final_row["localization"]["fn_fixed"])
    fp_fix = int(final_row["localization"]["fp_fixed"])
    if max_sp_diff < 1e-5 and abs(d_f1) < 1e-4:
        return "inert (=s0): species and commits unchanged"
    if d_f1 > 0.02 and fn_fix > fp_fix:
        return "promising: F1 gain with net FN fix"
    if d_f1 > 0.01:
        return "marginal F1 gain: check health / viz"
    if fi_imp > 1e-4 and d_f1 <= 0:
        return "species moved but commits stuck (gelation decouple)"
    if fp_fix == 0 and int(final_row["localization"]["fp_regress"]) > 0:
        return "FP regression: over-seeding risk"
    return "no clear gain vs s0"


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 Rung4 sweep leg diagnostics")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,7,15,22,27,40,53")
    ap.add_argument("--leg-dir", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()

    if args.ckpt.strip():
        ckpt = Path(args.ckpt)
        leg_dir = ckpt.parent
    else:
        leg_dir = Path(args.leg_dir)
        ckpt = leg_dir / "best.pth"
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not leg_dir.is_absolute():
        leg_dir = root / leg_dir

    bundle = load_sweep_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] missing checkpoint: {ckpt}")

    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph, map_location="cpu", weights_only=False)
    times = _parse_times(args.times, int(data.y.shape[0]))

    report = diagnose_leg(args.anchor, bundle, times=times, device=device)
    out = Path(args.out) if args.out.strip() else leg_dir / f"diagnostic_{args.anchor}.json"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    vs = report["vs_s0"]
    print(f"[OK] diagnostic -> {out}", flush=True)
    print(
        f"[i] {report['recipe_id']}: F1 delta={vs['final_f1_delta']:+.3f} "
        f"fi_mae_improve={vs['final_fi_mae_improve']:.2e} "
        f"max_sp_diff={vs['max_species_abs_diff']:.2e} identical_s0={vs['identical_to_s0']}",
        flush=True,
    )
    print(f"[i] verdict: {report['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
