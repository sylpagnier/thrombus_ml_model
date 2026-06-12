"""Pre-sweep T0 Rung4 diagnostics: oracle ceilings, rank barrier, s0 FN/FP, gelation oracle.

Run once before the arch sweep to set expectations for morning triage.

Usage::

    python scripts/diagnose_t0_r4_sweep_preflight.py --anchor patient007
    python scripts/diagnose_t0_r4_sweep_preflight.py --anchor patient007 --out outputs/biochem/sweep_t0_r4_arch_6h/preflight.json
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
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_r4_s2_species import (  # noqa: E402
    _risk_n_at_time,
    _s0_gate_from_species,
)
from src.core_physics.t0_rung4_ladder import (  # noqa: E402
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    _s0_onset_factor,
    _s0_spatial_weight,
    eval_rung4_step_clot,
    rollout_rung4_species_series,
    rung4_use_dgamma_wall_seed,
    species_log_mae_in_mask,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.training.train_t0_r4_s2_species import _fn_fp_masks  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

# Reuse oracle rollouts from s3 arch diagnostic
from scripts.diagnose_t0_r4_s3_arch import (  # noqa: E402
    _f1_at_t,
    _rank_rescue_stats,
    _rollout_gate_oracle,
    _s0_context,
)


def _parse_times(raw: str, n_steps: int) -> list[int]:
    if not raw.strip():
        return [0, 7, 15, 22, 27, 40, n_steps - 1]
    return sorted({max(0, min(n_steps - 1, int(x.strip()))) for x in raw.split(",") if x.strip()})


@torch.no_grad()
def _s0_species_timeline(data, phys, bio, device, times: list[int]) -> list[dict]:
    pred = rollout_rung4_species_series(data, phys, bio, device, step="s0")
    rows: list[dict] = []
    commits_prev = None
    for t in times:
        if t > 0:
            for ti in range(t):
                with t0_rung2_env():
                    phi, _ = predict_clot_phi_at_time(
                        data, ti, phys, bio, device,
                        gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
                    )
                commits_prev = (phi.reshape(-1) >= 0.5).bool()
        elig = resolve_nucleation_eligibility(
            data, t, device, phys, bio, commits_prev=commits_prev, growth_seed="pred",
            use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
        ).reshape(-1).bool()
        sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
        sp_p = pred[t, :, 4:16]
        mae = species_log_mae_in_mask(pred, data, t, elig, device)
        phi_gt = gt_clot_phi_at_time(data, t, phys, device).reshape(-1)
        s0_sp = _build_s0_deploy_species(
            data, t, device, bio, elig=elig, commits_prev=commits_prev,
        )
        gate = _s0_gate_from_species(s0_sp, data, device, bio, elig)
        fn, fp = _fn_fp_masks(s0_sp, sp_gt, phi_gt, elig, gate)
        with t0_rung2_env():
            phi_p, _ = predict_clot_phi_at_time(
                data, t, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
            )
        cm = _clot_metrics(phi_p.reshape(-1).float(), phi_gt, torch.ones_like(phi_gt, dtype=torch.bool))
        rows.append({
            "time": int(t),
            "n_elig": int(elig.sum().item()),
            "n_fn": int(fn.sum().item()),
            "n_fp": int(fp.sum().item()),
            "fi_log_mae_nuc": float(mae["fi_log_mae"]),
            "mat_log_mae_nuc": float(mae["mat_log_mae"]),
            "clot_f1": float(cm["clot_f1"]),
            "pred_pos_frac": float(cm["pred_pos_frac"]),
        })
    return rows


def run_preflight(anchor: str, *, times: list[int], loc_scale: float = 0.75) -> dict:
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
    data = torch.load(graph, map_location=device, weights_only=False)
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    t_last = int(data.y.shape[0]) - 1
    if t_last not in times:
        times = sorted(set(times + [t_last]))

    ctx = _s0_context(data, t_last, phys, bio, device)
    rank = _rank_rescue_stats(ctx["risk_n"], ctx["elig"], ctx["fn"], loc_scale)

    oracle: dict[str, float] = {}
    for name, fn in [
        ("s0", lambda: rollout_rung4_species_series(data, phys, bio, device, step="s0")),
        ("risk_max_fn", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="risk_max_fn", loc_scale=loc_scale)),
        ("gate_fn_only", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="gate_fn_only", loc_scale=loc_scale)),
        ("gate_fn_fp", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="gate_fn_fp", loc_scale=loc_scale)),
        ("species_fn_fp", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="species_fn_fp")),
    ]:
        pred = fn()
        oracle[name] = _f1_at_t(data, t_last, phys, bio, device, pred)

    fn_sets: list[set[int]] = []
    fn_timeline: list[dict] = []
    for t in times:
        c = _s0_context(data, t, phys, bio, device)
        ids = set(torch.where(c["fn"])[0].tolist())
        fn_sets.append(ids)
        fn_timeline.append({
            "time": int(t),
            "n_fn": len(ids),
            "n_fp": int(c["fp"].sum().item()),
            "n_elig": int(c["elig"].sum().item()),
            "onset": float(c["onset"]),
        })
    fn_persistent = len(set.intersection(*fn_sets)) if fn_sets else 0
    fn_union = len(set.union(*fn_sets)) if fn_sets else 0

    s0_timeline = _s0_species_timeline(data, phys, bio, device, times)

    deploy_eval = eval_rung4_step_clot(data, phys, bio, device, step="s0", times=times)
    deploy_final = next(r for r in deploy_eval["clot"] if r["time"] == t_last)

    gelation_note = (
        "Run scripts/diagnose_t0_carreau_gelation.py for full Carreau x gelation oracle; "
        "species_fn_fp ceiling here assumes GT FI/Mat patch in E(t)."
    )

    return {
        "anchor": anchor,
        "device": str(device),
        "t_last": t_last,
        "rank_barrier": rank,
        "oracle_f1_final_t": oracle,
        "fn_persistence": {
            "persistent_fn_all_times": fn_persistent,
            "union_fn": fn_union,
            "timeline": fn_timeline,
        },
        "s0_species_timeline": s0_timeline,
        "deploy_phi_eval": {
            "clot": deploy_eval["clot"],
            "rollout_health": deploy_eval["rollout_health"],
        },
        "gelation_note": gelation_note,
        "targets": {
            "s0_baseline_f1": float(deploy_final["clot_f1"]),
            "s0_species_only_f1": oracle.get("s0", 0.0),
            "gate_oracle_ceiling_f1": oracle.get("gate_fn_fp", 0.0),
            "species_oracle_ceiling_f1": oracle.get("species_fn_fp", 0.0),
            "beat_s0_min_f1": float(deploy_final["clot_f1"]) + 0.01,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 Rung4 sweep preflight diagnostics")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,7,15,22,27,40,53")
    ap.add_argument("--loc-scale", type=float, default=0.75)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph, map_location="cpu", weights_only=False)
    times = _parse_times(args.times, int(data.y.shape[0]))

    report = run_preflight(args.anchor, times=times, loc_scale=float(args.loc_scale))
    out = Path(args.out) if args.out.strip() else (
        root / "outputs/biochem/sweep_t0_r4_arch_6h/preflight.json"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] preflight -> {out}", flush=True)
    o = report["oracle_f1_final_t"]
    dep = report["deploy_phi_eval"]["clot"]
    dep_last = next(r for r in dep if r["time"] == report["t_last"])
    print(
        f"[i] deploy phi F1 @ t={report['t_last']}: {dep_last['clot_f1']:.3f} "
        f"(species-only s0={o['s0']:.3f})",
        flush=True,
    )
    print(
        f"[i] oracle F1 @ t={report['t_last']}: gate_fn_fp={o['gate_fn_fp']:.3f} "
        f"species_fn_fp={o['species_fn_fp']:.3f}",
        flush=True,
    )
    rb = report["rank_barrier"]
    print(
        f"[i] rank barrier: fn={rb['n_fn']} fn_rescued_by_max_boost={rb['n_fn_rescued_by_max_boost']} "
        f"/ hot={rb['n_hot_top_frac']}",
        flush=True,
    )
    print(
        f"[i] FN persistence: {report['fn_persistence']['persistent_fn_all_times']} "
        f"/ union {report['fn_persistence']['union_fn']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
