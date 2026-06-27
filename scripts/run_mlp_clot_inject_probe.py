"""A/B/C mu coupling probe on biochem anchors.

A = baseline teacher rollout (Lane A ckpt, no inject, no neighbor_wall gate)
B = Leg B v2: full MLP mu map in neighbor_wall + Carreau bulk (closed-loop DEQ)
C = neighbor_wall mask-only: cap_low_shear Carreau bulk + GNODE mu head on gt_clot mask

North-star scorecard: ``clot_shape`` (location-weighted F1 on full-mesh binary clots
from rollout channel-3 mu). ``flow_score`` / ``clot_recall`` / ``flow_ok`` listed
separately. See ``src/evaluation/clot_shape_score.py``.

Writes JSON to outputs/biochem/mlp_clot_inject_probe/abc_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.architecture.gnode_biochem import (
    apply_biochem_forward_policy_from_checkpoint_meta,
    restore_mlp_clot_inject_shell_env,
    snapshot_mlp_clot_inject_shell_env,
)
from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.inference.biochem_teacher_loader import build_biochem_teacher, resolve_rollout_mu_ratio_max
from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.anchor_mask import anchor_node_mask
from src.utils.metrics import rel_l2_uvp
from src.utils.nondim import to_t_nd
from src.utils.paths import get_project_root


def _apply_probe_speed_env(*, fast: bool) -> None:
    if not fast:
        return
    os.environ["BIOCHEM_ADJOINT_RK4_SUBSTEPS"] = "1"
    os.environ.setdefault("BIOCHEM_ODEINT_USE_ADJOINT", "1")


def _load_teacher(ckpt_path: Path, device: torch.device, mu_ratio_max: float, *, fast: bool = False):
    _apply_probe_speed_env(fast=fast)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    inject_shell = snapshot_mlp_clot_inject_shell_env()
    apply_biochem_forward_policy_from_checkpoint_meta(raw, quiet=True)
    restore_mlp_clot_inject_shell_env(inject_shell)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    teacher = build_biochem_teacher(
        raw,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        mu_ratio_max=mu_ratio_max,
        quiet=True,
    )
    if fast:
        teacher.max_inner_iters = max(3, min(int(getattr(teacher, "max_inner_iters", 10)), 4))
    return teacher, phys, bio


def _eval_times_for_rollout(
    t_si: torch.Tensor,
    t_ref: float,
    *,
    time_stride: int,
    fast: bool,
) -> torch.Tensor:
    """Build ND evaluation grid (fewer steps in --fast smoke mode)."""
    if fast:
        t_last = float(t_si[-1].item())
        t_mid = 0.5 * t_last
        times_si = sorted({0.0, t_mid, t_last})
        return to_t_nd(torch.tensor(times_si, device=t_si.device, dtype=t_si.dtype), t_ref)
    eval_t = to_t_nd(t_si, t_ref)
    stride = max(1, int(time_stride))
    if stride > 1 and eval_t.numel() > 2:
        idxs = list(range(0, int(eval_t.numel()), stride))
        if idxs[-1] != int(eval_t.numel()) - 1:
            idxs.append(int(eval_t.numel()) - 1)
        eval_t = eval_t[idxs]
    return eval_t


def _normalize_probe_leg(raw: str) -> str:
    u = raw.strip().upper().replace("-", "_")
    if u == "A":
        return "A"
    if u == "B":
        return "B"
    if u == "C":
        return "C"
    if u in ("B_DEPLOY", "BD", "B_DEP"):
        return "B_deploy"
    if u in ("B_WIRED", "BW", "WIRED"):
        return "B_wired"
    if u in ("B_SEED_GROWTH", "B_SEED", "SEED_GROWTH"):
        return "B_seed_growth"
    return raw.strip()


def _configure_leg(teacher, device, leg: str, *, clot_ckpt: Path, leg_b_v1: bool = False) -> None:
    """Set env + injector for one A/B/C leg."""
    os.environ["BIOCHEM_MLP_CLOT_INJECT"] = "0"
    os.environ["BIOCHEM_MLP_MU_MAP"] = "0"
    os.environ["BIOCHEM_MU_NEIGHBOR_WALL_ONLY"] = "0"
    teacher.clear_clot_phi_injector()

    if leg == "A":
        return
    if leg == "B":
        if leg_b_v1:
            os.environ["BIOCHEM_MLP_CLOT_INJECT"] = "1"
            os.environ["BIOCHEM_MLP_CLOT_REGION"] = "neighbor_wall"
        else:
            os.environ["BIOCHEM_MLP_MU_MAP"] = "1"
            os.environ["BIOCHEM_MLP_MU_MAP_PHI_GATE"] = "1"
            os.environ["BIOCHEM_MLP_MU_MAP_MASK"] = "gt_clot"
            os.environ["BIOCHEM_MLP_MU_MAP_BULK"] = "cap_low_shear"
            os.environ["BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND"] = "0.01"
            os.environ["BIOCHEM_MLP_MU_MAP_GEO_CAP"] = "0"
            os.environ.pop("BIOCHEM_MLP_CLOT_REGION", None)
            os.environ.pop("BIOCHEM_MLP_NEIGHBOR_SEED", None)
            os.environ.pop("BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI", None)
        attach_clot_phi_injector_to_teacher(teacher, device, str(clot_ckpt))
        return
    if leg in ("B_deploy", "Bd"):
        from src.inference.deploy_mu_map_env import apply_deploy_mu_map_env, clear_oracle_mu_map_env

        clear_oracle_mu_map_env()
        if (os.environ.get("BIOCHEM_MLP_MU_MAP_MASK") or "").strip().lower() != "neighbor":
            apply_deploy_mu_map_env()
        attach_clot_phi_injector_to_teacher(teacher, device, str(clot_ckpt))
        return
    if leg in ("B_wired", "Bw"):
        from src.inference.deploy_mu_map_env import apply_wired_deploy_mu_map_env, clear_oracle_mu_map_env

        clear_oracle_mu_map_env()
        apply_wired_deploy_mu_map_env()
        attach_clot_phi_injector_to_teacher(teacher, device, str(clot_ckpt))
        return
    if leg in ("B_seed_growth", "Bsg"):
        from src.inference.deploy_mu_map_env import apply_seed_growth_mu_map_env, clear_oracle_mu_map_env

        clear_oracle_mu_map_env()
        apply_seed_growth_mu_map_env()
        attach_clot_phi_injector_to_teacher(teacher, device, str(clot_ckpt))
        return
    if leg == "C":
        os.environ["BIOCHEM_MU_NEIGHBOR_WALL_ONLY"] = "1"
        os.environ["BIOCHEM_MU_NEIGHBOR_WALL_MASK"] = "gt_clot"
        os.environ["BIOCHEM_MU_NEIGHBOR_WALL_BULK"] = "cap_low_shear"
        os.environ["BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND"] = "0.01"
        return
    raise ValueError(f"unknown leg {leg!r}")


@torch.no_grad()
def _rollout(teacher, data, bio_cfg, device, *, time_stride: int = 1, fast: bool = False):
    data = infer_missing_schema(data, phase_hint="biochem").to(device)
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
    t_si = bio_cfg.resolve_biochem_times(data, device)
    t_ref = float(getattr(bio_cfg, "t_final", 30000.0))
    eval_t = _eval_times_for_rollout(t_si, t_ref, time_stride=time_stride, fast=fast)
    return teacher(
        data,
        eval_t,
        y_true_trajectory=data.y,
        teacher_forcing_ratio=1.0,
        start_idx=0,
        initial_species=None,
        detach_macro_state=True,
    )


def _mu_log_mae(pred_si: torch.Tensor, gt_si: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    p = pred_si.reshape(-1)
    g = gt_si.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1).bool()
        p = p[m]
        g = g[m]
    if p.numel() < 1:
        return float("nan")
    return float((torch.log(p.clamp(1e-8)) - torch.log(g.clamp(1e-8))).abs().mean().item())


def _metrics_at_time(
    pred_t: torch.Tensor,
    gt_t: torch.Tensor,
    phys: PhysicsConfig,
    *,
    edge_index: torch.Tensor | None = None,
    node_mask: torch.Tensor | None = None,
    gt_anchor_t: torch.Tensor | None = None,
) -> dict:
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    pred_mu = phys.viscosity_nd_to_si(pred_t[:, mu_ch])
    gt_mu = phys.viscosity_nd_to_si(gt_t[:, mu_ch])
    thr = max(0.055, float(phys.mu_inf) * 1.2)
    high = gt_mu.reshape(-1) >= thr
    out = {
        "mu_log_mae_all": _mu_log_mae(pred_mu, gt_mu),
        "mu_log_mae_high_mu": _mu_log_mae(pred_mu, gt_mu, high),
        "rel_l2": rel_l2_uvp(pred_t, gt_t, node_mask=node_mask),
        "mu_pred_mean_high": float(pred_mu.reshape(-1)[high].mean().item()) if bool(high.any()) else None,
        "mu_gt_mean_high": float(gt_mu.reshape(-1)[high].mean().item()) if bool(high.any()) else None,
    }
    if edge_index is not None:
        from src.evaluation.clot_shape_score import compute_clot_shape_metrics

        out.update(
            compute_clot_shape_metrics(
                pred_state=pred_t,
                gt_state=gt_t,
                edge_index=edge_index,
                phys_cfg=phys,
                node_mask=node_mask,
                gt_anchor_state=gt_anchor_t,
            )
        )
    return out


def _eval_anchor(
    teacher,
    leg: str,
    anchor: str,
    graph_dir: Path,
    device,
    bio_cfg,
    phys,
    *,
    time_index: int = -1,
    time_stride: int = 1,
    fast: bool = False,
) -> dict:
    mode = "fast" if fast else f"stride={time_stride}"
    print(f"   ->  leg {leg} anchor {anchor} rollout ({mode})...", flush=True)
    t0 = time.perf_counter()
    path = graph_dir / f"{anchor}.pt"
    data = torch.load(path, map_location=device, weights_only=False)
    pred = _rollout(teacher, data, bio_cfg, device, time_stride=time_stride, fast=fast)
    gt = data.y.to(device)
    ti = time_index if time_index >= 0 else int(gt.shape[0]) - 1
    ti = max(0, min(ti, int(gt.shape[0]) - 1))
    pred_ti = min(ti, int(pred.shape[0]) - 1)
    m = _metrics_at_time(
        pred[pred_ti],
        gt[ti],
        phys,
        edge_index=data.edge_index,
        node_mask=anchor_node_mask(data),
        gt_anchor_t=gt[0],
    )
    try:
        from src.evaluation.clot_shape_score import compute_clot_shape_trajectory

        traj = compute_clot_shape_trajectory(
            pred_traj=pred,
            gt_traj=gt,
            edge_index=data.edge_index,
            phys_cfg=phys,
            node_mask=anchor_node_mask(data),
        )
        for k, v in traj.items():
            if k not in m:
                m[k] = v
    except Exception:
        pass
    try:
        from src.core_physics.clot_phi_mu_inject import (
            cap_mu_eff_si,
            phi_gt_binary,
            supervision_region_mask,
        )
        from src.core_physics.clot_phi_simple import gt_mu_anchor_cap_si

        mu_gt = phys.viscosity_nd_to_si(gt[ti][:, STATE_CHANNEL_MU_EFF_ND])
        mu_cap = cap_mu_eff_si(mu_gt)
        region = supervision_region_mask(data, device, mu_cap, phys)
        anchor = gt_mu_anchor_cap_si(data, phys, device)
        clot_gate = phi_gt_binary(mu_cap, region, phys, mu_anchor_si=anchor).reshape(-1).bool()
        pred_mu = phys.viscosity_nd_to_si(pred[pred_ti][:, STATE_CHANNEL_MU_EFF_ND])
        if bool(clot_gate.any()):
            m["mu_log_mae_clot_gate"] = _mu_log_mae(pred_mu, mu_gt, clot_gate)
            m["clot_gate_frac"] = float(clot_gate.float().mean().item())
    except Exception:
        pass
    nw = getattr(teacher, "_last_neighbor_wall_mask", None)
    if nw is not None:
        m["neighbor_wall_mask_frac"] = float(nw.reshape(-1).float().mean().item())
    m["anchor"] = anchor
    m["time_index"] = ti
    m["pred_time_index"] = pred_ti
    m["rollout_s"] = round(time.perf_counter() - t0, 1)
    inj = getattr(teacher, "_clot_phi_injector", None)
    if inj is not None and hasattr(inj, "last_diag"):
        d = inj.last_diag
        m["inject_mode"] = d.mode
        m["inject_phi_mean"] = d.phi_mean
        m["inject_phi_frac_ge_05"] = d.phi_frac_ge_05
        m["inject_n_region"] = d.n_region
        m["inject_n_triggered"] = d.n_triggered
        if d.mu_mlp_mean_region > 0.0 or d.mode.startswith("mu_map_v2"):
            m["inject_mu_mlp_mean_region"] = d.mu_mlp_mean_region
            m["inject_mu_mlp_mean_high"] = d.mu_mlp_mean_high
    shape = m.get("clot_shape")
    recall = m.get("clot_recall")
    flow = m.get("flow_score")
    shape_s = f"{shape:.3f}" if shape is not None and shape == shape else "n/a"
    recall_s = f"{recall:.3f}" if recall is not None and recall == recall else "n/a"
    flow_s = f"{flow:.3f}" if flow is not None and flow == flow else "n/a"
    flow_ok = m.get("flow_ok")
    flow_tag = "flow_ok" if flow_ok else "flow_warn"
    print(
        f"   [OK]  leg {leg} {anchor}: clot_shape={shape_s} ({flow_tag}) "
        f"recall={recall_s} flow={flow_s} rel_l2={m.get('rel_l2', float('nan')):.4f} ({m['rollout_s']}s)",
        flush=True,
    )
    return m


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B/C mu coupling probe")
    ap.add_argument("--teacher-checkpoint", default="outputs/biochem/clot_baseline/teacher_best_high_mu.pth")
    ap.add_argument("--clot-phi-checkpoint", default="outputs/biochem/clot_baseline/clot_phi_best.pth")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--anchors", default="patient003,patient007,patient006")
    ap.add_argument("--time-index", type=int, default=-1, help="Label time index (-1 = last)")
    ap.add_argument("--time-stride", type=int, default=1, help="Subsample macro steps (1=full)")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Smoke: 3 keyframes (0, mid, t_final), DEQ iters<=4, RK4 substeps=1",
    )
    ap.add_argument("--mu-ratio-max", type=float, default=20.0)
    ap.add_argument(
        "--leg-b-v1",
        action="store_true",
        help="Use legacy Leg B v1 (constant mu trigger) instead of v2 mu map",
    )
    ap.add_argument(
        "--legs",
        default="A,B,C",
        help="Comma-separated subset of A,B,C (fair A vs B: --legs A,B)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["BIOCHEM_ROLLOUT_PROGRESS"] = "1"
    if not args.out:
        args.out = (
            "outputs/biochem/mlp_clot_inject_probe/abc_compare_fast.json"
            if args.fast
            else "outputs/biochem/mlp_clot_inject_probe/abc_compare.json"
        )
    print(f"[i]  device={device}", flush=True)
    if args.fast:
        print("[i]  FAST smoke: 3 keyframes, inner_iters<=4, RK4 substeps=1", flush=True)
    else:
        print(f"[i]  time_stride={args.time_stride}", flush=True)
    ckpt = root / args.teacher_checkpoint.replace("/", os.sep)
    if not ckpt.is_file():
        print(f"[ERR] teacher ckpt missing: {ckpt}")
        return 1
    clot_ckpt = root / args.clot_phi_checkpoint.replace("/", os.sep)
    if not clot_ckpt.is_file():
        print(f"[ERR] clot-phi ckpt missing: {clot_ckpt}")
        return 1
    graph_dir = root / args.graph_dir.replace("/", os.sep)
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    if args.leg_b_v1:
        os.environ.setdefault("BIOCHEM_MLP_CLOT_MU_SI", "0.10")
    os.environ.setdefault("BIOCHEM_MLP_CLOT_BLEND", "1.0")
    leg_b_label = "mlp_trigger_v1" if args.leg_b_v1 else "mlp_mu_map_v2"

    leg_ids = [_normalize_probe_leg(x) for x in args.legs.split(",") if x.strip()]
    leg_catalog = (
        ("A", "baseline"),
        ("B", leg_b_label),
        ("B_deploy", "mlp_mu_map_neighbor"),
        ("B_wired", "mlp_mu_map_wired"),
        ("B_seed_growth", "mlp_mu_map_seed_growth"),
        ("C", "neighbor_wall_mu"),
    )
    known = {k for k, _ in leg_catalog}
    unknown = [x for x in leg_ids if x not in known]
    if unknown:
        print(f"[ERR] unknown --legs entries: {unknown!r} (use A,B,B_deploy,B_wired,B_seed_growth,C)")
        return 1
    if not leg_ids:
        print("[ERR] --legs is empty")
        return 1

    mu_ratio = resolve_rollout_mu_ratio_max(BiochemConfig(phase="biochem"), cli_value=args.mu_ratio_max)
    teacher, phys, bio = _load_teacher(ckpt, device, mu_ratio, fast=args.fast)

    leg_rows: dict[str, list] = {}
    for leg, label in leg_catalog:
        if leg not in leg_ids:
            continue
        print(f"[NEW] leg {leg} ({label})", flush=True)
        _configure_leg(teacher, device, leg, clot_ckpt=clot_ckpt, leg_b_v1=args.leg_b_v1)
        leg_rows[leg] = [
            _eval_anchor(
                teacher,
                leg,
                a,
                graph_dir,
                device,
                bio,
                phys,
                time_index=args.time_index,
                time_stride=args.time_stride,
                fast=args.fast,
            )
            for a in anchors
        ]

    leg_key_map = {
        "A": "A_baseline",
        "B": f"B_{leg_b_label}",
        "B_deploy": "B_mlp_mu_map_neighbor",
        "B_wired": "B_mlp_mu_map_wired",
        "B_seed_growth": "B_mlp_mu_map_seed_growth",
        "C": "C_neighbor_wall_mu",
    }
    summary = {
        "teacher_checkpoint": str(ckpt),
        "clot_phi_checkpoint": str(clot_ckpt),
        "device": str(device),
        "fast_smoke": bool(args.fast),
        "time_index": args.time_index,
        "time_stride": args.time_stride,
        "leg_b_mode": leg_b_label,
        "legs_run": leg_ids,
        "legs": {leg_key_map[leg]: leg_rows[leg] for leg in leg_ids},
        "means": {},
        "delta_vs_A": {},
    }
    for leg in leg_ids:
        leg_key = leg_key_map[leg]
        rows = leg_rows[leg]
        summary["means"][leg_key] = {
            "clot_shape": _mean(rows, "clot_shape"),
            "clot_dice": _mean(rows, "clot_dice"),
            "clot_recall": _mean(rows, "clot_recall"),
            "flow_score": _mean(rows, "flow_score"),
            "clot_fp_distant": _mean(rows, "clot_fp_distant"),
            "clot_shape_mean": _mean(rows, "clot_shape_mean"),
            "mu_log_mae_all": _mean(rows, "mu_log_mae_all"),
            "mu_log_mae_high_mu": _mean(rows, "mu_log_mae_high_mu"),
            "mu_log_mae_clot_gate": _mean(rows, "mu_log_mae_clot_gate"),
            "rel_l2": _mean(rows, "rel_l2"),
            "clot_gate_frac": _mean(rows, "clot_gate_frac"),
            "flow_ok_frac": _mean(rows, "flow_ok"),
        }
    a_means = summary["means"].get("A_baseline")
    if a_means is not None:
        for leg in leg_ids:
            if leg == "A":
                continue
            leg_key = leg_key_map[leg]
            summary["delta_vs_A"][leg_key] = {}
            for metric in (
                "clot_shape",
                "clot_recall",
                "flow_score",
                "rel_l2",
                "mu_log_mae_all",
                "mu_log_mae_high_mu",
            ):
                a_val = a_means.get(metric)
                b_val = summary["means"][leg_key].get(metric)
                if a_val is not None and b_val is not None:
                    summary["delta_vs_A"][leg_key][metric] = float(b_val - a_val)

    out_path = root / args.out.replace("/", os.sep)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK]  A/B/C clot-shape scorecard (rank by clot_shape)")
    ranked = sorted(
        summary["means"].items(),
        key=lambda kv: (kv[1].get("clot_shape") or -1.0),
        reverse=True,
    )
    for leg_key, m in ranked:
        shape = m.get("clot_shape") or 0.0
        recall = m.get("clot_recall") or 0.0
        flow = m.get("flow_score") or 0.0
        dice = m.get("clot_dice") or 0.0
        rel_l2 = m.get("rel_l2") or float("nan")
        mu_all = m.get("mu_log_mae_all") or float("nan")
        print(
            f"      {leg_key:22s} clot_shape={shape:.3f}  recall={recall:.3f}  "
            f"flow={flow:.3f}  dice={dice:.3f}  rel_l2={rel_l2:.4f}  mu_all={mu_all:.4f}",
            flush=True,
        )
    for leg_key, deltas in summary.get("delta_vs_A", {}).items():
        if not deltas:
            continue
        parts = "  ".join(f"{k}={v:+.3f}" for k, v in sorted(deltas.items()))
        print(f"      delta_vs_A {leg_key}: {parts}", flush=True)
    print(f"      wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
