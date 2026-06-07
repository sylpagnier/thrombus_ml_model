"""Diagnose deploy MLP commit gate (phi / mu_mlp / allowed) on pred-kine rollout.

Step-1 diagnostic: why B_wired / B_deploy commit is empty in closed loop.

Usage (repo root):
  python scripts/diagnose_deploy_gate.py --anchor patient007 --leg B_wired
  python scripts/diagnose_deploy_gate.py --anchor patient007 --leg B_deploy --fast
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_mu_inject import (
    biochem_mlp_mu_map_enabled,
    diagnose_deploy_gate_rollout_series,
    mlp_mu_map_mask_mode,
)
from src.core_physics.clot_phi_simple import build_clot_phi_model
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.inference.biochem_teacher_loader import build_biochem_teacher, resolve_rollout_mu_ratio_max
from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema

sys.path.insert(0, str(ROOT / "scripts"))
from run_mlp_clot_inject_probe import (  # noqa: E402
    _configure_leg,
    _eval_times_for_rollout,
    _load_teacher,
    _normalize_probe_leg,
    _rollout,
)


def _load_clot_model(ckpt: Path, device: torch.device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = raw.get("config") or {}
    apply_clot_phi_config_from_checkpoint(cfg)
    apply_clot_phi_eval_defaults()
    os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "current")
    model = build_clot_phi_model(
        in_dim=int(cfg.get("in_dim", 6)),
        hidden=int(cfg.get("hidden", 32)),
    ).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, cfg


def _label_indices_for_rollout(data, bio_cfg, device, eval_t_nd: torch.Tensor) -> list[int]:
    t_si = bio_cfg.resolve_biochem_times(data, device)
    t_ref = float(bio_cfg.t_final)
    eval_times_si = eval_t_nd.reshape(-1).to(device=device, dtype=t_si.dtype) * t_ref
    out: list[int] = []
    for i in range(int(eval_times_si.numel())):
        t = float(eval_times_si[i].item())
        idx = int((t_si - t).abs().argmin().item())
        out.append(max(0, min(idx, int(data.y.shape[0]) - 1)))
    return out


def _print_table(rows: list) -> None:
    hdr = (
        " step |   t[s] | allow | phi% | mu% | both% | cmt% | roll% | gt | bottleneck"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.macro_step:5d} | {r.t_si:6.0f} | {r.n_allowed:5d} | "
            f"{100 * r.frac_phi_ge_thr_in_allowed:4.1f} | "
            f"{100 * r.frac_mu_mlp_ge_thr_in_allowed:4.1f} | "
            f"{100 * r.frac_both_in_allowed:4.1f} | "
            f"{100 * r.frac_commit_in_allowed:4.1f} | "
            f"{100 * r.frac_rollout_mu_ge_thr_in_allowed:4.1f} | "
            f"{r.n_gt_clot_in_allowed:3d} | {r.bottleneck}"
        )


def _primary_blocker(row) -> str:
    if row.no_commit_t0:
        return "no_commit_t0 (expected)"
    if row.frac_commit_in_allowed > 0.05:
        return "none (commit active)"
    if row.frac_phi_ge_thr_in_allowed < 0.05 and row.frac_mu_mlp_ge_thr_in_allowed < 0.05:
        return "phi_and_mu_mlp"
    if row.frac_mu_mlp_ge_thr_in_allowed < row.frac_phi_ge_thr_in_allowed * 0.5:
        return "mu_mlp"
    if row.frac_phi_ge_thr_in_allowed < 0.05:
        return "phi"
    if row.n_gt_clot_in_allowed > 0 and row.frac_both_in_allowed < 0.05:
        return "mu_mlp (phi ok)"
    return row.bottleneck


def _summarize(rows: list) -> dict:
    if not rows:
        return {}
    last = rows[-1]
    mid = rows[len(rows) // 2]
    bottlenecks = {}
    for r in rows:
        bottlenecks[r.bottleneck] = bottlenecks.get(r.bottleneck, 0) + 1
    return {
        "n_frames": len(rows),
        "mask_mode": last.mask_mode,
        "t_final_si": last.t_si,
        "t_final_bottleneck": last.bottleneck,
        "t_final_primary_blocker": _primary_blocker(last),
        "t_final_frac_phi_in_allowed": last.frac_phi_ge_thr_in_allowed,
        "t_final_frac_mu_in_allowed": last.frac_mu_mlp_ge_thr_in_allowed,
        "t_final_frac_both_in_allowed": last.frac_both_in_allowed,
        "t_final_frac_commit_in_allowed": last.frac_commit_in_allowed,
        "t_final_frac_rollout_clot_in_allowed": last.frac_rollout_mu_ge_thr_in_allowed,
        "t_final_n_gt_clot_in_allowed": last.n_gt_clot_in_allowed,
        "t_final_n_gt_clot_supervision_t": last.n_gt_clot_supervision_t,
        "t_final_allowed_vs_supervision_t": last.n_allowed - last.n_supervision_at_t,
        "t_final_phi_p90_allowed": last.phi_p90_allowed,
        "t_final_mu_mlp_p90_allowed": last.mu_mlp_p90_allowed,
        "t_mid_bottleneck": mid.bottleneck,
        "bottleneck_counts": bottlenecks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy commit gate diagnostics")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchors", default="", help="Comma list (overrides --anchor)")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--leg", default="B_wired", help="B_wired, B_deploy, B, ...")
    ap.add_argument(
        "--teacher-checkpoint",
        default="outputs/biochem/gnode10_sweep/gnode12_lane_a_promoted/biochem_teacher_best_high_mu.pth",
    )
    ap.add_argument(
        "--clot-phi-checkpoint",
        default="outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth",
    )
    ap.add_argument("--mu-ratio-max", type=float, default=20.0)
    ap.add_argument("--time-stride", type=int, default=5)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leg = _normalize_probe_leg(args.leg)
    teacher_ckpt = ROOT / args.teacher_checkpoint
    clot_ckpt = ROOT / args.clot_phi_checkpoint
    graph_dir = ROOT / args.graph_dir
    if not teacher_ckpt.is_file():
        print(f"[ERR] teacher ckpt missing: {teacher_ckpt}", file=sys.stderr)
        return 1
    if not clot_ckpt.is_file():
        print(f"[ERR] clot-phi ckpt missing: {clot_ckpt}", file=sys.stderr)
        return 1

    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()] or [args.anchor]
    mu_ratio = resolve_rollout_mu_ratio_max(args.mu_ratio_max)

    print(f"[NEW] deploy gate diagnose leg={leg} mask={os.environ.get('BIOCHEM_MLP_MU_MAP_MASK', '(unset)')}")
    print(f"[i]  device={device} fast={int(args.fast)} stride={args.time_stride}")

    all_out: dict = {"leg": leg, "anchors": {}, "summary": {}}
    for anchor in anchors:
        path = graph_dir / f"{anchor}.pt"
        if not path.is_file():
            print(f"[WARN] skip missing anchor {path}")
            continue
        print(f"\n[i]  anchor={anchor} rollout...", flush=True)
        t0 = time.perf_counter()
        teacher, phys, bio = _load_teacher(teacher_ckpt, device, mu_ratio, fast=args.fast)
        _configure_leg(teacher, device, leg, clot_ckpt=clot_ckpt)
        if not biochem_mlp_mu_map_enabled():
            print(f"[WARN] leg {leg} has BIOCHEM_MLP_MU_MAP=0; gate stats are offline-only", flush=True)
        attach_clot_phi_injector_to_teacher(teacher, device, str(clot_ckpt))
        clot_model, _ = _load_clot_model(clot_ckpt, device)

        data = infer_missing_schema(
            torch.load(path, map_location=device, weights_only=False),
            phase_hint="biochem",
        ).to(device)
        assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

        t_si = bio.resolve_biochem_times(data, device)
        eval_t = _eval_times_for_rollout(
            t_si, float(bio.t_final), time_stride=args.time_stride, fast=args.fast
        )
        pred = _rollout(teacher, data, bio, device, time_stride=args.time_stride, fast=args.fast)
        label_idx = _label_indices_for_rollout(data, bio, device, eval_t)
        eval_times_si = eval_t.reshape(-1) * float(bio.t_final)

        rows = diagnose_deploy_gate_rollout_series(
            clot_model,
            data,
            pred,
            eval_times_si,
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
            label_time_indices=label_idx,
        )
        elapsed = time.perf_counter() - t0
        print(f"[OK]  rollout+diag {elapsed:.1f}s  frames={len(rows)}  mask={mlp_mu_map_mask_mode()}")
        _print_table(rows)
        summary = _summarize(rows)
        summary["elapsed_s"] = elapsed
        all_out["anchors"][anchor] = {
            "frames": [r.to_dict() for r in rows],
            "summary": summary,
        }
        print(
            f"[i]  t_final: primary_blocker={summary.get('t_final_primary_blocker')} "
            f"phi={100 * summary.get('t_final_frac_phi_in_allowed', 0):.1f}% "
            f"mu_mlp={100 * summary.get('t_final_frac_mu_in_allowed', 0):.1f}% "
            f"both={100 * summary.get('t_final_frac_both_in_allowed', 0):.1f}% "
            f"commit={100 * summary.get('t_final_frac_commit_in_allowed', 0):.1f}% "
            f"rollout_clot={100 * summary.get('t_final_frac_rollout_clot_in_allowed', 0):.1f}% "
            f"gt_in_allowed={summary.get('t_final_n_gt_clot_in_allowed')} "
            f"(supervision_t gt_clot={summary.get('t_final_n_gt_clot_supervision_t')}) "
            f"phi_p90={summary.get('t_final_phi_p90_allowed', 0):.3f} "
            f"mu_mlp_p90={summary.get('t_final_mu_mlp_p90_allowed', 0):.4f}",
            flush=True,
        )

    out_path = Path(args.out) if args.out else ROOT / "outputs/biochem/diagnostics/deploy_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\n[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
