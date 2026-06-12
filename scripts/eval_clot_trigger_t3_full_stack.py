"""Star 3 (T3): frozen T1 hybrid trigger with pred kine + pred species (GNODE teacher).

Usage:
  python scripts/eval_clot_trigger_t3_full_stack.py
  python scripts/eval_clot_trigger_t3_full_stack.py --teacher outputs/biochem/biochem_teacher_best_high_mu.pth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.architecture.gnode_biochem import biochem_truth_node_mask
from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_rollout import clot_phi_rollout_detach_carry
from src.core_physics.clot_phi_simple import build_clot_phi_step, clot_phi_model_uses_mpnn
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
from src.core_physics.physics_kernels import PhysicsKernels
from src.evaluation.clot_phi_checkpoint_env import apply_clot_phi_eval_defaults
from src.training.biochem_supervision_masks import compute_supervised_species_log_mae
from src.training.clot_growth_eval import eval_phi_trajectory_on_anchor
from src.training.clot_trigger_stack import (
    advance_coupled_trigger_state,
    apply_star3_dumped_env,
    apply_star4_live_teacher_env,
    apply_star5_deploy_dumped_eval_env,
    apply_star5_deploy_teacher_eval_env,
    apply_star6_coupled_env,
    build_clot_trigger_coupled_step,
    build_clot_trigger_step_at_time,
    default_dumped_species_anchor_dir,
    default_gt_anchor_dir,
    default_t1_checkpoint_path,
    default_t5_deploy_paths,
    default_t5_deploy_teacher_checkpoint_path,
    default_t5_predkine_species_dump_dir,
    default_teacher_checkpoint_path,
    forward_clot_trigger_hybrid,
    forward_physics_trigger_phi,
    init_coupled_trigger_rollout,
    load_teacher_for_trigger,
    load_trigger_model,
    reset_star3_caches,
    reset_star6_caches,
    rollout_teacher_species_series,
)
from src.training.train_clot_phi_simple import _clot_metrics


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def _log(msg: str) -> None:
    print(msg, flush=True)


def eval_anchor_t3(
    graph_path: Path,
    model: torch.nn.Module,
    teacher,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    species_source: str = "dumped",
    coupling: str = "none",
    gt_graph_path: Path | None = None,
    anchor_idx: int = 1,
    n_anchors: int = 1,
    progress_step: int = 10,
    verbose: bool = True,
) -> dict:
    anchor = graph_path.stem
    tag = f"[{anchor_idx}/{n_anchors}] {anchor}"
    live = species_source.strip().lower() == "live"
    coupled = coupling.strip().lower() == "full"
    if coupled:
        inputs_label = (
            "pred_kine_coupled + live_species" if live else "pred_kine_coupled + dumped_species"
        )
    else:
        inputs_label = "pred_kine + live_species" if live else "pred_kine + dumped_species"

    def _v(msg: str) -> None:
        if verbose:
            _log(f"[i]  {tag}: {msg}")

    t0 = time.perf_counter()
    if coupled:
        reset_star6_caches()
    else:
        reset_star3_caches()
    _v("loading graph")
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])

    gt_data = None
    if gt_graph_path is not None and gt_graph_path.is_file():
        gt_data = torch.load(gt_graph_path, map_location=device, weights_only=False)

    pred_series = None
    if live:
        if teacher is None:
            raise ValueError("live species_source requires a loaded teacher")
        _v(f"teacher rollout ({n_steps} macro steps) -- may take several minutes")
        t_roll = time.perf_counter()
        pred_series = rollout_teacher_species_series(data, teacher, bio_cfg, device)
        _v(
            f"teacher rollout done in {time.perf_counter() - t_roll:.1f}s "
            f"pred_shape={tuple(pred_series.shape)}"
        )
    else:
        _v(f"using dumped species from y[:,4:16] (T={n_steps})")
    edge_index = data.edge_index.to(device) if clot_phi_model_uses_mpnn(model) else None
    phi_hyb_by_t: dict[int, torch.Tensor] = {}
    per_step: list[dict] = []
    truth_m = biochem_truth_node_mask(data, int(data.num_nodes), device)

    rollout_state = None
    kine_provider = None
    if coupled:
        _v("init coupled rollout (phi=0 -> mu_c -> GINO-DEQ MU_PRIOR)")
        rollout_state, kine_provider = init_coupled_trigger_rollout(data, device=device)

    step_stride = max(1, int(progress_step))
    for t in range(n_steps):
        if coupled:
            step = build_clot_trigger_coupled_step(
                data,
                t,
                pred_species_series=pred_series,
                rollout_state=rollout_state,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
            )
        elif live:
            step = build_clot_trigger_step_at_time(
                data,
                t,
                pred_species_series=pred_series,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
            )
        else:
            step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
        phi_gt = step.phi_gt.reshape(-1)
        phi_phys, _ = forward_physics_trigger_phi(
            step, data, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device, apply_region=True
        )
        bundle = forward_clot_trigger_hybrid(
            model, step, data, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device, edge_index=edge_index
        )
        phi_hyb = bundle["phi_hybrid"]
        phi_hyb_by_t[int(t)] = phi_hyb.reshape(-1)
        if coupled and rollout_state is not None and kine_provider is not None:
            advance_coupled_trigger_state(
                data,
                phi_hyb,
                bundle["mu_hybrid"],
                rollout_state=rollout_state,
                kine_provider=kine_provider,
                detach=clot_phi_rollout_detach_carry(),
            )
        mask = step.loss_mask.reshape(-1).bool()
        m_phys = _clot_metrics(phi_phys, phi_gt, mask)
        m_hyb = _clot_metrics(phi_hyb, phi_gt, mask)
        if live and pred_series is not None:
            pt = max(0, min(t, int(pred_series.shape[0]) - 1))
            pred_slice = pred_series[pt : pt + 1]
        else:
            pred_slice = data.y[t : t + 1]
        tgt_slice = data.y[t : t + 1]
        if gt_data is not None:
            gt_t = max(0, min(t, int(gt_data.y.shape[0]) - 1))
            tgt_slice = gt_data.y[gt_t : gt_t + 1].to(device)
        sp = compute_supervised_species_log_mae(
            pred_series=pred_slice,
            target_series=tgt_slice.to(device),
            node_mask=truth_m,
        )
        per_step.append(
            {
                "t": t,
                "f1_phys": float(m_phys["clot_f1"]),
                "f1_hybrid": float(m_hyb["clot_f1"]),
                "species_fi_log_mae": float(sp["species_fi_log_mae"]),
                "pred_pos_frac": float(m_hyb["pred_pos_frac"]),
                "gt_pos_frac": float(m_hyb["gt_pos_frac"]),
            }
        )
        if verbose and ((t + 1) % step_stride == 0 or t == n_steps - 1):
            _v(
                f"trigger {t + 1}/{n_steps} "
                f"hyb_F1={float(m_hyb['clot_f1']):.3f} phys_F1={float(m_phys['clot_f1']):.3f} "
                f"FI={float(sp['species_fi_log_mae']):.4f}"
            )

    _v("trajectory score")
    traj = eval_phi_trajectory_on_anchor(
        phi_hyb_by_t,
        data,
        anchor=anchor,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag=f"t{'5' if coupled else '3'}_{species_source}",
    )
    elapsed = time.perf_counter() - t0
    row = {
        "anchor": anchor,
        "inputs": inputs_label,
        "species_source": species_source,
        "coupling": coupling,
        "n_steps": n_steps,
        "mean_f1_phys": _mean(per_step, "f1_phys"),
        "mean_f1_hybrid": _mean(per_step, "f1_hybrid"),
        "final_f1_phys": per_step[-1]["f1_phys"] if per_step else float("nan"),
        "final_f1_hybrid": per_step[-1]["f1_hybrid"] if per_step else float("nan"),
        "mean_species_fi_log_mae": _mean(per_step, "species_fi_log_mae"),
        "final_species_fi_log_mae": per_step[-1]["species_fi_log_mae"] if per_step else float("nan"),
        "trajectory_score": float(traj.get("trajectory_score", float("nan"))),
        "tfinal_wall_ring_frac": float(traj.get("tfinal_wall_ring_frac", float("nan"))),
        "elapsed_s": float(elapsed),
        "per_step": per_step,
    }
    _log(
        f"[OK] {tag}: mean_hyb_F1={row['mean_f1_hybrid']:.3f} "
        f"final_hyb_F1={row['final_f1_hybrid']:.3f} "
        f"FI={row['final_species_fi_log_mae']:.4f} traj={row['trajectory_score']:.3f} "
        f"ring={row['tfinal_wall_ring_frac']:.3f} ({elapsed:.1f}s)"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="T3/T4/T5 clot trigger eval (pred kine + pred species)")
    ap.add_argument("--anchor-dir", default="")
    ap.add_argument("--gt-anchor-dir", default="", help="COMSOL GT graphs for species FI (default: graphs_biochem_anchors)")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--checkpoint", default="", help="Frozen T1 trigger ckpt")
    ap.add_argument("--teacher", default="", help="GNODE teacher ckpt (live mode only)")
    ap.add_argument("--kine-ckpt", default="outputs/kinematics/kinematics_best.pth")
    ap.add_argument(
        "--species-source",
        choices=("dumped", "live"),
        default="dumped",
        help="dumped=cached anchors_teacher_species; live=GNODE rollout (slow, T4)",
    )
    ap.add_argument(
        "--coupling",
        choices=("none", "full"),
        default="none",
        help="none=T3/T4/T5 frozen steady kine; full=T6 phi/mu -> GINO-DEQ feedback",
    )
    ap.add_argument(
        "--star",
        choices=("auto", "t3", "t4", "t5", "t6"),
        default="auto",
        help="Ladder star for env defaults (auto from species/coupling)",
    )
    ap.add_argument(
        "--out",
        default="",
        help="JSON out (default: t3_dumped_species.json or t4_live_teacher.json)",
    )
    ap.add_argument(
        "--progress-step",
        type=int,
        default=10,
        help="Log trigger progress every N macro steps (default 10)",
    )
    ap.add_argument(
        "--nucleation-band",
        action="store_true",
        help="Loss/F1 on deploy nucleation band only (default: full mesh)",
    )
    ap.add_argument(
        "--oracle-band",
        action="store_true",
        help="Legacy: GT-mu seeds + dgamma slice for loss/F1 (debug only)",
    )
    ap.add_argument("--quiet", action="store_true", help="Only print per-anchor summary lines")
    args = ap.parse_args()

    species_source = str(args.species_source).strip().lower()
    coupling = str(args.coupling).strip().lower()
    star = str(args.star).strip().lower()
    live = species_source == "live"
    coupled = coupling == "full"
    if star == "auto":
        if coupled:
            star = "t6"
        elif live:
            star = "t4"
        else:
            star = "t3"
    if coupled:
        dump_dir_arg = args.anchor_dir.strip() or None
        teacher_arg = str(Path(args.teacher)) if args.teacher.strip() else None
        apply_star6_coupled_env(
            kine_ckpt=args.kine_ckpt,
            teacher_ckpt=teacher_arg,
            species_live=live,
            dump_dir=dump_dir_arg,
        )
        teacher_path = (
            Path(args.teacher)
            if args.teacher.strip()
            else default_t5_deploy_teacher_checkpoint_path()
        )
    elif star == "t5":
        if live:
            teacher_path = (
                Path(args.teacher)
                if args.teacher.strip()
                else default_t5_deploy_teacher_checkpoint_path()
            )
            apply_star5_deploy_teacher_eval_env(
                kine_ckpt=args.kine_ckpt, teacher_ckpt=str(teacher_path)
            )
        else:
            teacher_path = None
            dump_dir = args.anchor_dir.strip() or str(default_t5_predkine_species_dump_dir())
            apply_star5_deploy_dumped_eval_env(kine_ckpt=args.kine_ckpt, dump_dir=dump_dir)
    elif live:
        teacher_path = Path(args.teacher) if args.teacher.strip() else default_teacher_checkpoint_path()
        apply_star4_live_teacher_env(kine_ckpt=args.kine_ckpt, teacher_ckpt=str(teacher_path))
    else:
        teacher_path = None
        dump_dir = args.anchor_dir.strip() or str(default_dumped_species_anchor_dir())
        apply_star3_dumped_env(kine_ckpt=args.kine_ckpt, dump_dir=dump_dir)
    apply_clot_phi_eval_defaults()
    if bool(args.oracle_band):
        from src.training.clot_trigger_stack import apply_oracle_neighbor_mask_env

        apply_oracle_neighbor_mask_env()
    elif bool(args.nucleation_band):
        from src.training.clot_trigger_stack import apply_deploy_nucleation_mask_env

        apply_deploy_nucleation_mask_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(
        f"[i]  device={device} cuda_available={torch.cuda.is_available()} "
        f"species_source={species_source} coupling={coupling}"
    )
    trigger_path = Path(args.checkpoint) if args.checkpoint.strip() else default_t1_checkpoint_path()
    if not trigger_path.is_absolute():
        trigger_path = _REPO / trigger_path
    if not trigger_path.is_file():
        print(f"[ERR] missing T1 trigger checkpoint: {trigger_path}", file=sys.stderr)
        return 2

    model, cfg = load_trigger_model(trigger_path, device)
    teacher = None
    teacher_resolved = None
    if live:
        teacher, bio_cfg, phys_cfg, teacher_resolved = load_teacher_for_trigger(device, teacher_path)
    else:
        bio_cfg = BiochemConfig(phase="biochem")
        phys_cfg = PhysicsConfig(phase="biochem")
    _ = BiochemPhysicsKernels(bio_cfg, PhysicsKernels(phys_cfg=phys_cfg))

    _log(
        f"[i]  eval trigger={trigger_path.name} star={cfg.get('clot_trigger_star', '?')} "
        f"teacher={teacher_resolved.name if teacher_resolved else 'dumped_cache'} kine={args.kine_ckpt}"
    )

    if args.anchor_dir:
        anchor_dir = Path(args.anchor_dir)
        if not anchor_dir.is_absolute():
            anchor_dir = _REPO / anchor_dir
    elif live or (coupled and not args.anchor_dir.strip()):
        anchor_dir = _REPO / VesselConfig(phase="biochem_anchors").graph_output_dir
    elif star == "t5":
        anchor_dir = default_t5_predkine_species_dump_dir()
    else:
        anchor_dir = default_dumped_species_anchor_dir()

    gt_anchor_dir = Path(args.gt_anchor_dir) if args.gt_anchor_dir.strip() else default_gt_anchor_dir()
    if not gt_anchor_dir.is_absolute():
        gt_anchor_dir = _REPO / gt_anchor_dir

    paths = sorted(anchor_dir.glob("*.pt"))
    if not paths:
        print(f"[ERR] no graphs in {anchor_dir}", file=sys.stderr)
        if not live and not coupled:
            print(
                "[i]  run dump first:\n"
                "  powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\go_clot_trigger_t3_dump_species.ps1",
                file=sys.stderr,
            )
        return 2

    if coupled:
        out_default = (
            "outputs/biochem/clot_trigger/t6_coupled_live.json"
            if live
            else "outputs/biochem/clot_trigger/t6_coupled_dumped.json"
        )
    elif star == "t5":
        out_default = str(
            default_t5_deploy_paths().eval_live_json
            if live
            else default_t5_deploy_paths().eval_dumped_json
        )
    elif live:
        out_default = "outputs/biochem/clot_trigger/t4_live_teacher.json"
    else:
        out_default = "outputs/biochem/clot_trigger/t3_dumped_species.json"
    out_path = _REPO / (args.out.strip() or out_default)
    n_anchors = len(paths)
    _log(f"[i]  eval {n_anchors} anchors from {anchor_dir}")
    progress_path = out_path.with_suffix(".progress.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")

    rows: list[dict] = []
    t_all = time.perf_counter()
    for i, p in enumerate(paths, start=1):
        row = eval_anchor_t3(
            p,
            model,
            teacher,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            species_source=species_source,
            coupling=coupling,
            gt_graph_path=gt_anchor_dir / p.name,
            anchor_idx=i,
            n_anchors=n_anchors,
            progress_step=int(args.progress_step),
            verbose=not bool(args.quiet),
        )
        rows.append(row)
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({k: v for k, v in row.items() if k != "per_step"}) + "\n"
            )
        _log(f"[i]  progress: {i}/{n_anchors} anchors done ({time.perf_counter() - t_all:.1f}s elapsed)")

    for r in rows:
        if args.quiet:
            print(
                f"[OK] {r['anchor']}: mean_hyb_F1={r['mean_f1_hybrid']:.3f} "
                f"final_hyb_F1={r['final_f1_hybrid']:.3f} "
                f"FI={r['final_species_fi_log_mae']:.4f} traj={r['trajectory_score']:.3f} "
                f"ring={r['tfinal_wall_ring_frac']:.3f}",
                flush=True,
            )

    val_row = next((r for r in rows if r["anchor"] == args.val), None)
    if coupled:
        step_id = "t6_coupled_live" if live else "t6_coupled_dumped"
        inputs_desc = (
            "pred_kine_coupled + live_GNODE_species"
            if live
            else "pred_kine_coupled + cached_teacher_species_dump"
        )
    elif star == "t5":
        step_id = "t5_deploy_live" if live else "t5_deploy_dumped"
        inputs_desc = (
            "pred_GINO_DEQ_flow + T5_deploy_live_species"
            if live
            else "pred_GINO_DEQ_flow + T5_predkine_species_dump"
        )
    elif live:
        step_id = "t4_live_teacher"
        inputs_desc = "pred_GINO_DEQ_flow + live_GNODE_species"
    else:
        step_id = "t3_dumped_species"
        inputs_desc = "pred_GINO_DEQ_flow + cached_teacher_species_dump"
    summary = {
        "step": step_id,
        "star": star,
        "species_source": species_source,
        "coupling": coupling,
        "loss_scope": (
            "oracle"
            if bool(args.oracle_band)
            else ("nucleation" if bool(args.nucleation_band) else "full_mesh")
        ),
        "inputs": inputs_desc,
        "anchor_dir": str(
            anchor_dir.relative_to(_REPO) if anchor_dir.is_relative_to(_REPO) else anchor_dir
        ),
        "checkpoint": str(
            trigger_path.relative_to(_REPO) if trigger_path.is_relative_to(_REPO) else trigger_path
        ),
        "teacher_checkpoint": (
            str(teacher_resolved.relative_to(_REPO))
            if teacher_resolved and teacher_resolved.is_relative_to(_REPO)
            else (str(teacher_resolved) if teacher_resolved else "")
        ),
        "kine_ckpt": args.kine_ckpt,
        "mean_f1_hybrid": _mean(rows, "mean_f1_hybrid"),
        "mean_f1_phys": _mean(rows, "mean_f1_phys"),
        "mean_trajectory_score": _mean(rows, "trajectory_score"),
        "mean_species_fi_log_mae": _mean(rows, "mean_species_fi_log_mae"),
        "val_anchor": args.val,
        "val_mean_f1_hybrid": float(val_row["mean_f1_hybrid"]) if val_row else float("nan"),
        "val_final_f1_hybrid": float(val_row["final_f1_hybrid"]) if val_row else float("nan"),
        "val_trajectory_score": float(val_row["trajectory_score"]) if val_row else float("nan"),
        "val_tfinal_wall_ring_frac": float(val_row["tfinal_wall_ring_frac"]) if val_row else float("nan"),
        "per_anchor": [{k: v for k, v in r.items() if k != "per_step"} for r in rows],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"[save] {out_path}")
    _log(f"[save] {progress_path} (per-anchor progress log)")
    _log(
        f"[summary] mean_hyb_F1={summary['mean_f1_hybrid']:.3f} "
        f"mean_traj={summary['mean_trajectory_score']:.3f} "
        f"mean_FI={summary['mean_species_fi_log_mae']:.4f} "
        f"total={time.perf_counter() - t_all:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
