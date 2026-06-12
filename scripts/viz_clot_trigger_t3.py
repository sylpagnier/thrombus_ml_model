"""T3 clot trigger timeline viz: GT | physics (pred kine+species) | hybrid (deploy)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.evaluation.clot_phi_checkpoint_env import apply_clot_phi_eval_defaults
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1, clot_trigger_viz_phis, scatter_clot_vessel
from src.core_physics.clot_phi_simple import build_clot_phi_step
from src.core_physics.clot_phi_rollout import clot_phi_rollout_detach_carry
from src.training.clot_trigger_stack import (
    advance_coupled_trigger_state,
    apply_deploy_nucleation_mask_env,
    apply_star3_dumped_env,
    apply_star4_live_teacher_env,
    apply_star5_deploy_dumped_eval_env,
    apply_star5_deploy_teacher_eval_env,
    apply_star6_coupled_env,
    build_clot_trigger_coupled_step,
    build_clot_trigger_step_at_time,
    default_dumped_species_anchor_dir,
    default_t1_checkpoint_path,
    default_t5_deploy_teacher_checkpoint_path,
    default_t5_predkine_species_dump_dir,
    default_teacher_checkpoint_path,
    init_coupled_trigger_rollout,
    load_teacher_for_trigger,
    load_trigger_model,
    forward_clot_trigger_hybrid,
    reset_star3_caches,
    reset_star6_caches,
    rollout_teacher_species_series,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="T3 clot trigger timeline viz (full deploy stack)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--kine-ckpt", default="outputs/kinematics/kinematics_best.pth")
    ap.add_argument(
        "--species-source",
        choices=("dumped", "live"),
        default="dumped",
        help="dumped=anchors_teacher_species cache; live=GNODE rollout (T4)",
    )
    ap.add_argument(
        "--coupling",
        choices=("none", "full"),
        default="none",
        help="none=T3/T4/T5 frozen kine; full=T6 mu/phi -> GINO-DEQ feedback",
    )
    ap.add_argument(
        "--star",
        choices=("auto", "t3", "t4", "t5", "t6"),
        default="auto",
        help="Ladder star for env defaults",
    )
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument(
        "--band-mask",
        action="store_true",
        help="Grey out nodes outside the supervision band (legacy debug)",
    )
    ap.add_argument(
        "--nucleation-band",
        action="store_true",
        help="Loss env: nucleation band only (default: honest full mesh)",
    )
    ap.add_argument(
        "--oracle-band",
        action="store_true",
        help="Legacy loss env: GT-mu seeds + dgamma slice (debug only)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
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
    teacher_path = (
        Path(args.teacher)
        if args.teacher.strip()
        else (
            default_t5_deploy_teacher_checkpoint_path()
            if star in ("t5", "t6")
            else default_teacher_checkpoint_path()
        )
    )
    if coupled:
        apply_star6_coupled_env(
            kine_ckpt=args.kine_ckpt,
            teacher_ckpt=str(teacher_path),
            species_live=live,
            dump_dir=args.anchor_dir.strip() or None,
        )
    elif star == "t5":
        if live:
            apply_star5_deploy_teacher_eval_env(
                kine_ckpt=args.kine_ckpt, teacher_ckpt=str(teacher_path)
            )
        else:
            dump_dir = args.anchor_dir.strip() or str(default_t5_predkine_species_dump_dir())
            apply_star5_deploy_dumped_eval_env(kine_ckpt=args.kine_ckpt, dump_dir=dump_dir)
    elif live:
        apply_star4_live_teacher_env(kine_ckpt=args.kine_ckpt, teacher_ckpt=str(teacher_path))
    else:
        dump_dir = args.anchor_dir.strip() or str(default_dumped_species_anchor_dir())
        apply_star3_dumped_env(kine_ckpt=args.kine_ckpt, dump_dir=dump_dir)
    apply_clot_phi_eval_defaults()
    if bool(args.oracle_band):
        from src.training.clot_trigger_stack import apply_oracle_neighbor_mask_env

        apply_oracle_neighbor_mask_env()
    elif bool(args.nucleation_band):
        apply_deploy_nucleation_mask_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_tag = star.upper()
    print(
        f"[i]  {step_tag} viz device={device} anchor={args.anchor} "
        f"species={species_source} coupling={coupling}",
        flush=True,
    )

    ckpt_path = Path(args.checkpoint) if args.checkpoint.strip() else default_t1_checkpoint_path()
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    model, _cfg = load_trigger_model(ckpt_path, device)
    bio = BiochemConfig(phase="biochem")
    phys = PhysicsConfig(phase="biochem")
    teacher = None
    teacher_resolved = None
    if live:
        teacher, bio, phys, teacher_resolved = load_teacher_for_trigger(device, teacher_path)

    if coupled:
        reset_star6_caches()
    else:
        reset_star3_caches()
    if args.anchor_dir.strip():
        anchor_dir = Path(args.anchor_dir)
    elif not live:
        if star == "t5":
            anchor_dir = default_t5_predkine_species_dump_dir()
        else:
            anchor_dir = default_dumped_species_anchor_dir()
    else:
        anchor_dir = root / "data/processed/graphs_biochem_anchors"
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    graph_path = anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
    if live:
        print(f"[i]  teacher rollout ({int(data.y.shape[0])} macro steps)...", flush=True)
        t0 = time.perf_counter()
        pred_series = rollout_teacher_species_series(data, teacher, bio, device)
        print(
            f"[OK] teacher rollout done in {time.perf_counter() - t0:.1f}s "
            f"shape={tuple(pred_series.shape)}",
            flush=True,
        )
    else:
        pred_series = None
        print(f"[i]  using dumped species in y[:,4:16] (T={int(data.y.shape[0])})", flush=True)

    pos = data.x[:, :2].detach().cpu().numpy()
    n_steps = int(data.y.shape[0])
    edge_index = data.edge_index.to(device)
    frame_times = _pick_times(n_steps, int(args.max_frames))

    rollout_state = None
    kine_provider = None
    if coupled:
        rollout_state, kine_provider = init_coupled_trigger_rollout(data, device=device)

    frames: list[dict] = []
    for fi, t in enumerate(frame_times, start=1):
        print(f"[i]  viz frame {fi}/{len(frame_times)} t={t}", flush=True)
        if coupled:
            step = build_clot_trigger_coupled_step(
                data,
                t,
                pred_species_series=pred_series,
                rollout_state=rollout_state,
                phys_cfg=phys,
                bio_cfg=bio,
                device=device,
            )
        elif live:
            step = build_clot_trigger_step_at_time(
                data, t, pred_species_series=pred_series, phys_cfg=phys, bio_cfg=bio, device=device
            )
        else:
            step = build_clot_phi_step(data, t, phys, bio, device)
        mask = step.loss_mask.reshape(-1).bool()
        display = clot_trigger_viz_phis(
            step,
            data,
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
            model=model,
            edge_index=edge_index,
        )
        m_phys = clot_trigger_viz_f1(display["phi_phys"], display["phi_gt"], mask)
        m_hyb = clot_trigger_viz_f1(display["phi_hybrid"], display["phi_gt"], mask)
        if coupled and rollout_state is not None and kine_provider is not None:
            bundle = forward_clot_trigger_hybrid(
                model, step, data, phys_cfg=phys, bio_cfg=bio, device=device, edge_index=edge_index
            )
            advance_coupled_trigger_state(
                data,
                bundle["phi_hybrid"],
                bundle["mu_hybrid"],
                rollout_state=rollout_state,
                kine_provider=kine_provider,
                detach=clot_phi_rollout_detach_carry(),
            )
        frames.append(
            {
                "t": int(t),
                "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                "phi_gt": display["phi_gt"].detach().cpu().numpy(),
                "phi_phys": display["phi_phys"].detach().cpu().numpy(),
                "phi_hybrid": display["phi_hybrid"].detach().cpu().numpy(),
                "region": mask.detach().cpu().numpy(),
                "f1_phys": float(m_phys["clot_f1"]),
                "f1_hybrid": float(m_hyb["clot_f1"]),
            }
        )

    ncols = len(frames)
    fig, axes = plt.subplots(3, ncols, figsize=(max(2.5 * ncols, 10), 7.5), squeeze=False)
    fig.suptitle(
        f"{step_tag} clot trigger -- {args.anchor} | full vessel (F1 in support) | "
        f"row0=GT | row1=physics | row2=hybrid",
        fontsize=11,
    )
    row_labels = (
        ("GT", "physics (coupled kine+species)", "hybrid (coupled kine+species)")
        if coupled
        else ("GT", "physics (pred kine+species)", "hybrid (pred kine+species)")
    )
    keys = ("phi_gt", "phi_phys", "phi_hybrid")
    for j, fr in enumerate(frames):
        title = (
            f"t={fr['t']} tau={fr['tau']:.2f}\n"
            f"phys F1={fr['f1_phys']:.2f} hyb F1={fr['f1_hybrid']:.2f}"
        )
        for ri, (label, key) in enumerate(zip(row_labels, keys)):
            scatter_clot_vessel(
                axes[ri, j],
                pos,
                fr[key],
                label if j == 0 else "",
                scatter_size=float(args.scatter_size),
                mask_outside_region=bool(args.band_mask),
                region=fr["region"],
            )
        axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/{step_tag.lower()}_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    summary_path = out_path.with_suffix(".json")
    teacher_ckpt_str = ""
    if teacher_resolved is not None:
        teacher_ckpt_str = str(
            teacher_resolved.relative_to(root)
            if teacher_resolved.is_relative_to(root)
            else teacher_resolved
        )
    summary_path.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "step": f"{step_tag.lower()}_{species_source}" + ("_coupled" if coupled else ""),
                "species_source": species_source,
                "coupling": coupling,
                "checkpoint": str(
                    ckpt_path.relative_to(root) if ckpt_path.is_relative_to(root) else ckpt_path
                ),
                "teacher_checkpoint": teacher_ckpt_str,
                "species_cache_dir": str(
                    anchor_dir.relative_to(root) if anchor_dir.is_relative_to(root) else anchor_dir
                )
                if not live
                else "",
                "kine_ckpt": args.kine_ckpt,
                "frames": [
                    {k: v for k, v in fr.items() if k not in ("phi_gt", "phi_phys", "phi_hybrid", "region")}
                    for fr in frames
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
