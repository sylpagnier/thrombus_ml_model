"""GT vs deploy s0 vs Rung4 mini-ladder step (CUDA).

Default rows: GT clot | R4.s0 (rules) | R4.<step>
Optional --include-ceilings adds Rung2 (GT species) and R4 teacher rows for audit.
"""

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

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_rung4_ladder import (  # noqa: E402
    describe_rung4_step,
    rollout_rung4_phi_trajectory,
)
from src.core_physics.t0_rung_config import (  # noqa: E402
    DEFAULT_SPECIES_DUMP_DIR,
    RUNG2_GAMMA_MODE,
    resolve_default_teacher_ckpt,
    rollout_t0_pred_species_series,
    t0_rung2_env,
    t0_rung4_env,
)
from src.core_physics.t0_mu_physics import rollout_t0_clot_phi  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="Rung 4 mini-ladder clot viz (CUDA)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--step", default="s0")
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument(
        "--include-ceilings",
        action="store_true",
        help="Add Rung2 (GT species) and R4 teacher rows (audit only)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    step_info = describe_rung4_step(args.step)
    step_s = step_info.step

    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    times = _pick_times(int(data.y.shape[0]), int(args.max_frames))
    full_region = np.ones(n_nodes, dtype=bool)
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] step={step_s} deploy={step_info.deploy} gt_species={step_info.uses_gt_species}", flush=True)

    t0 = time.perf_counter()
    phi_s0_traj = rollout_rung4_phi_trajectory(data, phys, bio, device, step="s0")
    if step_s == "s0":
        phi_step_traj = phi_s0_traj
    else:
        phi_step_traj = rollout_rung4_phi_trajectory(data, phys, bio, device, step=step_s)
    print(f"[i] R4.{step_s} rollout {time.perf_counter() - t0:.1f}s", flush=True)

    traj2 = None
    traj4 = None
    if args.include_ceilings:
        teacher = Path(args.teacher_ckpt) if args.teacher_ckpt.strip() else Path(resolve_default_teacher_ckpt())
        if not teacher.is_absolute():
            teacher = root / teacher
        species_dump = root / DEFAULT_SPECIES_DUMP_DIR / f"{args.anchor}.pt"
        with t0_rung2_env():
            traj2 = rollout_t0_clot_phi(
                data, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", nucleation=True, nucleation_hops=1,
            )
        if teacher.is_file():
            pred_teacher = rollout_t0_pred_species_series(
                data,
                str(teacher),
                device,
                bio_cfg=bio,
                dumped_graph=str(species_dump) if species_dump.is_file() else None,
                time_stride=6,
            )
            with t0_rung4_env(teacher_ckpt=str(teacher)):
                traj4 = rollout_t0_clot_phi(
                    data, phys, bio, device,
                    gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
                    pred_species_series=pred_teacher, nucleation=True, nucleation_hops=1,
                )

    step_label = f"R4.{step_s}"
    if step_info.uses_gt_species:
        step_label += " [AUDIT]"
    row_labels = [
        "GT clot",
        "R4.s0 (rules)",
        step_label,
    ]
    if args.include_ceilings:
        row_labels = ["GT clot", "Rung2 (GT species)", "R4 (teacher)"] + row_labels[1:]

    ncols = len(times)
    nrows = len(row_labels)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.5 * ncols, 2.2 * nrows), squeeze=False
    )
    deploy_tag = "deploy" if step_info.deploy else "audit"
    fig.suptitle(
        f"T0 Rung4 ladder -- {args.anchor} | step={step_s} ({deploy_tag}) | CUDA",
        fontsize=9,
    )

    frames_meta: list[dict] = []
    for j, t in enumerate(times):
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        phi_s0 = phi_s0_traj[int(t)]
        phi_step = phi_step_traj[int(t)]
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        m_s0 = clot_trigger_viz_f1(phi_s0, phi_gt, mask)
        m_step = clot_trigger_viz_f1(phi_step, phi_gt, mask)
        title = f"t={t} tau={tau:.2f}\ns0={m_s0['clot_f1']:.2f} {step_s}={m_step['clot_f1']:.2f}"
        panels = [
            phi_gt.detach().cpu().numpy(),
            phi_s0.detach().cpu().numpy(),
            phi_step.detach().cpu().numpy(),
        ]
        if args.include_ceilings and traj2 is not None:
            phi2 = traj2[int(t)]["phi"]
            m2 = clot_trigger_viz_f1(phi2, phi_gt, mask)
            title = f"t={t} tau={tau:.2f}\nR2={m2['clot_f1']:.2f} s0={m_s0['clot_f1']:.2f} {step_s}={m_step['clot_f1']:.2f}"
            panels = [panels[0], phi2.detach().cpu().numpy()]
            if traj4 is not None:
                phi4 = traj4[int(t)]["phi"]
                m4 = clot_trigger_viz_f1(phi4, phi_gt, mask)
                title = (
                    f"t={t} tau={tau:.2f}\n"
                    f"R2={m2['clot_f1']:.2f} R4={m4['clot_f1']:.2f} "
                    f"s0={m_s0['clot_f1']:.2f} {step_s}={m_step['clot_f1']:.2f}"
                )
                panels.append(phi4.detach().cpu().numpy())
            panels.extend([phi_s0.detach().cpu().numpy(), phi_step.detach().cpu().numpy()])

        for i, vals in enumerate(panels):
            _scatter_fullmesh_region(
                axes[i, j], pos, vals, full_region,
                row_labels[i] if j == 0 else "",
                cmap="bwr", vmin=0.0, vmax=1.0,
                s=float(args.scatter_size), layer_positive_on_top=True,
            )
        axes[0, j].set_title(title, fontsize=5)

        frame = {
            "time": int(t),
            "tau": tau,
            "s0_f1": float(m_s0["clot_f1"]),
            "step_f1": float(m_step["clot_f1"]),
            "step": step_s,
        }
        if args.include_ceilings and traj2 is not None:
            frame["rung2_f1"] = float(m2["clot_f1"])
            if traj4 is not None:
                frame["rung4_f1"] = float(m4["clot_f1"])
        frames_meta.append(frame)

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t0_rung4_{step_s}_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "device": "cuda",
                "step": step_s,
                "step_deploy": step_info.deploy,
                "step_uses_gt_species": step_info.uses_gt_species,
                "row_labels": row_labels,
                "frames": frames_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
