"""Timeline viz: biochem_gnn clot phi on a synthetic vessel (deploy, no GT).

Usage::

    python scripts/viz_biochem_gnn_timeline.py --seed 42
    python scripts/viz_biochem_gnn_timeline.py --graph path/to/vessel.pt --flow kinematics
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

from scripts.build_synthetic_biochem_graph import build_synthetic_biochem_graph  # noqa: E402
from src.biochem_gnn import BiochemGNN, FlowMode, apply_deploy_env, load_manifest, reference_manifest_path  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.species_gnn_ladder_viz import ladder_viz_times  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import rollout_inc40_phi_trajectory  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _flow_mode(name: str) -> FlowMode:
    raw = (name or "frozen_kine").strip().lower()
    if raw in ("coupled", "mu_coupled", "feedback"):
        return FlowMode.COUPLED
    if raw in ("gt", "comsol"):
        return FlowMode.GT
    return FlowMode.FROZEN_KINE


def _clot_volume_frac(phi: torch.Tensor) -> float:
    p = phi.reshape(-1).float()
    return float((p >= 0.5).float().mean().item())


def main() -> int:
    ap = argparse.ArgumentParser(description="biochem_gnn synthetic vessel clot timeline")
    ap.add_argument("--graph", default="", help="Existing .pt graph (skip generation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--level", type=int, default=1, help="Synthetic geometry level 0-2")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument(
        "--flow",
        default="frozen_kine",
        choices=("frozen_kine", "coupled", "gt", "kinematics"),
        help="frozen_kine=pred kine held; coupled=mu feedback; gt=oracle only",
    )
    ap.add_argument("--n-steps", type=int, default=0, help="Resample timeline length (0=default)")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--scatter-size", type=float, default=2.5)
    ap.add_argument("--no-inc40", action="store_true", help="Skip inc40 rules baseline row")
    ap.add_argument("--no-s0", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    manifest_path = args.manifest.strip() or str(reference_manifest_path())
    manifest = load_manifest(manifest_path)
    apply_deploy_env(manifest, overrides={"T0_R4_FLOW_SOURCE": "kinematics"})

    if args.graph.strip():
        graph_path = Path(args.graph)
        if not graph_path.is_absolute():
            graph_path = root / graph_path
        data = torch.load(graph_path, map_location=device, weights_only=False)
        case = graph_path.stem
    else:
        print(f"[NEW] build synthetic biochem graph seed={args.seed} level={args.level}", flush=True)
        data, graph_path = build_synthetic_biochem_graph(
            seed=int(args.seed),
            level=int(args.level),
            regenerate=bool(args.regenerate),
            n_time_steps=int(args.n_steps) if int(args.n_steps) > 0 else None,
        )
        data = data.to(device)
        case = f"synthetic_seed{int(args.seed)}"

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    n_steps = int(data.y.shape[0])
    times = ladder_viz_times(n_steps, max_frames=int(args.max_frames))
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    full_region = np.ones(n_nodes, dtype=bool)

    fm = _flow_mode(args.flow)
    if args.flow == "kinematics":
        fm = FlowMode.FROZEN_KINE

    print(f"[i] case={case} nodes={n_nodes} steps={n_steps} flow={fm.value}", flush=True)
    print(f"[i] manifest={manifest_path}", flush=True)

    t0 = time.perf_counter()
    model = BiochemGNN.from_manifest(manifest, device=device, flow_mode=fm)
    rollout = model.rollout(data)
    phi_gnn = rollout.phi_by_time

    phi_rules: dict[int, torch.Tensor] = {}
    skip_rules = bool(args.no_inc40 or args.no_s0)
    if not skip_rules:
        vel = "kinematics" if fm != FlowMode.GT else "gt"
        phi_rules = rollout_inc40_phi_trajectory(data, phys, bio, device, vel_source=vel)
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    rows: list[tuple[str, dict[int, torch.Tensor]]] = [("biochem_gnn", phi_gnn)]
    if phi_rules:
        rows.insert(0, ("inc40 rules", phi_rules))

    fig_h = 2.4 * len(rows) + 1.8
    fig, axes = plt.subplots(
        len(rows), len(times),
        figsize=(2.4 * len(times), fig_h),
        squeeze=False,
    )
    fig.suptitle(
        f"biochem_gnn deploy -- {case} | flow={fm.value} | synthetic (no GT)",
        fontsize=9,
    )

    frames: list[dict] = []
    for j, t in enumerate(times):
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        row_metrics: dict[str, float] = {}
        for i, (label, traj) in enumerate(rows):
            phi = traj[int(t)].detach().cpu().numpy()
            _scatter_fullmesh_region(
                axes[i, j], pos, phi, full_region,
                label if j == 0 else "",
                cmap="hot", vmin=0.0, vmax=1.0,
                s=float(args.scatter_size), layer_positive_on_top=True,
            )
            row_metrics[f"{label}_vol"] = _clot_volume_frac(traj[int(t)])
        title = f"t={t} tau={tau:.0f}s\n" + " ".join(
            f"{k.split()[0]}={v:.3f}" for k, v in row_metrics.items()
        )
        axes[0, j].set_title(title, fontsize=5)
        frames.append({"time": int(t), "tau_s": tau, **row_metrics})

    fig.tight_layout(rect=[0, 0.12, 1, 0.96])

    # Volume-over-time inset
    ax_ts = fig.add_axes([0.08, 0.02, 0.84, 0.08])
    all_t = list(range(n_steps))
    for label, traj in rows:
        vols = [_clot_volume_frac(traj[t]) for t in all_t]
        ax_ts.plot(all_t, vols, label=label, linewidth=1.2)
    ax_ts.set_xlim(0, n_steps - 1)
    ax_ts.set_ylabel("clot vol", fontsize=7)
    ax_ts.set_xlabel("time index", fontsize=7)
    ax_ts.tick_params(labelsize=6)
    ax_ts.legend(fontsize=6, loc="upper left", ncol=2)
    ax_ts.grid(True, alpha=0.3)

    out_dir = root / "outputs/biochem/viz/biochem_gnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out.strip():
        out_png = Path(args.out)
    else:
        out_png = out_dir / f"{case}_{fm.value}.png"
    if not out_png.is_absolute():
        out_png = root / out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}", flush=True)

    meta = {
        "case": case,
        "graph": str(graph_path),
        "manifest": manifest_path,
        "flow_mode": fm.value,
        "n_nodes": n_nodes,
        "n_steps": n_steps,
        "viz_times": times,
        "frames": frames,
        "final_clot_volume": {
            label: _clot_volume_frac(traj[n_steps - 1]) for label, traj in rows
        },
    }
    meta_path = out_png.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save] {meta_path}", flush=True)
    if frames:
        last = frames[-1]
        print(
            f"[OK] t={last['time']} tau={last['tau_s']:.0f}s "
            + " ".join(f"{k}={v:.4f}" for k, v in last.items() if k.endswith("_vol")),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
