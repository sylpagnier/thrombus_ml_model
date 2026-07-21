"""Interactive and headless visualizer for synthetic vessel 3x timescale rollout.

Runs the combined baseline wall + off-wall GraphSAGE deploy stack on vessel_0.pt
over a 3x timescale (90,000s) and displays an interactive time slider.
"""

from __future__ import annotations

import argparse
import json
import os
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
from src.core_physics.species_gnn_clot_rollout import (
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_species_series,
    species_gnn_rollout_ckpt,
)
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import rollout_t0_clot_phi
from src.inference.species_gnn_deploy_env import load_deploy_manifest, species_gnn_deploy_env
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic 3x timescale rollout slider")
    ap.add_argument("--headless", action="store_true", help="Save snapshots instead of showing GUI")
    ap.add_argument("--out-dir", default="outputs/biochem/viz/synthetic_3x")
    args = ap.parse_args()

    device = require_cuda_device()
    manifest = load_deploy_manifest()
    phys = PhysicsConfig(phase="biochem")
    # Set t_final to 90000s (3x timescale)
    bio = BiochemConfig(phase="biochem", t_final=90000.0)

    graph_path = REPO / "data/phase_comparison_test/graphs_biochem/vessel_0.pt"
    print(f"[i] Loading synthetic graph: {graph_path}", flush=True)
    data = torch.load(graph_path, map_location=device, weights_only=False)

    # 1. Tile data.y and data.t to run over 3x timescale
    T_orig = data.y.shape[0]
    data_3x = data.clone()
    # Replicate y 3 times along time dimension
    data_3x.y = torch.cat([data.y, data.y, data.y], dim=0)
    T_3x = data_3x.y.shape[0]

    # Re-linspace time vector to span 90000s
    data_3x.t = torch.linspace(0.0, 90000.0, steps=T_3x, device=device, dtype=torch.float32)

    print(f"[i] Rescaled synthetic timeline: {T_orig} -> {T_3x} steps (max time: {data_3x.t[-1].item():.0f}s)", flush=True)

    # 2. Set up deploy environment and run rollout
    t0 = time.perf_counter()
    with species_gnn_deploy_env(manifest, overrides={"T0_R4_FLOW_SOURCE": "kinematics"}, anchor="patient007"):
        ckpt = Path(species_gnn_rollout_ckpt())
        print(f"[i] Loading species GNN checkpoint: {ckpt.name}", flush=True)
        bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
        if bundle is None:
            raise FileNotFoundError(f"missing species GNN ckpt: {ckpt}")

        static = prepare_species_gnn_rollout_static(data_3x, device=device)
        
        # GNN species timeline
        pred_species = rollout_species_gnn_species_series(
            data_3x, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device,
        )

        from src.core_physics.species_viscosity_calibration import resolve_deploy_gelation_beta
        gel_beta = resolve_deploy_gelation_beta(device)
        from src.core_physics.t0_rung_config import t0_rung2_env, RUNG2_GAMMA_MODE
        import os
        nuc_hops = int(os.environ.get("CLOT_V2_NUCLEATION_HOPS", "1"))

        # Physics/gelation trigger rollout
        with t0_rung2_env():
            traj = rollout_t0_clot_phi(
                data_3x, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="kinematics",
                pred_species_series=pred_species, nucleation=True, nucleation_hops=nuc_hops,
                gelation_beta=gel_beta,
            )

    elapsed = time.perf_counter() - t0
    ms_per_step = (elapsed / T_3x) * 1000
    speed_note = f"ML Rollout Speed: {T_3x} steps in {elapsed:.2f}s ({ms_per_step:.1f} ms/step)"
    print(f"[OK] {speed_note}", flush=True)

    # Extract coordinates and arrays
    pos = data_3x.x[:, :2].detach().cpu().numpy()
    
    # Compute dynamic coupled flow (base velocity + local corrector)
    print("[i] Computing dynamic velocity redirect trajectory...", flush=True)
    from src.inference.corrector_coupling import CorrectorCoupledFlow
    flow_provider = CorrectorCoupledFlow(device=device, phys_cfg=phys)
    vel_all = {}
    for t in sorted(traj.keys()):
        mu_eff_si = traj[t]["mu"].to(device)
        u, v = flow_provider.couple(data_3x, mu_eff_si, publish=False)
        vel_all[t] = torch.sqrt(u**2 + v**2).detach().cpu().numpy()
        
    phi_all = {t: v["phi"].detach().cpu().numpy() for t, v in traj.items()}
    mu_all = {t: v["mu"].detach().cpu().numpy() for t, v in traj.items()}

    # Set up matplotlib layout
    if args.headless or os.environ.get("MPLBACKEND") == "Agg":
        # Save snapshot files
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshot time points: start, 1/3, 2/3, final
        step_indices = [0, T_3x // 3, (2 * T_3x) // 3, T_3x - 1]
        for step_idx in step_indices:
            t_sec = float(data_3x.t[step_idx].item())
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"Synthetic Vessel 3x Rollout Snapshot -- t={t_sec:.0f}s\n{speed_note}", fontsize=11, fontweight="bold")
            
            # Left panel: Velocity magnitude
            full_region = np.ones(pos.shape[0], dtype=bool)
            _scatter_fullmesh_region(
                axes[0], pos, vel_all[step_idx], full_region,
                f"Velocity Mag (t={t_sec:.0f}s)", cmap="bwr", vmin=0.0, vmax=1.5, s=2.5
            )
            
            # Right panel: Dynamic Viscosity mu
            _scatter_fullmesh_region(
                axes[1], pos, mu_all[step_idx], full_region,
                "Dynamic Viscosity mu (Pa*s)", cmap="bwr", vmin=0.0, vmax=0.3, s=2.5
            )
            
            out_path = out_dir / f"snapshot_t_{t_sec:.0f}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"[save] Headless snapshot saved to: {out_path}", flush=True)

        # Also save JSON metadata
        meta_path = out_dir / "synthetic_3x_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "T_steps": T_3x,
                "elapsed_s": elapsed,
                "ms_per_step": ms_per_step,
                "speed_note": speed_note,
            }, f, indent=2)
        print(f"[save] Headless metadata saved to: {meta_path}", flush=True)
    else:
        # Interactive mode using matplotlib Widgets
        from matplotlib.widgets import Slider

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.subplots_adjust(bottom=0.20)
        fig.suptitle(f"Synthetic Vessel 3x Rollout Interactive Slider\n{speed_note}", fontsize=11, fontweight="bold")

        # Initial plots at t=0
        full_region = np.ones(pos.shape[0], dtype=bool)
        
        # We need custom scatter references to update on slide
        ax_vel = axes[0]
        ax_mu = axes[1]

        sc_vel = ax_vel.scatter(pos[:, 0], pos[:, 1], c=vel_all[0], cmap="bwr", vmin=0.0, vmax=1.5, s=2.0)
        ax_vel.set_aspect("equal")
        ax_vel.axis("off")
        ax_vel.set_title("Velocity Magnitude")
        fig.colorbar(sc_vel, ax=ax_vel, label="Velocity (ND)", shrink=0.7)

        sc_mu = ax_mu.scatter(pos[:, 0], pos[:, 1], c=mu_all[0], cmap="bwr", vmin=0.0, vmax=0.3, s=2.0)
        ax_mu.set_aspect("equal")
        ax_mu.axis("off")
        ax_mu.set_title("Effective Viscosity mu (Pa*s)")
        fig.colorbar(sc_mu, ax=ax_mu, label="Viscosity (Pa*s)", shrink=0.7)

        # Add Slider
        ax_slider = fig.add_axes([0.15, 0.08, 0.70, 0.03])
        time_slider = Slider(
            ax=ax_slider,
            label="Time Step",
            valmin=0,
            valmax=T_3x - 1,
            valinit=0,
            valstep=1,
            color="red"
        )

        def update(val):
            idx = int(time_slider.val)
            t_sec = float(data_3x.t[idx].item())
            
            # Update data arrays
            sc_vel.set_array(vel_all[idx])
            sc_mu.set_array(mu_all[idx])
            
            ax_vel.set_title(f"Velocity Mag (t={t_sec:.0f}s)")
            ax_mu.set_title(f"Effective Viscosity mu (t={t_sec:.0f}s)")
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        print("[i] Starting Matplotlib GUI. Drag slider to browse timeline.", flush=True)
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
