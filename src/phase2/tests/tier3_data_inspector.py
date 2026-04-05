import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

from src.config import VesselConfig, STATE_CHANNEL_MU_EFF_ND
from src.utils.paths import get_project_root


class Tier3DataInspector:
    """
    Load processed PyTorch Geometric graphs and visualize nodal fields
    across extracted time steps using an interactive slider.
    """

    def __init__(self, proc_dir=None, tier="tier3_patients"):
        self.root = get_project_root()
        self.vessel_cfg = VesselConfig(tier=tier)

        # Default to graph output directory if no explicit directory is provided.
        self.proc_dir = Path(proc_dir) if proc_dir else self.root / self.vessel_cfg.graph_output_dir

    def inspect(self, stem):
        filepath = self.proc_dir / f"{stem}.pt"

        if not filepath.exists():
            print(f"Error: Could not find processed graph file at {filepath}")
            return

        print(f"Loading data for {stem}...")
        data = torch.load(filepath, weights_only=False)

        # Static geometry features.
        x = data.x[ :, 0 ].cpu().numpy()
        y = data.x[ :, 1 ].cpu().numpy()
        sdf = data.x[ :, 2 ].cpu().numpy()

        # Time metadata.
        time_steps = data.t.cpu().numpy()
        num_steps = len(time_steps)

        # Setup plot.
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        plt.subplots_adjust(bottom=0.08, hspace=0.2)
        fig.suptitle(f"Patient Inspector: {stem} (t={time_steps[ 0 ]:.4f}s)", fontsize=18, fontweight="bold")

        def get_data_at_step(idx):
            u_comp = data.y[ idx, :, 0 ].cpu().numpy()
            v_comp = data.y[ idx, :, 1 ].cpu().numpy()
            vel_mag = np.sqrt(u_comp ** 2 + v_comp ** 2)
            p_rel = data.y[ idx, :, 2 ].cpu().numpy()
            mu_eff = data.y[ idx, :, STATE_CHANNEL_MU_EFF_ND ].cpu().numpy()
            # Index 9 is Thrombin: 4 kinematics + 5th species.
            thrombin = data.y[ idx, :, 9 ].cpu().numpy()
            return vel_mag, p_rel, mu_eff, thrombin

        # Load initial step (t = 0).
        init_idx = 0
        vel_mag, p_rel, mu_eff, thrombin = get_data_at_step(init_idx)

        # 1) Normalized velocity.
        sc1 = axs[ 0, 0 ].scatter(x, y, c=vel_mag, cmap="viridis", s=2)
        axs[ 0, 0 ].set_title(r"Normalized Velocity ($|U| / u_{ref}$)")
        fig.colorbar(sc1, ax=axs[ 0, 0 ], label="ND")

        # 2) Relative pressure.
        sc2 = axs[ 0, 1 ].scatter(x, y, c=p_rel, cmap="RdBu_r", s=2)
        axs[ 0, 1 ].set_title("Non-Dimensional Pressure (Relative)")
        fig.colorbar(sc2, ax=axs[ 0, 1 ], label="ND (p / p_ref)")

        # 3) Effective viscosity.
        sc3 = axs[ 1, 0 ].scatter(x, y, c=mu_eff, cmap="magma", s=2)
        axs[ 1, 0 ].set_title("ND effective viscosity (μ_si / μ_viscosity_nd_scale)")
        fig.colorbar(sc3, ax=axs[ 1, 0 ], label="ND Ratio")

        # 4) Thrombin concentration.
        sc4 = axs[ 1, 1 ].scatter(x, y, c=thrombin, cmap="plasma", s=2)
        axs[ 1, 1 ].set_title(r"Thrombin $\ln(1 + \hat{T})$")
        fig.colorbar(sc4, ax=axs[ 1, 1 ], label="Transformed ND Units")

        # 5) Wall distance (static).
        sc5 = axs[ 2, 0 ].scatter(x, y, c=sdf, cmap="coolwarm", s=2)
        axs[ 2, 0 ].set_title("Wall Distance (SDF)")
        fig.colorbar(sc5, ax=axs[ 2, 0 ])

        # 6) Boundary masks (static).
        wall_mask = data.mask_wall.cpu().numpy().astype(bool)
        inlet_mask = data.mask_inlet.cpu().numpy().astype(bool)
        outlet_mask = data.mask_outlet.cpu().numpy().astype(bool)

        axs[ 2, 1 ].scatter(x, y, c="gray", s=1, alpha=0.05, label="Internal")
        axs[ 2, 1 ].scatter(x[ wall_mask ], y[ wall_mask ], c="black", s=5, label="Wall")
        axs[ 2, 1 ].scatter(x[ inlet_mask ], y[ inlet_mask ], c="blue", s=8, label="Inlet")
        axs[ 2, 1 ].scatter(x[ outlet_mask ], y[ outlet_mask ], c="red", s=8, label="Outlet")
        axs[ 2, 1 ].set_title("Boundary Node Verification")
        axs[ 2, 1 ].legend(loc="upper right")

        for ax in axs.flat:
            ax.axis("equal")
            ax.axis("off")

        # Slider setup.
        ax_slider = plt.axes([ 0.2, 0.02, 0.6, 0.03 ])
        time_slider = Slider(
            ax=ax_slider,
            label="Time Step Index",
            valmin=0,
            valmax=num_steps - 1,
            valinit=init_idx,
            valstep=1,
            color="teal",
        )

        def update(_):
            idx = int(time_slider.val)
            v_m, p_r, m_e, thr = get_data_at_step(idx)

            sc1.set_array(v_m)
            sc2.set_array(p_r)
            sc3.set_array(m_e)
            sc4.set_array(thr)

            # Keep color contrast visible at each selected time.
            sc1.set_clim(vmin=v_m.min(), vmax=v_m.max())
            sc2.set_clim(vmin=p_r.min(), vmax=p_r.max())
            sc3.set_clim(vmin=m_e.min(), vmax=m_e.max())
            sc4.set_clim(vmin=thr.min(), vmax=thr.max())

            fig.suptitle(f"Patient Inspector: {stem} (t={time_steps[ idx ]:.4f}s)", fontsize=18, fontweight="bold")
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()


def _pick_stem_interactively(proc_dir):
    stems = sorted([ p.stem for p in Path(proc_dir).glob("*.pt") ])
    if len(stems) == 0:
        print(f"No .pt files found in {proc_dir}")
        return None

    print("\nAvailable patients:")
    for idx, stem in enumerate(stems):
        print(f"  [ {idx} ] {stem}")

    while True:
        user_input = input(f"\nSelect index [0-{len(stems) - 1}] or q to quit: ").strip()
        if user_input.lower() in [ "q", "quit", "exit" ]:
            return None

        try:
            idx = int(user_input)
            if 0 <= idx < len(stems):
                return stems[ idx ]
            print(f"Invalid selection. Enter a value in [ 0, {len(stems) - 1} ].")
        except ValueError:
            print("Invalid input. Enter an integer index.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Tier 3 patient PyG data")
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Stem name of the patient file (for example: patient_01). If omitted, you'll be prompted to pick one.",
    )
    parser.add_argument(
        "--proc-dir",
        type=str,
        default=None,
        help="Optional directory containing processed .pt files",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="tier3_patients",
        help="Vessel tier to resolve default processed graph directory",
    )
    args = parser.parse_args()

    inspector = Tier3DataInspector(proc_dir=args.proc_dir, tier=args.tier)
    if args.stem is None:
        selected_stem = _pick_stem_interactively(inspector.proc_dir)
        if selected_stem is None:
            print("Exiting without action.")
        else:
            inspector.inspect(selected_stem)
    else:
        inspector.inspect(args.stem)
