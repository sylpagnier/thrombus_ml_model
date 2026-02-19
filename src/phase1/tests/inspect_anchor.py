import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.paths import get_project_root
from src.config import VesselConfig


def inspect_data(sample_idx=0):
    root = get_project_root()
    cfg = VesselConfig()

    # Dynamically resolve path from Config
    if Path(cfg.output_dir).is_absolute():
        data_dir = Path(cfg.output_dir)
    else:
        data_dir = root / cfg.output_dir

    file_path = data_dir / f"vessel_{sample_idx}.npz"

    if not file_path.exists():
        print(f" File not found: {file_path}")
        print(f"   Checked directory: {data_dir}")
        return

    print(f"Loading: {file_path.name}")
    try:
        data = np.load(file_path)
        keys = list(data.keys())
        print(f"   Available Keys: {keys}")

        if 'x' not in keys or 'y' not in keys:
            print("Spatial coordinates (x, y) missing. Cannot plot spatial map.")
            return

        # FORCE 1D SHAPES
        x = data['x'].flatten()
        y = data['y'].flatten()
        u = data['u'].flatten()
        v = data['v'].flatten()
        p = data['p'].flatten()

        vel_mag = np.sqrt(u ** 2 + v ** 2)

        # Check for Tier 2 Viscosity
        has_mu = 'mu' in keys
        if has_mu:
            mu = data['mu'].flatten()

        print(f"--- Data Summary (Sample {sample_idx}) ---")
        if 'd_bar' in keys:
            print(f"   Mean Diameter (d_bar): {data['d_bar']:.4f} m")
        print(f"   Nodes: {len(x)}")
        print(f"   Velocity Range: {vel_mag.min():.4f} - {vel_mag.max():.4f} m/s")
        print(f"   Pressure Range: {p.min():.4f} - {p.max():.4f} Pa")
        if has_mu:
            print(f"   Viscosity Range: {mu.min():.6f} - {mu.max():.6f} Pa·s")

        # --- Spatial Visualization ---
        num_subplots = 4 if has_mu else 3
        fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 5))

        # Plot Velocity Magnitude
        sc0 = axes[0].scatter(x, y, c=vel_mag, cmap='viridis', s=2)
        plt.colorbar(sc0, ax=axes[0], label='|U| (m/s)')
        axes[0].set_title(f"Velocity Magnitude (Sample {sample_idx})")
        axes[0].set_aspect('equal')

        # Plot Relative Pressure
        sc1 = axes[1].scatter(x, y, c=p, cmap='plasma', s=2)
        plt.colorbar(sc1, ax=axes[1], label='Relative Pressure (Pa)')
        axes[1].set_title("Relative Pressure Field")
        axes[1].set_aspect('equal')

        # Plot Dynamic Viscosity (Tier 2)
        if has_mu:
            sc2 = axes[2].scatter(x, y, c=mu, cmap='magma', s=2)
            plt.colorbar(sc2, ax=axes[2], label=r'Viscosity $\mu$ (Pa·s)')
            axes[2].set_title("Dynamic Viscosity Field")
            axes[2].set_aspect('equal')
            vec_ax = axes[3]
        else:
            vec_ax = axes[2]

        # Plot Vector Field
        k = 20 if len(x) > 1000 else 1
        vec_ax.quiver(x[::k], y[::k], u[::k], v[::k], color='white', alpha=0.8, scale=vel_mag.max() * 10)
        vec_ax.set_facecolor('black')
        vec_ax.set_title("Velocity Vector Field")
        vec_ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error inspecting data: {e}")


if __name__ == "__main__":
    inspect_data(sample_idx=0)