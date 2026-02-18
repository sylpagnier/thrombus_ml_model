import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
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

        # FORCE 1D SHAPES (Fixes the (1, N) vs (N,) mismatch)
        x = data['x'].flatten()
        y = data['y'].flatten()
        u = data['u'].flatten()
        v = data['v'].flatten()
        p = data['p'].flatten()

        d_bar = data['d_bar'] if 'd_bar' in keys else "N/A"

        # Calculate Magnitude
        vel_mag = np.sqrt(u ** 2 + v ** 2)

        print(f"   Shape: {u.shape} points (Flattened)")
        print(f"   Diameter (d_bar): {d_bar}")
        print(f"   Velocity Range: {vel_mag.min():.4f} - {vel_mag.max():.4f} m/s")
        print(f"   Pressure Range: {p.min():.4f} - {p.max():.4f} Pa")

        # --- Spatial Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot Velocity Magnitude
        sc0 = axes[0].scatter(x, y, c=vel_mag, cmap='viridis', s=2)
        plt.colorbar(sc0, ax=axes[0], label='|U| (m/s)')
        axes[0].set_title(f"Velocity Magnitude (Sample {sample_idx})")
        axes[0].set_aspect('equal')

        # Plot Pressure
        sc1 = axes[1].scatter(x, y, c=p, cmap='plasma', s=2)
        plt.colorbar(sc1, ax=axes[1], label='Pressure (Pa)')
        axes[1].set_title("Pressure Field")
        axes[1].set_aspect('equal')

        # Plot Vector Field (Downsampled for clarity)
        # Skip every k points to make arrows visible
        k = 20 if len(x) > 1000 else 1

        # Quiver requires inputs to be the same size, which .flatten() ensures
        axes[2].quiver(x[::k], y[::k], u[::k], v[::k], vel_mag[::k], cmap='viridis')
        axes[2].set_title(f"Velocity Vectors (Every {k}th point)")
        axes[2].set_aspect('equal')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error inspecting data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # You can pass an index argument or default to 0
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    inspect_data(idx)