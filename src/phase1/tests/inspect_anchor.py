import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from src.utils.paths import get_project_root
from src.config import VesselConfig


def inspect_data(sample_idx=0, active_tier="tier1"):
    root = get_project_root()
    cfg = VesselConfig(tier=active_tier)

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

            # --- Spatial Visualization (2x2 Grid) ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # Flatten axes array for easy indexing: [0,0]->0, [0,1]->1, etc.
            ax = axes.flatten()

            # 1. Plot Velocity Magnitude
            sc0 = ax[0].scatter(x, y, c=vel_mag, cmap='viridis', s=2)
            plt.colorbar(sc0, ax=ax[0], label='|U| (m/s)')
            ax[0].set_title(f"Velocity Magnitude (Sample {sample_idx})")
            ax[0].set_aspect('equal')

            # 2. Plot Relative Pressure
            sc1 = ax[1].scatter(x, y, c=p, cmap='plasma', s=2)
            plt.colorbar(sc1, ax=ax[1], label='Relative Pressure (Pa)')
            ax[1].set_title("Relative Pressure Field")
            ax[1].set_aspect('equal')

            # 3. Plot Dynamic Viscosity (if available) or skip
            if has_mu:
                sc2 = ax[2].scatter(x, y, c=mu, cmap='magma', s=2)
                plt.colorbar(sc2, ax=ax[2], label=r'Viscosity $\mu$ (Pa·s)')
                ax[2].set_title("Dynamic Viscosity Field")
                ax[2].set_aspect('equal')
            else:
                ax[2].axis('off')  # Hide if no viscosity data

            # 4. Plot Vector Field
            k = 20 if len(x) > 1000 else 1
            ax[3].quiver(x[::k], y[::k], u[::k], v[::k], color='white', alpha=0.8, scale=vel_mag.max() * 10)
            ax[3].set_facecolor('black')
            ax[3].set_title("Velocity Vector Field")
            ax[3].set_aspect('equal')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error inspecting data: {e}")


def _pick_sample_idx_interactively(data_dir):
    files = sorted(Path(data_dir).glob("vessel_*.npz"))
    if len(files) == 0:
        print(f"No vessel_*.npz files found in {data_dir}")
        return None

    indexed = []
    print("\nAvailable anchor files:")
    for i, f in enumerate(files):
        try:
            sample_idx = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        indexed.append(sample_idx)
        print(f"  [ {i} ] vessel_{sample_idx}.npz")

    if len(indexed) == 0:
        print("No parsable vessel indices found.")
        return None

    while True:
        raw = input(f"\nSelect index [0-{len(indexed) - 1}] or q to quit: ").strip()
        if raw.lower() in [ "q", "quit", "exit" ]:
            return None
        try:
            choice = int(raw)
            if 0 <= choice < len(indexed):
                return indexed[choice]
            print(f"Invalid selection. Enter a value in [ 0, {len(indexed) - 1} ].")
        except ValueError:
            print("Invalid input. Enter an integer index.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect anchor CFD outputs")
    parser.add_argument(
        "--tier",
        type=str,
        default="tier1",
        help="Tier used to resolve default output directory",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Sample index to inspect (for example: 12). If omitted, you'll be prompted.",
    )
    args = parser.parse_args()

    cfg = VesselConfig(tier=args.tier)
    data_dir = cfg.output_dir if Path(cfg.output_dir).is_absolute() else get_project_root() / cfg.output_dir

    sample_idx = args.sample_idx if args.sample_idx is not None else _pick_sample_idx_interactively(data_dir)
    if sample_idx is None:
        print("Exiting without action.")
    else:
        inspect_data(sample_idx=sample_idx, active_tier=args.tier)