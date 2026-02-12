import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
project_root = Path(__file__).parent.parent.parent.parent  # Adjust based on where you save this
data_path = project_root / 'data/raw/cfd_anchors/vessel_0.npz'


def inspect_data():
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    print(f"Loading {data_path}...")
    data = np.load(data_path)
    u = data['u']
    v = data['v']
    p = data['p']

    print(f"Shapes -> u: {u.shape}, v: {v.shape}, p: {p.shape}")
    print(f"Velocity Magnitude stats -> Min: {np.min(u):.4f}, Max: {np.max(u):.4f}")

    # Simple Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Note: These are raw point clouds from the mesh.
    # For a proper heatmap, we usually interpolate, but a scatter works for a quick check.
    # We assume x/y coordinates are needed, but for now we just plot the index vs value
    # or just a histogram to check distribution.
    # If you saved coordinates in the npz, we could plot spatially.

    ax[0].hist(u.flatten(), bins=50)
    ax[0].set_title('Velocity U Distribution')

    ax[1].hist(v.flatten(), bins=50)
    ax[1].set_title('Velocity V Distribution')

    ax[2].hist(p.flatten(), bins=50)
    ax[2].set_title('Pressure Distribution')

    plt.show()


if __name__ == "__main__":
    inspect_data()