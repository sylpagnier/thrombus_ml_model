import os
import torch
import numpy as np
import meshio
import matplotlib.pyplot as plt
from pathlib import Path  # Added for robust path handling
from scipy.spatial import KDTree
from torch_geometric.data import Data
from tqdm import tqdm


class MeshToGraphConverter:
    def __init__(self, raw_dir="data/raw/synthetic_v1", proc_dir="data/processed/tier1_graphs"):
        """
        Processes GMSH files into Dynamic Non-Dimensionalized PyG Graphs.
        Scaling: Mean Diameter (D_bar) per unique geometry.
        """
        # Find the path to this script file and create the directory
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent.parent
        self.raw_dir = project_root / raw_dir
        self.proc_dir = project_root / proc_dir
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_dynamic_dbar(self, nodes, sdf):
        """Calculates mean diameter (D_bar) for per-sample non-dimensionalization."""
        x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
        x_steps = np.linspace(x_min, x_max, 25)
        radii = []
        for i in range(len(x_steps) - 1):
            mask = (nodes[:, 0] >= x_steps[i]) & (nodes[:, 0] < x_steps[i + 1])
            if np.any(mask):
                radii.append(np.max(sdf[mask]))
        return 2.0 * np.mean(radii) if radii else 1.5

    def _generate_unified_grid(self, sanity_data):
        """Generates a single figure containing all sanity check samples."""
        n = len(sanity_data)
        if n == 0: return  # Handle case with 0 samples

        fig, axes = plt.subplots(n, 2, figsize=(15, 3 * n))
        if n == 1: axes = np.expand_dims(axes, 0)  # Handle single sample case

        fig.suptitle("Unified Sanity Check: ND-SDF & Capped Shear Potential", fontsize=16)

        for i, (data, filename, d_bar) in enumerate(sanity_data):
            nodes = data.x.numpy()
            sdf_nd = data.sdf.numpy().flatten()
            shear_nd = data.shear_pot.numpy().flatten()

            # Column 1: ND-SDF
            ax1 = axes[i, 0]
            sc1 = ax1.scatter(nodes[:, 0], nodes[:, 1], c=sdf_nd, cmap='viridis', s=1)
            ax1.set_title(f"{filename} | SDF (D_bar: {d_bar:.2f})")
            ax1.set_aspect('equal')
            plt.colorbar(sc1, ax=ax1)

            # Column 2: Capped Shear Potential
            ax2 = axes[i, 1]
            sc2 = ax2.scatter(nodes[:, 0], nodes[:, 1], c=shear_nd, cmap='magma', s=1)
            ax2.set_title(f"Shear Potential (Max: {shear_nd.max():.2f})")
            ax2.set_aspect('equal')
            plt.colorbar(sc2, ax=ax2)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        print("\nDisplaying Unified Sanity Grid. Close window to finish processing...")
        plt.show()

    def process_file(self, filename):
        msh_path = self.raw_dir / filename
        pt_path = self.proc_dir / filename.replace(".msh", ".pt")

        mesh = meshio.read(msh_path)
        nodes = mesh.points[:, :2]

        wall_indices = np.unique(mesh.cells_dict["line"].flatten())
        wall_coords = nodes[wall_indices]
        tree = KDTree(wall_coords)
        dist, _ = tree.query(nodes)
        sdf_dim = dist.reshape(-1, 1).astype(np.float32)

        d_bar = self._calculate_dynamic_dbar(nodes, sdf_dim)
        nodes_nd = nodes / d_bar
        sdf_nd = sdf_dim / d_bar
        shear_pot_nd = 1.0 / (sdf_nd + 0.1)  # Capped at 10.0

        triangles = mesh.cells_dict["triangle"]
        edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)

        data = Data(
            x=torch.tensor(nodes_nd, dtype=torch.float32),
            edge_index=edge_index,
            sdf=torch.tensor(sdf_nd, dtype=torch.float32),
            shear_pot=torch.tensor(shear_pot_nd, dtype=torch.float32),
            d_bar=torch.tensor([d_bar], dtype=torch.float32)
        )

        torch.save(data, pt_path)
        return data, d_bar

    def run(self, n_sanity=5):
        if not self.raw_dir.exists():
            print(f"Error: Raw directory not found at {self.raw_dir}")
            return

        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".msh")]
        sanity_samples = []

        print(f"Starting conversion of {len(files)} files...")
        print(f"Reading from: {self.raw_dir}")
        print(f"Saving to:   {self.proc_dir}")

        # Process and collect sanity samples first
        for i in range(min(n_sanity, len(files))):
            data, d_bar = self.process_file(files[i])
            sanity_samples.append((data, files[i], d_bar))

        # Show the unified plot
        self._generate_unified_grid(sanity_samples)

        # Process the rest silently
        for i in tqdm(range(n_sanity, len(files))):
            try:
                self.process_file(files[i])
            except Exception as e:
                print(f"Error in {files[i]}: {e}")


if __name__ == "__main__":
    converter = MeshToGraphConverter()
    converter.run(n_sanity=3)