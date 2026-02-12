import mph
import numpy as np
import meshio
from pathlib import Path
from tqdm import tqdm


class AnchorGenerator:
    def __init__(self, template_path, mesh_dir, output_dir):
        current_script = Path(__file__).resolve()
        project_root = current_script.parent.parent.parent.parent

        abs_template = project_root / template_path
        self.mesh_dir = project_root / mesh_dir
        self.output_dir = project_root / output_dir

        if not abs_template.exists():
            raise FileNotFoundError(f"COMSOL template not found at: {abs_template}")

        print(f"Connecting to COMSOL... Loading: {abs_template.name}")
        self.client = mph.start()
        self.model = self.client.load(str(abs_template))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _measure_inlet_width(self, msh_path):
        """
        Reads the .msh file, finds the Inlet (Tag 101), and calculates diameter.
        """
        try:
            mesh = meshio.read(msh_path)
        except Exception as e:
            print(f"Warning: Could not read {msh_path.name} to measure diameter. ({e})")
            return 0.0015  # Fallback

        # Gmsh Physical Groups: 'line' block usually holds the boundaries
        if "line" not in mesh.cell_data_dict["gmsh:physical"]:
            return 0.0015

        lines = mesh.cells_dict["line"]
        tags = mesh.cell_data_dict["gmsh:physical"]["line"]

        # Filter for Inlet Tag (101 is standard from vessel_generator.py)
        inlet_indices = [i for i, t in enumerate(tags) if t == 101]

        if not inlet_indices:
            return 0.0015

        # Get all node indices for the inlet lines
        inlet_node_indices = np.unique(lines[inlet_indices].flatten())

        # Get coordinates
        inlet_coords = mesh.points[inlet_node_indices]

        # Diameter = Max Y - Min Y (Assuming inlet is vertical at x=0)
        y_coords = inlet_coords[:, 1]
        diameter = np.max(y_coords) - np.min(y_coords)

        return float(diameter)

    def run_batch(self, start_idx=0, end_idx=50):
        print("\n--- Java Layer Setup ---")
        try:
            comp_j = self.model.java.component('comp1')
            mesh_j = comp_j.mesh('mesh1')
        except Exception as e:
            raise RuntimeError(f"Critical: Could not access 'comp1' or 'mesh1'. Error: {e}")

        import_tag = None
        all_tags = mesh_j.feature().tags()
        for tag in all_tags:
            if mesh_j.feature(tag).getType() == 'Import':
                import_tag = tag
                break

        if not import_tag:
            import_tag = 'imp1' if 'imp1' in all_tags else None
            if not import_tag: raise RuntimeError("No Import feature found.")

        print(f"Targeting Import Feature: {import_tag}")

        for i in tqdm(range(start_idx, end_idx), desc="Solving Anchors"):
            nas_file = self.mesh_dir / f"vessel_{i}.nas"
            msh_file = self.mesh_dir / f"vessel_{i}.msh"  # Helper for metadata

            if not nas_file.exists(): continue

            try:
                # 1. Measure ACTUAL Diameter from Mesh
                D_actual = self._measure_inlet_width(msh_file)

                # 2. Update COMSOL Parameter
                # This updates the 'D_inlet' parameter we defined earlier.
                # COMSOL will auto-calculate U_in = (Re*mu)/(rho*D) based on this.
                self.model.parameter('D_inlet', f'{D_actual:.6f} [m]')

                # 3. Load Mesh Geometry
                feat = mesh_j.feature(import_tag)
                feat.set('filename', str(nas_file))
                mesh_j.run()

                # 4. Solve
                self.model.solve()

                # 5. Extract Data
                data = self.model.evaluate(['x', 'y', 'u', 'v', 'p'])
                x_sol, y_sol, u, v, p = data

                # 6. Save
                if np.isnan(u).any() or np.isinf(u).any():
                    print(f"Skipping {i}: Solver produced NaNs")
                    continue
                np.savez(
                    self.output_dir / f"vessel_{i}.npz",
                    x=x_sol, y=y_sol, u=u, v=v, p=p,
                    d_inlet=D_actual  # Save metadata for verification
                )
            except Exception as e:
                print(f"Error solving vessel_{i}: {e}")


if __name__ == "__main__":
    generator = AnchorGenerator(
        template_path='comsol_models/phase1_template.mph',
        mesh_dir='data/raw/synthetic_v1',
        output_dir='data/raw/cfd_anchors'
    )
    generator.run_batch(start_idx=0, end_idx=125)