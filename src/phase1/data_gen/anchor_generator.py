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

        if "line" not in mesh.cells_dict:
            return 0.0015

        lines = mesh.cells_dict["line"]
        tags = mesh.cell_data_dict["gmsh:physical"]["line"]
        inlet_indices = [i for i, t in enumerate(tags) if t == 101]

        if not inlet_indices:
            return 0.0015

        inlet_node_indices = np.unique(lines[inlet_indices].flatten())
        inlet_coords = mesh.points[inlet_node_indices]
        y_coords = inlet_coords[:, 1]
        return float(np.max(y_coords) - np.min(y_coords))

    def _evaluate_at_coords(self, coords):
        """
        Uses COMSOL Java API to interpolate solution at specific (x, y) coordinates.
        coords: (N, 2) numpy array of [x, y]
        Returns: u, v, p (all shape (N,))
        """
        # 1. Prepare coordinates for COMSOL (Must be 2D array: [[x1, x2...], [y1, y2...]])
        coords_T = coords.T  # Shape (2, N)

        # 2. Get the Java 'Results' object
        # 'dset1' is the standard default solution dataset in COMSOL.
        # If your model uses a different one, check COMSOL GUI -> Results -> Datasets.
        model_j = self.model.java

        # Create a numerical "Interp" feature to evaluate arbitrary points
        # This is faster and more robust than 'model.evaluate' for unstructured lists
        results = model_j.result()
        interp_tag = results.numerical().create("interp1", "Interp").tag()
        interp = results.numerical(interp_tag)

        interp.set("data", "dset1")  # Target the solution dataset
        interp.set("expr", ["u", "v", "p"])  # Variables to evaluate

        # Pass coordinates. COMSOL expects specific Java double[][] format,
        # but JPype (used by mph) usually handles list-of-lists.
        interp.setInterpolationCoordinates(coords_T.tolist())

        # 3. Compute
        # getData() returns a flattened 1D array or 2D array depending on version.
        # Typically returns double[num_expr][num_points]
        data = interp.getData()

        # Cleanup (remove the temp feature to save memory)
        results.numerical().remove("interp1")

        # 4. Parse
        # data is typically [[u1, u2...], [v1, v2...], [p1, p2...]]
        u = np.array(data[0])
        v = np.array(data[1])
        p = np.array(data[2])

        return u, v, p

    def run_batch(self, start_idx=0, end_idx=50):
        print("\n--- Java Layer Setup ---")
        try:
            comp_j = self.model.java.component('comp1')
            mesh_j = comp_j.mesh('mesh1')
        except Exception as e:
            raise RuntimeError(f"Critical: Could not access 'comp1' or 'mesh1'. Error: {e}")

        # Find the Import feature
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
            msh_file = self.mesh_dir / f"vessel_{i}.msh"

            if not nas_file.exists(): continue

            try:
                # 1. Measure Diameter
                D_actual = self._measure_inlet_width(msh_file)
                self.model.parameter('D_inlet', f'{D_actual:.6f} [m]')

                # 2. Update Mesh
                feat = mesh_j.feature(import_tag)
                feat.set('filename', str(nas_file))
                mesh_j.run()

                # 3. Solve
                self.model.solve()

                # 4. Extract Data EXACTLY at Mesh Nodes
                # Load the mesh used to generate the graph
                mesh = meshio.read(msh_file)
                # Use only x, y (ignore z if 2D)
                target_nodes = mesh.points[:, :2]

                u, v, p = self._evaluate_at_coords(target_nodes)

                # 5. Save
                if np.isnan(u).any() or np.isinf(u).any():
                    print(f"Skipping {i}: Solver produced NaNs")
                    continue

                np.savez(
                    self.output_dir / f"vessel_{i}.npz",
                    x=target_nodes[:, 0],
                    y=target_nodes[:, 1],
                    u=u, v=v, p=p,
                    d_inlet=D_actual
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