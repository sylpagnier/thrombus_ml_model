import mph
import numpy as np
from pathlib import Path
from tqdm import tqdm


class AnchorGenerator:
    def __init__(self, template_path, mesh_dir, output_dir):
        current_script = Path(__file__).resolve()
        project_root = current_script.parent.parent.parent

        abs_template = project_root / template_path
        self.mesh_dir = project_root / mesh_dir
        self.output_dir = project_root / output_dir

        if not abs_template.exists():
            raise FileNotFoundError(f"COMSOL template not found at: {abs_template}")

        print(f"Connecting to COMSOL... Loading: {abs_template.name}")
        self.client = mph.start()
        self.model = self.client.load(str(abs_template))
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            if not nas_file.exists(): continue

            try:
                # 1. Load Mesh
                feat = mesh_j.feature(import_tag)
                feat.set('filename', str(nas_file))
                mesh_j.run()

                # 2. Solve
                self.model.parameter('D_eff', '0.0015 [m]')
                self.model.solve()

                # 3. Extract Data WITH COORDINATES
                # We extract x, y, u, v, p simultaneously to ensure alignment
                data = self.model.evaluate(['x', 'y', 'u', 'v', 'p'])
                x_sol, y_sol, u, v, p = data

                # 4. Save
                np.savez(
                    self.output_dir / f"vessel_{i}.npz",
                    x=x_sol, y=y_sol, u=u, v=v, p=p
                )
            except Exception as e:
                print(f"Error solving vessel_{i}: {e}")


if __name__ == "__main__":
    generator = AnchorGenerator(
        template_path='comsol_models/phase1_template.mph',
        mesh_dir='data/raw/synthetic_v1',
        output_dir='data/raw/cfd_anchors'
    )
    generator.run_batch(start_idx=0, end_idx=50)