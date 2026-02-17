import logging
import json
import numpy as np
import mph
import meshio
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
from src.config import VesselConfig, PhysicsConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class AnchorGenerator:
    """
    Automates COMSOL CFD simulations based on synthetic vessel meshes.
    """

    def __init__(self):
        self.vessel_config = VesselConfig
        self.phys_cfg = PhysicsConfig
        # Resolve root relative to this script location (src/anchor_generator.py -> root)
        self.root_dir = Path(__file__).resolve().parent.parent

        # --- 1. Resolve Template Path ---
        self.template_path = self.root_dir / VesselConfig.template_path

        # --- 2. Resolve Input/Output Paths ---
        # Handle absolute vs relative paths automatically from Config
        self.output_dir = (Path(VesselConfig.output_dir) if Path(VesselConfig.output_dir).is_absolute()
                           else self.root_dir / VesselConfig.output_dir)

        self.mesh_dir = (Path(VesselConfig.mesh_input_dir) if Path(VesselConfig.mesh_input_dir).is_absolute()
                         else self.root_dir / VesselConfig.mesh_input_dir)

        self.client: Optional[mph.Client] = None
        self.model: Optional[mph.Model] = None

        # Validation
        if not self.template_path.exists():
            raise FileNotFoundError(f"COMSOL template not found at: {self.template_path}")
        if not self.mesh_dir.exists():
            logger.warning(f"Mesh input directory does not exist: {self.mesh_dir}")

    def __enter__(self):
        """Context manager entry: Start COMSOL."""
        logger.info(f"Connecting to COMSOL... Loading: {self.template_path.name}")
        self.client = mph.start()
        self.model = self.client.load(str(self.template_path))
        self._set_global_physics_parameters()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            logger.info("Disconnecting from COMSOL...")
            self.client.clear()

    def _set_global_physics_parameters(self):
        """Initializes COMSOL global parameters from PhysicsConfig."""
        logger.info("Setting global physics parameters in COMSOL.")
        # Fluid density and Newtonian viscosity
        self.model.parameter('rho_fluid', f'{self.phys_cfg.rho} [kg/m^3]')
        self.model.parameter('mu_ref', f'{self.phys_cfg.mu_newtonian} [Pa*s]')
        self.model.parameter('Re_target', str(self.phys_cfg.re_target))

        # Carreau-Yasuda parameters (for non-Newtonian sims)
        self.model.parameter('mu_inf', f'{self.phys_cfg.mu_inf} [Pa*s]')
        self.model.parameter('mu_0', f'{self.phys_cfg.mu_0} [Pa*s]')
        self.model.parameter('lambda_cy', f'{self.phys_cfg.lam} [s]')
        self.model.parameter('n_index', str(self.phys_cfg.n))
        self.model.parameter('a_yasuda', str(self.phys_cfg.a))

    def _evaluate_at_coords(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        High-performance evaluation using COMSOL Java API Interp feature.
        """
        # Prepare coordinates (Transpose for Java: 2 rows, N columns)
        coords_T = coords.T

        # Access Java layer
        model_j = self.model.java
        results = model_j.result()

        # Unique tag for this operation to avoid collisions
        interp_name = "py_interp_temp"

        try:
            # Create numerical interpolation feature
            interp_tag = results.numerical().create(interp_name, "Interp").tag()
            interp = results.numerical(interp_tag)

            interp.set("data", "dset1")  # Ensure 'dset1' matches your COMSOL study
            interp.set("expr", ["u", "v", "p"])

            # Pass coordinates
            interp.setInterpolationCoordinates(coords_T.tolist())

            # Compute
            data = interp.getData()

            # Map results
            u = np.array(data[0])
            v = np.array(data[1])
            p = np.array(data[2])

            return u, v, p

        except Exception as e:
            logger.error(f"COMSOL Evaluation failed: {e}")
            # Return NaNs so the batch loop can handle it gracefully
            nan_arr = np.full(coords.shape[0], np.nan)
            return nan_arr, nan_arr, nan_arr

        finally:
            # Cleanup: Always remove the temporary feature
            try:
                results.numerical().remove(interp_name)
            except Exception:
                pass

    def _get_import_feature_tag(self, mesh_j) -> str:
        """Helper to dynamically find the Import feature in the COMSOL mesh sequence."""
        all_tags = mesh_j.feature().tags()
        for tag in all_tags:
            if mesh_j.feature(tag).getType() == 'Import':
                return tag

        # Fallback check
        if 'imp1' in all_tags:
            return 'imp1'

        raise RuntimeError("No 'Import' feature found in the COMSOL model mesh sequence.")

    def run_batch(self, start_idx: int = 0, end_idx: int = 50, max_anchors: int = 500):
        """
        Args:
            start_idx: Start index for batch
            end_idx: End index for batch
            max_anchors: The index cutoff. Meshes > this ID will NOT be simulated (Physics Set).
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")

        logger.info(f"Batch processing ID {start_idx} to {end_idx}. Anchors limit: {max_anchors}")

        # Get the Java mesh object once
        try:
            mesh_j = self.model.java.component('comp1').mesh('mesh1')
            import_tag = self._get_import_feature_tag(mesh_j)
        except Exception as e:
            logger.critical(f"Setup failed: {e}")
            return

        for i in tqdm(range(start_idx, end_idx), desc="Processing"):
            # Logic: If we are past the anchor limit, skip simulation.
            if i >= max_anchors:
                continue

            # Paths
            nas_file = self.mesh_dir / f"vessel_{i}.nas"
            msh_file = self.mesh_dir / f"vessel_{i}.msh" # Check existence only
            json_file = self.mesh_dir / f"vessel_{i}.json"
            out_file = self.output_dir / f"vessel_{i}.npz"

            if not nas_file.exists() or not json_file.exists():
                continue

            if out_file.exists():
                continue

            try:
                with open(json_file, 'r') as f:
                    meta = json.load(f)
                    d_bar = meta['d_bar']
                    num_outlets = meta.get('num_outlets', 1)

                # Dynamic Tag Selection
                # Inlet is always 101, Outlet_1 is 102.
                # Walls shift based on num_outlets.
                wall_tag = 104 if num_outlets == 2 else 103
                outlet_2_active = (num_outlets == 2)

                # Update COMSOL Boundary Conditions
                # Assuming 'wall_selection' and 'outlet2_feature' are named in your .mph
                self.model.java.component('comp1').physics('spf').feature('wall1').selection().set(wall_tag)

                if outlet_2_active:
                    # Ensure the second outlet is enabled and set to Tag 103
                    self.model.java.component('comp1').physics('spf').feature('out2').active(True)
                    self.model.java.component('comp1').physics('spf').feature('out2').selection().set(103)
                else:
                    self.model.java.component('comp1').physics('spf').feature('out2').active(False)

                # 1. Update dynamic parameters per vessel
                self.model.parameter('D_eff', f'{d_bar:.8f} [m]')

                # 2. Calculate u_ref to match MeshToGraph scaling
                # COMSOL can calculate this itself if you set a formula in the .mph,
                # but pushing it explicitly ensures 100% parity with your graph labels.
                u_ref = (self.phys_cfg.re_target * self.phys_cfg.mu_newtonian) / (self.phys_cfg.rho * d_bar)
                self.model.parameter('U_inlet', f'{u_ref:.8f} [m/s]')

                # 3. Update Mesh & Solve
                feat = mesh_j.feature(import_tag)
                feat.set('filename', str(nas_file))
                mesh_j.run()
                self.model.solve()

                # 5. Extract & Validate
                mesh = meshio.read(msh_file)
                target_nodes = mesh.points[:, :2]

                u, v, p = self._evaluate_at_coords(target_nodes)

                # PRESSURE PINNING: Subtract mean outlet pressure
                if "line" in mesh.cells_dict and "gmsh:physical" in mesh.cell_data_dict:
                    line_cells = mesh.cells_dict["line"]
                    line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
                    outlet_node_indices = []

                    # Use standard tags: Outlet_1 (102) and Outlet_2 (103)
                    for j, tag in enumerate(line_tags):
                        if tag in [102, 103]:
                            outlet_node_indices.extend(line_cells[j])

                    if outlet_node_indices:
                        # Identify unique nodes and calculate mean pressure at the outlet
                        p_offset = np.mean(p[np.unique(outlet_node_indices)])
                        p = p - p_offset

                if not np.isfinite(u).all():
                    logger.warning(f"NaNs or infinities detected in {i}")
                    continue

                np.savez(
                    out_file,
                    x=target_nodes[:, 0], y=target_nodes[:, 1],
                    u=u, v=v, p=p,
                    d_bar=d_bar,
                    config_id=i
                )

            except Exception as e:
                logger.error(f"Error on {i}: {e}")
                continue

if __name__ == "__main__":
    try:
        # Run the pipeline
        with AnchorGenerator() as generator:
            generator.run_batch(start_idx=0, end_idx=50)

    except KeyboardInterrupt:
        logger.info("Batch run interrupted by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")