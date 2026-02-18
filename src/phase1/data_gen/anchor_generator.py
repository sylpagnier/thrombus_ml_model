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


def _get_import_feature_tag(mesh_j) -> str:
    all_tags = mesh_j.feature().tags()
    for tag in all_tags:
        if mesh_j.feature(tag).getType() == 'Import':
            return tag
    if 'imp1' in all_tags:
        return 'imp1'
    raise RuntimeError("No 'Import' feature found in the COMSOL model mesh sequence.")


class AnchorGenerator:
    """
    Automates COMSOL CFD simulations based on synthetic vessel meshes.
    """

    def __init__(self, mesh_dir=None, output_dir=None, template_path=None):
        self.vessel_config = VesselConfig()
        self.phys_cfg = PhysicsConfig()
        self.root_dir = Path(__file__).resolve().parent.parent

        # --- 1. Resolve Template Path ---
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = self.root_dir / VesselConfig.template_path

        # --- 2. Resolve Input/Output Paths ---
        # Handle Output Directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = (Path(VesselConfig.output_dir) if Path(VesselConfig.output_dir).is_absolute()
                               else self.root_dir / VesselConfig.output_dir)

        # Handle Mesh Directory
        if mesh_dir:
            self.mesh_dir = Path(mesh_dir)
        else:
            self.mesh_dir = (Path(VesselConfig.mesh_input_dir) if Path(VesselConfig.mesh_input_dir).is_absolute()
                             else self.root_dir / VesselConfig.mesh_input_dir)

        self.client: Optional[mph.Client] = None
        self.model: Optional[mph.Model] = None

        if not self.template_path.exists():
            raise FileNotFoundError(f"COMSOL template not found at: {self.template_path}")
        if not self.mesh_dir.exists():
            logger.warning(f"Mesh input directory does not exist: {self.mesh_dir}")

    def __enter__(self):
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
        logger.info("Setting global physics parameters in COMSOL.")
        self.model.parameter('rho_fluid', f'{self.phys_cfg.rho} [kg/m^3]')
        self.model.parameter('mu_ref', f'{self.phys_cfg.mu_newtonian} [Pa*s]')
        self.model.parameter('Re_target', str(self.phys_cfg.re_target))

        self.model.parameter('mu_inf', f'{self.phys_cfg.mu_inf} [Pa*s]')
        self.model.parameter('mu_0', f'{self.phys_cfg.mu_0} [Pa*s]')
        self.model.parameter('lambda_cy', f'{self.phys_cfg.lam} [s]')
        self.model.parameter('n_index', str(self.phys_cfg.n))
        self.model.parameter('a_yasuda', str(self.phys_cfg.a))

    def _evaluate_at_coords(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        High-performance evaluation using COMSOL Java API Interp feature.
        """
        coords_T = coords.T
        model_j = self.model.java
        results = model_j.result()
        interp_name = "py_interp_temp"

        try:
            interp_tag = results.numerical().create(interp_name, "Interp").tag()
            interp = results.numerical(interp_tag)

            interp.set("data", "dset1")
            # Explicitly cast to Java String array to ensure stability
            interp.set("expr", ["u", "v", "p"])

            interp.setInterpolationCoordinates(coords_T.tolist())
            data = interp.getData()

            # Robust unpacking: check if data is 2D
            if len(data) < 3:
                raise ValueError(f"COMSOL returned incomplete data. Shape: {len(data)}")

            u = np.array(data[0])
            v = np.array(data[1])
            p = np.array(data[2])

            return u, v, p

        except Exception as e:
            logger.error(f"COMSOL Evaluation failed: {e}")
            nan_arr = np.full(coords.shape[0], np.nan)
            return nan_arr, nan_arr, nan_arr

        finally:
            try:
                results.numerical().remove(interp_name)
            except Exception:
                pass

    def run_batch(self, start_idx: int = 0, end_idx: int = 50, max_anchors: int = 500):
        if not self.model:
            raise RuntimeError("Model not loaded.")

        logger.info(f"Batch processing ID {start_idx} to {end_idx}. Anchors limit: {max_anchors}")

        try:
            mesh_j = self.model.java.component('comp1').mesh('mesh1')
            import_tag = _get_import_feature_tag(mesh_j)
        except Exception as e:
            logger.critical(f"Setup failed: {e}")
            return

        for i in tqdm(range(start_idx, end_idx), desc="Processing"):
            if i >= max_anchors:
                continue

            nas_file = self.mesh_dir / f"vessel_{i}.nas"
            msh_file = self.mesh_dir / f"vessel_{i}.msh"
            json_file = self.mesh_dir / f"vessel_{i}.json"
            out_file = self.output_dir / f"vessel_{i}.npz"

            if not nas_file.exists() or not json_file.exists():
                continue

            # Skip empty files which cause Solver/Import crashes
            if nas_file.stat().st_size == 0:
                logger.warning(f"Skipping empty mesh file: {nas_file}")
                continue

            if out_file.exists():
                continue

            try:
                with open(json_file, 'r') as f:
                    meta = json.load(f)
                    d_bar = meta['d_bar']

                self.model.parameter('D_eff', f'{d_bar:.8f} [m]')

                # Select viscosity based on physics tier.
                # For Tier 1 (Newtonian), use mu_newtonian. For Tier 2 (Non-Newtonian), typically mu_inf is the reference.
                ref_mu = self.phys_cfg.mu_inf if self.phys_cfg.mu_inf != self.phys_cfg.mu_newtonian else self.phys_cfg.mu_newtonian
                u_ref = (self.phys_cfg.re_target * ref_mu) / (self.phys_cfg.rho * d_bar)

                self.model.parameter('U_inlet', f'{u_ref:.8f} [m/s]')

                feat = mesh_j.feature(import_tag)
                safe_nas_path = str(nas_file).replace("\\", "/")
                feat.set('filename', safe_nas_path)

                # Rebuild mesh and Solve
                mesh_j.run()
                # --- VERIFICATION BLOCK ---
                try:
                    # Get vertex count (Java integer)
                    n_verts = mesh_j.getNumVertex()

                    # 1. Log it so you can see it changing in the terminal
                    logger.info(f"Sample {i}: Loaded Mesh with {n_verts} vertices.")

                    # 2. Safety Check: If it's too small, the import probably failed
                    if n_verts < 10:
                        raise RuntimeError(f"Mesh {i} is empty/corrupt (Vertices: {n_verts})")

                except Exception as e:
                    # If the specific API call fails (version mismatch), just warn and continue
                    logger.warning(f"Could not verify mesh stats for {i}: {e}")
                # --------------------------

                self.model.solve()

                # --- Extraction & Pinning ---
                mesh = meshio.read(msh_file)
                target_nodes = mesh.points[:, :2]

                u, v, p = self._evaluate_at_coords(target_nodes)

                # ---Robust Meshio Access ---
                # Check if 'line' cells exist using modern meshio API
                # This handles both list-of-arrays and single-array internals
                try:
                    line_cells = mesh.get_cells_type("line")  # Returns (N, 2)
                    line_tags = mesh.get_cell_data("gmsh:physical", "line")  # Returns (N,)

                    has_lines = len(line_cells) > 0
                except Exception:
                    has_lines = False

                if has_lines:
                    outlet_node_indices = []
                    # Dynamically fetch tags for any 'Outlet' defined in config
                    outlet_tags = [
                        tag_id for name, tag_id in self.vessel_config.TAGS.items()
                        if "Outlet" in name
                    ]

                    for j, tag in enumerate(line_tags):
                        if tag in outlet_tags:
                            # line_cells[j] contains the vertex indices for that line segment
                            outlet_node_indices.extend(line_cells[j])

                    if outlet_node_indices:
                        unique_indices = np.unique(outlet_node_indices)
                        # Ensure indices are within the bounds of the pressure array
                        valid_indices = unique_indices[unique_indices < len(p)]
                        if len(valid_indices) > 0:
                            # Pinning: Subtract mean outlet pressure to set relative gauge pressure
                            p_offset = np.mean(p[valid_indices])
                            p = p - p_offset
                # ---------------------------------

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
                # Clear model results to save memory if solve failed
                self.model.clear()
                continue


if __name__ == "__main__":
    try:
        with AnchorGenerator() as generator:
            generator.run_batch(start_idx=0, end_idx=50)
    except KeyboardInterrupt:
        logger.info("Batch run interrupted by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")