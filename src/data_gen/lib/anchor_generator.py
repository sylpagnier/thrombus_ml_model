import logging
import json
import random
import numpy as np
import mph
import meshio
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any, List
from src.config import VesselConfig, PhysicsConfig
from src.utils.paths import get_project_root
from scipy.interpolate import NearestNDInterpolator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def list_anchor_candidate_json_paths(
    mesh_dir: Path,
    output_dir: Path,
    *,
    include_existing_npz: bool = False,
) -> List[Path]:
    """Meshes eligible for anchor CFD: ``vessel_*.json`` with non-empty ``.nas`` and ``.msh``.

    By default skips stems that already have ``.npz``. With ``include_existing_npz=True``,
    those stems are included so runs can overwrite outputs.
    """
    mesh_dir = Path(mesh_dir)
    output_dir = Path(output_dir)
    candidates: List[Path] = []
    if not mesh_dir.exists():
        return candidates
    for json_file in sorted(mesh_dir.glob("vessel_*.json")):
        stem = json_file.stem
        try:
            int(stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        nas_file = mesh_dir / f"{stem}.nas"
        msh_file = mesh_dir / f"{stem}.msh"
        if not nas_file.exists() or nas_file.stat().st_size == 0:
            continue
        if not msh_file.exists():
            continue
        if (output_dir / f"{stem}.npz").exists() and not include_existing_npz:
            continue
        candidates.append(json_file)
    return candidates


def summarize_anchor_inventory(mesh_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Count existing CFD outputs and meshes still missing ``.npz`` (compatible with incremental runs).

    By default ``run_batch`` skips existing ``.npz``; use ``allow_overwrite=True`` to replace them.

    ``candidate_pool_ready`` counts meshes with ``.msh`` beside ``.nas`` and no ``.npz`` yet.
    ``candidate_pool_including_npz`` is the same but includes stems that already have ``.npz``.
    """
    mesh_dir = Path(mesh_dir)
    output_dir = Path(output_dir)
    existing_npz = 0
    if output_dir.exists():
        existing_npz = len(list(output_dir.glob("vessel_*.npz")))
    json_files = sorted(mesh_dir.glob("vessel_*.json")) if mesh_dir.exists() else []
    mesh_with_nas = 0
    pending_missing_npz = 0
    for json_file in json_files:
        stem = json_file.stem
        nas_file = mesh_dir / f"{stem}.nas"
        out_file = output_dir / f"{stem}.npz"
        if not nas_file.exists() or nas_file.stat().st_size == 0:
            continue
        mesh_with_nas += 1
        if not out_file.exists():
            pending_missing_npz += 1
    return {
        "existing_npz": existing_npz,
        "mesh_json_with_valid_nas": mesh_with_nas,
        "pending_missing_npz": pending_missing_npz,
        "candidate_pool_ready": len(
            list_anchor_candidate_json_paths(mesh_dir, output_dir, include_existing_npz=False)
        ),
        "candidate_pool_including_npz": len(
            list_anchor_candidate_json_paths(mesh_dir, output_dir, include_existing_npz=True)
        ),
    }


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

    def __init__(self, phase="kinematics", mesh_dir=None, output_dir=None, template_path=None):
        self.vessel_config = VesselConfig(phase=phase)
        self.phys_cfg = PhysicsConfig(phase=phase)

        self.root_dir = get_project_root()

        # --- 1. Resolve Template Path ---
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = self.vessel_config.template_path

        # --- 2. Resolve Input/Output Paths ---
        # Handle Output Directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.vessel_config.output_dir

        # Handle Mesh Directory
        if mesh_dir:
            self.mesh_dir = Path(mesh_dir)
        else:
            self.mesh_dir = self.vessel_config.mesh_input_dir

        self.client: Optional[mph.Client] = None
        self.model: Optional[mph.Model] = None

        if not self.template_path.exists():
            raise FileNotFoundError(f"COMSOL template not found at: {self.template_path}")
        if not self.mesh_dir.exists():
            logger.warning(f"Mesh input directory does not exist: {self.mesh_dir}")

    def _final_target_output_dir(self) -> Path:
        """Directory for the final target n outputs.

        Kinematics now always writes final anchors to ``n_<target>`` for consistency with
        continuation layouts. Kinematics keeps the base output directory.
        """
        if self.vessel_config.phase == "kinematics":
            return self.output_dir / f"n_{float(self.phys_cfg.n):.3f}"
        return self.output_dir

    def target_output_dir(self) -> Path:
        """Public accessor for current final-target output directory."""
        return self._final_target_output_dir()

    def __enter__(self):
        logger.info(f"Connecting to COMSOL... Loading: {self.template_path.name}")
        self.client = mph.start()
        self.model = self.client.load(str(self.template_path))
        self._set_global_physics_parameters()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._final_target_output_dir().mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            logger.info("Disconnecting from COMSOL...")
            self.client.clear()

    def _set_global_physics_parameters(self):
        logger.info(f"Setting global physics in {self.phys_cfg.viscosity_model} mode.")

        # Update Parameters (Global)
        self.model.parameter('rho_fluid', f'{self.phys_cfg.rho} [kg/m^3]')
        self.model.parameter('Re_target', str(self.phys_cfg.re_target))
        self.model.parameter('mu_ref', f'{self.phys_cfg.mu_newtonian} [Pa*s]')
        self.model.parameter('mu_inf', f'{self.phys_cfg.mu_inf} [Pa*s]')
        self.model.parameter('mu_0', f'{self.phys_cfg.mu_0} [Pa*s]')
        self.model.parameter('lambda_cy', f'{self.phys_cfg.lam} [s]')
        self.model.parameter('n_index', str(self.phys_cfg.n))
        self.model.parameter('a_yasuda', str(self.phys_cfg.a))

        # Update Variables (Component Level)
        var_node = self.model.java.component('comp1').variable('var1')

        if self.phys_cfg.viscosity_model == "carreau":
            carreau_expr = 'mu_inf + (mu_0 - mu_inf) * (1 + (lambda_cy * spf.sr)^a_yasuda)^((n_index - 1) / a_yasuda)'
            var_node.set('mu_final', carreau_expr)
        else:
            # Newtonian fallback
            var_node.set('mu_final', 'mu_ref')

    def _evaluate_at_coords(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            interp.set("expr", ["u", "v", "p", "mu_final"])
            interp.setInterpolationCoordinates(coords_T.tolist())
            data = interp.getData()

            # Robust unpacking: check if data has all 4 fields
            if len(data) < 4:
                raise ValueError(f"COMSOL returned incomplete data. Shape: {len(data)}")

            u = np.array(data[0])
            v = np.array(data[1])
            p = np.array(data[2])
            mu = np.array(data[3])

            return u, v, p, mu

        except Exception as e:
            logger.error(f"COMSOL Evaluation failed: {e}")
            nan_arr = np.full(coords.shape[0], np.nan)
            return nan_arr, nan_arr, nan_arr, nan_arr

        finally:
            try:
                results.numerical().remove(interp_name)
            except Exception:
                pass

    def _process_single_anchor(
        self, json_file: Path, mesh_j, import_tag, *, allow_overwrite: bool = False, continuation_steps: Optional[List[float]] = None
    ) -> bool:
        """Run COMSOL for one vessel; write ``.npz`` if field checks pass. Returns True if saved."""
        file_stem = json_file.stem
        try:
            i = int(file_stem.split("_")[1])
        except (ValueError, TypeError, IndexError):
            return False

        nas_file = self.mesh_dir / f"{file_stem}.nas"
        msh_file = self.mesh_dir / f"{file_stem}.msh"

        # Build sequence of n_values to solve (continuation steps + target)
        n_sequence = continuation_steps.copy() if continuation_steps else []
        if self.phys_cfg.n not in n_sequence:
            n_sequence.append(self.phys_cfg.n)

        if not nas_file.exists() or nas_file.stat().st_size == 0:
            return False

        try:
            logger.debug(f"[{i}] Purging old solution data from COMSOL memory...")
            for tag in self.model.java.sol().tags():
                self.model.java.sol(tag).clearSolutionData()

            with open(json_file, "r") as f:
                meta = json.load(f)
                d_bar = meta["d_bar"]

            self.model.parameter("D_eff", f"{d_bar:.8f} [m]")
            u_ref = self.phys_cfg.get_u_ref(d_bar)
            self.model.parameter("U_inlet", f"{u_ref:.8f} [m/s]")

            feat = mesh_j.feature(import_tag)
            safe_nas_path = str(nas_file).replace("\\", "/")
            feat.set("filename", safe_nas_path)

            mesh_j.run()
            try:
                n_verts = mesh_j.getNumVertex()
                logger.info(f"Sample {i}: Loaded Mesh with {n_verts} vertices.")
                if n_verts < 10:
                    raise RuntimeError(f"Mesh {i} is empty/corrupt (Vertices: {n_verts})")
            except Exception as e:
                logger.warning(f"Could not verify mesh stats for {i}: {e}")

            mesh = meshio.read(msh_file)
            target_nodes = mesh.points[:, :2]

            # --- THE CONTINUATION LOOP ---
            for step_idx, n_val in enumerate(n_sequence):
                is_target = (n_val == self.phys_cfg.n and step_idx == len(n_sequence) - 1)

                # Setup output paths
                if is_target:
                    out_file = self._final_target_output_dir() / f"{file_stem}.npz"
                else:
                    step_dir = self.output_dir / f"n_{n_val:.3f}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    out_file = step_dir / f"{file_stem}.npz"

                if out_file.exists() and not allow_overwrite:
                    logger.debug(f"[{i}] Skipping step n={n_val}, already exists.")
                    continue

                logger.info(f"[{i}] Solving for n_index = {n_val}...")
                self.model.parameter('n_index', str(n_val))
                self.model.solve()

                u, v, p, mu = self._evaluate_at_coords(target_nodes)
                u, v, p, mu = u.flatten(), v.flatten(), p.flatten(), mu.flatten()

                def fix_boundary_nans(field, coords):
                    mask = np.isnan(field)
                    if mask.any() and not mask.all():
                        interpolator = NearestNDInterpolator(coords[~mask], field[~mask])
                        field[mask] = interpolator(coords[mask])
                    return field

                u = fix_boundary_nans(u, target_nodes)
                v = fix_boundary_nans(v, target_nodes)
                p = fix_boundary_nans(p, target_nodes)
                mu = fix_boundary_nans(mu, target_nodes)

                try:
                    line_cells = mesh.get_cells_type("line")
                    line_tags = mesh.get_cell_data("gmsh:physical", "line")
                    has_lines = len(line_cells) > 0
                except Exception:
                    has_lines = False

                if has_lines:
                    outlet_node_indices = []
                    outlet_tags = [
                        tag_id for name, tag_id in self.vessel_config.TAGS.items()
                        if "Outlet" in name
                    ]
                    for j, tag in enumerate(line_tags):
                        if tag in outlet_tags:
                            outlet_node_indices.extend(line_cells[j])
                    if outlet_node_indices:
                        unique_indices = np.unique(outlet_node_indices)
                        valid_indices = unique_indices[unique_indices < len(p)]
                        if len(valid_indices) > 0:
                            p_offset = np.mean(p[valid_indices])
                            p = p - p_offset

                nan_u = np.isnan(u).sum()
                nan_v = np.isnan(v).sum()
                nan_p = np.isnan(p).sum()
                nan_mu = np.isnan(mu).sum()
                total_nodes = len(u)

                if nan_u > 0 or nan_v > 0 or nan_p > 0 or nan_mu > 0:
                    logger.warning(
                        f"NaNs detected in {nas_file.name} at n={n_val} | Total Nodes: {total_nodes} | "
                        f"NaN counts -> u: {nan_u}, v: {nan_v}, p: {nan_p}, mu: {nan_mu}"
                    )
                    return False

                p_std = np.std(p)
                u_max = np.max(np.abs(u))
                if p_std < 1e-9 or u_max < 1e-7:
                    logger.warning(f"[{i}] Skipping: Trivial solution detected at n={n_val}")
                    return False

                np.savez(
                    out_file,
                    x=target_nodes[:, 0],
                    y=target_nodes[:, 1],
                    u=u,
                    v=v,
                    p=p,
                    mu=mu,
                    d_bar=d_bar,
                    config_id=i,
                    carreau_n=n_val  # Crucial for your dataloader later
                )
            return True

        except Exception as e:
            logger.error(f"Error on {i}: {e}")
            logger.debug(f"[{i}] Purging old solution data from COMSOL memory...")
            for tag in self.model.java.sol().tags():
                self.model.java.sol(tag).clearSolutionData()
            return False

    def run_batch(
        self,
        max_new: int = 500,
        max_json_to_scan: Optional[int] = None,
        shuffle_candidates: bool = False,
        shuffle_seed: Optional[int] = None,
        allow_overwrite: bool = False,
        continuation_steps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Write up to ``max_new`` new healthy ``vessel_*.npz`` files.

        Builds the pool of geometries (valid ``.nas`` + ``.msh``). By default only stems without
        ``.npz`` are eligible; with ``allow_overwrite=True``, existing ``.npz`` files may be replaced.
        Then walks candidates **in order** until enough saves succeed or the list is exhausted.
        Failed solves (NaNs, trivial flow, exceptions) **do not** count toward ``max_new``; each
        failure skips to the **next** CFD-ready geometry so the batch still aims for ``max_new``
        successes when enough candidates exist.

        Parameters
        ----------
        max_new
            Target number of new ``.npz`` files to write this run.
        max_json_to_scan
            Optional cap on how many **candidates** to attempt (after building the pool). ``None`` = try
            every candidate. Prefer a large pool of meshes or leave unset so failures can be offset by
            later indices (the old behavior truncated the global sorted list *before* filtering, which
            could hide viable vessels).
        shuffle_candidates
            If True, randomize candidate order (e.g. spread load across geometry types).
        shuffle_seed
            Seed for shuffling; only used when ``shuffle_candidates`` is True.
        allow_overwrite
            If True, include stems that already have ``.npz`` and replace files after a successful solve.
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target_output_dir = self._final_target_output_dir()
        target_output_dir.mkdir(parents=True, exist_ok=True)
        existing_npz = len(list(target_output_dir.glob("vessel_*.npz")))
        candidates = list_anchor_candidate_json_paths(
            self.mesh_dir, target_output_dir, include_existing_npz=allow_overwrite
        )
        pool_full = len(candidates)

        if shuffle_candidates:
            rng = random.Random(shuffle_seed)
            rng.shuffle(candidates)

        if max_json_to_scan is not None:
            candidates = candidates[: int(max_json_to_scan)]

        logger.info(
            f"Anchor batch: existing .npz={existing_npz}, target new successes={max_new}, "
            f"candidate pool (no .npz, has .nas+.msh)={pool_full}, "
            f"will attempt min(len(pool), cap)={len(candidates)} geometries."
        )
        logger.info(
            "Failed solves (exceptions, NaNs, trivial flow) do not count toward the target; "
            "the batch continues with the next CFD-ready geometry until the target is met or "
            "the candidate list is exhausted."
        )

        try:
            mesh_j = self.model.java.component("comp1").mesh("mesh1")
            import_tag = _get_import_feature_tag(mesh_j)
        except Exception as e:
            logger.critical(f"Setup failed: {e}")
            return {
                "existing_before": existing_npz,
                "requested_new": max_new,
                "new_written": 0,
                "attempted": 0,
                "failed_or_discarded": 0,
                "pool_full": pool_full,
                "pool_attempted": len(candidates),
                "pool_exhausted": True,
                "setup_failed": True,
            }

        new_written = 0
        attempted = 0
        n_failed = 0
        for json_file in tqdm(candidates, desc="Anchors"):
            if new_written >= max_new:
                break
            attempted += 1
            if self._process_single_anchor(
                json_file,
                mesh_j,
                import_tag,
                allow_overwrite=allow_overwrite,
                continuation_steps=continuation_steps,
            ):
                new_written += 1
            else:
                n_failed += 1

        pool_exhausted = new_written < max_new and attempted >= len(candidates)
        if new_written < max_new:
            logger.warning(
                f"Anchor batch finished short: {new_written}/{max_new} new .npz after "
                f"{attempted} attempt(s) ({n_failed} failed or discarded, {new_written} saved). "
                + (
                    "All CFD-ready candidates in this pass were tried — no more geometries left to "
                    "reach the target; generate more vessel meshes for this phase or raise "
                    "max_json_to_scan / remove the scan cap."
                    if pool_exhausted
                    else "Raise max_json_to_scan (or leave it unset) to try more existing geometries."
                )
            )
        else:
            logger.info(f"Anchor batch: wrote {new_written} new .npz (target was {max_new}).")

        return {
            "existing_before": existing_npz,
            "requested_new": max_new,
            "new_written": new_written,
            "attempted": attempted,
            "failed_or_discarded": n_failed,
            "pool_full": pool_full,
            "pool_attempted": len(candidates),
            "pool_exhausted": pool_exhausted,
            "setup_failed": False,
        }


def _prompt_int_choice(label: str, allowed: Tuple[int, ...]) -> int:
    """Read an integer from stdin until it is one of ``allowed``."""
    allowed_str = "/".join(str(x) for x in allowed)
    while True:
        raw = input(f"{label} ({allowed_str}): ").strip()
        try:
            v = int(raw)
        except ValueError:
            print(f"  Enter an integer: {allowed_str}")
            continue
        if v in allowed:
            return v
        print(f"  Must be one of: {allowed_str}")


if __name__ == "__main__":
    try:
        def _prompt_int(label, default):
            while True:
                raw = input(f"{label} [{default}]: ").strip()
                if raw == "":
                    return int(default)
                try:
                    v = int(raw)
                    if v < 0:
                        print("Enter a non-negative integer.")
                        continue
                    return v
                except ValueError:
                    print("Invalid input. Enter an integer value.")

        def _prompt_write_mode() -> bool:
            """Return True if overwrite mode, False for add-only."""
            while True:
                raw = input("Write mode [1=add new files only / 2=overwrite existing .npz] [1]: ").strip()
                if raw in ("", "1"):
                    return False
                if raw == "2":
                    return True
                print("  Enter 1 or 2.")

        phase_n = _prompt_int_choice("Phase", (1, 2))
        phase = f"phase{phase_n}"
        gen = AnchorGenerator(phase=phase)
        target_dir = gen.target_output_dir()
        inv = summarize_anchor_inventory(gen.mesh_dir, target_dir)
        total_v = int(inv["mesh_json_with_valid_nas"])
        have_npz = int(inv["existing_npz"])
        remaining = int(inv["pending_missing_npz"])
        ready_add = int(inv["candidate_pool_ready"])
        ready_all = int(inv["candidate_pool_including_npz"])
        print("\n--- Anchor CFD inventory ---")
        print(f"  Output: {target_dir}")
        print(f"  Mesh:   {gen.mesh_dir}")
        print(f"  Total number of phase vessels: {total_v}")
        print(f"  Number of anchors already generated: {have_npz}")
        print(f"  Number of non-anchors remaining: {remaining}")
        if remaining > ready_add:
            print(
                f"  ({remaining - ready_add} of those still need a .msh export before CFD.)"
            )
        print()
        allow_overwrite = _prompt_write_mode()
        pool = ready_all if allow_overwrite else ready_add
        default_more = min(pool, 50) if pool > 0 else 0
        if pool == 0:
            if allow_overwrite:
                print("No meshes are CFD-ready (need .json + non-empty .nas + .msh).")
            else:
                msg = "Nothing to add (need .json + non-empty .nas + .msh, and no .npz yet)."
                if remaining > 0:
                    msg += " Some meshes lack .msh — re-run mesh export for those vessels."
                elif total_v == 0:
                    msg = "No vessel meshes found in the mesh directory."
                print(msg)
            raise SystemExit(0)
        mode_note = "CFD runs to attempt" if allow_overwrite else "new CFD samples to generate"
        asked = _prompt_int(f"How many {mode_note}", default_more)
        if asked == 0:
            print("Exiting (0 requested).")
            raise SystemExit(0)
        max_new = min(asked, pool)
        if asked > pool:
            print(f"Requested {asked} but only {pool} mesh(es) match this mode; running {max_new}.")
        with gen:
            gen.run_batch(max_new=max_new, allow_overwrite=allow_overwrite)
    except SystemExit:
        raise
    except Exception as e:
        print(e)