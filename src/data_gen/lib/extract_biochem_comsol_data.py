import os
import json
import torch
import numpy as np
import pandas as pd
import meshio
from pathlib import Path
from scipy.spatial import cKDTree, KDTree
from torch_geometric.data import Data
from tqdm import tqdm
import glob
import re
from src.config import BIOCHEM_T_MAX, NodeFeat, VesselConfig, PhysicsConfig, BiochemConfig, biochem_comsol_time_cap_s
from src.utils.kinematics_paths import BIOCHEM_ANCHOR_KINE_RHEOLOGY, kinematics_anchor_graph_dir
from src.utils.paths import get_project_root
from src.data_gen.lib.node_feature_assembly import build_biochem_bc_x_tensor
from src.utils.channel_schema import BIO_Y_SCHEMA, attach_patient_anchor_graph_metadata
from src.data_gen.lib.centerline_utils import write_anchor_sidecar_from_masks
from src.data_gen.lib.kinematics_graph_builder import (
    build_kinematics_graph_from_comsol_steady,
    resolve_d_bar_si_from_sidecar_or_inlet,
)
from src.utils.units import MESH_UNIT_CM, assert_mesh_unit


class PatientDataExtractor:
    """
    Extracts and processes Eulerian node-wise COMSOL data into PyTorch Geometric Data objects.

    State ``y`` layout: indices 0–2 are u, v, p (non-dimensional); channel index 3
    (``STATE_CHANNEL_MU_EFF_ND`` in ``src.config``) is ``mu_effective`` via
    ``PhysicsConfig.viscosity_si_to_nd`` (canonical cross-phase ND viscosity reference).

    Default entry: ``python -m src.data_gen.lib.extract_biochem_comsol_data`` pulls solved
    ``comsol_models/phase2_nowound_XXX.mph`` (``patientXXX``) via ``pull_comsol_exports``, then
    builds graphs. Manual COMSOL txt only with ``--no-from-comsol``.

    --- Manual COMSOL Export Instructions ---
    Alternatively, export the exact node-wise data from COMSOL to match the .msh topology.
    IMPORTANT: export from ``Component 1 -> Mesh 1`` geometry coordinates (the solved component mesh),
    not directly from the raw Mesh Import object, otherwise node coordinates/order can drift and mapping fails.

    1. In COMSOL: Go to Results > Export > Data.
    2. Main Domain: Export domain nodes to `data/processed/cfd_results_biochem/<stem>.txt`
       Headers must map exactly to:
       x, y, u, v, p, mu_effective, rp, ap, apr, aps, PT, th, at, fg, fi, M, Mas, Mat
    3. Boundaries: Export Edge 2D coordinates (x, y) with "Time Selection: Last" to:
       - <stem>_inlet.txt
       - <stem>_outlet.txt
       - <stem>_wall.txt
    ----------------------------------
    """

    def __init__(self, phase="biochem_anchors", raw_dir=None, label_dir=None, proc_dir=None):
        self.root = get_project_root()
        self.vessel_cfg = VesselConfig(phase=phase)
        self.phys_cfg = PhysicsConfig(phase=phase)

        # Directory handling
        self.raw_dir = Path(raw_dir) if raw_dir else self.root / self.vessel_cfg.mesh_input_dir
        self.label_dir = Path(label_dir) if label_dir else self.root / self.vessel_cfg.output_dir
        self.proc_dir = Path(proc_dir) if proc_dir else self.root / self.vessel_cfg.graph_output_dir
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.kine_anchor_dir = kinematics_anchor_graph_dir(rheology=BIOCHEM_ANCHOR_KINE_RHEOLOGY)
        self.kine_anchor_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary mapping exact COMSOL export names to standardized internal names
        self.species_map = {
            'rp': 'RP', 'ap': 'AP', 'apr': 'APR', 'aps': 'APS',
            'PT': 'PT', 'th': 'T', 'at': 'AT', 'fg': 'FG',
            'fi': 'FI', 'M': 'M', 'Mas': 'Mas', 'Mat': 'Mat'
        }

        self.csv_fields = [ 'x', 'y', 'u', 'v', 'p', 'mu_effective' ] + list(self.species_map.keys())

    def _precompute_wls(self, edge_index, num_nodes, pos_tensor):
        """Computes the 2nd Order Polynomial Basis, WLS Inverse, and Condition Number."""
        row, col = edge_index
        pos_diff = pos_tensor[ col, :2 ] - pos_tensor[ row, :2 ]
        dx, dy = pos_diff[ :, 0 ], pos_diff[ :, 1 ]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        V = torch.stack([ dx, dy, 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2 ], dim=1)
        W = 1.0 / dist_sq

        V_unsqueezed = V.unsqueeze(2)
        V_T_unsqueezed = V.unsqueeze(1)
        M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

        M_e_flat = M_e.view(-1, 25)
        out = torch.zeros((num_nodes, 25), dtype=M_e_flat.dtype, device=M_e_flat.device)
        M_flat = out.scatter_add_(0, row.view(-1, 1).expand_as(M_e_flat), M_e_flat)

        M = M_flat.view(num_nodes, 5, 5)
        epsilon = 1e-6
        I = torch.eye(5, device=M.device).unsqueeze(0).expand(num_nodes, 5, 5)
        M_reg = M + epsilon * I

        # --- NEW: Compute Max Condition Number for Stability Check ---
        cond_numbers = torch.linalg.cond(M_reg)
        max_cond = cond_numbers.max().item()

        M_inv = torch.linalg.pinv(M_reg)
        return V, W, M_inv.squeeze(1), max_cond

    def _precompute_sparse_operators(self, edge_index, num_nodes, M_inv, V, W):
        """Converts WLS polynomial weights into global sparse matrices."""
        row, col = edge_index
        M_inv_edges = M_inv[ row ]
        WV = (W.unsqueeze(1) * V).unsqueeze(2)
        C = torch.bmm(M_inv_edges, WV).squeeze(2)

        Cx = C[ :, 0 ]
        Cy = C[ :, 1 ]
        C_laplacian = C[ :, 2 ] + C[ :, 4 ]

        def build_sparse_matrix(edge_weights):
            off_diag_indices = edge_index
            off_diag_values = edge_weights
            diag_values = torch.zeros(num_nodes, dtype=torch.float32, device=C.device)
            diag_values.scatter_add_(0, row, -edge_weights)
            diag_indices = torch.arange(num_nodes, device=C.device).repeat(2, 1)

            indices = torch.cat([ off_diag_indices, diag_indices ], dim=1)
            values = torch.cat([ off_diag_values, diag_values ])
            return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()

        return build_sparse_matrix(Cx), build_sparse_matrix(Cy), build_sparse_matrix(C_laplacian)

    def _compute_gradient_wls(self, f_node, row, col, W, V, M_inv, num_nodes):
        """Generic WLS gradient computer for any scalar field f_node."""
        df = f_node[ col ] - f_node[ row ]
        sum_W_V_df = torch.zeros((num_nodes, 5), dtype=torch.float32, device=f_node.device)
        integrand = W.unsqueeze(1) * V * df.unsqueeze(1)
        sum_W_V_df.scatter_add_(0, row.unsqueeze(1).expand(-1, 5), integrand)
        grad_f = torch.bmm(M_inv, sum_W_V_df.unsqueeze(2)).squeeze(2)
        return grad_f[ :, :2 ]

    def _compute_boundary_normals(self, edge_index, boundary_mask, pos_tensor, num_nodes):
        """Computes geometric unit normals for any boundary mask using adjacent edges."""
        normals = torch.zeros((num_nodes, 2), dtype=torch.float32, device=pos_tensor.device)

        row, col = edge_index
        # Use unique undirected boundary segments only; duplicated edge directions cancel exactly.
        b_edges = boundary_mask[ row ] & boundary_mask[ col ] & (row < col)

        r = row[ b_edges ]
        c = col[ b_edges ]

        # Compute edge vectors (dx, dy)
        edge_vecs = pos_tensor[ c ] - pos_tensor[ r ]

        # Perpendicular vector (-dy, dx)
        edge_normals = torch.stack([ -edge_vecs[ :, 1 ], edge_vecs[ :, 0 ] ], dim=1)

        # Orient normals consistently toward mesh center (inward) to avoid random sign flips.
        center_pt = pos_tensor.mean(dim=0)
        midpoints = (pos_tensor[ r ] + pos_tensor[ c ]) / 2.0
        inward_vecs = center_pt - midpoints
        dot_prods = (edge_normals * inward_vecs).sum(dim=1, keepdim=True)
        edge_normals = torch.where(dot_prods < 0, -edge_normals, edge_normals)

        # Accumulate normals at the respective nodes
        normals.scatter_add_(0, r.unsqueeze(1).expand(-1, 2), edge_normals)
        normals.scatter_add_(0, c.unsqueeze(1).expand(-1, 2), edge_normals)

        # Normalize to create unit vectors
        norm_mag = torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-9
        normals_unit = normals / norm_mag

        return normals_unit

    @staticmethod
    def _empty_boundary_diagnostics(*, missing: bool = False) -> dict:
        """Diagnostic dict returned when a boundary file is missing or empty."""
        return {
            "n_csv_unique": 0,
            "n_vertex_hits": 0,
            "vertex_hit_rate": 0.0,
            "unmapped_ratio": 1.0 if missing else 0.0,
            "d_median_m": float("nan"),
            "d_p90_m": float("nan"),
            "d_max_m": float("nan"),
            "p2_inferred": False,
            "status": "missing" if missing else "empty",
        }

    def _load_spatial_mask(
        self,
        file_path,
        tree,
        num_nodes,
        *,
        mesh_edge_scale_m: float | None = None,
        vertex_tol_m: float = 1e-5,
        unit_floor_m: float = 1.0e-3,
    ):
        """Map COMSOL boundary coords to mesh-vertex indices, with health diagnostics.

        Returns ``(mask, diagnostics)``. ``diagnostics`` is a small ``dict`` recording
        the nearest-vertex distance distribution, the vertex-hit rate, and a
        ``p2_inferred`` flag set when the unmapped half of the export sits at
        ~½ ``mesh_edge_scale_m`` (textbook signature of COMSOL exporting P2
        Lagrange mid-edge nodes against a P1 mesh).

        Three-tier policy:

        1. **Hard error** if ``d_median > unit_floor_m`` (median residual that
           large can only be a unit/coordinate-frame bug, *not* a P2 expansion).
        2. **Hard error** if zero vertex matches were found at ``vertex_tol_m``
           and the file had unique COMSOL coords (the original guard, kept).
        3. **Loud warning** if ``vertex_hit_rate < 0.30`` (the mesh genuinely
           doesn't cover the wall vertices that COMSOL exported).
        4. **Info note** if a P2 export is inferred -- this is the common,
           benign case and should not look like an alarm.
        5. **Plain warning** otherwise when ``unmapped_ratio > 0.10``.
        """
        mask = torch.zeros(num_nodes, dtype=torch.bool)

        if not file_path.exists():
            return mask, self._empty_boundary_diagnostics(missing=True)

        bnd_df = pd.read_csv(file_path, comment='%', sep=r'\s+', header=None)

        # USE CENTRALIZED SCALE
        bnd_coords = np.unique(bnd_df.iloc[:, -2:].values, axis=0) * self.phys_cfg.cm_to_m
        n_unique = int(len(bnd_coords))
        if n_unique == 0:
            return mask, self._empty_boundary_diagnostics(missing=False)

        distances, indices = tree.query(bnd_coords)
        d_median = float(np.median(distances))
        d_p90 = float(np.percentile(distances, 90))
        d_max = float(distances.max())

        # 1) Unit / coordinate-frame sanity floor: a real bug, not a P2 expansion.
        if d_median > unit_floor_m:
            raise ValueError(
                f"\nCRITICAL ERROR: median nearest-vertex distance for {file_path.name} "
                f"is {d_median * 1e3:.3f} mm (> {unit_floor_m * 1e3:.3f} mm).\n"
                f"This usually means the COMSOL export is in a different unit / "
                f"coordinate frame than the mesh.\n"
                f"Verify: boundary export is in cm, same origin/axes as the .nas mesh."
            )

        valid_matches = indices[distances < vertex_tol_m]
        n_vertex_hits = int(len(np.unique(valid_matches)))
        vertex_hit_rate = n_vertex_hits / n_unique

        # 2) No vertex matches at all -- existing hard error, sharper diagnostic.
        if n_vertex_hits == 0:
            raise ValueError(
                f"\nCRITICAL ERROR: zero boundary nodes mapped for {file_path.name}!\n"
                f"Attempted to map {n_unique} unique COMSOL coords; nearest "
                f"distances span [{distances.min() * 1e6:.1f}, {d_max * 1e6:.1f}] um, "
                f"none within the {vertex_tol_m * 1e6:.1f} um vertex tolerance.\n"
                f"Verify that boundary exports use the same units (cm) as the domain export."
            )

        mask[valid_matches] = True
        unmapped_ratio = 1.0 - vertex_hit_rate

        # P2 (quadratic-element) inference: the unmapped cluster sits at ~edge/2.
        p2_inferred = False
        if (
            mesh_edge_scale_m is not None
            and unmapped_ratio > 0.10
            and vertex_hit_rate >= 0.30
        ):
            far = distances[distances >= 5 * vertex_tol_m]
            if far.size > 0:
                far_median = float(np.median(far))
                lo = 0.30 * mesh_edge_scale_m
                hi = 0.70 * mesh_edge_scale_m
                p2_inferred = lo <= far_median <= hi

        # 3) Loud warning: mesh genuinely lacks the wall vertices.
        if vertex_hit_rate < 0.30:
            print(
                f"⚠️ {file_path.name}: only {vertex_hit_rate:.1%} of {n_unique} "
                f"COMSOL boundary coords matched a mesh vertex within "
                f"{vertex_tol_m * 1e6:.0f} um (median residual {d_median * 1e6:.1f} um, "
                f"max {d_max * 1e6:.1f} um). The mesh may not contain the wall "
                f"vertices COMSOL is exporting; check that the .nas mesh and the "
                f"COMSOL geometry share the same vertex set."
            )
        elif p2_inferred:
            # 4) Benign P2 export -- one informational line, no alarm bell.
            far_median_um = float(
                np.median(distances[distances >= 5 * vertex_tol_m])
            ) * 1e6
            print(
                f"[note] {file_path.name}: P2 export inferred -- "
                f"{n_vertex_hits}/{n_unique} vertex hits, residual cluster at "
                f"~{far_median_um:.0f} um (~1/2 mesh edge "
                f"{mesh_edge_scale_m * 1e6:.0f} um). Mid-edge nodes have no P1 "
                f"counterpart and are correctly ignored."
            )
        elif unmapped_ratio > 0.10:
            # 5) Unexplained partial: still worth a heads-up.
            print(
                f"⚠️ {file_path.name}: {unmapped_ratio:.1%} of boundary nodes "
                f"unmapped (median {d_median * 1e6:.1f} um, p90 "
                f"{d_p90 * 1e6:.1f} um, max {d_max * 1e6:.1f} um); not a clean "
                f"P2 pattern -- inspect the export."
            )

        diagnostics = {
            "n_csv_unique": n_unique,
            "n_vertex_hits": n_vertex_hits,
            "vertex_hit_rate": vertex_hit_rate,
            "unmapped_ratio": unmapped_ratio,
            "d_median_m": d_median,
            "d_p90_m": d_p90,
            "d_max_m": d_max,
            "p2_inferred": p2_inferred,
            "status": "ok",
        }
        return mask, diagnostics

    def _compute_analytic_inlet_mu_nd(self, mask_inlet, mesh_nodes, u_raw_si, v_raw_si):
        """Compute analytical Carreau inlet viscosity from a Poiseuille-style profile."""
        num_nodes = mesh_nodes.shape[0]
        mu_inlet_nd = torch.zeros((num_nodes, 1), dtype=torch.float32)

        inlet_idx = torch.where(mask_inlet)[0]
        if inlet_idx.numel() < 2:
            return mu_inlet_nd

        inlet_coords = torch.tensor(mesh_nodes[inlet_idx.cpu().numpy()], dtype=torch.float32)
        inlet_center = inlet_coords.mean(dim=0, keepdim=True)
        r = torch.linalg.norm(inlet_coords - inlet_center, dim=1)
        R = torch.max(r)
        if torch.isclose(R, torch.tensor(0.0), atol=1e-12):
            return mu_inlet_nd

        inlet_speed = torch.sqrt(u_raw_si[inlet_idx] ** 2 + v_raw_si[inlet_idx] ** 2)
        Umax = torch.max(inlet_speed)
        gamma_dot = 2.0 * Umax * (r / (R ** 2 + 1e-12))

        shear_term = 1.0 + (self.phys_cfg.lam * gamma_dot) ** self.phys_cfg.a
        power = (self.phys_cfg.n - 1.0) / self.phys_cfg.a
        mu_inlet_si = self.phys_cfg.mu_inf + (self.phys_cfg.mu_0 - self.phys_cfg.mu_inf) * (shear_term ** power)
        mu_inlet_nd[inlet_idx, 0] = self.phys_cfg.viscosity_si_to_nd(mu_inlet_si)
        return mu_inlet_nd

    def load_comsol_trajectory(self, filepath):
        """Parses a single 'wide-format' COMSOL Spreadsheet export."""

        # 1. Read the header to extract the time steps dynamically
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header_line = ""
        for line in lines:
            if line.startswith('% x') and '@ t=' in line:
                header_line = line
                break

        if not header_line:
            raise ValueError(f"Could not find time-step header in {filepath.name}")

        # Find all unique time values in the header
        times = []
        for match in re.finditer(r't=([0-9.]+)', header_line):
            t_val = float(match.group(1))
            if t_val not in times:
                times.append(t_val)

        times_arr = np.asarray(times, dtype=np.float64)
        t_cap = biochem_comsol_time_cap_s()
        if t_cap is None:
            times = [float(x) for x in times_arr]
            print(
                f"[i]  {filepath.name}: keeping full COMSOL horizon "
                f"({len(times)} steps, t_max={float(times_arr.max()):.1f} s).",
                flush=True,
            )
        else:
            valid_time_indices = times_arr <= float(t_cap)
            n_kept = int(valid_time_indices.sum())
            if n_kept == 0:
                raise ValueError(
                    f"No COMSOL export time steps <= t_cap={t_cap} s in {filepath.name!r}."
                )
            if n_kept < int(times_arr.size):
                print(
                    f"[i]  {filepath.name}: truncating COMSOL trajectory to "
                    f"t <= {t_cap} s (kept {n_kept}/{int(times_arr.size)} steps).",
                    flush=True,
                )
            times = [float(x) for x in times_arr[valid_time_indices]]

        # 2. Load the numeric data (skipping comment lines)
        df_full = pd.read_csv(filepath, comment='%', sep=r'\s+', header=None)

        # 3. Slice the wide dataframe into time blocks
        time_blocks = {}
        vars_per_step = 18  # x, y, u, v, p, mu, + 12 species

        for i, t_val in enumerate(times):
            # Base coords take columns. Step 0 starts at col 2.
            start_col = 2 + (i * vars_per_step)
            end_col = start_col + vars_per_step

            # Extract the block
            df_step = df_full.iloc[ :, start_col:end_col ].copy()

            # Assign consistent internal column names
            df_step.columns = [
                'x', 'y', 'u', 'v', 'p', 'mu_effective',
                'rp', 'ap', 'apr', 'aps', 'PT', 'th', 'at', 'fg', 'fi', 'M', 'Mas', 'Mat'
            ]
            time_blocks[ t_val ] = df_step

        return time_blocks

    def pull_comsol_exports(
        self,
        stem: str,
        *,
        model_path: Path | None = None,
        force: bool = False,
    ) -> Path:
        """Sample a solved ``.mph`` onto the anchor mesh and write ``cfd_results_biochem`` txt."""
        from src.data_gen.lib.biochem_comsol_auto_export import pull_biochem_comsol_exports

        return pull_biochem_comsol_exports(
            stem,
            label_dir=self.label_dir,
            raw_dir=self.raw_dir,
            model_path=model_path,
            force=force,
        )

    def process_patient(self, stem):
        """
        Full extraction pipeline with Physics-Informed Sanity Checks and
        Training Metadata generation.
        """
        # 1. Path Setup and Mesh Loading
        msh_path_nas = self.raw_dir / f"{stem}.nas"
        msh_path_msh = self.raw_dir / f"{stem}.msh"
        msh_path = msh_path_nas if msh_path_nas.exists() else msh_path_msh

        if not msh_path.exists():
            print(f"[ERR] Skipping {stem}: Mesh file (.nas/.msh) not found.", flush=True)
            return

        sidecar_path = self.raw_dir / f"{stem}.json"
        sidecar_meta = None
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as _f:
                sidecar_meta = json.load(_f)
        assert_mesh_unit(sidecar_meta, MESH_UNIT_CM, stem=stem, builder="PatientDataExtractor")

        txt_path = self.label_dir / f"{stem}.txt"
        inlet_path = self.label_dir / f"{stem}_inlet.txt"
        outlet_path = self.label_dir / f"{stem}_outlet.txt"
        wall_path = self.label_dir / f"{stem}_wall.txt"

        if not txt_path.exists():
            print(f"[ERR] Skipping {stem}: COMSOL domain data (.txt) missing.", flush=True)
            return

        # 2. Topology & Enhanced Boundary Mapping
        mesh = meshio.read(msh_path)
        mesh_nodes = mesh.points[ :, :2 ] * self.phys_cfg.cm_to_m
        num_nodes = len(mesh_nodes)
        mesh_tree = cKDTree(mesh_nodes)

        # Mean nearest-neighbour spacing -- used to recognise the COMSOL P2 mid-edge
        # signature (unmatched cluster sits at ~½ this value) without misclassifying
        # genuine alignment failures.
        if num_nodes >= 2:
            nn_d, _ = mesh_tree.query(mesh_nodes, k=2)
            mesh_edge_scale_m = float(np.mean(nn_d[:, 1]))
        else:
            mesh_edge_scale_m = None

        mask_inlet, diag_inlet = self._load_spatial_mask(
            inlet_path, mesh_tree, num_nodes, mesh_edge_scale_m=mesh_edge_scale_m
        )
        mask_outlet, diag_outlet = self._load_spatial_mask(
            outlet_path, mesh_tree, num_nodes, mesh_edge_scale_m=mesh_edge_scale_m
        )
        mask_wall, diag_wall = self._load_spatial_mask(
            wall_path, mesh_tree, num_nodes, mesh_edge_scale_m=mesh_edge_scale_m
        )

        d_bar = resolve_d_bar_si_from_sidecar_or_inlet(
            sidecar_meta,
            stem=stem,
            mesh_nodes_si=mesh_nodes,
            mask_inlet=mask_inlet,
        )

        # 4. Connectivity and Edge Construction
        if "triangle" in mesh.cells_dict:
            all_tris = mesh.cells_dict[ "triangle" ]
        elif "triangle6" in mesh.cells_dict:
            all_tris = mesh.cells_dict[ "triangle6" ][ :, :3 ]
        else:
            print(f"[WARN] {stem}: Unsupported cell type.", flush=True)
            return

        edges = np.unique(np.sort(np.vstack([
            all_tris[ :, [ 0, 1 ] ], all_tris[ :, [ 1, 2 ] ], all_tris[ :, [ 2, 0 ] ]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([ edges.T, edges[ :, [ 1, 0 ] ].T ]), dtype=torch.long)
        row, col = edge_index

        needs_sidecar = (
            sidecar_meta is None
            or sidecar_meta.get("centerline_pts") is None
            or sidecar_meta.get("centerline_tangents") is None
            or sidecar_meta.get("d_bar") is None
        )
        if needs_sidecar:
            level_hint = int((sidecar_meta or {}).get("level", 2))
            write_anchor_sidecar_from_masks(
                sidecar_path,
                mesh_nodes_si=mesh_nodes,
                mask_inlet=mask_inlet,
                mask_outlet=mask_outlet,
                mask_wall=mask_wall,
                edge_index=edge_index,
                d_bar_si=d_bar,
                stem=stem,
                unit="cm",
                level=level_hint,
                existing=sidecar_meta,
            )
            with open(sidecar_path, encoding="utf-8") as _f:
                sidecar_meta = json.load(_f)
            d_bar = resolve_d_bar_si_from_sidecar_or_inlet(
                sidecar_meta,
                stem=stem,
                mesh_nodes_si=mesh_nodes,
                mask_inlet=mask_inlet,
            )
            print(f"[i] {stem}: wrote sidecar (d_bar, centerline) from COMSOL boundary masks", flush=True)

        # --- 5. DYNAMIC EULERIAN FIELD MAPPING (TRAJECTORY EXTRACTION) ---
        trajectory_file = self.label_dir / f"{stem}.txt"

        if not trajectory_file.exists():
            print(f"[ERR] Skipping {stem}: Trajectory file not found.", flush=True)
            return

        print(f"Parsing transient trajectory for {stem}...")
        time_blocks = self.load_comsol_trajectory(trajectory_file)

        eval_times = sorted(list(time_blocks.keys()))
        eval_times_tensor = torch.tensor(eval_times, dtype=torch.float32)

        # Pre-compute geometry-normalized node/edge features once
        nodes_nd = torch.tensor(mesh_nodes / d_bar, dtype=torch.float32)
        edge_attr = torch.cat([
            nodes_nd[ row ] - nodes_nd[ col ],
            torch.linalg.norm(nodes_nd[ row ] - nodes_nd[ col ], dim=1, keepdim=True)
        ], dim=1)

        y_trajectory = []
        u_raw_list = []
        v_raw_list = []

        # --- Pre-Compute KDTree Mapping ONCE outside the loop ---
        # Load just the first timestep block to establish the spatial mapping
        df_first = time_blocks[ eval_times[ 0 ] ]
        csv_coords_static = df_first[ [ 'x', 'y' ] ].values * self.phys_cfg.cm_to_m

        domain_tree = cKDTree(csv_coords_static)
        match_distances, match_indices = domain_tree.query(mesh_nodes)
        tol_m = self.phys_cfg.comsol_spatial_match_tol_m
        is_anchor = torch.tensor(match_distances < tol_m, dtype=torch.bool)
        if int(is_anchor.sum()) == 0:
            print(
                f"[WARN] {stem}: no nodes within comsol_spatial_match_tol_m={tol_m} m of COMSOL export; "
                f"raise PhysicsConfig.comsol_spatial_match_tol_m or verify mesh/CSV alignment.",
                flush=True,
            )

        # --- Pre-Compute Normals and initialize accumulator ---
        pos_tensor = torch.tensor(mesh_nodes, dtype=torch.float32)
        inlet_normals = self._compute_boundary_normals(edge_index, mask_inlet, pos_tensor, num_nodes)
        outlet_normals = self._compute_boundary_normals(edge_index, mask_outlet, pos_tensor, num_nodes)
        total_flux_imbalance = 0.0

        # Iterate through the parsed time steps
        for t_idx, t_val in enumerate(eval_times):
            df_csv = time_blocks[t_val]
            df_matched = df_csv.iloc[match_indices].reset_index(drop=True)

            # --- USE CENTRALIZED SCALES ---
            u_raw = torch.tensor(df_matched['u'].values, dtype=torch.float32) * self.phys_cfg.cm_to_m
            v_raw = torch.tensor(df_matched['v'].values, dtype=torch.float32) * self.phys_cfg.cm_to_m
            p_raw = torch.tensor(df_matched['p'].values, dtype=torch.float32) * self.phys_cfg.cgs_p_to_pa
            mu_eff = torch.tensor(df_matched['mu_effective'].values, dtype=torch.float32) * self.phys_cfg.cgs_mu_to_pa_s

            u_ref_actual = self.phys_cfg.get_u_ref(d_bar)
            p_ref = self.phys_cfg.rho * (u_ref_actual ** 2)
            p_relative = p_raw - (p_raw[mask_outlet].mean() if mask_outlet.any() else p_raw.min())

            u_nd = u_raw / u_ref_actual
            v_nd = v_raw / u_ref_actual
            p_nd = p_relative / p_ref
            mu_nd = self.phys_cfg.viscosity_si_to_nd(mu_eff)

            # --- USE CENTRALIZED BIOCHEM SCALES ---
            bio_cfg = BiochemConfig(phase=self.vessel_cfg.phase)
            species_cols = list(self.species_map.keys())
            raw_bulk_cgs = torch.tensor(df_matched[species_cols[:9]].values, dtype=torch.float32)
            raw_surf_cgs = torch.tensor(df_matched[species_cols[9:]].values, dtype=torch.float32)

            # COMSOL export convention for phase-3 species is mixed-CGS:
            # - rp, ap: platelet number density in plt/ml  -> scaled linear field via x bulk_scale
            # - apr, aps, PT, th, at, fg, fi: concentration in uM -> mol/m^3 (x1e-3), then x bulk_scale
            # This keeps transformed ND channels consistent with BiochemConfig.get_species_scales().
            bulk_si = torch.zeros_like(raw_bulk_cgs)
            bulk_si[:, 0:2] = raw_bulk_cgs[:, 0:2] * bio_cfg.bulk_scale
            bulk_si[:, 2:9] = raw_bulk_cgs[:, 2:9] * (bio_cfg.bulk_scale * 1e-3)
            surf_si = raw_surf_cgs * bio_cfg.surface_scale
            species = torch.clamp(torch.cat([bulk_si, surf_si], dim=1), min=0.0)

            scales = bio_cfg.get_species_scales(device='cpu')
            species_nd = species / scales
            species_transformed = torch.log1p(species_nd)

            # Combine to [Nodes, 16]
            y_t = torch.cat([
                u_nd.unsqueeze(1), v_nd.unsqueeze(1), p_nd.unsqueeze(1),
                mu_nd.unsqueeze(1), species_transformed
            ], dim=1)

            y_trajectory.append(y_t)
            u_raw_list.append(u_raw)
            v_raw_list.append(v_raw)

            # --- DYNAMIC MASS FLUX CALCULATION ---
            inlet_v = torch.stack([ u_raw[ mask_inlet ], v_raw[ mask_inlet ] ], dim=1)
            outlet_v = torch.stack([ u_raw[ mask_outlet ], v_raw[ mask_outlet ] ], dim=1)

            inlet_flux = torch.abs(torch.sum(inlet_v * inlet_normals[ mask_inlet ])).item()
            outlet_flux = torch.abs(torch.sum(outlet_v * outlet_normals[ mask_outlet ])).item()

            step_imbalance = abs(inlet_flux - outlet_flux) / (inlet_flux + 1e-8)
            total_flux_imbalance += step_imbalance

            # Save the inlet/wall BCs explicitly from the FIRST timestep (t=0)
            if t_idx == 0:
                u_nd_0 = u_nd
                v_nd_0 = v_nd
                mu_nd_0 = mu_nd
                p_nd_0 = p_nd

        # Stack into shape: [Time, Nodes, 16]
        y_tensor_series = torch.stack(y_trajectory, dim=0)
        if y_tensor_series.shape[0] != len(eval_times):
            raise ValueError(
                f"{stem}: trajectory length {y_tensor_series.shape[0]} != time stamps {len(eval_times)}; "
                "check COMSOL export headers (@ t=... columns)."
            )
        avg_flux_imbalance = total_flux_imbalance / len(eval_times)

        # 6. Gradients & Numerical Stability
        V, W, M_inv, max_cond = self._precompute_wls(edge_index, num_nodes, nodes_nd)
        G_x, G_y, Laplacian = self._precompute_sparse_operators(edge_index, num_nodes, M_inv, V, W)

        # 8. Data-Driven ML Scaling
        # Compute U_ref from the 99th percentile to clamp data strictly to ~[-1.0, 1.0]
        u_ref_actual = self.phys_cfg.get_u_ref(d_bar)  # Matches Kinematics/2 logic exactly
        re_actual = self.phys_cfg.re_target  # Locks the ML Re to your target

        # Ensure your bio_cfg scales match these new SI units
        bio_cfg = BiochemConfig(phase=self.vessel_cfg.phase)
        scales = bio_cfg.get_species_scales(device='cpu')

        # Outlet normals for species BCs (Bio_IO); distinct from wall normals in x.
        outlet_normals = self._compute_boundary_normals(
            edge_index, mask_outlet, pos_tensor, num_nodes
        )

        u_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
        v_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
        u_bc[mask_inlet, 0] = u_nd_0[mask_inlet]
        v_bc[mask_inlet, 0] = v_nd_0[mask_inlet]
        p_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)

        mu_bc = mu_nd_0

        geometry_level = None
        if sidecar_meta is not None and sidecar_meta.get("level") is not None:
            geometry_level = int(sidecar_meta["level"])
        from src.data_gen.lib.node_feature_assembly import resolve_anchor_kine_phys_cfg

        kine_phys = resolve_anchor_kine_phys_cfg()
        kine_data = build_kinematics_graph_from_comsol_steady(
            mesh=mesh,
            mesh_nodes_si=mesh_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            u_nd=u_nd_0,
            v_nd=v_nd_0,
            p_nd=p_nd_0,
            mu_nd=mu_nd_0,
            d_bar_si=d_bar,
            u_ref=u_ref_actual,
            sidecar_meta=sidecar_meta,
            stem=stem,
            G_x=G_x,
            G_y=G_y,
            V=V,
            W=W,
            M_inv=M_inv,
            phys_cfg=kine_phys,
            raw_sidecar_dir=self.raw_dir,
            geometry_level=geometry_level,
            prior_mode=os.environ.get("KINE_ANCHOR_PRIOR_MODE", "gt_flow"),
        )
        x_kine = kine_data.x
        u_prior = kine_data.u_prior
        mu_prior = kine_data.mu_prior
        centerline_source = str(getattr(kine_data, "centerline_source", ""))
        torch.save(kine_data, self.kine_anchor_dir / f"{stem}.pt")

        x_biochem = build_biochem_bc_x_tensor(
            pos_nd=x_kine[:, NodeFeat.XY],
            sdf_nd=x_kine[:, NodeFeat.SDF],
            wall_normal=x_kine[:, NodeFeat.WALL_NORMAL],
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            u_bc=u_bc,
            v_bc=v_bc,
            p_bc=p_bc,
            mu_bc_nd=mu_bc,
        )

        # --- Use index assignment for the tensor ---
        inlet_species_si = torch.zeros(9, dtype=torch.float32)
        inlet_species_si[0] = bio_cfg.c_RP0 * bio_cfg.bulk_scale  # RP
        inlet_species_si[1] = bio_cfg.c_AP0 * bio_cfg.bulk_scale  # AP
        inlet_species_si[4] = bio_cfg.c_pT0 * bio_cfg.bulk_scale  # PT
        inlet_species_si[6] = bio_cfg.cAT0 * bio_cfg.bulk_scale  # AT
        inlet_species_si[7] = bio_cfg.c_Fg0 * bio_cfg.bulk_scale  # FG
        # Note: APR, APS, T, FI remain 0.0 at the inlet

        # Scale and transform
        inlet_species_nd = inlet_species_si / scales[ :9 ]
        inlet_species_transformed = torch.log1p(inlet_species_nd)

        # Broadcast to all nodes
        bio_inlet_bc = inlet_species_transformed.unsqueeze(0).expand(num_nodes, -1)

        # 10. Metadata Export
        boundary_diagnostics = {
            "inlet": diag_inlet,
            "outlet": diag_outlet,
            "wall": diag_wall,
        }
        metadata = {
            "stem": stem,
            "quality": {
                "max_wls_condition_number": max_cond,
                "mass_flux_imbalance": avg_flux_imbalance,
                # Legacy field, kept for back-compat. Per-boundary diagnostics live in `boundaries`.
                "boundary_unmapped_ratio": max(
                    diag_inlet["unmapped_ratio"],
                    diag_outlet["unmapped_ratio"],
                    diag_wall["unmapped_ratio"],
                ),
                "mesh_edge_scale_m": mesh_edge_scale_m,
                "boundaries": boundary_diagnostics,
            },
            "field_stats": {
                "u_max": u_raw.max().item(),
                "u_ref_ml": u_ref_actual,
                "re_ml": re_actual,
                "d_bar": d_bar
            }
        }
        with open(self.proc_dir / f"{stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # 11. Final PyG Data Save
        uv_inlet_bc = torch.cat([u_nd_0.unsqueeze(1), v_nd_0.unsqueeze(1)], dim=1)
        mu_inlet_bc = self._compute_analytic_inlet_mu_nd(
            mask_inlet=mask_inlet,
            mesh_nodes=mesh_nodes,
            u_raw_si=u_raw_list[0],
            v_raw_si=v_raw_list[0],
        )
        data = Data(
            x=x_kine,
            x_biochem=x_biochem,
            y=y_tensor_series,
            t=eval_times_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            is_anchor=is_anchor,
            d_bar=torch.tensor([d_bar], dtype=torch.float32),
            u_ref=torch.tensor([u_ref_actual], dtype=torch.float32),
            re_actual=torch.tensor([re_actual], dtype=torch.float32),
            G_x=G_x,
            G_y=G_y,
            Laplacian=Laplacian,
            V=V,
            W=W,
            M_inv=M_inv,
            u_inlet_bc=uv_inlet_bc,
            mu_inlet_bc=mu_inlet_bc,
            bio_inlet_bc=bio_inlet_bc,
            outlet_normal=outlet_normals,
            u_prior=u_prior,
            mu_prior=mu_prior,
        )
        data = attach_patient_anchor_graph_metadata(data, mask_wall=mask_wall)
        data.centerline_source = centerline_source

        torch.save(data, self.proc_dir / f"{stem}.pt")
        print(
            f"[OK] Saved {stem}: D={d_bar * 1000:.1f}mm | Re_ML={re_actual:.0f} | "
            f"Imbal={avg_flux_imbalance:.2%} | centerline={centerline_source} | "
            f"uv_prior_max={float(x_kine[:, 11:13].abs().max()):.3f}"
        )

    def run(
        self,
        *,
        from_comsol: bool = True,
        force_comsol_pull: bool = False,
        stems: list[str] | None = None,
    ) -> None:
        """Batch-extract all anchor meshes (optionally pull COMSOL fields first)."""
        if stems is None:
            files = [f for f in os.listdir(self.raw_dir) if f.endswith(".nas") or f.endswith(".msh")]
            if len(files) == 0:
                print(f"CRITICAL ERROR: No .msh or .nas files found in {self.raw_dir}")
                return
            stems = sorted({Path(f).stem for f in files})

        from src.data_gen.lib.biochem_comsol_auto_export import resolve_biochem_comsol_model_path

        for stem in tqdm(stems, desc="Extracting Biochem anchor data"):
            if from_comsol:
                domain_txt = self.label_dir / f"{stem}.txt"
                mph_path = resolve_biochem_comsol_model_path(stem)
                if mph_path is None:
                    if not domain_txt.is_file():
                        print(
                            f"[WARN] Skipping {stem}: no domain .txt and no "
                            f"phase2_nowound_XXX.mph for patientXXX in comsol_models/.",
                            flush=True,
                        )
                        continue
                elif force_comsol_pull or not domain_txt.is_file():
                    print(f"[i] COMSOL pull {stem} <- {mph_path.name}", flush=True)
                    try:
                        self.pull_comsol_exports(stem, model_path=mph_path, force=force_comsol_pull)
                    except Exception as exc:
                        print(f"[ERR] COMSOL pull failed for {stem}: {exc}", flush=True)
                        if not domain_txt.is_file():
                            continue
            self.process_patient(stem)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract biochem anchor graphs. Default: pull solved COMSOL fields from "
            "comsol_models/phase2_nowound_XXX.mph (patientXXX) via mph, then write .pt graphs."
        )
    )
    parser.add_argument("--stem", type=str, default="", help="Only this stem (default: all meshes).")
    parser.add_argument("--force", action="store_true", help="Re-pull COMSOL txt and overwrite graphs.")
    parser.add_argument(
        "--no-from-comsol",
        action="store_true",
        help="Require manual cfd_results_biochem/*.txt exports (legacy).",
    )
    args = parser.parse_args(argv)

    from src.data_gen.pipeline_biochem import _auto_scaffold_anchor_sidecars

    extractor = PatientDataExtractor(phase="biochem_anchors")
    _auto_scaffold_anchor_sidecars(extractor.raw_dir)

    stem_list = [args.stem.strip()] if args.stem.strip() else None
    extractor.run(
        from_comsol=not args.no_from_comsol,
        force_comsol_pull=args.force,
        stems=stem_list,
    )


if __name__ == "__main__":
    main()