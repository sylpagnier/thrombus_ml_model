"""
Centerline-based registration logic for 2D vascular geometries.
Replaces coordinate/angular sorting with flow-path topological mapping.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.linalg import orthogonal_procrustes

def calculate_centerline_2d(coords: np.ndarray):
    """
    Extracts a longitudinal skeleton to identify the vessel's flow axis.
    """
    # 1. Sort by X to get general flow direction
    sorted_idx = np.argsort(coords[:, 0])
    pts = coords[sorted_idx]

    # 2. Slice the vessel and find midpoints of the cross-sections
    num_bins = 20
    bins = np.linspace(pts[:, 0].min(), pts[:, 0].max(), num_bins)
    centerline = []

    for i in range(len(bins)-1):
        mask = (pts[:, 0] >= bins[i]) & (pts[:, 0] < bins[i+1])
        if np.any(mask):
            centerline.append(np.mean(pts[mask], axis=0))

    return np.array(centerline)

def map_vessel_topology(coords: np.ndarray):
    """
    Separates the point cloud into 'Walls' and 'Caps' (Inlet/Outlet).
    """
    # 1. Get the skeleton
    skeleton = calculate_centerline_2d(coords)
    inlet_ref = skeleton[0]   # The 'Leftmost' end
    outlet_ref = skeleton[-1] # The 'Rightmost' end

    # 2. Identify Landmark: Node 0 is the node closest to the inlet center
    dist_to_inlet = np.linalg.norm(coords - inlet_ref, axis=1)
    start_idx = np.argmin(dist_to_inlet)

    # 3. Path Traversal: Order points by their projection along the centerline
    # This prevents the 'spiderweb' by ensuring points follow the flow path
    vec_flow = outlet_ref - inlet_ref
    projections = np.dot(coords - inlet_ref, vec_flow) / np.linalg.norm(vec_flow)

    # We split into 'Top' and 'Bottom' walls based on Y relative to centerline
    # For a 2D tube, this creates a clean, non-intersecting loop
    center_y = np.interp(coords[:, 0], skeleton[:, 0], skeleton[:, 1])
    top_wall = coords[coords[:, 1] > center_y]
    bottom_wall = coords[coords[:, 1] <= center_y]

    # Sort both by flow projection
    top_wall = top_wall[np.argsort(projections[coords[:, 1] > center_y])]
    bottom_wall = bottom_wall[np.argsort(projections[coords[:, 1] <= center_y])][::-1]

    # Combine into a single continuous loop
    ordered_loop = np.vstack([top_wall, bottom_wall])
    return ordered_loop

def resample_vessel(ordered_coords: np.ndarray, n_nodes: int = 1000):
    """Equidistant arc-length resampling along the flow-path."""
    # Close loop and fit spline
    data = np.vstack((ordered_coords, ordered_coords[0]))
    tck, _ = splprep([data[:, 0], data[:, 1]], s=0, per=True)

    # Resample
    u_new = np.linspace(0, 1, n_nodes, endpoint=False)
    return np.column_stack(splev(u_new, tck))

class CenterlineAligner:
    """Standardizes rotation/scale after the topological map is established."""
    def fit_transform(self, data_dict: dict):
        ref_name = list(data_dict.keys())[0]
        ref = data_dict[ref_name]
        ref_s = (ref - np.mean(ref, axis=0)) / np.linalg.norm(ref - np.mean(ref, axis=0), 'fro')

        results = {}
        for name, coords in data_dict.items():
            c = coords - np.mean(coords, axis=0)
            s = c / np.linalg.norm(c, 'fro')
            R, _ = orthogonal_procrustes(ref_s, s)
            results[name] = s @ R.T
        return results