"""Probe: Dynamic SDF wall-ification for RGP-DEQ macro-resolves.

Tests whether treating clot nodes as walls (SDF=0, UV_PRIOR=0) produces
physically valid flow diversions when re-solving with the RGP-DEQ model.

Probes:
  1. Wall-ify a synthetic clot region on patient007, re-solve, check no-slip
  2. Verify SDF field smoothness after recompute
  3. Compare z_kin magnitude (in-distribution vs OOD check)
  4. Time the full macro-resolve pipeline
"""
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_device import require_cuda_device
from src.config import NodeFeat, PhysicsConfig
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics_and_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import data_root


def recompute_sdf_euclidean(pos_nd, mask_wall):
    """Recompute SDF from the updated wall mask (including clot-as-wall nodes)."""
    from scipy.spatial import cKDTree
    pos_np = pos_nd.cpu().numpy()
    wall_np = pos_np[mask_wall.cpu().numpy()]
    if wall_np.shape[0] == 0:
        return torch.zeros(pos_nd.shape[0], dtype=torch.float32)
    tree = cKDTree(wall_np)
    dists, _ = tree.query(pos_np)
    # Use raw distance (already in ND frame since pos_nd is ND)
    sdf_nd = np.clip(dists, 1e-6, None)
    return torch.tensor(sdf_nd, dtype=torch.float32)


def find_synthetic_clot_nodes(data, device, frac=0.05):
    """Pick nodes near the wall in a localized region to simulate a clot."""
    pos = data.x[:, 0:2].to(device)
    sdf = data.x[:, NodeFeat.SDF.start].to(device)

    # Find wall-adjacent nodes (small SDF) in a spatial window
    x_range = pos[:, 0].max() - pos[:, 0].min()
    x_center = pos[:, 0].min() + 0.4 * x_range  # 40% along vessel

    spatial_mask = (
        (pos[:, 0] - x_center).abs() < 0.08 * x_range
    )
    wall_adjacent = sdf < 0.15  # close to wall (ND)

    clot_candidates = spatial_mask & wall_adjacent & ~data.mask_wall.to(device)
    clot_idx = torch.where(clot_candidates)[0]

    # Cap to frac of total nodes
    max_clot = max(int(data.num_nodes * frac), 10)
    if clot_idx.numel() > max_clot:
        clot_idx = clot_idx[:max_clot]

    return clot_idx


def wallify_clot_nodes(data, clot_idx, device):
    """Create a modified data object with clot nodes treated as walls."""
    data_mod = data.clone()
    x_new = data_mod.x.clone().to(device)

    # 1. Set SDF = 0 at clot nodes (they ARE the wall now)
    x_new[clot_idx, NodeFeat.SDF.start] = 0.0

    # 2. Set UV_PRIOR = 0 at clot nodes (no-slip)
    x_new[clot_idx, NodeFeat.UV_PRIOR] = 0.0

    # 3. Set WSS_PRIOR = 0 at clot nodes
    if x_new.shape[1] > NodeFeat.WSS_PRIOR.start:
        x_new[clot_idx, NodeFeat.WSS_PRIOR] = 0.0

    # 4. Expand wall mask
    mask_wall_new = data_mod.mask_wall.clone().to(device)
    mask_wall_new[clot_idx] = True

    # 5. Recompute SDF for ALL nodes from the expanded wall boundary
    pos_nd = x_new[:, 0:2]
    sdf_recomputed = recompute_sdf_euclidean(pos_nd, mask_wall_new).to(device)
    x_new[:, NodeFeat.SDF.start] = sdf_recomputed

    # 6. Update shear potential (derived from SDF): |1 - 2*SDF|
    if x_new.shape[1] > NodeFeat.SHEAR_POT.start:
        x_new[:, NodeFeat.SHEAR_POT.start] = torch.abs(1.0 - 2.0 * sdf_recomputed)

    data_mod.x = x_new
    data_mod.mask_wall = mask_wall_new
    return data_mod


def main():
    device = require_cuda_device()
    print(f"[i] Device: {device}")

    # --- Load graph ---
    graph_path = data_root() / "processed" / "graphs_biochem_anchors" / "patient007.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n = data.num_nodes
    n_wall_orig = int(data.mask_wall.sum())
    print(f"[i] Graph: {graph_path.name}, N={n}, wall_nodes={n_wall_orig}")

    # --- Load kine model ---
    ckpt = resolve_kinematics_checkpoint()
    kine = load_kinematics_predictor(ckpt, device)
    print(f"[i] Kinematics checkpoint: {ckpt.name}")

    # ========== PROBE 1: Baseline solve (no clot) ==========
    print("\n===== PROBE 1: Baseline RGP-DEQ solve (no clot) =====")
    t0 = time.perf_counter()
    pred_base, z_kin_base = predict_kinematics_and_latent(kine, data.to(device))
    dt_base = time.perf_counter() - t0
    u_base, v_base = pred_base[:, 0], pred_base[:, 1]
    speed_base = torch.sqrt(u_base**2 + v_base**2)

    print(f"  Solve time: {dt_base*1000:.1f} ms")
    print(f"  z_kin shape: {z_kin_base.shape}, mag: {z_kin_base.norm(dim=1).mean():.4f} +/- {z_kin_base.norm(dim=1).std():.4f}")
    print(f"  Speed: mean={speed_base.mean():.4f}, max={speed_base.max():.4f}")
    print(f"  Wall node speed (should be ~0): mean={speed_base[data.mask_wall.to(device)].mean():.6f}")

    # ========== PROBE 2: Synthetic clot, wall-ified solve ==========
    print("\n===== PROBE 2: Wall-ified clot RGP-DEQ solve =====")
    clot_idx = find_synthetic_clot_nodes(data, device, frac=0.05)
    n_clot = clot_idx.numel()
    print(f"  Synthetic clot: {n_clot} nodes ({100*n_clot/n:.1f}% of mesh)")

    data_mod = wallify_clot_nodes(data, clot_idx, device)
    n_wall_new = int(data_mod.mask_wall.sum())
    print(f"  Wall nodes: {n_wall_orig} -> {n_wall_new}")

    t0 = time.perf_counter()
    pred_clot, z_kin_clot = predict_kinematics_and_latent(kine, data_mod.to(device))
    dt_clot = time.perf_counter() - t0
    u_clot, v_clot = pred_clot[:, 0], pred_clot[:, 1]
    speed_clot = torch.sqrt(u_clot**2 + v_clot**2)

    print(f"  Solve time: {dt_clot*1000:.1f} ms")
    print(f"  z_kin shape: {z_kin_clot.shape}, mag: {z_kin_clot.norm(dim=1).mean():.4f} +/- {z_kin_clot.norm(dim=1).std():.4f}")
    print(f"  Speed: mean={speed_clot.mean():.4f}, max={speed_clot.max():.4f}")

    # ========== PROBE 3: No-slip check at clot nodes ==========
    print("\n===== PROBE 3: No-slip enforcement at clot nodes =====")
    clot_speed = speed_clot[clot_idx]
    print(f"  Clot node speed (MUST be ~0 for no-slip):")
    print(f"    mean = {clot_speed.mean():.6f}")
    print(f"    max  = {clot_speed.max():.6f}")
    print(f"    std  = {clot_speed.std():.6f}")
    noslip_ok = clot_speed.max().item() < 0.01
    print(f"  No-slip verdict: {'[OK] PASS' if noslip_ok else '[FAIL] clot nodes have nonzero velocity'}")

    # Compare to original wall nodes
    orig_wall_speed = speed_clot[data.mask_wall.to(device)]
    print(f"  Original wall node speed: mean={orig_wall_speed.mean():.6f}, max={orig_wall_speed.max():.6f}")

    # ========== PROBE 4: z_kin distribution shift check ==========
    print("\n===== PROBE 4: z_kin in-distribution check =====")
    z_mag_base = z_kin_base.norm(dim=1)
    z_mag_clot = z_kin_clot.norm(dim=1)
    z_diff = (z_kin_clot - z_kin_base).norm(dim=1)

    print(f"  Baseline z_kin L2 norm: {z_mag_base.mean():.4f} +/- {z_mag_base.std():.4f}")
    print(f"  Clot     z_kin L2 norm: {z_mag_clot.mean():.4f} +/- {z_mag_clot.std():.4f}")
    print(f"  Delta z_kin norm:       {z_diff.mean():.4f} +/- {z_diff.std():.4f}")
    print(f"  Max delta z_kin:        {z_diff.max():.4f}")
    print(f"  Has NaN: {bool(z_kin_clot.isnan().any())}")
    print(f"  Has Inf: {bool(z_kin_clot.isinf().any())}")

    ratio = z_mag_clot.mean() / z_mag_base.mean()
    print(f"  Magnitude ratio (clot/base): {ratio:.4f}")
    in_dist = 0.5 < ratio.item() < 2.0 and not z_kin_clot.isnan().any() and not z_kin_clot.isinf().any()
    print(f"  In-distribution verdict: {'[OK] PASS' if in_dist else '[FAIL] z_kin is OOD'}")

    # ========== PROBE 5: SDF field smoothness after recompute ==========
    print("\n===== PROBE 5: SDF field smoothness =====")
    sdf_orig = data.x[:, NodeFeat.SDF.start].to(device)
    sdf_mod = data_mod.x[:, NodeFeat.SDF.start].to(device)

    # Check for discontinuities: max SDF gradient across edges
    edge_index = data.edge_index.to(device)
    row, col = edge_index[0], edge_index[1]
    sdf_diff_orig = (sdf_orig[row] - sdf_orig[col]).abs()
    sdf_diff_mod = (sdf_mod[row] - sdf_mod[col]).abs()

    print(f"  Original SDF: range [{sdf_orig.min():.4f}, {sdf_orig.max():.4f}], edge grad max={sdf_diff_orig.max():.4f}")
    print(f"  Modified SDF: range [{sdf_mod.min():.4f}, {sdf_mod.max():.4f}], edge grad max={sdf_diff_mod.max():.4f}")
    smooth = sdf_diff_mod.max().item() < 2.0 * sdf_diff_orig.max().item()
    print(f"  Smoothness verdict: {'[OK] PASS' if smooth else '[WARN] SDF may have discontinuities'}")

    # ========== PROBE 6: Flow diversion quality ==========
    print("\n===== PROBE 6: Flow diversion quality =====")
    # How much did the flow change near the clot vs far from it?
    from torch_geometric.utils import k_hop_subgraph
    near_clot, _, _, _ = k_hop_subgraph(clot_idx, num_hops=3, edge_index=edge_index, num_nodes=n)
    far_mask = torch.ones(n, dtype=torch.bool, device=device)
    far_mask[near_clot] = False

    speed_delta = (speed_clot - speed_base).abs()
    print(f"  Speed change near clot (3-hop): mean={speed_delta[near_clot].mean():.4f}, max={speed_delta[near_clot].max():.4f}")
    print(f"  Speed change far from clot:     mean={speed_delta[far_mask].mean():.4f}, max={speed_delta[far_mask].max():.4f}")
    local_effect = speed_delta[near_clot].mean() > 5 * speed_delta[far_mask].mean()
    print(f"  Localized diversion verdict: {'[OK] PASS - effect concentrated near clot' if local_effect else '[WARN] effect is diffuse'}")

    # ========== PROBE 7: Timing multiple resolves ==========
    print("\n===== PROBE 7: Timing budget for multiple macro-resolves =====")
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        _ = predict_kinematics_and_latent(kine, data_mod.to(device))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  RGP-DEQ solve: {avg_ms:.1f} +/- {std_ms:.1f} ms (n=5)")
    print(f"  Budget for 10 macro-resolves: {10*avg_ms/1000:.2f}s")
    print(f"  Budget for 60 macro-resolves: {60*avg_ms/1000:.2f}s")
    print(f"  2-min budget utilization (10 resolves): {10*avg_ms/1000/120*100:.1f}%")

    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = noslip_ok and in_dist and smooth
    print(f"  No-slip at clot:      {'PASS' if noslip_ok else 'FAIL'}")
    print(f"  z_kin in-distribution: {'PASS' if in_dist else 'FAIL'}")
    print(f"  SDF smoothness:        {'PASS' if smooth else 'WARN'}")
    print(f"  Localized diversion:   {'PASS' if local_effect else 'WARN'}")
    print(f"  Solve time:            {avg_ms:.0f}ms per macro-resolve")
    print(f"  Overall verdict:       {'[OK] SDF wall-ification is VIABLE' if all_pass else '[WARN] Issues detected - see above'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
