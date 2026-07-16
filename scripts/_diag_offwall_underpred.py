"""Diagnostic: off-wall clot underprediction analysis.

Loads the WC_v2_dilation checkpoint (current best off-wall baseline) and a
patient graph, then produces a detailed breakdown of:

  1. GT off-wall clot counts by hop distance from wall (hop 1, 2, 3+)
  2. Predicted off-wall clot counts under the current model
  3. Mat level at each hop distance (is the model outputting low values or are
     they clamped post-hoc by the ceiling/nucleation mask?)
  4. Summary of env flags actually used during evaluation
  5. Which env flags are blocking off-wall gradients in the MAT_GROWTH_SIMPLE_RECIPE

Run:
    python scripts/_diag_offwall_underpred.py
    python scripts/_diag_offwall_underpred.py --ckpt outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth
    python scripts/_diag_offwall_underpred.py --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- repo root ---
_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root))

import torch


def _hop_distance_mask(
    wall_mask: torch.Tensor,
    edge_index: torch.Tensor,
    max_hops: int = 6,
) -> torch.Tensor:
    """Return per-node hop distance from the nearest wall node (BFS)."""
    n = wall_mask.shape[0]
    dist = torch.full((n,), fill_value=max_hops + 1, dtype=torch.long)
    dist[wall_mask.bool()] = 0

    src, dst = edge_index[0], edge_index[1]
    for h in range(1, max_hops + 1):
        frontier = dist == (h - 1)
        # propagate: dst gets distance h if src is at h-1
        prop = torch.zeros(n, dtype=torch.bool)
        prop.scatter_reduce_(0, dst, frontier[src], reduce="amax", include_self=True)
        # also propagate in the reverse direction (undirected)
        prop.scatter_reduce_(0, src, frontier[dst], reduce="amax", include_self=True)
        new_reach = prop & (dist > h)
        dist[new_reach] = h

    return dist


def _load_anchor(anchor: str) -> object:
    """Load a patient graph from the standard biochem data location."""
    from src.data_gen.lib.biochem_graph_loader import load_biochem_graph
    return load_biochem_graph(anchor)


def _rollout_baseline(
    ckpt_path: Path,
    data,
    device: torch.device,
) -> dict:
    """Run the deploy rollout and return phi_by_time + mu_by_time."""
    from src.core_physics.species_gnn_clot_rollout import load_species_gnn_rollout_bundle
    from src.biochem_gnn.model import BiochemDeployStack, FlowMode

    bundle = load_species_gnn_rollout_bundle(ckpt_path, device=device)
    if bundle is None:
        raise FileNotFoundError(f"Cannot load bundle from {ckpt_path}")

    stack = BiochemDeployStack(
        bundle,
        device=device,
    )
    result = stack.rollout(data)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Off-wall clot underprediction diagnostics")
    ap.add_argument(
        "--ckpt",
        default="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth",
        help="Path to species GNN checkpoint (default: WC_v2_dilation best.pth)",
    )
    ap.add_argument("--anchor", default="patient007", help="Patient anchor to evaluate on")
    ap.add_argument("--device", default="", help="cuda / cpu (auto-detect if empty)")
    ap.add_argument("--t-eval", type=int, default=-1, help="Time step to evaluate at (-1 = last)")
    ap.add_argument("--out", default="", help="Optional JSON output path for CI")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = _root / ckpt_path

    print(f"[i] Device      : {device}")
    print(f"[i] Checkpoint  : {ckpt_path}")
    print(f"[i] Anchor      : {args.anchor}")

    # ── Section 1: env flag audit ─────────────────────────────────────────────
    print("\n[1] ENVIRONMENT FLAG AUDIT")
    print("    (flags that control off-wall gelation / supervision)")
    KEY_FLAGS = [
        "CLOT_PHI_PHYSICS_WALL_MAT_ONLY",
        "CLOT_V2_NUCLEATION_HOPS",
        "CLOT_PHI_CEILING_HOPS",
        "SPECIES_GROWTH_DILATION",
        "SPECIES_SNAPSHOT_WALL_HOPS",
        "SPECIES_DYNAMIC_OCCLUSION",
        "BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION",
        "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT",
        "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT",
        "SPECIES_PUSHFORWARD_GROWTH_THRESH_MAT",
        "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT",
    ]
    for k in KEY_FLAGS:
        v = os.environ.get(k, "(not set)")
        print(f"    {k:55s} = {v}")

    # ── Section 2: GT hop analysis ────────────────────────────────────────────
    print(f"\n[2] GT HOP-DISTANCE ANALYSIS  (anchor={args.anchor})")
    try:
        data = _load_anchor(args.anchor)
    except Exception as exc:
        print(f"    [WARN] Could not load anchor: {exc}")
        print("    Attempting fallback via graph dataset...")
        try:
            from src.data_gen.lib.biochem_graph_loader import iter_biochem_graphs
            graphs = list(iter_biochem_graphs(anchor_filter=args.anchor))
            if not graphs:
                raise RuntimeError("No graphs found")
            data = graphs[0]
        except Exception as exc2:
            print(f"    [ERROR] Cannot load graph: {exc2}")
            sys.exit(1)

    data = data.to(device)
    n_nodes = int(data.num_nodes)
    n_steps = int(data.y.shape[0])
    t_eval = args.t_eval if args.t_eval >= 0 else n_steps - 1

    # Wall mask
    wall_mask = None
    for attr in ("mask_wall", "wall_mask"):
        v = getattr(data, attr, None)
        if v is not None:
            wall_mask = v.bool().reshape(-1).to(device)
            break
    if wall_mask is None:
        print("    [WARN] No wall mask found; using SDF < 0.01 as proxy")
        sdf = data.x[:, 2].reshape(-1)
        wall_mask = (sdf < 0.01).to(device)

    n_wall = int(wall_mask.sum().item())
    print(f"    Total nodes   : {n_nodes}")
    print(f"    Wall nodes    : {n_wall}  ({100*n_wall/n_nodes:.1f}%)")
    print(f"    Off-wall      : {n_nodes - n_wall}  ({100*(n_nodes-n_wall)/n_nodes:.1f}%)")
    print(f"    Eval time step: {t_eval} / {n_steps-1}")

    # Hop distances
    edge_index = data.edge_index.to(device)
    hop_dist = _hop_distance_mask(wall_mask, edge_index, max_hops=8)

    print("\n    Hop distribution (node counts):")
    for h in range(0, 9):
        cnt = int((hop_dist == h).sum().item())
        bar = "#" * int(cnt / max(1, n_nodes / 40))
        print(f"      Hop {h}: {cnt:5d} nodes  {bar}")

    # GT clot at t_eval
    from src.config import PhysicsConfig
    phys = PhysicsConfig(phase="biochem")
    from src.core_physics.clot_growth_masks import gt_clot_mask_at_time
    gt_clot = gt_clot_mask_at_time(data, t_eval, phys, device).bool().reshape(-1)
    n_gt_clot = int(gt_clot.sum().item())
    n_gt_offwall = int((gt_clot & ~wall_mask).sum().item())
    n_gt_wall = int((gt_clot & wall_mask).sum().item())

    print(f"\n    GT clot at t={t_eval}: {n_gt_clot} total | {n_gt_wall} on-wall | {n_gt_offwall} off-wall")
    print("    GT clot by hop:")
    for h in range(0, 7):
        at_h = (hop_dist == h)
        gt_h = int((gt_clot & at_h).sum().item())
        tot_h = int(at_h.sum().item())
        pct = 100 * gt_h / max(1, tot_h)
        bar = "#" * min(40, int(gt_h * 2))
        print(f"      Hop {h}: {gt_h:4d} / {tot_h:5d} nodes clotted ({pct:5.1f}%)  {bar}")

    # ── Section 3: Model prediction analysis ─────────────────────────────────
    print(f"\n[3] MODEL PREDICTION ANALYSIS")
    # Apply the leg's env overrides first
    print("    Loading checkpoint metadata...")
    try:
        ckpt_raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        meta = ckpt_raw.get("meta", {})
        env_overrides = meta.get("env_overrides") or {}
        if env_overrides:
            print(f"    Applying {len(env_overrides)} env overrides from checkpoint meta:")
            for k, v in env_overrides.items():
                os.environ[k] = str(v)
                print(f"      {k} = {v}")
        else:
            print("    [NOTE] No env_overrides in checkpoint meta. Using current environment.")
            print("    [NOTE] For WC_v2_dilation, manually applying its known overrides:")
            # WC_v2_dilation overrides from mat_growth_simple.py:
            dilation_overrides = {
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_GROWTH_DILATION": "2",
                "CLOT_V2_NUCLEATION_HOPS": "2",
                "CLOT_PHI_CEILING_HOPS": "5",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
            }
            for k, v in dilation_overrides.items():
                os.environ.setdefault(k, v)
                print(f"      {k} = {v}")
    except Exception as exc:
        print(f"    [WARN] Could not load ckpt meta: {exc}")

    print("\n    Running rollout (this may take 1-2 min)...")
    try:
        result = _rollout_baseline(ckpt_path, data, device)
        phi_by_t = result.phi_by_time
        phi_pred = phi_by_t.get(t_eval)
        if phi_pred is None:
            phi_pred = phi_by_t[max(phi_by_t.keys())]
        phi_pred = phi_pred.reshape(-1)

        pred_clot = (phi_pred >= 0.5).bool()
        n_pred_clot = int(pred_clot.sum().item())
        n_pred_offwall = int((pred_clot & ~wall_mask).sum().item())
        n_pred_wall = int((pred_clot & wall_mask).sum().item())

        print(f"    Pred clot at t={t_eval}: {n_pred_clot} total | {n_pred_wall} on-wall | {n_pred_offwall} off-wall")
        print(f"    GT   clot at t={t_eval}: {n_gt_clot} total | {n_gt_wall} on-wall | {n_gt_offwall} off-wall")
        print(f"\n    Off-wall recall : {n_pred_offwall}/{n_gt_offwall} = {n_pred_offwall/max(1,n_gt_offwall):.3f}")

        print("\n    Predicted clot by hop:")
        for h in range(0, 7):
            at_h = (hop_dist == h)
            pred_h = int((pred_clot & at_h).sum().item())
            gt_h = int((gt_clot & at_h).sum().item())
            tot_h = int(at_h.sum().item())
            bar = "#" * min(40, pred_h * 2)
            print(f"      Hop {h}: pred={pred_h:4d} | gt={gt_h:4d} | total={tot_h:5d}  {bar}")

        print("\n    Mean phi score by hop (are off-wall scores near 0 or just below threshold?):")
        for h in range(0, 7):
            at_h = (hop_dist == h)
            if not at_h.any():
                continue
            phi_h = phi_pred[at_h]
            gt_h_mask = gt_clot[at_h]
            mean_all = float(phi_h.mean().item())
            mean_gt = float(phi_h[gt_h_mask].mean().item()) if gt_h_mask.any() else float("nan")
            mean_nongt = float(phi_h[~gt_h_mask].mean().item()) if (~gt_h_mask).any() else float("nan")
            print(f"      Hop {h}: mean_phi={mean_all:.3f} | gt_nodes={mean_gt:.3f} | non-gt={mean_nongt:.3f}")

    except Exception as exc:
        import traceback
        print(f"    [ERROR] Rollout failed: {exc}")
        traceback.print_exc()

    # ── Section 4: Training recipe flag check ─────────────────────────────────
    print("\n[4] TRAINING RECIPE FLAG IMPACT SUMMARY")
    print("    Flags blocking off-wall learning (MAT_GROWTH_SIMPLE_RECIPE defaults):")
    blocking = {
        "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": (
            "1",
            "BLOCKS: masks gelation at Hop>=1. Off-wall grad = 0.",
            "Must be 0 for off-wall learning.",
        ),
    }
    for k, (bad_val, impact, fix) in blocking.items():
        cur = os.environ.get(k, "0")
        status = "[BLOCKED]" if cur == bad_val else "[OK]"
        print(f"    {status} {k}={cur}  |  {impact}")
        if cur == bad_val:
            print(f"           Fix: {fix}")

    print("\n    Current nucleation reach per step:")
    nuc_hops = int(os.environ.get("CLOT_V2_NUCLEATION_HOPS", "1"))
    ceiling = int(os.environ.get("CLOT_PHI_CEILING_HOPS", "3"))
    dil = int(os.environ.get("SPECIES_GROWTH_DILATION", "1"))
    print(f"      CLOT_V2_NUCLEATION_HOPS = {nuc_hops}  (front advance per step)")
    print(f"      CLOT_PHI_CEILING_HOPS   = {ceiling}  (max allowed hop distance)")
    print(f"      SPECIES_GROWTH_DILATION = {dil}  (dilate seed each step)")
    max_reach = min(ceiling, nuc_hops * n_steps)
    print(f"      Max reachable at t={t_eval}: Hop {min(ceiling, nuc_hops * (t_eval+1))}")
    if n_gt_offwall > 0:
        print(f"      GT off-wall: {n_gt_offwall} nodes (need model to reach them)")

    print("\n[i] Diagnostic complete.")

    if args.out:
        out_data = {
            "anchor": args.anchor,
            "ckpt": str(ckpt_path),
            "t_eval": t_eval,
            "n_nodes": n_nodes,
            "n_wall": n_wall,
            "n_gt_clot": n_gt_clot,
            "n_gt_offwall": n_gt_offwall,
            "key_env": {k: os.environ.get(k, "") for k in KEY_FLAGS},
        }
        try:
            out_data["n_pred_offwall"] = n_pred_offwall
            out_data["n_pred_clot"] = n_pred_clot
        except NameError:
            pass
        Path(args.out).write_text(json.dumps(out_data, indent=2))
        print(f"[i] Results saved -> {args.out}")


if __name__ == "__main__":
    main()
