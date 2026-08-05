"""Why does the pocket gate devastate patient037 specifically?

diag_pocket_gate_sweep.py found pct=5 helps patient021 (+0.451 F1) and patient035
(+0.314) but wrecks patient037 (-0.186, recall 54%->6.5%) and patient032 (-0.378) --
even though patient037's own wall-band flow percentile distribution is not anomalous
(p5=0.167, in range with the other three vessels). So it isn't a flow-scale mismatch.

Working hypothesis: patient037's baseline prediction is already weak (54% recall, 19%
precision off-gate) -- the model struggles to grow this vessel correctly to begin with.
Its few correct (TP) components may be partial/boundary fragments of the true GT pocket
rather than confident, deep commits, so their minimum hop-2 speed sits higher than a
fully-formed pocket's would -- and a gate built around "does this reach a stagnant core"
kills exactly those marginal-but-correct fragments first.

This rolls out once and checks directly:
  1. TP vs FP component h2min distributions (does the gate's discriminator even
     separate them for this vessel, the way it does on patient020/28-vessel probe)?
  2. Per-GT-pocket coverage: for each ground-truth clot component, what fraction does
     the model's prediction actually capture? Low coverage = partial fragments,
     confirming the hypothesis. High coverage with high h2min = a different failure
     (the true pocket itself sits at moderate flow for this vessel).

    python scripts/diag_pocket037_mechanism.py --anchor patient037
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _load_static  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import (  # noqa: E402
    reset_species_rollout_flow_cache,
    resolve_species_rollout_uv,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_clot_phi_fields,
    deploy_species_rollout_series,
    load_continuous_bundle,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.evaluation.pocket_gate import hop2_speed_field  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth")
    ap.add_argument("--anchor", default="patient037")
    ap.add_argument("--compare-anchor", default="patient021",
                    help="A vessel where the gate worked, for side-by-side contrast")
    args = ap.parse_args()

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    root = get_project_root()
    device = require_cuda_device()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="pocket037_diag", ckpt_path=ckpt)
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    for anc in (args.anchor, args.compare_anchor):
        print(f"\n{'='*70}\n=== {anc} ===\n{'='*70}", flush=True)
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = _load_static(data, device, kine, wall_hops, anc)
        static["n_times"] = int(data.y.shape[0])
        print("  rolling out...", flush=True)
        series, data = deploy_species_rollout_series(
            model, data, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
        )
        phi_pred, phi_gt, wall_mask, t_eval = deploy_clot_phi_fields(
            data, series, static, phys, bio, device, flow_source="kinematics", gelation_beta=None,
        )
        n = int(data.num_nodes)
        u0, v0 = resolve_species_rollout_uv(data, 0, device, for_training=False)
        h2 = hop2_speed_field(data, device, u0, v0).cpu().numpy()
        wall = wall_mask.reshape(-1).bool().cpu().numpy() if wall_mask is not None else np.ones(n, bool)
        pcts = np.percentile(h2[wall], [5, 10, 25, 50])
        print(f"  vessel flow percentiles: p5={pcts[0]:.4f} p10={pcts[1]:.4f} p25={pcts[2]:.4f} p50={pcts[3]:.4f}")

        pred = (phi_pred > 0.5).cpu().numpy()
        gt = (phi_gt > 0.5).cpu().numpy()
        ei = data.edge_index.cpu().numpy()

        def components(mask):
            idx = np.where(mask)[0]
            if idx.size == 0:
                return [], idx
            keep = np.isin(ei[0], idx) & np.isin(ei[1], idx)
            remap = {v: i for i, v in enumerate(idx)}
            rr = np.array([remap[x] for x in ei[0][keep]], dtype=int)
            cc = np.array([remap[x] for x in ei[1][keep]], dtype=int)
            if rr.size == 0:
                return [(np.array([i]),) for i in idx], idx
            A = coo_matrix((np.ones(rr.size), (rr, cc)), shape=(idx.size, idx.size))
            nc, lab = connected_components(A, directed=False)
            return [idx[lab == k] for k in range(nc)], idx

        pred_comps, pred_idx = components(pred)
        gt_comps, gt_idx = components(gt)
        print(f"  n_gt={gt.sum()} in {len(gt_comps)} GT pockets  |  "
              f"n_pred={pred.sum()} in {len(pred_comps)} predicted components")

        # 1. TP vs FP predicted-component h2min
        tp_h2min, fp_h2min, tp_sizes, fp_sizes = [], [], [], []
        for nd in pred_comps:
            is_tp = bool(gt[nd].any())
            (tp_h2min if is_tp else fp_h2min).append(float(h2[nd].min()))
            (tp_sizes if is_tp else fp_sizes).append(len(nd))
        print(f"\n  Predicted components: {len(tp_h2min)} touch GT (TP), {len(fp_h2min)} do not (FP)")
        if tp_h2min:
            print(f"    TP h2min: mean={np.mean(tp_h2min):.4f} min={min(tp_h2min):.4f} max={max(tp_h2min):.4f}"
                  f"  sizes={sorted(tp_sizes, reverse=True)[:8]}")
        if fp_h2min:
            print(f"    FP h2min: mean={np.mean(fp_h2min):.4f} min={min(fp_h2min):.4f} max={max(fp_h2min):.4f}"
                  f"  sizes={sorted(fp_sizes, reverse=True)[:8]}")
        if tp_h2min and fp_h2min:
            a = np.array(tp_h2min).reshape(-1, 1)
            b = np.array(fp_h2min).reshape(1, -1)
            auc = float(((a < b).mean() + 0.5 * (a == b).mean()))
            print(f"    AUC(TP h2min < FP h2min) = {auc:.4f}  (1.0 = perfect separation, gate-friendly)")

        # 2. Per-GT-pocket coverage: does the model even find each true pocket, fully or partially?
        print(f"\n  Per-GT-pocket coverage (does the model's prediction reach each true pocket?):")
        print(f"  {'gt_size':>8} {'covered':>8} {'coverage':>9} {'covering_comp_h2min':>20}")
        for gnd in sorted(gt_comps, key=len, reverse=True)[:10]:
            covered = pred[gnd].sum()
            cov_frac = covered / len(gnd)
            # h2min of whichever predicted component(s) overlap this GT pocket
            touching_h2min = []
            for pnd in pred_comps:
                if np.isin(pnd, gnd).any():
                    touching_h2min.append(float(h2[pnd].min()))
            th = f"{min(touching_h2min):.4f}" if touching_h2min else "none (missed)"
            print(f"  {len(gnd):8d} {covered:8d} {cov_frac:9.2%} {th:>20}")

        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
