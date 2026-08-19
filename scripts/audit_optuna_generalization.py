"""Audit: does the Optuna generalization model actually generalize, and to what baseline?

``scripts/sweep_optuna_generalization.py`` trains on ONE vessel (patient020), selects the
checkpoint by that same vessel's deploy score, and reports the median over a 5-vessel
"validation cohort".  Every member of that cohort is either in WALL_COHORT_V2_TRAIN or on
the excluded no-clot list -- **none is held out**:

    patient012  TRAIN      patient015  TRAIN (and a TRUNCATED run, T=83)
    patient016  TRAIN      patient017  EXCLUDED, zero GT clot -> a free 1.0
    patient020  TRAIN      <- the vessel the weights were fitted on

So the reported number is an in-sample maximum over 28 Optuna trials, not a
generalization estimate.  This script re-scores the saved checkpoint under three
protocols, alongside the zero-parameter physics model, so the comparison is like-for-like.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
CKPT = Path("outputs/biochem/best_generalization_model.pth")
OPTUNA_COHORT = ["patient012", "patient015", "patient016", "patient017", "patient020"]
TRAIN_VESSEL = "patient020"
RELAX, GROW, STENCIL = 2.0, 6, {"gt": 3, "pred": 4}


def physics_score(d, bio, phys, flow="gt"):
    w = d.mask_wall.reshape(-1).bool().numpy()
    f = t0_flow_fields(d, bio, hops=STENCIL[flow], flow_source=flow)
    ei = d.edge_index.numpy()
    n = len(w)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    cur = (f.gate > 0) & w
    adm = (f.sr < float(bio.lss) * RELAX) & w
    for _ in range(GROW):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return _score(d, torch.tensor(cur.astype(np.float32)), phys)


def _score(d, pred, phys):
    w = d.mask_wall.reshape(-1).bool()
    te = resolve_deploy_eval_time_index(int(d.y.shape[0]))
    gt = gt_clot_phi_at_time(d, te, phys, device=torch.device("cpu")).reshape(-1) * w.float()
    m = compute_clot_relaxed_metrics(pred.reshape(-1) * w.float(), gt, d.edge_index, wall_mask=w)
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def build_model(sd, in_channels, device):
    from src.differentiable_wall_model.advanced_models import MeshGraphNetCorrector
    from src.differentiable_wall_model.temporal_models import (
        PseudoCGNODE, TemporalDifferentiableWallModel,
    )
    import re
    hidden = sd["node_encoder.0.weight"].shape[0]
    steps = len({int(m.group(1)) for k in sd
                 for m in [re.search(r"node_mlps\.(\d+)\.", k)] if m})
    base = TemporalDifferentiableWallModel(temporal_corrector=PseudoCGNODE())

    class Custom(MeshGraphNetCorrector):
        def forward(self, data, *, flow_source="pred", device=None):
            from torch_geometric.utils import scatter
            device = device or torch.device("cpu")
            wall = data.mask_wall.reshape(-1).float().to(device)
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            bp, bg = base_out["prob_clot"], base_out["gate_init"]
            x = torch.nan_to_num(data.x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            ea = torch.nan_to_num(data.edge_attr.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            v = self.node_encoder(torch.cat([x, bp.unsqueeze(-1), bg.unsqueeze(-1)], -1))
            e = self.edge_encoder(ea)
            row, col = data.edge_index.to(device)
            for i in range(self.processor_steps):
                e = e + self.edge_mlps[i](torch.cat([e, v[row], v[col]], -1))
                agg = scatter(e, row, dim=0, dim_size=v.size(0), reduce="sum")
                v = v + self.node_mlps[i](torch.cat([v, agg], -1))
            dl = self.node_decoder(v).squeeze(-1)
            p = torch.clamp(bp, 1e-5, 1 - 1e-5)
            base_out["prob_clot"] = torch.sigmoid(torch.log(p / (1 - p)) + dl) * wall
            return base_out

    m = Custom(base_model=base, in_channels=in_channels,
               hidden_channels=int(hidden), num_layers=int(steps)).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    dev = torch.device("cpu")
    if not CKPT.exists():
        print("[!] no checkpoint at %s" % CKPT)
        return 1
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    probe = torch.load(DIR / "patient020.pt", map_location="cpu", weights_only=False)
    model = build_model(sd, probe.x.size(1), dev)
    print("loaded %s  (hidden=%d, steps=%d)\n"
          % (CKPT.name, sd["node_encoder.0.weight"].shape[0], model.processor_steps))

    sealed = list(WALL_COHORT_V2_GENERALIZATION)
    universe = sorted(set(OPTUNA_COHORT) | set(sealed))
    rows = {}
    for a in universe:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        with torch.no_grad():
            out = model(d, flow_source="gt", device=dev)
            ml = _score(d, (out["prob_clot"] >= 0.5).float().cpu(), phys)
        rows[a] = dict(ml=ml, phys=physics_score(d, bio, phys, "gt"),
                       T=int(d.y.shape[0]))

    print("%-12s %-22s %5s %8s %8s %8s" % ("vessel", "role", "T", "ML", "physics", "delta"))
    for a in universe:
        if a not in rows:
            continue
        role = ("TRAINED ON" if a == TRAIN_VESSEL else
                "train-cohort" if a in WALL_COHORT_V2_TRAIN else
                "SEALED" if a in sealed else "excluded/no-clot")
        r = rows[a]
        print("%-12s %-22s %5d %8.4f %8.4f %+8.4f%s"
              % (a, role, r["T"], r["ml"], r["phys"], r["ml"] - r["phys"],
                 "  <-- truncated" if r["T"] < 150 else ""))

    def block(name, names, fn):
        v_ml = [rows[a]["ml"] for a in names if a in rows]
        v_ph = [rows[a]["phys"] for a in names if a in rows]
        if not v_ml:
            return
        print("%-46s ML %.4f | physics %.4f | delta %+.4f"
              % (name, fn(v_ml), fn(v_ph), fn(v_ml) - fn(v_ph)))

    print("\n--- as the Optuna script reports it ---")
    block("median of the 5-vessel 'validation cohort'", OPTUNA_COHORT, np.median)
    print("\n--- same cohort, cleaned up ---")
    block("mean of the 5", OPTUNA_COHORT, np.mean)
    clean = [a for a in OPTUNA_COHORT if a != "patient017" and rows.get(a, {}).get("T", 0) >= 150]
    block("drop no-clot 017 + truncated 015 (n=%d)" % len(clean), clean, np.mean)
    block("drop those AND the trained-on vessel (n=%d)"
          % len([a for a in clean if a != TRAIN_VESSEL]),
          [a for a in clean if a != TRAIN_VESSEL], np.mean)
    print("\n--- the honest question: the TRUE sealed set (0 of these seen in any form) ---")
    block("SEALED mean (n=%d)" % len([a for a in sealed if a in rows]), sealed, np.mean)
    block("SEALED median", sealed, np.median)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
