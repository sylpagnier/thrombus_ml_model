"""Load and run the locked clot-GNN ensemble by name.

    from src.clot_ml.locked import load_ensemble, predict_scores
    ens = load_ensemble()                      # reads data/reference/clot_gnn_locked.json
    score = predict_scores(ens, sample)        # [N] per-node probability

For the temporal model (v3 and on), use the dispatcher instead of ``load_ensemble``
directly -- it reads the pointer's ``kind`` and returns whatever is currently shipped:

    from src.clot_ml.locked import load_default, predict_default_series
    bundle = load_default()
    out = predict_default_series(bundle, data, times)   # {score, mask, onset, series}
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
POINTER = REPO / "data/reference/clot_gnn_locked.json"


def load_ensemble(name: str | None = None, device=None) -> dict:
    ptr = json.loads(POINTER.read_text())
    root = REPO / (ptr["path"] if name is None else f"outputs/clot_ml/locked/{name}")
    manifest = json.loads((root / "manifest.json").read_text())
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = np.load(root / "feature_norm.npz", allow_pickle=True)

    from src.clot_ml.gnn import ClotGNN

    members = []
    for m in manifest["members"]:
        blob = torch.load(root / m["file"], map_location=dev, weights_only=False)
        edim = 7  # edge_features() width; asserted at call time
        net = ClotGNN(blob["in_dim"], edim, dim=m["dim"], layers=m["layers"], drop=0.0,
                      extra_dim=blob["extra_dim"]).to(dev)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        members.append(dict(net=net, rounds=int(m["rounds"]), file=m["file"]))
    return dict(members=members, mu=norm["mu"], sd=norm["sd"],
                cols=[str(c) for c in norm["cols"]], manifest=manifest, device=dev)


def ensemble_variant(ens: dict) -> str:
    """Which feature block this artifact was trained on -- read from its own manifest."""
    return "v4" if ens.get("manifest", {}).get("v4_channels") else "v3"


def sample_for_ensemble(ens: dict, data, bio_cfg=None, phys_cfg=None, *,
                        flow: str = "gt") -> dict:
    """Build the sample the loaded ensemble actually expects, and check the width.

    A `clot_gnn_v4` member takes 69 columns and a v2/v3 member takes 56; feeding one the
    other's block fails deep inside the first linear layer with an unhelpful shape error.
    Both counts include ``phys_mask``.
    """
    S = build_sample(data, bio_cfg, phys_cfg, flow=flow, variant=ensemble_variant(ens))
    # `n_features` in the manifest is the FULL width, phys_mask included -- it is written
    # from the cache's own `cols` after `attach_physics` has appended it.
    want = int(ens["manifest"].get("n_features", S["X"].shape[1]))
    got = S["X"].shape[1]
    if got != want:
        raise ValueError("%s expects %d features, sample has %d"
                         % (ens["manifest"].get("name", "ensemble"), want, got))
    return S


@torch.no_grad()
def predict_scores(ens: dict, sample: dict) -> np.ndarray:
    """Mean per-node probability over the ensemble.  ``sample`` is a clot-ml cache entry."""
    from scripts.train_clot_gnn import build_graph, rollout  # noqa: PLC0415

    out = None
    for m in ens["members"]:
        g = build_graph(sample, ens["mu"], ens["sd"], ens["device"], need_fb=m["rounds"] > 1)
        logit, _ = rollout(m["net"], g, m["rounds"])
        p = torch.sigmoid(logit).cpu().numpy()
        out = p if out is None else out + p
    return out / max(len(ens["members"]), 1)


# Thresholds the locked readout was tuned at (docs/PHASE9_ML.md 8, per domain).
THRESH_WALL, THRESH_OFF = 0.73, 0.92

# Off-wall onset = the time the node's OWNER wall trajectory reaches ``crit / OFF_ATT``.
# 0.80 means "just after its owner commits".  Chosen for a PHYSICAL reason, not a scored
# one: an off-wall node cannot clot before the wall node feeding it, and freezing off-wall
# at the final mask (the alternative) puts off-wall clot on screen at t=0 with an empty
# wall, which is nonsense.  On score it is a wash -- the att sweep reads
# 0.490 / 0.494 / 0.459 / 0.510 at 0.16 / 0.30 / 0.50 / 0.80 against frozen's 0.5015, all
# inside noise -- so the constraint is doing the work, not a fit.
OFF_ATT = 0.80


def build_sample(data, bio_cfg=None, phys_cfg=None, *, flow: str = "gt",
                 variant: str = "v3") -> dict:
    """Feature dict for one raw pack, matching the locked ensemble's training layout.

    ``variant="v4"`` additionally applies :func:`src.clot_ml.features_v4.augment_sample`,
    which appends the 13 advective-transport / indicator-gate channels a `clot_gnn_v4`
    member expects.  Order matters and is pinned by the cache builder: the v4 block goes
    **after** the 55 base channels and **before** ``phys_mask``.  Use
    :func:`sample_for_ensemble` rather than choosing the variant by hand.
    """
    from src.clot_ml.data import physics_mask
    from src.clot_ml.features import build_features, feature_matrix
    from src.config import BiochemConfig, PhysicsConfig

    bio = bio_cfg or BiochemConfig(phase="biochem")
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    S = build_features(data, bio, phys, flow=flow)
    X, cols = feature_matrix(S["F"])
    out = dict(X=X, cols=np.array(cols), y=S["y"], mat_gt=S["mat_gt"], wall=S["wall"],
               shell=S["shell"], owner=S["owner"], edge_index=S["edge_index"],
               pos=S["pos"], mat_phys=S["mat_phys"], gate=S["gate"], sr=S["sr"],
               spd=S["spd"], u=S["u"], v=S["v"])
    if variant == "v4":
        from src.clot_ml.features_v4 import augment_sample
        Xv, colsv = augment_sample(data, out, bio)
        out["X"], out["cols"] = Xv, np.array(colsv)
    m = physics_mask(out)
    out["phys_mask"] = m
    out["X"] = np.concatenate([out["X"], m.astype(np.float32).reshape(-1, 1)], axis=1)
    out["cols"] = np.array([str(c) for c in out["cols"]] + ["phys_mask"])
    return out


def predict_clot_series(ens: dict, data, times, *, flow: str = "gt",
                        sample: dict | None = None) -> dict:
    """Clot mask at each requested time index.

    The SET is the locked ensemble's; the WALL timing is the zero-parameter surface ODE's
    first crossing of ``viscosity_mat_crit``.  Off-wall stays frozen at the final mask --
    every off-wall timing rule measured so far scores below frozen because it depends on the
    ``Mat`` magnitude field (docs/PHASE9_ML.md 12.4).

    Returns ``{score, mask, onset, series}`` where ``series`` maps time index -> bool mask.
    """
    from src.clot_ml.temporal import mask_series, ode_trajectory, onset_from_ode
    from src.config import BiochemConfig

    bio = BiochemConfig(phase="biochem")
    S = sample if sample is not None else build_sample(data, bio, flow=flow)
    score = predict_scores(ens, S)
    wall = S["wall"]
    mask = ((score >= THRESH_WALL) & wall) | ((score >= THRESH_OFF) & ~wall)

    traj, _ = ode_trajectory(data, bio, flow=flow)
    crit = float(bio.viscosity_mat_crit)
    onset = onset_from_ode(traj, mask, wall, S["pos"].astype(np.float64), crit,
                           attenuation=OFF_ATT)
    return dict(score=score, mask=mask, onset=onset,
                series=mask_series(onset, mask, times))


@torch.no_grad()
def predict_mat(ens: dict, sample: dict) -> np.ndarray:
    """Mean predicted ``log1p(Mat/crit)`` over the ensemble's REGRESSION head.

    That head exists in every locked member (physics-based, zero-init residual on the
    backbone's own ``Mat``) but has never been the readout -- the deploy score uses the
    classifier.  It is the natural place to read the magnitude field from.
    """
    from scripts.train_clot_gnn import build_graph, rollout  # noqa: PLC0415

    out = None
    for m in ens["members"]:
        g = build_graph(sample, ens["mu"], ens["sd"], ens["device"], need_fb=m["rounds"] > 1)
        _, reg = rollout(m["net"], g, m["rounds"])
        r = reg.cpu().numpy()
        out = r if out is None else out + r
    return out / max(len(ens["members"]), 1)


# ---------------------------------------------------------------------------
# v3: time-conditioned model (docs/PHASE9_ML.md 13.9)
# ---------------------------------------------------------------------------
def load_temporal_v3(name: str | None = None) -> dict:
    """Load a v3-kind artifact: the base GNN (the SET) plus the time-conditioned head."""
    ptr = json.loads(POINTER.read_text())
    root = REPO / (ptr["path"] if name is None else f"outputs/clot_ml/locked/{name}")
    manifest = json.loads((root / "manifest.json").read_text())
    ens = load_ensemble(name=manifest["base_set_model"])
    with (root / manifest["clf_file"]).open("rb") as fh:
        clf = pickle.load(fh)
    return dict(ens=ens, clf=clf, manifest=manifest,
               thresh_wall=float(manifest["thresh_wall"]),
               thresh_off=float(manifest["thresh_off"]),
               n_times_trained=int(manifest["n_times"]))


def enforce_owner_and_monotone(series: dict[int, np.ndarray], wall: np.ndarray,
                               owner: np.ndarray, times) -> dict[int, np.ndarray]:
    """Two physical constraints applied to a raw per-time mask series, in place order:

    1. MONOTONE in time -- the production law has no sink, so a node once clot stays clot.
    2. An off-wall node cannot be clot before its OWNER wall node is (it is fed by it).

    Pure and model-free, so it is unit-testable without loading any weights.
    """
    out: dict[int, np.ndarray] = {}
    prev = np.zeros_like(wall)
    for ti in times:
        m = series[ti] | prev
        m = m & (wall | m[owner])
        out[int(ti)] = m
        prev = m
    return out


def predict_temporal_v3(bundle: dict, data, times, *, flow: str = "gt",
                        sample: dict | None = None) -> dict:
    """Time-conditioned prediction: P(clot at t) is a direct model output, not a schedule
    imposed on a static onset ranking.  Returns ``{score, mask, onset, series}`` with the
    same shape as :func:`predict_clot_series`, so callers do not need to know which
    generation shipped."""
    from scripts.promote_clot_gnn_v3 import node_feats, time_row  # noqa: PLC0415
    from src.clot_ml.temporal import ode_trajectory  # noqa: PLC0415
    from src.config import BiochemConfig  # noqa: PLC0415

    bio = BiochemConfig(phase="biochem")
    S = sample if sample is not None else build_sample(data, bio, flow=flow)
    wall = S["wall"]
    crit = float(bio.viscosity_mat_crit)

    sc = predict_scores(bundle["ens"], S)
    gnn_mask = ((sc >= THRESH_WALL) & wall) | ((sc >= THRESH_OFF) & ~wall)

    traj, t_grid = ode_trajectory(data, bio, flow=flow)
    r0 = traj[1] / max(float(t_grid[1] - t_grid[0]), 1e-9)
    hot = traj >= crit
    T = traj.shape[0]
    oon = np.where(hot.any(0), hot.argmax(0), T)
    Xn = node_feats(S, r0, oon, T, sc)

    clf = bundle["clf"]
    P = np.zeros((len(times), len(wall)), dtype=np.float32)
    for j, ti in enumerate(times):
        P[j] = clf.predict_proba(time_row(Xn, int(ti), T, oon))[:, 1]
    Pmono = np.maximum.accumulate(P, axis=0)

    widx = np.flatnonzero(wall)
    owner = S["owner"] if "owner" in S else widx[np.zeros(len(wall), dtype=int)]

    raw = {int(ti): gnn_mask & (Pmono[j] >= np.where(wall, bundle["thresh_wall"],
                                                     bundle["thresh_off"]))
           for j, ti in enumerate(times)}
    series = enforce_owner_and_monotone(raw, wall, owner, times)

    onset = np.full(len(wall), -1, dtype=int)
    seen = np.zeros(len(wall), dtype=bool)
    for ti in times:
        newly = series[int(ti)] & ~seen
        onset[newly] = int(ti)
        seen |= series[int(ti)]
    score = Pmono[-1] if len(times) else sc
    return dict(score=score, mask=series[int(times[-1])], onset=onset, series=series)


def load_default(device=None) -> tuple[dict, str]:
    """Follow the pointer and load whatever generation is currently shipped.

    Returns ``(bundle, kind)``; ``kind`` is ``"gnn_ensemble"`` (v1/v2, ODE-only timing) or
    ``"temporal_v3"``.  Use with :func:`predict_default_series`.
    """
    ptr = json.loads(POINTER.read_text())
    kind = ptr.get("kind", "gnn_ensemble")
    if kind == "temporal_v3":
        return load_temporal_v3(), kind
    return load_ensemble(device=device), kind


def predict_default_series(bundle: dict, kind: str, data, times, *, flow: str = "gt",
                           sample: dict | None = None) -> dict:
    if kind == "temporal_v3":
        return predict_temporal_v3(bundle, data, times, flow=flow, sample=sample)
    return predict_clot_series(bundle, data, times, flow=flow, sample=sample)
