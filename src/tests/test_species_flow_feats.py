"""Clot-aware flow features for the species teacher (speed + shear + divergence + geometry)."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.core_physics.species_pushforward_gnn import (
    GEOM_FEATS_DIM,
    GEOM_FEATS_RICH_DIM,
    _flow_band_features,
    _flow_feats_series_from_y,
    _geometry_band_features,
    _resolve_flow_uv,
    flow_feats_ablate,
    flow_feats_drop_xy,
    flow_feats_dynamic,
    flow_feats_enabled,
    geom_feats_enabled,
    geom_feats_rich_enabled,
    flow_feats_source,
)
from src.core_physics.species_pushforward_continuous import (
    SpeciesDualHeadContinuousGNN,
    continuous_feature_dim,
    continuous_frontier_hops,
    continuous_gate_temp,
    continuous_nucleation_topk,
    maybe_drop_latent,
    splice_dynamic_flow,
)
from src.core_physics.species_snapshot_gnn import build_snapshot_features


def _line_graph(device: torch.device, *, n_times: int = 3) -> Data:
    """4-node chain with a velocity field stored in y[:, :, 0:2] (GT-source readout)."""
    x = torch.zeros(4, 16, dtype=torch.float32, device=device)
    x[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)  # px
    x[:, 1] = torch.zeros(4, device=device)  # py
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long, device=device)
    data = Data(x=x, edge_index=ei)
    data.num_nodes = 4
    y = torch.zeros(n_times, 4, 18, dtype=torch.float32, device=device)
    # decelerating flow along x at the last time -> nonzero shear + negative divergence
    y[-1, :, 0] = torch.tensor([1.0, 0.6, 0.2, 0.0], device=device)  # u
    y[-1, :, 1] = torch.zeros(4, device=device)  # v
    data.y = y
    return data


def test_geom_feats_rich_flag_and_dim(monkeypatch):
    """SPECIES_GEOM_FEATS_RICH implies geom feats and appends two 2-hop channels."""
    monkeypatch.delenv("SPECIES_GEOM_FEATS", raising=False)
    monkeypatch.delenv("SPECIES_GEOM_FEATS_RICH", raising=False)
    assert geom_feats_enabled() is False
    assert geom_feats_rich_enabled() is False

    dev = torch.device("cpu")
    data = _line_graph(dev)
    data.x[:, 15] = torch.tensor([1.0, 1.2, 2.0, 2.2], device=dev)  # WIDTH_ND varies along chain
    node_idx = torch.arange(4, device=dev)

    base = _geometry_band_features(data, dev, node_idx)
    assert base.shape == (4, GEOM_FEATS_DIM)

    monkeypatch.setenv("SPECIES_GEOM_FEATS_RICH", "1")
    assert geom_feats_rich_enabled() is True
    assert geom_feats_enabled() is True  # rich implies base
    rich = _geometry_band_features(data, dev, node_idx)
    assert rich.shape == (4, GEOM_FEATS_RICH_DIM)
    assert GEOM_FEATS_RICH_DIM == GEOM_FEATS_DIM + 2
    # first 3 channels are the same construction (per-band standardized identically)
    assert torch.allclose(rich[:, :GEOM_FEATS_DIM], base, atol=1e-5)


def test_gate_temp_sharpens_spatial_gate(monkeypatch):
    """GATE_TEMP < 1 pushes sigmoid(logits/T) toward 0/1 vs the T=1 baseline (sparser support)."""
    monkeypatch.delenv("SPECIES_CONTINUOUS_GATE_TEMP", raising=False)
    assert continuous_gate_temp() == 1.0
    monkeypatch.setenv("SPECIES_CONTINUOUS_GATE_TEMP", "0.5")
    assert continuous_gate_temp() == 0.5

    dev = torch.device("cpu")
    latent_dim = 8
    in_dim = continuous_feature_dim(latent_dim)
    model = SpeciesDualHeadContinuousGNN(in_dim, hidden=16).to(dev)
    model.eval()
    x = torch.randn(6, in_dim, device=dev)
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long, device=dev)
    with torch.no_grad():
        _, logits, _ = model.forward_decoupled(x, ei)
        sharp = torch.sigmoid(logits / 0.5)
        base = torch.sigmoid(logits)
    # sharpening moves probabilities away from 0.5 (toward the hard decision) wherever logits != 0.
    nz = logits.abs().reshape(-1) > 1e-4
    assert torch.all((sharp.reshape(-1)[nz] - 0.5).abs() >= (base.reshape(-1)[nz] - 0.5).abs() - 1e-6)


def test_frontier_nucleation_mask_is_deployable_and_local(monkeypatch):
    """Frontier mask: derives from PREDICTED log_state + the model's own gate logits only (no GT),
    confines growth to the k-hop neighbourhood of committed mass, and seeds via top-k confidence."""
    monkeypatch.delenv("SPECIES_CONTINUOUS_FRONTIER_HOPS", raising=False)
    monkeypatch.delenv("SPECIES_CONTINUOUS_NUCLEATION_TOPK", raising=False)
    assert continuous_frontier_hops() == 0 and continuous_nucleation_topk() == 0.0

    dev = torch.device("cpu")
    from src.training.biochem_species_scope import MAT_CHANNEL
    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DUAL_HEAD", "1")
    monkeypatch.setenv("SPECIES_CONTINUOUS_FRONTIER_HOPS", "1")
    monkeypatch.setenv("SPECIES_CONTINUOUS_MAT_COMMIT_THRESH", "0.5")
    monkeypatch.setenv("SPECIES_CONTINUOUS_NUCLEATION_TOPK", "0.0")  # frontier-only for this check

    latent_dim = 8
    in_dim = continuous_feature_dim(latent_dim)
    model = SpeciesDualHeadContinuousGNN(in_dim, hidden=16).to(dev)
    # 5-node chain; only node 0 is committed (predicted Mat above thresh).
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    log_state = torch.zeros(5, 1)
    log_state[0, 0] = 1.0  # committed seed (PREDICTED state, not GT)
    logits = torch.zeros(5, 1)
    mask = model._frontier_nucleation_mask(logits, log_state, ei)
    assert mask.shape == (5, 1)
    m = mask.reshape(-1).bool()
    # committed node 0 and its 1-hop neighbour node 1 are allowed; far nodes 3,4 are not.
    assert bool(m[0]) and bool(m[1])
    assert not bool(m[3]) and not bool(m[4])
    # the mask is detached (a structural gate, not a grad path).
    assert not mask.requires_grad

    # top-k nucleation seeds from the model's own logits even with NO committed mass (deployable t0).
    monkeypatch.setenv("SPECIES_CONTINUOUS_NUCLEATION_TOPK", "0.2")  # ~1 of 5 nodes
    cold = torch.zeros(5, 1)  # nothing committed yet (phi=0 at deploy t0)
    strong = torch.tensor([[-9.0], [-9.0], [5.0], [-9.0], [-9.0]])  # node 2 most confident
    seed_mask = model._frontier_nucleation_mask(strong, cold, ei).reshape(-1).bool()
    assert bool(seed_mask[2])  # confident node nucleates without any GT


def test_flow_feats_flag(monkeypatch):
    monkeypatch.delenv("SPECIES_FLOW_FEATS", raising=False)
    assert flow_feats_enabled() is False
    monkeypatch.setenv("SPECIES_FLOW_FEATS", "1")
    assert flow_feats_enabled() is True
    monkeypatch.setenv("SPECIES_FLOW_FEATS_SOURCE", "gt")
    assert flow_feats_source() == "gt"


def test_flow_band_features_gt_source(monkeypatch):
    monkeypatch.setenv("SPECIES_FLOW_FEATS_SOURCE", "gt")
    monkeypatch.setenv("SPECIES_FLOW_FEATS_TIME", "-1")  # last time = formed clot
    dev = torch.device("cpu")
    data = _line_graph(dev)
    node_idx = torch.arange(4, device=dev)
    feats = _flow_band_features(data, None, dev, node_idx)  # kine_model unused for gt source
    assert feats.shape == (4, 5)  # [log1p(speed), log1p(shear), tanh(div), x_n, y_n]
    # divergence channel is tanh-bounded
    assert float(feats[:, 2].abs().max()) <= 1.0
    # node 0 has the fastest flow -> largest log1p(speed)
    assert int(torch.argmax(feats[:, 0])) == 0
    # decelerating flow (du<0 along +x) -> negative divergence somewhere
    assert float(feats[:, 2].min()) < 0.0


def test_flow_band_features_kine_source(monkeypatch):
    """kine source pulls velocity from the kine model (patched), not data.y."""
    monkeypatch.setenv("SPECIES_FLOW_FEATS_SOURCE", "kine")
    dev = torch.device("cpu")
    data = _line_graph(dev)
    node_idx = torch.arange(4, device=dev)

    import src.utils.kinematics_inference as ki

    monkeypatch.setattr(ki, "predict_kinematics", lambda model, d: torch.zeros(int(d.num_nodes), 3))
    feats = _flow_band_features(data, object(), dev, node_idx)
    assert feats.shape == (4, 5)
    # zero flow -> zero speed/shear/divergence; only geometry channels carry signal
    assert torch.allclose(feats[:, 0], torch.zeros(4))
    assert torch.allclose(feats[:, 2], torch.zeros(4))


def test_flow_feats_ablate_flag(monkeypatch):
    monkeypatch.delenv("SPECIES_FLOW_FEATS_ABLATE", raising=False)
    assert flow_feats_ablate() is False
    monkeypatch.setenv("SPECIES_FLOW_FEATS_ABLATE", "1")
    assert flow_feats_ablate() is True


def test_flow_feats_drop_xy_flag(monkeypatch):
    monkeypatch.delenv("SPECIES_FLOW_FEATS_DROP_XY", raising=False)
    assert flow_feats_drop_xy() is False
    monkeypatch.setenv("SPECIES_FLOW_FEATS_DROP_XY", "1")
    assert flow_feats_drop_xy() is True


def test_flow_band_features_drop_xy_zeros_coordinate_channels(monkeypatch):
    monkeypatch.setenv("SPECIES_FLOW_FEATS_SOURCE", "gt")
    monkeypatch.setenv("SPECIES_FLOW_FEATS_DROP_XY", "1")
    dev = torch.device("cpu")
    data = _line_graph(dev)
    node_idx = torch.arange(4, device=dev)
    feats = _flow_band_features(data, None, dev, node_idx)
    assert feats.shape == (4, 5)
    assert torch.allclose(feats[:, 3], torch.zeros(4))
    assert torch.allclose(feats[:, 4], torch.zeros(4))
    # dynamic channels remain informative
    assert float(feats[:, 0].max()) > 0.0


class _LeashModel:
    """Minimal stand-in carrying the latent-leash attributes maybe_drop_latent reads."""

    def __init__(self, latent_dim: int, p: float):
        self.kin_latent_dim = latent_dim
        self.latent_dropout_p = p


def test_maybe_drop_latent_zeros_zkin_slice():
    base = torch.ones(6, 10)  # [z_kin(4), sdf(1), flow(5)]
    model = _LeashModel(latent_dim=4, p=1.0)
    out = maybe_drop_latent(base, model, training=True)  # p=1 -> always drop
    assert torch.allclose(out[:, :4], torch.zeros(6, 4))  # z_kin zeroed
    assert torch.allclose(out[:, 4:], torch.ones(6, 6))   # sdf + flow untouched
    assert torch.allclose(base, torch.ones(6, 10))        # input not mutated in place


def test_maybe_drop_latent_eval_and_off_are_identity():
    base = torch.ones(6, 10)
    # eval mode: never drops even with p=1
    assert torch.allclose(maybe_drop_latent(base, _LeashModel(4, 1.0), training=False), base)
    # p=0: identity even in training
    assert torch.allclose(maybe_drop_latent(base, _LeashModel(4, 0.0), training=True), base)
    # no leash attrs: identity
    assert torch.allclose(maybe_drop_latent(base, object(), training=True), base)


def test_zkin_is_leading_slice_of_features():
    """Trap G1: the leash zeros base_feats[:, :kin_latent_dim]; lock that z_kin leads, sdf trails."""
    z = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 1.0  # latent_dim=3, nonzero
    sdf = torch.tensor([0.2, 0.4, 0.6, 0.8])
    feats = build_snapshot_features(z, sdf)  # no kin_mean/std -> z passes through unnormalized
    assert feats.shape == (4, 4)  # [z_kin(3), sdf(1)]
    assert torch.allclose(feats[:, :3], z)            # z_kin occupies the FIRST latent_dim columns
    assert float(feats[:, 3].max()) <= 1.0            # sdf normalized into the trailing column
    # zeroing the leading latent_dim columns leaves sdf intact (what maybe_drop_latent does)
    dropped = feats.clone()
    dropped[:, :3] = 0.0
    assert torch.allclose(dropped[:, 3], feats[:, 3])


def test_resolve_flow_uv_auto_coupling_override(monkeypatch):
    """Trap F: the real deploy path (source=auto + coupling) returns the coupled field, else kine."""
    monkeypatch.delenv("SPECIES_FLOW_FEATS_SOURCE", raising=False)  # default 'auto'
    assert flow_feats_source() == "auto"
    dev = torch.device("cpu")
    data = _line_graph(dev)

    import src.utils.kinematics_inference as ki
    from src.inference.corrector_coupling import (
        reset_coupled_flow_registry,
        set_coupled_flow,
    )

    # kine base flow = constant 1.0 (clot-blind)
    monkeypatch.setattr(ki, "predict_kinematics", lambda m, d: torch.ones(int(d.num_nodes), 3))

    # coupling OFF -> kine field
    monkeypatch.delenv("BIOCHEM_CORRECTOR_COUPLING", raising=False)
    reset_coupled_flow_registry()
    u, v = _resolve_flow_uv(data, object(), dev)
    assert torch.allclose(u, torch.ones(4))

    # coupling ON but registry empty -> falls back to kine
    monkeypatch.setenv("BIOCHEM_CORRECTOR_COUPLING", "1")
    reset_coupled_flow_registry()
    u, v = _resolve_flow_uv(data, object(), dev)
    assert torch.allclose(u, torch.ones(4))

    # coupling ON + coupled field published -> override wins
    coupled_u = torch.tensor([5.0, 6.0, 7.0, 8.0])
    coupled_v = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    set_coupled_flow(data, coupled_u, coupled_v)
    u, v = _resolve_flow_uv(data, object(), dev)
    assert torch.allclose(u, coupled_u)
    assert torch.allclose(v, coupled_v)
    reset_coupled_flow_registry()


def test_resolve_flow_uv_prefers_u0_pred_without_deq(monkeypatch):
    """Pack-build baseline UV must not trigger a second GINO-DEQ when u0_pred is present."""
    monkeypatch.delenv("SPECIES_FLOW_FEATS_SOURCE", raising=False)  # default auto
    monkeypatch.delenv("BIOCHEM_CORRECTOR_COUPLING", raising=False)
    dev = torch.device("cpu")
    data = _line_graph(dev)
    data.u0_pred = torch.tensor([9.0, 8.0, 7.0, 6.0])
    data.v0_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])

    import src.utils.kinematics_inference as ki

    calls: list[int] = []

    def _boom(model, d):
        calls.append(1)
        raise AssertionError("predict_kinematics must not run when u0_pred is set")

    monkeypatch.setattr(ki, "predict_kinematics", _boom)
    u, v = _resolve_flow_uv(data, object(), dev)
    assert calls == []
    assert torch.allclose(u, data.u0_pred)
    assert torch.allclose(v, data.v0_pred)


def test_flow_feats_dynamic_flag(monkeypatch):
    monkeypatch.delenv("SPECIES_FLOW_FEATS_DYNAMIC", raising=False)
    assert flow_feats_dynamic() is False
    monkeypatch.setenv("SPECIES_FLOW_FEATS_DYNAMIC", "1")
    assert flow_feats_dynamic() is True


def test_flow_feats_series_is_time_varying():
    """Trap C: the per-time series has shape [n_t, n_band, 5] and differs across times."""
    dev = torch.device("cpu")
    data = _line_graph(dev, n_times=3)  # only the last time has nonzero velocity
    node_idx = torch.arange(4, device=dev)
    series = _flow_feats_series_from_y(data, dev, node_idx)
    assert series.shape == (3, 4, 5)
    # t0 velocity is zero -> zero speed; last time is fast -> nonzero speed (temporal sharpening)
    assert torch.allclose(series[0, :, 0], torch.zeros(4))
    assert float(series[-1, :, 0].max()) > 0.0
    assert not torch.allclose(series[0], series[-1])


def test_splice_dynamic_flow_replaces_block_and_clamps():
    base = torch.ones(4, 10)  # [z_kin(4), sdf(1), flow(5)]
    flow_cols = (5, 5)
    series = torch.zeros(3, 4, 5)
    series[2] = 9.0  # distinct last-time block
    out = splice_dynamic_flow(base, series, flow_cols, time_index=2)
    assert torch.allclose(out[:, 5:10], torch.full((4, 5), 9.0))  # flow block replaced
    assert torch.allclose(out[:, :5], torch.ones(4, 5))           # z_kin + sdf untouched
    # time index clamps to n_t-1
    out_clamp = splice_dynamic_flow(base, series, flow_cols, time_index=99)
    assert torch.allclose(out_clamp[:, 5:10], torch.full((4, 5), 9.0))
    # no-op when series/cols/time missing
    assert torch.allclose(splice_dynamic_flow(base, None, flow_cols, 2), base)
    assert torch.allclose(splice_dynamic_flow(base, series, None, 2), base)


def test_dual_head_gate_fp_penalizes_inactive_nodes(monkeypatch):
    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DUAL_HEAD", "1")
    monkeypatch.setenv("SPECIES_CONTINUOUS_GATE_FP_WEIGHT", "4.0")
    from src.core_physics.species_pushforward_continuous import dual_head_step_loss

    n = 6
    mask = torch.ones(n, dtype=torch.bool)
    spatial_logits = torch.zeros(n, 1, requires_grad=True)
    magnitude = torch.zeros(n, 1)
    tgt_delta = torch.zeros(n, 1)
    tgt_delta[0, 0] = 1e-4  # one active growth node
    loss = dual_head_step_loss(spatial_logits, magnitude, tgt_delta, mask)
    assert loss is not None
    loss.backward()
    assert spatial_logits.grad is not None
    # inactive nodes: positive grad on logits -> SGD pushes gate prob toward 0
    assert float(spatial_logits.grad[1].item()) > 0.0
    # active node grad should differ from inactive (growth supervision path)
    assert float(spatial_logits.grad[0].item()) != float(spatial_logits.grad[1].item())
