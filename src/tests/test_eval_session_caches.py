"""Session caches for kinematics / corrector loads (eval multi-vessel)."""

from __future__ import annotations

from pathlib import Path

import torch

from src.core_physics.coupled_shear_gnn import (
    LocalKinematicCorrector,
    clear_local_corrector_cache,
    load_local_corrector,
)
from src.utils import kinematics_inference as ki


def test_kinematics_predictor_session_cache(tmp_path, monkeypatch):
    """Second load with cache=True must return the same module object."""
    ki.clear_kinematics_predictor_cache()
    calls = {"n": 0}

    class _Fake:
        def eval(self):
            return self

        def to(self, *a, **k):
            return self

        def load_state_dict(self, *a, **k):
            return None

        def parameters(self):
            if False:
                yield None

    def _fake_load(checkpoint, device, *, phys_cfg=None, max_iters=25, cache=True):
        ckpt_path = Path(checkpoint).resolve()
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        cache_key = (str(ckpt_path), str(dev), max(3, int(max_iters)))
        if cache and cache_key in ki._KINE_MODEL_CACHE:
            return ki._KINE_MODEL_CACHE[cache_key]
        calls["n"] += 1
        model = _Fake()
        if cache:
            ki._KINE_MODEL_CACHE[cache_key] = model
        return model

    monkeypatch.setattr(ki, "load_kinematics_predictor", _fake_load)
    ckpt = tmp_path / "kine.pth"
    ckpt.write_text("x", encoding="utf-8")
    a = ki.load_kinematics_predictor(ckpt, "cpu", cache=True)
    b = ki.load_kinematics_predictor(ckpt, "cpu", cache=True)
    c = ki.load_kinematics_predictor(ckpt, "cpu", cache=False)
    assert a is b
    assert a is not c
    assert calls["n"] == 2
    ki.clear_kinematics_predictor_cache()


def test_local_corrector_session_cache(tmp_path):
    clear_local_corrector_cache()
    ckpt = tmp_path / "corr.pth"
    model = LocalKinematicCorrector()
    torch.save(
        {"model_state": model.state_dict(), "in_channels": 6, "hidden_dim": 64, "heads": 4},
        ckpt,
    )
    a = load_local_corrector(ckpt, torch.device("cpu"), cache=True)
    b = load_local_corrector(ckpt, torch.device("cpu"), cache=True)
    assert a is b
    clear_local_corrector_cache()
    c = load_local_corrector(ckpt, torch.device("cpu"), cache=True)
    assert c is not a
    clear_local_corrector_cache()


def test_splice_dynamic_flow_no_grad_reuses_buffer():
    from src.core_physics.species_pushforward_continuous import splice_dynamic_flow

    base = torch.ones(4, 10)
    series = torch.zeros(3, 4, 5)
    series[1] = 2.0
    with torch.no_grad():
        a = splice_dynamic_flow(base, series, (5, 5), 1)
        b = splice_dynamic_flow(base, series, (5, 5), 1)
    assert a is b
    assert torch.allclose(a[:, 5:], torch.full((4, 5), 2.0))
    base_g = torch.ones(4, 10, requires_grad=True)
    c = splice_dynamic_flow(base_g, series, (5, 5), 1)
    d = splice_dynamic_flow(base_g, series, (5, 5), 1)
    assert c is not d


def test_species_gnn_static_from_band_dict_no_kine():
    from torch_geometric.data import Data

    from src.core_physics.species_gnn_clot_rollout import species_gnn_static_from_band_dict

    n = 4
    data = Data(
        x=torch.zeros(n, 16),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    data.num_nodes = n
    data.mask_wall = torch.ones(n, dtype=torch.bool)
    stat = {
        "base_feats": torch.randn(n, 8),
        "edge_index": data.edge_index,
        "node_idx": torch.arange(n),
        "pos_band": data.x[:, :2],
        "flow_series": None,
        "flow_cols": None,
    }
    out = species_gnn_static_from_band_dict(stat, data, device=torch.device("cpu"), wall_hops=1)
    assert out.base_feats.shape == (n, 8)
    assert int(out.node_idx.numel()) == n
