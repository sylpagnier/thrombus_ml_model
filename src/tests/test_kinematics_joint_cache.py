"""Joint GINO-DEQ UV+latent cache (one Anderson solve per vessel)."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.utils import kinematics_inference as ki


class _FakeGino:
    """Minimal stand-in that counts DEQ solves and returns fixed UV/latent."""

    def __init__(self, n: int, latent_dim: int = 4):
        self.n = n
        self.latent_dim = latent_dim
        self.n_solves = 0
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._p

    def predict_uv_and_latent(self, data, **kwargs):
        self.n_solves += 1
        n = int(data.num_nodes)
        pred = torch.arange(n * 5, dtype=torch.float32).reshape(n, 5)
        z = torch.ones(n, self.latent_dim)
        return pred, z


def _tiny_graph(n: int = 3) -> Data:
    x = torch.zeros(n, 4)
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=ei)
    data.num_nodes = n
    return data


def test_predict_kinematics_and_latent_one_solve_fills_both_caches():
    model = _FakeGino(3)
    data = _tiny_graph(3)
    pred1, z1 = ki.predict_kinematics_and_latent(model, data)
    assert model.n_solves == 1
    pred2 = ki.predict_kinematics(model, data)
    z2 = ki.predict_kinematics_latent(model, data)
    assert model.n_solves == 1
    assert torch.equal(pred1, pred2)
    assert torch.equal(z1, z2)


def test_predict_kinematics_then_latent_still_one_solve():
    model = _FakeGino(3)
    data = _tiny_graph(3)
    _ = ki.predict_kinematics(model, data)
    assert model.n_solves == 1
    _ = ki.predict_kinematics_latent(model, data)
    assert model.n_solves == 1


def test_clone_busts_cache_and_resolves_again():
    model = _FakeGino(3)
    data = _tiny_graph(3)
    _ = ki.predict_kinematics_and_latent(model, data)
    assert model.n_solves == 1
    _ = ki.predict_kinematics_and_latent(model, data.clone())
    assert model.n_solves == 2
