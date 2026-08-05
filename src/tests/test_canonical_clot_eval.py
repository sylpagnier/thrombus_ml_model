"""Canonical deploy clot protocol must preserve active PushforwardConfig overrides."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.architecture.pushforward_config import (
    PushforwardConfig,
    get_active_config,
    use_pushforward_config,
)
from src.architecture.runtime_config import BiochemRuntimeConfig, use_biochem_runtime
from src.evaluation.canonical_clot_eval import (
    bind_canonical_deploy_protocol,
    canonical_deploy_clot_metrics,
)
from src.evaluation.seed_growth_diagnostics import (
    classify_seed_growth_mode,
    seed_growth_diagnostic_panel,
)


def test_bind_canonical_deploy_preserves_sparse_commit_overrides():
    """Fresh build_deploy_configs defaults must not wipe CLI / recipe gate knobs."""
    pf = PushforwardConfig(
        frontier_hops=2,
        nucleation_topk=0.05,
        gate_temp=0.8,
        mat_commit_thresh=1.5e-5,
        flow_feats_source="auto",
    )
    rt = BiochemRuntimeConfig()
    with use_pushforward_config(pf), use_biochem_runtime(rt):
        bound_pf, _bound_rt = bind_canonical_deploy_protocol(flow="kinematics")
        assert bound_pf.frontier_hops == 2
        assert bound_pf.nucleation_topk == 0.05
        assert bound_pf.gate_temp == 0.8
        assert bound_pf.mat_commit_thresh == 1.5e-5
        active = get_active_config()
        assert active is not None
        assert active.frontier_hops == 2
        assert active.nucleation_topk == 0.05


def test_canonical_deploy_clot_metrics_sees_active_overrides(monkeypatch):
    """Regression: clot path used to rebuild PushforwardConfig and ignore sparse gates."""
    seen: dict[str, float | int] = {}

    def _fake_eval_deploy_clot_f1(*_a, **_k):
        cfg = get_active_config()
        assert cfg is not None
        seen["frontier_hops"] = int(cfg.frontier_hops)
        seen["nucleation_topk"] = float(cfg.nucleation_topk)
        seen["gate_temp"] = float(cfg.gate_temp)
        seen["mat_commit_thresh"] = float(cfg.mat_commit_thresh)
        return {
            "deploy_clot_f1": 0.12,
            "deploy_clot_score": 0.11,
            "deploy_clot_mass_ratio": 1.0,
        }

    monkeypatch.setattr(
        "src.core_physics.species_pushforward_continuous.eval_deploy_clot_f1",
        _fake_eval_deploy_clot_f1,
    )
    monkeypatch.setattr(
        "src.core_physics.species_deploy_rollout.reset_species_rollout_flow_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        "src.biochem_gnn.config.build_deploy_configs",
        lambda *a, **k: (
            PushforwardConfig(frontier_hops=0, nucleation_topk=0.0, gate_temp=1.0),
            BiochemRuntimeConfig(),
            {},
        ),
    )

    pf = PushforwardConfig(
        frontier_hops=2,
        nucleation_topk=0.05,
        gate_temp=0.7,
        mat_commit_thresh=2.0e-5,
    )
    data = SimpleNamespace(clone=lambda: SimpleNamespace())
    with use_pushforward_config(pf), use_biochem_runtime(BiochemRuntimeConfig()):
        out = canonical_deploy_clot_metrics(
            model=torch.nn.Identity(),
            data=data,
            static={},
            phys_cfg=None,
            bio_cfg=None,
            device=torch.device("cpu"),
            flow_source="kinematics",
        )
        # Caller config restored after the protocol.
        restored = get_active_config()
        assert restored is not None
        assert restored.frontier_hops == 2
        assert restored.nucleation_topk == 0.05

    assert seen["frontier_hops"] == 2
    assert seen["nucleation_topk"] == 0.05
    assert seen["gate_temp"] == 0.7
    assert seen["mat_commit_thresh"] == 2.0e-5
    assert out["deploy_clot_f1"] == 0.12
    assert out["canonical_protocol"] == 1.0


def test_seed_growth_diagnostic_panel_modes():
    under = seed_growth_diagnostic_panel(
        {
            "mat_seed_prec": 0.0,
            "mat_seed_count": 1.0,
            "mat_front_speed_ratio": 0.2,
            "mat_overpaint_frac": 0.01,
            "deploy_clot_mass_ratio": 0.9,
            "deploy_clot_offwall_strict_f1": 0.0,
            "deploy_clot_fn": 60.0,
            "deploy_clot_fp": 10.0,
            "deploy_clot_f1": 0.25,
        }
    )
    assert under["mode"] == "underseed"
    assert "seed" in under["hint"].lower() or "front" in under["hint"].lower()

    over = classify_seed_growth_mode(
        {
            "mat_seed_prec": 0.4,
            "mat_seed_count": 20.0,
            "mat_front_speed_ratio": 1.2,
            "mat_overpaint_frac": 0.2,
            "deploy_clot_mass_ratio": 2.5,
            "deploy_clot_offwall_strict_f1": 0.3,
            "deploy_clot_fn": 5.0,
            "deploy_clot_fp": 80.0,
        }
    )
    assert over == "overspray"
