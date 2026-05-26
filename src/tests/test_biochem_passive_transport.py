"""Step 2 passive biochemistry: 1-way transport (ADR + species, frozen flow, mu_ratio=1)."""
from __future__ import annotations

import os

import pytest
import torch

from src.utils.channel_schema import BIO_Y_SCHEMA, Y_SCHEMAS


def test_passive_transport_preset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_preset_passive_transport_if_requested

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_PRESET", "passive_transport")

    _apply_biochem_preset_passive_transport_if_requested()

    assert os.environ["BIOCHEM_LOSS_ISOLATE"] == "PASSIVE"
    assert float(os.environ["BIOCHEM_TEACHER_MU_RATIO_MAX"]) == pytest.approx(1.0)
    assert os.environ["BIOCHEM_TEACHER_FORCE_MIN"] == "1.0"
    assert os.environ["BIOCHEM_MU_DISABLE_EXPLICIT_GELATION"] == "1"
    assert os.environ["BIOCHEM_TRAIN_MU_ENCODER"] == "0"
    assert os.environ["BIOCHEM_TRAIN_KIN_LORA"] == "0"
    assert os.environ["BIOCHEM_TRAIN_BIO_DECODER"] == "1"
    assert float(os.environ["BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT"]) == pytest.approx(0.0)
    assert os.environ["BIOCHEM_STOP_AFTER_TEACHER"] == "1"
    assert os.environ["BIOCHEM_GT_KINE_VEL"] == "1"
    assert os.environ["BIOCHEM_GT_KINE_SKIP_DEQ"] == "1"
    assert os.environ["BIOCHEM_DETACH_MACRO_STATE"] == "0"
    assert os.environ["BIOCHEM_PASSIVE_ADR_BACKPROP"] == "0"
    assert float(os.environ["BIOCHEM_TEACHER_LR"]) == pytest.approx(5e-4)
    assert os.environ["BIOCHEM_TEACHER_GRAD_SCALE_ON_CAP"] == "1"
    assert float(os.environ["BIOCHEM_PASSIVE_DATA_KINE_WEIGHT"]) == pytest.approx(0.0)


@pytest.mark.parametrize("alias", ("one_way", "step2_passive"))
def test_passive_transport_preset_aliases(monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_preset_passive_transport_if_requested

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_PRESET", alias)
    monkeypatch.delenv("BIOCHEM_LOSS_ISOLATE", raising=False)

    _apply_biochem_preset_passive_transport_if_requested()

    assert os.environ["BIOCHEM_LOSS_ISOLATE"] == "PASSIVE"


def test_passive_transport_preset_skipped_when_stock_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_preset_passive_transport_if_requested

    monkeypatch.setenv("BIOCHEM_STOCK_DEFAULTS", "1")
    monkeypatch.setenv("BIOCHEM_PRESET", "passive_transport")
    monkeypatch.delenv("BIOCHEM_LOSS_ISOLATE", raising=False)

    _apply_biochem_preset_passive_transport_if_requested()

    assert "BIOCHEM_LOSS_ISOLATE" not in os.environ


def test_teacher_mu_ratio_max_respects_passive_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import BiochemConfig
    from src.training.train_biochem_corrector import _biochem_teacher_mu_ratio_max

    monkeypatch.setenv("BIOCHEM_TEACHER_MU_RATIO_MAX", "1.0")
    bio_cfg = BiochemConfig(phase="biochem")
    bio_cfg.mu_ratio_max = 80.0
    assert _biochem_teacher_mu_ratio_max(bio_cfg) == pytest.approx(1.0)


def test_passive_loss_isolate_data_bio_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _biochem_resolve_isolated_loss

    monkeypatch.setenv("BIOCHEM_PASSIVE_ADR_BACKPROP", "0")
    monkeypatch.setenv("BIOCHEM_PASSIVE_DATA_KINE_WEIGHT", "0.0")
    monkeypatch.setenv("BIOCHEM_PASSIVE_DATA_BIO_WEIGHT", "2.0")

    def _t(v: float) -> torch.Tensor:
        return torch.tensor(v, dtype=torch.float32)

    loss = _biochem_resolve_isolated_loss(
        "PASSIVE",
        pred_final=_t(0.0),
        l_adr_fast=_t(1.0),
        l_adr_slow=_t(2.0),
        l_wall_bio=_t(0.0),
        l_wall_phys=_t(0.0),
        l_bio_io=_t(0.0),
        l_mom=_t(0.0),
        l_data_kine=_t(4.0),
        l_data_bio=_t(8.0),
        l_pseudo=_t(0.0),
        pseudo_loss_weight=0.0,
        l_latent_reg=_t(0.0),
        latent_scale=0.0,
        l_visc_reg=_t(0.0),
        visc_reg_w=0.0,
        l_kine_prior=_t(0.0),
        w_kp=0.0,
        l_phys_temp=_t(0.0),
        w_pt=0.0,
        l_mu_si_anchor=_t(0.0),
        w_mu_aux=0.0,
        l_mu_log_anchor=_t(0.0),
        w_mu_log=0.0,
        l_mu_log_wall=_t(0.0),
        w_mu_log_wall=0.0,
        l_mu_log_high=_t(0.0),
        w_mu_log_high=0.0,
        l_mu_mse_anchor=_t(0.0),
        l_mu_wall_bypass=_t(0.0),
        w_mu_wall_bypass=0.0,
        l_mu_log_adjacent=_t(0.0),
        l_k10e_bulk_delta=_t(0.0),
        l_fi_gate_start=_t(0.0),
        w_fi_gate_start_eff=0.0,
        l_residual_sparse=_t(0.0),
        lambda_residual_sparse=0.0,
    )
    # data-only backward: 2*8 = 16 (ADR terms excluded by default)
    assert float(loss.item()) == pytest.approx(16.0)


def test_passive_loss_includes_adr_when_backprop_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _biochem_resolve_isolated_loss

    monkeypatch.setenv("BIOCHEM_PASSIVE_ADR_BACKPROP", "1")
    monkeypatch.setenv("BIOCHEM_PASSIVE_DATA_KINE_WEIGHT", "0.0")
    monkeypatch.setenv("BIOCHEM_PASSIVE_DATA_BIO_WEIGHT", "1.0")

    loss = _biochem_resolve_isolated_loss(
        "PASSIVE",
        pred_final=torch.tensor(0.0),
        l_adr_fast=torch.tensor(1.0),
        l_adr_slow=torch.tensor(2.0),
        l_wall_bio=torch.tensor(0.0),
        l_wall_phys=torch.tensor(0.0),
        l_bio_io=torch.tensor(0.0),
        l_mom=torch.tensor(0.0),
        l_data_kine=torch.tensor(0.0),
        l_data_bio=torch.tensor(8.0),
        l_pseudo=torch.tensor(0.0),
        pseudo_loss_weight=0.0,
        l_latent_reg=torch.tensor(0.0),
        latent_scale=0.0,
        l_visc_reg=torch.tensor(0.0),
        visc_reg_w=0.0,
        l_kine_prior=torch.tensor(0.0),
        w_kp=0.0,
        l_phys_temp=torch.tensor(0.0),
        w_pt=0.0,
        l_mu_si_anchor=torch.tensor(0.0),
        w_mu_aux=0.0,
        l_mu_log_anchor=torch.tensor(0.0),
        w_mu_log=0.0,
        l_mu_log_wall=torch.tensor(0.0),
        w_mu_log_wall=0.0,
        l_mu_log_high=torch.tensor(0.0),
        w_mu_log_high=0.0,
        l_mu_mse_anchor=torch.tensor(0.0),
        l_mu_wall_bypass=torch.tensor(0.0),
        w_mu_wall_bypass=0.0,
        l_mu_log_adjacent=torch.tensor(0.0),
        l_k10e_bulk_delta=torch.tensor(0.0),
        l_fi_gate_start=torch.tensor(0.0),
        w_fi_gate_start_eff=0.0,
        l_residual_sparse=torch.tensor(0.0),
        lambda_residual_sparse=0.0,
    )
    assert float(loss.item()) == pytest.approx(11.0)


def test_passive_loss_one_way_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _biochem_resolve_isolated_loss

    monkeypatch.setenv("BIOCHEM_PASSIVE_ADR_BACKPROP", "1")
    loss = _biochem_resolve_isolated_loss(
        "ONE_WAY",
        pred_final=torch.tensor(0.0),
        l_adr_fast=torch.tensor(3.0),
        l_adr_slow=torch.tensor(0.0),
        l_wall_bio=torch.tensor(0.0),
        l_wall_phys=torch.tensor(0.0),
        l_bio_io=torch.tensor(0.0),
        l_mom=torch.tensor(0.0),
        l_data_kine=torch.tensor(0.0),
        l_data_bio=torch.tensor(0.0),
        l_pseudo=torch.tensor(0.0),
        pseudo_loss_weight=0.0,
        l_latent_reg=torch.tensor(0.0),
        latent_scale=0.0,
        l_visc_reg=torch.tensor(0.0),
        visc_reg_w=0.0,
        l_kine_prior=torch.tensor(0.0),
        w_kp=0.0,
        l_phys_temp=torch.tensor(0.0),
        w_pt=0.0,
        l_mu_si_anchor=torch.tensor(0.0),
        w_mu_aux=0.0,
        l_mu_log_anchor=torch.tensor(0.0),
        w_mu_log=0.0,
        l_mu_log_wall=torch.tensor(0.0),
        w_mu_log_wall=0.0,
        l_mu_log_high=torch.tensor(0.0),
        w_mu_log_high=0.0,
        l_mu_mse_anchor=torch.tensor(0.0),
        l_mu_wall_bypass=torch.tensor(0.0),
        w_mu_wall_bypass=0.0,
        l_mu_log_adjacent=torch.tensor(0.0),
        l_k10e_bulk_delta=torch.tensor(0.0),
        l_fi_gate_start=torch.tensor(0.0),
        w_fi_gate_start_eff=0.0,
        l_residual_sparse=torch.tensor(0.0),
        lambda_residual_sparse=0.0,
    )
    assert float(loss.item()) == pytest.approx(3.0)


def test_data_bio_supervision_includes_fi_and_mat_channels() -> None:
    schema = Y_SCHEMAS[BIO_Y_SCHEMA]
    names = schema.channels
    assert names[12] == "FI_log1p_nd"
    assert names[15] == "Mat_log1p_nd"
    fi_idx = names.index("FI_log1p_nd") - 4
    mat_idx = names.index("Mat_log1p_nd") - 4
    assert fi_idx == 8
    assert mat_idx == 11


def test_resolve_gt_kine_uvp_blends_truth_nodes() -> None:
    from types import SimpleNamespace

    from src.architecture.gnode_biochem import resolve_gt_kine_uvp_at_step

    n = 4
    y = torch.zeros(2, n, 16)
    y[0, :, 0] = 1.0
    y[0, :, 1] = 2.0
    y[0, :, 2] = 3.0
    y[1, :, 0] = 10.0
    batch = SimpleNamespace(y=y)
    truth = torch.tensor([True, True, False, False])
    fallback = torch.zeros(n, 3)
    out = resolve_gt_kine_uvp_at_step(
        batch, None, 0, truth, torch.device("cpu"), torch.float32, fallback_uvp=fallback
    )
    assert out is not None
    assert float(out[0, 0]) == pytest.approx(1.0)
    assert float(out[2, 0]) == pytest.approx(0.0)


def test_explicit_gelation_off_under_passive_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.architecture import gnode_biochem as gb

    monkeypatch.setenv("BIOCHEM_MU_DISABLE_EXPLICIT_GELATION", "1")
    from unittest.mock import MagicMock

    model = MagicMock()
    model.mu1_sigmoid = lambda x: torch.ones_like(x) * 50.0
    model.mu2_sigmoid = lambda x: torch.ones_like(x) * 50.0
    fi = torch.tensor([[1e6]])
    mat = torch.tensor([[1e6]])
    mu1, mu2 = gb.biochem_explicit_gelation_terms(model, fi, mat)
    assert float(mu1.sum()) == 0.0
    assert float(mu2.sum()) == 0.0
