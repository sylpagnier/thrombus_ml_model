"""Red-console trivial-flow warnings in biochem teacher training."""
from __future__ import annotations

import pytest


def test_trivial_flow_warn_prints_red_when_above_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.training.train_biochem_corrector import _biochem_warn_trivial_flow_if_needed

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("BIOCHEM_NO_COLOR", "0")
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("BIOCHEM_FLOW_TRIVIAL_WARN_THRESHOLD", "0.9")

    _biochem_warn_trivial_flow_if_needed(
        0.95,
        stage="teacher",
        epoch=3,
        batch_idx=1,
        metrics={
            "DBG_Q_pred_inlet": 1e-6,
            "DBG_Q_ref_re": 1.0,
            "DBG_Q_flow_ratio": 0.01,
        },
    )
    out = capsys.readouterr().out
    assert "TRIVIAL FLOW" in out
    assert "\033[91m" in out
    assert "flow_trivial_score=0.950" in out


def test_trivial_flow_warn_silent_below_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.training.train_biochem_corrector import _biochem_warn_trivial_flow_if_needed

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("BIOCHEM_FLOW_TRIVIAL_WARN_THRESHOLD", "0.9")

    _biochem_warn_trivial_flow_if_needed(0.12, stage="teacher", epoch=0)
    assert capsys.readouterr().out == ""
