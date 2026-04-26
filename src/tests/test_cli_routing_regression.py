"""Regression tests for CLI routing and orchestration order."""

from __future__ import annotations

import pytest

import src.bin.main as cli_main
import src.bin.orchestrate as orchestrate


def test_bin_main_routes_orchestrate_phase_and_args(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str, forwarded_args: list[str]) -> None:
        calls.append((module_name, forwarded_args))

    monkeypatch.setattr(cli_main, "_run_module", _fake_run_module)
    cli_main.main(["orchestrate", "all"])

    assert calls == [("src.bin.orchestrate", ["all"])]


def test_bin_main_routes_train_targets_to_predictor_modules(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str, forwarded_args: list[str]) -> None:
        calls.append((module_name, forwarded_args))

    monkeypatch.setattr(cli_main, "_run_module", _fake_run_module)
    cli_main.main(["train", "t1"])
    cli_main.main(["train", "t2", "--", "--epochs", "5"])
    cli_main.main(["train", "kinematics"])

    assert calls[0] == ("src.training.train_kinematics_predictor", [])
    assert calls[1] == ("src.training.train_kinematics_predictor", ["--epochs", "5"])
    assert calls[2] == ("src.training.train_kinematics_predictor", [])


def test_orchestrate_all_runs_kinematics_then_biochem_in_order(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str) -> None:
        calls.append(module_name)

    monkeypatch.setattr(orchestrate, "_run_module", _fake_run_module)
    orchestrate.main(["all"])

    assert calls == [
        "src.training.train_kinematics_predictor",
        "src.training.train_biochem_corrector",
    ]


def test_orchestrate_kinematics_runs_unified_kinematics(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str) -> None:
        calls.append(module_name)

    monkeypatch.setattr(orchestrate, "_run_module", _fake_run_module)
    orchestrate.main(["kinematics"])

    assert calls == ["src.training.train_kinematics_predictor"]


def test_orchestrate_rejects_legacy_a_b_aliases():
    with pytest.raises(SystemExit):
        orchestrate.main(["a"])
    with pytest.raises(SystemExit):
        orchestrate.main(["b"])
