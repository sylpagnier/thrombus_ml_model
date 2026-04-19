"""Regression tests for CLI routing and orchestration order."""

from __future__ import annotations

import src.bin.main as cli_main
import src.bin.orchestrate as orchestrate


def test_bin_main_routes_orchestrate_stage_and_args(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str, forwarded_args: list[str]) -> None:
        calls.append((module_name, forwarded_args))

    monkeypatch.setattr(cli_main, "_run_module", _fake_run_module)
    cli_main.main(["orchestrate", "all", "--", "--skip-tier1"])

    assert calls == [("src.bin.orchestrate", ["all", "--skip-tier1"])]


def test_bin_main_routes_train_targets_to_predictor_modules(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str, forwarded_args: list[str]) -> None:
        calls.append((module_name, forwarded_args))

    monkeypatch.setattr(cli_main, "_run_module", _fake_run_module)
    cli_main.main(["train", "t1"])
    cli_main.main(["train", "t2", "--", "--epochs", "5"])

    assert calls[0] == ("src.training.train_t1_predictor", [])
    assert calls[1] == ("src.training.train_t2_predictor", ["--epochs", "5"])


def test_orchestrate_all_runs_t1_t2_t3_in_order(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str) -> None:
        calls.append(module_name)

    monkeypatch.setattr(orchestrate, "_run_module", _fake_run_module)
    orchestrate.main(["all"])

    assert calls == [
        "src.training.train_t1_predictor",
        "src.training.train_t2_predictor",
        "src.training.train_t3_corrector",
    ]


def test_orchestrate_stage_a_skip_tier1_runs_only_t2(monkeypatch):
    calls = []

    def _fake_run_module(module_name: str) -> None:
        calls.append(module_name)

    monkeypatch.setattr(orchestrate, "_run_module", _fake_run_module)
    orchestrate.main(["a", "--skip-tier1"])

    assert calls == ["src.training.train_t2_predictor"]
