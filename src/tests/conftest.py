"""Pytest suite selection helpers for kinematics vs biochem runs."""

from __future__ import annotations

from pathlib import Path

import pytest


# Biochem-focused modules that are not required for kinematics-only validation.
BIOCHEM_ONLY_FILES = {
    "test_biochem_physics.py",
    "test_transport_pde.py",
    "test_rheology_feedback.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--suite",
        action="store",
        default="all",
        choices=("all", "kinematics", "biochem"),
        help=(
            "Select test suite: 'kinematics' skips biochem-only tests; "
            "'biochem' runs full coverage including kinematics."
        ),
    )


def _is_biochem_item(item: pytest.Item) -> bool:
    path_name = Path(str(getattr(item, "fspath", ""))).name
    nodeid = item.nodeid.lower()
    test_name = item.name.lower()
    return (
        path_name in BIOCHEM_ONLY_FILES
        or "biochem" in nodeid
        or "phase3" in test_name
        or "tier3" in test_name
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    suite = config.getoption("--suite")
    if suite == "all":
        return

    skip_biochem = pytest.mark.skip(reason="Excluded from kinematics suite (--suite=kinematics).")

    for item in items:
        is_biochem = _is_biochem_item(item)
        if is_biochem:
            item.add_marker("biochem")
        else:
            item.add_marker("kinematics")

        if suite == "kinematics" and is_biochem:
            item.add_marker(skip_biochem)


# --- Test isolation for the process-wide typed-config binding -------------------------------
# `src/biochem_gnn/config.py::_bind_typed_configs` binds PushforwardConfig / BiochemRuntimeConfig
# process-wide via module-global contextvar tokens, deliberately: deploy and eval scripts want a
# single active config for the life of the process. In a pytest session that persistence leaks
# across tests. Any test that calls apply_train_recipe_env / apply_mat_growth_leg_env leaves a
# config bound, and every `*_enabled()` helper checks `resolve_config()` BEFORE falling back to
# os.environ -- so later tests that set env vars are silently ignored.
#
# Concretely this made 8 tests in test_species_flow_feats.py fail in aggregate runs while passing
# in isolation (test_mat_growth_simple_scope / test_runtime_config / test_seed_aux_loss all sort
# before it). A suite that only fails in aggregate hides real regressions, which matters now that
# the shared eval path is being edited.
#
# This restores the binding to whatever it was before each test. Production behaviour is
# untouched -- the fixture only runs under pytest.
@pytest.fixture(autouse=True)
def _isolate_active_typed_configs():
    from src.architecture.pushforward_config import _ACTIVE_CONFIG
    from src.architecture.runtime_config import _ACTIVE_RUNTIME
    from src.biochem_gnn import config as _bio_config

    pf_before = _ACTIVE_CONFIG.get()
    rt_before = _ACTIVE_RUNTIME.get()
    pf_tok_before = _bio_config._ACTIVE_PF_TOKEN
    rt_tok_before = _bio_config._ACTIVE_RT_TOKEN
    try:
        yield
    finally:
        _ACTIVE_CONFIG.set(pf_before)
        _ACTIVE_RUNTIME.set(rt_before)
        _bio_config._ACTIVE_PF_TOKEN = pf_tok_before
        _bio_config._ACTIVE_RT_TOKEN = rt_tok_before
