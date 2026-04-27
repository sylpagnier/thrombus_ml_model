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
