"""Training orchestrator: kinematics and biochem phases."""

from __future__ import annotations

import argparse
import runpy
import sys


def _run_module(module_name: str) -> None:
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run kinematics and/or biochem training.")
    p.add_argument(
        "phase",
        choices=("kinematics", "biochem", "all"),
        help=(
            "kinematics: unified kinematics pretraining. "
            "biochem: biochem deploy (GraphSAGE). 'all' runs kinematics then biochem."
        ),
    )
    args = p.parse_args(argv)
    phase = args.phase

    if phase in ("kinematics", "all"):
        print("=== Kinematics training ===")
        _run_module("src.training.train_kinematics_predictor")
    if phase in ("biochem", "all"):
        print("=== Biochem training ===")
        _run_module("src.training.train_biochem_gnn")


if __name__ == "__main__":
    main(sys.argv[1:])
