"""Unified CLI for runnable scripts across the repository.

Examples:
    python -m src.bin.main train kinematics
    python -m src.bin.main train biochem -- --epochs 10
    python -m src.bin.main data kinematics
    python -m src.bin.main eval benchmark
    python -m src.bin.main inspect graph
    python -m src.bin.main orchestrate all
    python -m src.bin.main orchestrate biochem
"""

from __future__ import annotations

import argparse
import runpy
import sys


MODULE_MAP: dict[tuple[str, str], str] = {
    ("train", "kinematics"): "src.training.train_kinematics_predictor",
    ("train", "biochem"): "src.training.train_biochem_corrector",
    # Backward-compatible aliases; all map to unified kinematics trainer.
    ("train", "t1"): "src.training.train_kinematics_predictor",
    ("train", "t2"): "src.training.train_kinematics_predictor",
    ("train", "t3"): "src.training.train_biochem_corrector",
    ("train", "explore"): "src.training.train_kinematics_predictor",
    ("data", "kinematics"): "src.data_gen.pipeline_kinematics",
    ("data", "biochem"): "src.data_gen.pipeline_biochem",
    ("eval", "benchmark"): "src.evaluation.run_benchmark",
    ("eval", "visualize"): "src.evaluation.visualize_pipeline",
    ("inspect", "anchor"): "src.tools.inspect_kinematics_data",
    ("inspect", "graph"): "src.tools.inspect_graph_sample",
    ("inspect", "kinematics"): "src.tools.inspect_kinematics_data",
    ("inspect", "biochem"): "src.tools.inspect_biochem_data",
    ("inspect", "comsol"): "src.tools.inspect_comsol_model",
    ("inspect", "deq"): "src.tools.verify_deq_convergence",
    ("orchestrate", "kinematics"): "src.bin.orchestrate",
    ("orchestrate", "biochem"): "src.bin.orchestrate",
    ("orchestrate", "all"): "src.bin.orchestrate",
}


def _run_module(module_name: str, forwarded_args: list[str]) -> None:
    sys.argv = [module_name, *forwarded_args]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Central runnable-entrypoint router. Keeps scripts discoverable in one "
            "place while preserving existing module structure."
        )
    )
    parser.add_argument(
        "group",
        choices=("train", "data", "eval", "inspect", "orchestrate"),
        help="High-level command group.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Sub-command in the selected group (required unless group=orchestrate).",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the selected script. Prefix with --.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)

    forwarded_args = list(ns.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    if not ns.target:
        parser.error(f"{ns.group} requires a target")

    key = (ns.group, ns.target)
    module = MODULE_MAP.get(key)
    if module is None:
        valid_targets = sorted(k[1] for k in MODULE_MAP if k[0] == ns.group)
        parser.error(f"unknown target '{ns.target}' for group '{ns.group}'. valid: {valid_targets}")

    if ns.group == "train" and ns.target in {"t1", "t2", "t3", "explore"}:
        canonical = "biochem" if ns.target == "t3" else "kinematics"
        print(
            f"⚠️ Deprecated train target '{ns.target}'. "
            f"Use 'train {canonical}' instead."
        )

    if ns.group == "orchestrate":
        _run_module(module, [ns.target, *forwarded_args])
    else:
        _run_module(module, forwarded_args)


if __name__ == "__main__":
    main(sys.argv[1:])
