"""Unified CLI for runnable scripts across the repository.

Examples:
    python -m src.bin.main train t1
    python -m src.bin.main train t3 -- --epochs 10
    python -m src.bin.main data tier12
    python -m src.bin.main eval benchmark
    python -m src.bin.main inspect graph
    python -m src.bin.main orchestrate all
"""

from __future__ import annotations

import argparse
import runpy
import sys


MODULE_MAP: dict[tuple[str, str], str] = {
    ("train", "t1"): "src.training.train_t1",
    ("train", "t2"): "src.training.train_t2",
    ("train", "t3"): "src.training.train_t3_corrector",
    ("train", "explore"): "src.training.t1_explorer",
    ("data", "tier12"): "src.data_gen.pipeline_tier12",
    ("data", "tier3"): "src.data_gen.pipeline_tier3",
    ("eval", "benchmark"): "src.evaluation.run_benchmark",
    ("eval", "visualize"): "src.evaluation.visualize_tiers",
    ("inspect", "anchor"): "src.tools.inspect_phase1_data",
    ("inspect", "graph"): "src.tools.inspect_graph_sample",
    ("inspect", "phase1"): "src.tools.inspect_phase1_data",
    ("inspect", "tier3"): "src.tools.inspect_tier3_data",
    ("inspect", "comsol"): "src.tools.inspect_comsol_model",
    ("inspect", "deq"): "src.tools.verify_deq_convergence",
    ("orchestrate", "a"): "src.bin.orchestrate",
    ("orchestrate", "b"): "src.bin.orchestrate",
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

    if ns.group == "orchestrate":
        _run_module(module, [ns.target, *forwarded_args])
    else:
        _run_module(module, forwarded_args)


if __name__ == "__main__":
    main(sys.argv[1:])
