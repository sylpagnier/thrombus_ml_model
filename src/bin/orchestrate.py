"""Stage orchestrator: Stage A (Tier 1/2) and Stage B (Tier 3)."""

from __future__ import annotations

import argparse
import runpy
import sys


def _run_module(module_name: str) -> None:
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run Stage A and/or Stage B training.")
    p.add_argument(
        "stage",
        choices=("a", "b", "all"),
        help=(
            "Stage A: Tier 1 + Tier 2 predictor warm-up. "
            "Stage B: Tier 3 corrector. 'all' runs A then B."
        ),
    )
    p.add_argument(
        "--skip-tier1",
        action="store_true",
        help="With stage 'a' or 'all', skip Tier 1 and start at Tier 2.",
    )
    args = p.parse_args(argv)

    if args.stage in ("a", "all") and not args.skip_tier1:
        print("=== Stage A: Tier 1 ===")
        _run_module("src.training.train_t1_predictor")
    if args.stage in ("a", "all"):
        print("=== Stage A: Tier 2 ===")
        _run_module("src.training.train_t2_predictor")
    if args.stage in ("b", "all"):
        print("=== Stage B: Tier 3 (corrector) ===")
        _run_module("src.training.train_t3_corrector")


if __name__ == "__main__":
    main(sys.argv[1:])
