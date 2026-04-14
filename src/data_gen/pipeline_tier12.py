"""
Interactive Tier 1/2 data pipeline: vessel meshes, optional COMSOL anchors, PyG graphs.

Runs the same logical steps as ``vessel_generator``, ``anchor_generator``, and ``mesh_to_graph``.
**Interactive mode asks every question first** (per tier), then runs Gmsh / COMSOL / mesh-to-graph
with **no further prompts** so you can leave the machine unattended after the planning phase.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from dataclasses import dataclass
from typing import Optional

from src.data_gen.lib.mesh_to_graph import MeshToGraphComplete
from src.data_gen.lib.vessel_generator import (
    VesselGenerator,
    summarize_vessel_mesh_inventory,
    _prompt_int_choice as _vg_prompt_int_choice,
    _prompt_positive_int as _vg_prompt_positive_int,
    _prompt_write_mode_vessel as _vg_prompt_write_mode_vessel,
)


def _prompt_yes_no(label: str, default: bool = True) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{default_hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y or n.")


def _prompt_optional_int(label: str) -> Optional[int]:
    raw = input(f"{label} [empty = auto]: ").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        print("  Invalid integer; using auto.")
        return None


def _prompt_anchor_write_mode() -> bool:
    """Return True if overwriting existing .npz, False for add-only."""
    while True:
        raw = input("Anchor .npz write mode [1=add new only / 2=overwrite existing] [1]: ").strip()
        if raw in ("", "1"):
            return False
        if raw == "2":
            return True
        print("  Enter 1 or 2.")


def _prompt_nonnegative_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return int(default)
        try:
            v = int(raw)
            if v < 0:
                print("Enter a non-negative integer.")
                continue
            return v
        except ValueError:
            print("Invalid input. Enter an integer value.")


def _tier_str_from_n(tier_n: int) -> str:
    return f"tier{tier_n}"


@dataclass
class TierInteractivePlan:
    """All interactive choices for one tier (collected before any long-running step)."""

    run_vessel: bool
    level: Optional[int]
    overwrite: Optional[bool]
    n_vessels: Optional[int]
    seed: Optional[int]
    num_workers: Optional[int]
    chunk_size: Optional[int]
    no_plot: bool
    run_anchors: bool
    allow_overwrite_anchor: bool
    anchor_max_json_scan: Optional[int]
    anchor_shuffle: bool
    anchor_shuffle_seed: Optional[int]
    # When anchor_target == 0 and run_anchors: how many new CFD samples to aim for.
    anchor_manual_max_new: Optional[int]
    run_mesh: bool


def run_interactive_pipeline() -> None:
    print("\n=== Tier 1/2 data generation pipeline (vessel, anchors, graphs) ===\n")

    tier_scope = _vg_prompt_int_choice(
        "Run datagen for (1 = Tier 1 only, 2 = Tier 2 only, 3 = Tier 1 and Tier 2 sequentially)",
        (1, 2, 3),
    )
    tier_sequence = (1, 2) if tier_scope == 3 else (tier_scope,)

    if len(tier_sequence) == 2:
        print(
            "\nBoth tiers run one after the other (separate `data/raw/tier*`, CFD dirs, and graphs). "
            "Each tier gets an independent random cohort unless you set an explicit RNG seed.\n"
        )

    print(
        "\nEach COMSOL anchor needs meshes with .json + non-empty .nas + .msh. "
        "Set a target anchor count to size the cohort; use 0 to skip COMSOL anchors in this run "
        "(you can still enable them in a follow-up pass).\n"
    )
    if len(tier_sequence) == 2:
        print(
            "When running both tiers, you can set **different** anchor targets (and vessel defaults) "
            "for Tier 1 vs Tier 2.\n"
        )
        anchor_by_tier = {
            1: _prompt_nonnegative_int(
                "Tier 1 — how many COMSOL anchor CFD samples (.npz) should this run target?",
                0,
            ),
            2: _prompt_nonnegative_int(
                "Tier 2 — how many COMSOL anchor CFD samples (.npz) should this run target?",
                0,
            ),
        }
    else:
        only = tier_sequence[0]
        anchor_by_tier = {
            only: _prompt_nonnegative_int(
                "How many COMSOL anchor CFD samples (.npz) do you want this run to target?",
                0,
            ),
        }

    print(
        "\n--- Planning: answer all questions now; the run after this has **no further prompts** ---\n"
    )
    plans: dict[int, TierInteractivePlan] = {}
    for tier_n in tier_sequence:
        plans[tier_n] = _prompt_tier_interactive_plan(tier_n, anchor_by_tier[tier_n])

    print(
        f"\n{'=' * 60}\n"
        "  All prompts complete — unattended run starting.\n"
        f"{'=' * 60}\n"
    )

    for tier_n in tier_sequence:
        _execute_tier_interactive_plan(tier_n, anchor_by_tier[tier_n], plans[tier_n])

    print("\n=== Pipeline finished ===\n")


def _prompt_tier_interactive_plan(tier_n: int, anchor_target: int) -> TierInteractivePlan:
    tier = _tier_str_from_n(tier_n)
    print(f"\n{'=' * 60}\n  PLAN — {tier.upper()} (independent cohort)\n{'=' * 60}\n")

    run_vessel = _prompt_yes_no("Run vessel mesh generation (Gmsh)?", default=True)
    level: Optional[int] = None
    overwrite: Optional[bool] = None
    n_vessels: Optional[int] = None
    seed: Optional[int] = None
    num_workers: Optional[int] = None
    chunk_size: Optional[int] = None
    no_plot = True

    if run_vessel:
        level = _vg_prompt_int_choice("Geometry level", (0, 1))
        vg = VesselGenerator(tier=tier)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        n_on_disk = int(inv["count"])
        max_idx = int(inv["max_idx"])
        index_span = max_idx + 1 if max_idx >= 0 else 0
        unused_slots = index_span - n_on_disk if max_idx >= 0 else 0
        print("\n--- Vessel mesh inventory (now) ---")
        print(f"  Output: {vg.output_dir}")
        print(f"  Total index span: {index_span}")
        print(f"  Meshes on disk: {n_on_disk}")
        print(f"  Unused index slots: {unused_slots}\n")

        overwrite = _vg_prompt_write_mode_vessel()
        base_default = 50 if n_on_disk > 0 else 500
        if anchor_target > 0:
            default_n = max(base_default, anchor_target)
            print(
                f"  Default vessel count = {default_n} (at least your anchor target of {anchor_target}).\n"
            )
        else:
            default_n = base_default
        n_vessels = _vg_prompt_positive_int("How many vessels to generate", default_n)
        seed = _prompt_optional_int(
            "RNG seed (empty = random cohort — default; set an integer only to reproduce a run)"
        )
        num_workers = _prompt_optional_int("Worker processes")
        chunk_size = _prompt_optional_int("Chunk size (samples per worker chunk)")
        no_plot = _prompt_yes_no("Skip matplotlib preview of saved meshes?", default=True)

    if anchor_target > 0:
        run_anchors = True
        print(
            f"\n--- {tier.upper()} COMSOL anchors: will target up to {anchor_target} new .npz "
            "(subject to CFD-ready pool at run time) ---\n"
        )
    else:
        run_anchors = _prompt_yes_no(
            "Run COMSOL anchor CFD (requires COMSOL + MPh, writes .npz)?", default=False
        )

    allow_overwrite_anchor = False
    anchor_max_json_scan: Optional[int] = None
    anchor_shuffle = False
    anchor_shuffle_seed: Optional[int] = None
    anchor_manual_max_new: Optional[int] = None

    if run_anchors:
        from src.data_gen.lib.anchor_generator import (
            AnchorGenerator,
            summarize_anchor_inventory,
        )

        gen = AnchorGenerator(tier=tier)
        inv = summarize_anchor_inventory(gen.mesh_dir, gen.output_dir)
        total_v = int(inv["mesh_json_with_valid_nas"])
        have_npz = int(inv["existing_npz"])
        remaining = int(inv["pending_missing_npz"])
        ready_add = int(inv["candidate_pool_ready"])
        ready_all = int(inv["candidate_pool_including_npz"])
        print("\n--- Anchor CFD inventory (now; may change after vessel generation) ---")
        print(f"  Output: {gen.output_dir}")
        print(f"  Mesh:   {gen.mesh_dir}")
        print(f"  Meshes with valid .nas: {total_v}")
        print(f"  Anchors already generated: {have_npz}")
        print(f"  Pending (no .npz yet): {remaining}")
        if remaining > ready_add:
            print(f"  ({remaining - ready_add} still need a .msh export before CFD.)")
        print()

        allow_overwrite_anchor = _prompt_anchor_write_mode()
        pool = ready_all if allow_overwrite_anchor else ready_add
        default_more = min(pool, 50) if pool > 0 else 0
        if pool == 0:
            if allow_overwrite_anchor:
                print("No meshes are CFD-ready (need .json + non-empty .nas + .msh).")
            else:
                msg = "Nothing to add (need .json + non-empty .nas + .msh, and no .npz yet)."
                if remaining > 0:
                    msg += " Some meshes lack .msh — export meshes for those vessels first."
                elif total_v == 0:
                    msg = "No vessel meshes found in the mesh directory."
                print(msg)
        else:
            if anchor_target > 0:
                asked = min(anchor_target, pool)
                if anchor_target > pool:
                    print(
                        f"  Only {pool} CFD-ready mesh(es) available now; at run time at most {asked} "
                        f"anchors toward target {anchor_target} (more may exist after vessel generation).\n"
                    )
                else:
                    print(
                        f"  Toward anchor target {anchor_target}: up to {asked} in current pool "
                        "(run will re-check pool after vessels).\n"
                    )
            else:
                mode_note = (
                    "CFD runs to attempt" if allow_overwrite_anchor else "new CFD samples to generate"
                )
                anchor_manual_max_new = _prompt_nonnegative_int(f"How many {mode_note}", default_more)

        if anchor_target == 0 and anchor_manual_max_new is None:
            mode_note = (
                "CFD runs to attempt" if allow_overwrite_anchor else "new CFD samples to generate"
            )
            print(
                "\n  Pool is empty now; if vessel generation creates CFD-ready meshes, "
                "the run will use your count below.\n"
            )
            anchor_manual_max_new = _prompt_nonnegative_int(
                f"How many {mode_note} (0 = skip anchor CFD for this tier)",
                0,
            )

        cap_raw = input(
            "Cap candidate JSON files to scan [empty = no cap, try all candidates]: "
        ).strip()
        anchor_max_json_scan = int(cap_raw) if cap_raw else None
        anchor_shuffle = _prompt_yes_no("Shuffle candidate order?", default=False)
        if anchor_shuffle:
            s = input("Shuffle seed [empty = random]: ").strip()
            anchor_shuffle_seed = int(s) if s else None

    run_mesh = _prompt_yes_no(
        "Convert all vessel_*.msh under raw/ to graphs (clears graphs *.pt first)?",
        default=True,
    )

    return TierInteractivePlan(
        run_vessel=run_vessel,
        level=level,
        overwrite=overwrite,
        n_vessels=n_vessels,
        seed=seed,
        num_workers=num_workers,
        chunk_size=chunk_size,
        no_plot=no_plot,
        run_anchors=run_anchors,
        allow_overwrite_anchor=allow_overwrite_anchor,
        anchor_max_json_scan=anchor_max_json_scan,
        anchor_shuffle=anchor_shuffle,
        anchor_shuffle_seed=anchor_shuffle_seed,
        anchor_manual_max_new=anchor_manual_max_new,
        run_mesh=run_mesh,
    )


def _execute_tier_interactive_plan(
    tier_n: int, anchor_target: int, plan: TierInteractivePlan
) -> None:
    tier = _tier_str_from_n(tier_n)
    print(f"\n{'=' * 60}\n  RUN — {tier.upper()}\n{'=' * 60}\n")

    if plan.run_vessel:
        assert plan.level is not None and plan.overwrite is not None and plan.n_vessels is not None
        vg = VesselGenerator(tier=tier)
        start_idx = 0 if plan.overwrite else None
        print("\n--- Running vessel generator ---\n")
        vg.run_pipeline(
            n=plan.n_vessels,
            level=plan.level,
            seed=plan.seed,
            num_workers=plan.num_workers,
            chunk_size=plan.chunk_size,
            start_idx=start_idx,
        )

        if not plan.no_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)

    if plan.run_anchors:
        from src.data_gen.lib.anchor_generator import (
            AnchorGenerator,
            summarize_anchor_inventory,
        )

        gen = AnchorGenerator(tier=tier)
        inv = summarize_anchor_inventory(gen.mesh_dir, gen.output_dir)
        ready_add = int(inv["candidate_pool_ready"])
        ready_all = int(inv["candidate_pool_including_npz"])
        remaining = int(inv["pending_missing_npz"])
        total_v = int(inv["mesh_json_with_valid_nas"])

        pool = ready_all if plan.allow_overwrite_anchor else ready_add
        print("\n--- Anchor CFD inventory (at run time) ---")
        print(f"  CFD-ready pool: {pool} (add-only pool: {ready_add})\n")

        if pool == 0:
            if plan.allow_overwrite_anchor:
                print("No meshes are CFD-ready — skipping anchor batch.\n")
            else:
                msg = "Nothing to add (need .json + non-empty .nas + .msh, and no .npz yet)."
                if remaining > 0:
                    msg += " Some meshes lack .msh — export meshes for those vessels first."
                elif total_v == 0:
                    msg = "No vessel meshes found in the mesh directory."
                print(msg + "\nSkipping anchor batch.\n")
            max_new = 0
        elif anchor_target > 0:
            asked = min(anchor_target, pool)
            if anchor_target > pool:
                print(
                    f"  Only {pool} CFD-ready mesh(es); running at most {asked} anchors "
                    f"(target was {anchor_target}).\n"
                )
            else:
                print(
                    f"  Running up to {asked} anchor CFD sample(s) toward target {anchor_target}.\n"
                )
            max_new = asked
        else:
            assert plan.anchor_manual_max_new is not None
            asked = plan.anchor_manual_max_new
            if asked == 0:
                print("Skipping anchor batch (0 requested).\n")
                max_new = 0
            else:
                max_new = min(asked, pool)
                if asked > pool:
                    print(f"Requested {asked} but only {pool} mesh(es) match; running {max_new}.\n")

        if pool > 0 and max_new > 0:
            print("\n--- Running anchor CFD ---\n")
            with gen:
                gen.run_batch(
                    max_new=max_new,
                    max_json_to_scan=plan.anchor_max_json_scan,
                    shuffle_candidates=plan.anchor_shuffle,
                    shuffle_seed=plan.anchor_shuffle_seed,
                    allow_overwrite=plan.allow_overwrite_anchor,
                )

    if plan.run_mesh:
        print("\n--- Mesh to graph ---\n")
        MeshToGraphComplete(tier=tier).run()


def _parse_batch_args(argv: list[str]) -> Optional[argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="Tier 1/2 data pipeline: vessel meshes, optional COMSOL anchors, PyG graphs.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Non-interactive mode (requires --tier or --both-tiers; optional anchor flags).",
    )
    p.add_argument(
        "--both-tiers",
        action="store_true",
        help="Run tier 1 then tier 2 sequentially (independent cohorts; use --seed-tier1/--seed-tier2).",
    )
    p.add_argument("--tier", type=int, choices=(1, 2), help="Tier (1 or 2); omit when using --both-tiers")
    p.add_argument("--level", type=int, choices=(0, 1), help="Geometry complexity")
    p.add_argument(
        "-n",
        "--num-vessels",
        type=int,
        metavar="N",
        help="Number of vessels to generate (both tiers use this if --num-vessels-tier* omitted)",
    )
    p.add_argument(
        "--num-vessels-tier1",
        type=int,
        default=None,
        metavar="N",
        help="With --both-tiers: vessel count for tier 1 (falls back to -n).",
    )
    p.add_argument(
        "--num-vessels-tier2",
        type=int,
        default=None,
        metavar="N",
        help="With --both-tiers: vessel count for tier 2 (falls back to -n).",
    )
    p.add_argument("--overwrite", action="store_true", help="Start vessel indices at 0")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Gmsh RNG seed for vessel generation (single-tier batch only; empty default = random).",
    )
    p.add_argument(
        "--seed-tier1",
        type=int,
        default=None,
        metavar="INT",
        help="With --both-tiers: Gmsh seed for tier 1 (omit for random).",
    )
    p.add_argument(
        "--seed-tier2",
        type=int,
        default=None,
        metavar="INT",
        help="With --both-tiers: Gmsh seed for tier 2 (omit for random).",
    )
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument(
        "--show-vessel-plot",
        action="store_true",
        help="Show matplotlib mesh preview after vessel generation (default: skip; avoids blocking on plot windows).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--skip-vessel", action="store_true", help="Skip Gmsh vessel generation")
    p.add_argument("--skip-anchor", action="store_true", help="Skip COMSOL anchor step")
    p.add_argument("--skip-mesh", action="store_true", help="Skip mesh-to-graph conversion")
    p.add_argument(
        "--anchor-max-new",
        type=int,
        default=None,
        metavar="K",
        help="COMSOL: target new .npz per tier if tier-specific flags omitted (omit with --skip-anchor)",
    )
    p.add_argument(
        "--anchor-max-new-tier1",
        type=int,
        default=None,
        metavar="K",
        help="With --both-tiers: anchor target for tier 1 (falls back to --anchor-max-new).",
    )
    p.add_argument(
        "--anchor-max-new-tier2",
        type=int,
        default=None,
        metavar="K",
        help="With --both-tiers: anchor target for tier 2 (falls back to --anchor-max-new).",
    )
    p.add_argument(
        "--anchor-overwrite",
        action="store_true",
        help="COMSOL: allow replacing existing .npz",
    )
    p.add_argument("--anchor-max-json-scan", type=int, default=None)
    p.add_argument("--anchor-shuffle", action="store_true")
    p.add_argument("--anchor-shuffle-seed", type=int, default=None)

    args = p.parse_args(argv)
    if not args.batch:
        return None
    if args.both_tiers and args.tier is not None:
        p.error("Do not pass --tier with --both-tiers")
    if args.both_tiers and args.seed is not None:
        p.error("With --both-tiers use --seed-tier1 and --seed-tier2 (not --seed)")
    missing = []
    if not args.skip_vessel:
        if not args.both_tiers and args.tier is None:
            missing.append("--tier or --both-tiers")
        if args.level is None:
            missing.append("--level")
        if args.both_tiers:
            ok_nv = args.num_vessels is not None or (
                args.num_vessels_tier1 is not None and args.num_vessels_tier2 is not None
            )
            if not ok_nv:
                missing.append(
                    "-n / --num-vessels, or both --num-vessels-tier1 and --num-vessels-tier2"
                )
        elif args.num_vessels is None:
            missing.append("-n / --num-vessels")
    else:
        if not args.both_tiers and args.tier is None:
            missing.append("--tier or --both-tiers (needed for mesh step paths)")
    if missing:
        p.error(f"--batch mode missing: {', '.join(missing)}")
    if not args.skip_anchor:
        if not getattr(args, "both_tiers", False):
            if args.anchor_max_new is None:
                p.error("--batch: specify --anchor-max-new or --skip-anchor")
        else:
            for tn in (1, 2):
                av = getattr(args, f"anchor_max_new_tier{tn}", None)
                if av is None and args.anchor_max_new is None:
                    p.error(
                        "--both-tiers: set --anchor-max-new, or both "
                        "--anchor-max-new-tier1 and --anchor-max-new-tier2"
                    )
    if getattr(args, "both_tiers", False) and not args.skip_vessel:
        for tn in (1, 2):
            nv = getattr(args, f"num_vessels_tier{tn}", None)
            if nv is None and args.num_vessels is None:
                p.error(
                    "--both-tiers: set -n / --num-vessels, or both "
                    "--num-vessels-tier1 and --num-vessels-tier2"
                )
    return args


def _batch_num_vessels_for_tier(tier_num: int, args: argparse.Namespace) -> int:
    v = getattr(args, f"num_vessels_tier{tier_num}", None)
    if v is not None:
        return int(v)
    assert args.num_vessels is not None
    return int(args.num_vessels)


def _batch_anchor_max_for_tier(tier_num: int, args: argparse.Namespace) -> int:
    v = getattr(args, f"anchor_max_new_tier{tier_num}", None)
    if v is not None:
        return int(v)
    assert args.anchor_max_new is not None
    return int(args.anchor_max_new)


def _run_batch_for_tier(
    tier_num: int,
    args: argparse.Namespace,
    *,
    vessel_seed: Optional[int],
    num_vessels: Optional[int] = None,
    anchor_max_new: Optional[int] = None,
) -> None:
    tier = _tier_str_from_n(tier_num)

    if not args.skip_vessel:
        assert num_vessels is not None
        vg = VesselGenerator(tier=tier)
        start_idx = 0 if args.overwrite else None
        print(
            f"--- Vessel generation: tier={tier} level={args.level} n={num_vessels} "
            f"seed={vessel_seed!r} ---\n"
        )
        vg.run_pipeline(
            n=num_vessels,
            level=int(args.level),
            seed=vessel_seed,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            start_idx=start_idx,
        )
        if args.show_vessel_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)

    if not args.skip_anchor:
        assert anchor_max_new is not None
        from src.data_gen.lib.anchor_generator import AnchorGenerator

        gen = AnchorGenerator(tier=tier)
        print(f"--- Anchor CFD: tier={tier} max_new={anchor_max_new} ---\n")
        with gen:
            gen.run_batch(
                max_new=anchor_max_new,
                max_json_to_scan=args.anchor_max_json_scan,
                shuffle_candidates=bool(args.anchor_shuffle),
                shuffle_seed=args.anchor_shuffle_seed,
                allow_overwrite=bool(args.anchor_overwrite),
            )

    if not args.skip_mesh:
        print(f"--- Mesh to graph (tier={tier}) ---\n")
        MeshToGraphComplete(tier=tier).run()


def run_batch_pipeline(args: argparse.Namespace) -> None:
    if getattr(args, "both_tiers", False):
        tiers = (1, 2)
        seeds = (args.seed_tier1, args.seed_tier2)
    else:
        tiers = (int(args.tier),)
        seeds = (args.seed,)

    for i, tier_num in enumerate(tiers):
        if len(tiers) > 1:
            print(f"\n========== Batch tier {tier_num} ==========\n")
        nv = _batch_num_vessels_for_tier(tier_num, args) if not args.skip_vessel else None
        am = _batch_anchor_max_for_tier(tier_num, args) if not args.skip_anchor else None
        _run_batch_for_tier(
            tier_num,
            args,
            vessel_seed=seeds[i],
            num_vessels=nv,
            anchor_max_new=am,
        )

    print("\n=== Pipeline finished ===\n")


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    mp.freeze_support()

    batch_args = _parse_batch_args(argv)
    if batch_args is not None:
        run_batch_pipeline(batch_args)
        return

    if argv:
        print(
            "Unknown arguments (use --batch for non-interactive). "
            "Re-run without arguments for interactive mode.",
            file=sys.stderr,
        )
        sys.exit(2)

    run_interactive_pipeline()


if __name__ == "__main__":
    main()
