"""
Interactive Kinematics/2 data pipeline: vessel meshes, optional COMSOL anchors, PyG graphs.

Runs the same logical steps as ``vessel_generator``, ``anchor_generator``, and ``mesh_to_graph``.
**Interactive mode asks every question first** (per phase), then runs Gmsh / COMSOL / mesh-to-graph
with **no further prompts** so you can leave the machine unattended after the planning phase.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from dataclasses import dataclass
from typing import Optional

from src.data_gen.lib.mesh_to_graph import MeshToGraph
from src.data_gen.lib.vessel_generator import (
    VesselGenerator,
    summarize_vessel_mesh_inventory,
    _prompt_int_choice as _vg_prompt_int_choice,
    _prompt_write_mode_vessel as _vg_prompt_write_mode_vessel,
)


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


def _rheology_from_n(choice_n: int) -> str:
    if choice_n == 1:
        return "newtonian"
    if choice_n == 2:
        return "carreau"
    raise ValueError(f"Unsupported rheology choice: {choice_n}")


def _final_subdir_for_rheology(rheology: str) -> str:
    return str(rheology).strip().lower()


@dataclass
class PhaseInteractivePlan:
    """All interactive choices for one rheology pass (collected before any long-running step)."""

    anchor_target: int
    run_vessel: bool
    level: Optional[int]
    overwrite: Optional[bool]
    n_vessels: Optional[int]
    seed: Optional[int]
    num_workers: Optional[int]
    chunk_size: Optional[int]
    run_anchors: bool
    allow_overwrite_anchor: bool
    anchor_max_json_scan: Optional[int]
    anchor_shuffle: bool
    anchor_shuffle_seed: Optional[int]
    # When anchor_target == 0 and run_anchors: how many new CFD samples to aim for.
    anchor_manual_max_new: Optional[int]
    run_mesh: bool


def run_interactive_pipeline() -> None:
    print("\n=== Kinematics/2 data generation pipeline (vessel, anchors, graphs) ===\n")

    rheology_scope = _vg_prompt_int_choice(
        "Run Kinematics datagen for (1 = Newtonian Primer, 2 = Carreau Target, 3 = Both sequentially)",
        (1, 2, 3),
    )
    rheology_sequence = (1, 2) if rheology_scope == 3 else (rheology_scope,)

    if len(rheology_sequence) == 2:
        print(
            "\nBoth rheology passes run one after the other (newtonian then carreau), "
            "writing to separate CFD and graph subfolders.\n"
        )

    print(
        "\nEach rheology plan asks vessel generation -> COMSOL anchors (optional) first; "
        "mesh-to-graph runs automatically after. No prompts during execution.\n"
    )

    print(
        "\n--- Planning: answer all questions now; the run after this has **no further prompts** ---\n"
    )
    plans: dict[int, PhaseInteractivePlan] = {}
    for rheology_n in rheology_sequence:
        plans[rheology_n] = _prompt_phase_interactive_plan(rheology_n)

    print(
        f"\n{'=' * 60}\n"
        "  All prompts complete — unattended run starting.\n"
        f"{'=' * 60}\n"
    )

    for rheology_n in rheology_sequence:
        _execute_phase_interactive_plan(rheology_n, plans[rheology_n].anchor_target, plans[rheology_n])

    print("\n=== Pipeline finished ===\n")


def _prompt_phase_interactive_plan(rheology_n: int) -> PhaseInteractivePlan:
    rheology = _rheology_from_n(rheology_n)
    print(f"\n{'=' * 60}\n  PLAN — {rheology.upper()} (independent cohort)\n{'=' * 60}\n")

    # ==========================================================
    # 1. VESSELS
    # ==========================================================
    vg = VesselGenerator(phase="kinematics")
    inv = summarize_vessel_mesh_inventory(vg.output_dir)
    n_on_disk = int(inv["count"])

    print("\n--- Vessel mesh inventory ---")
    print(f"  Meshes currently on disk: {n_on_disk}\n")

    default_n = 50 if n_on_disk > 0 else 500
    n_vessels = _prompt_nonnegative_int("How many vessels to generate? (0 = skip)", default=default_n)
    run_vessel = n_vessels > 0

    level: Optional[int] = None
    overwrite: Optional[bool] = None
    seed: Optional[int] = None
    num_workers: Optional[int] = None
    chunk_size: Optional[int] = None

    if run_vessel:
        level = _vg_prompt_int_choice("Geometry level", (0, 1))
        if n_on_disk == 0:
            overwrite = True
            print("  No meshes on disk — starting indices at 0 (overwrite).\n")
        else:
            overwrite = _vg_prompt_write_mode_vessel()

    # ==========================================================
    # 2. ANCHORS
    # ==========================================================
    from src.data_gen.lib.anchor_generator import (
        AnchorGenerator,
        summarize_anchor_inventory,
    )

    print(f"\n--- {rheology.upper()} COMSOL anchors ---")

    # Match anchor write mode to vessel overwrite (new cohort replaces old meshes + anchors).
    if run_vessel and overwrite is True:
        allow_overwrite_anchor = True
        print(
            "  Vessel generation overwrites mesh indices — using anchor overwrite (existing .npz can be replaced).\n"
        )
    else:
        allow_overwrite_anchor = _prompt_anchor_write_mode()

    anchor_output_dir = vg.vessel_cfg.output_dir / _final_subdir_for_rheology(rheology)
    gen = AnchorGenerator(phase="kinematics", output_dir=anchor_output_dir)
    anchor_inv = summarize_anchor_inventory(gen.mesh_dir, gen.target_output_dir())
    have_npz = int(anchor_inv["existing_npz"])
    ready_add = int(anchor_inv["candidate_pool_ready"])
    ready_all = int(anchor_inv["candidate_pool_including_npz"])

    if not allow_overwrite_anchor:
        print(f"  Anchors already generated: {have_npz}\n")
        pool = ready_add
    else:
        print("  Overwriting existing anchors.\n")
        pool = ready_all

    default_anchors = min(pool, 50) if pool > 0 else 0
    anchor_manual_max_new = _prompt_nonnegative_int(
        "How many anchors to generate? (0 = skip)", default=default_anchors
    )
    run_anchors = anchor_manual_max_new > 0

    # No JSON scan cap; shuffle candidates with a random seed (interactive kinematics defaults).
    anchor_max_json_scan: Optional[int] = None
    anchor_shuffle = True
    anchor_shuffle_seed: Optional[int] = None
    # ==========================================================
    # 3. MESH TO GRAPH (Automatic)
    # ==========================================================
    run_mesh = True

    return PhaseInteractivePlan(
        anchor_target=0,  # Hardcoded to 0 to trigger the manual count in execution
        run_vessel=run_vessel,
        level=level,
        overwrite=overwrite,
        n_vessels=n_vessels if run_vessel else None,
        seed=seed,
        num_workers=num_workers,
        chunk_size=chunk_size,
        run_anchors=run_anchors,
        allow_overwrite_anchor=allow_overwrite_anchor,
        anchor_max_json_scan=anchor_max_json_scan,
        anchor_shuffle=anchor_shuffle,
        anchor_shuffle_seed=anchor_shuffle_seed,
        anchor_manual_max_new=anchor_manual_max_new,
        run_mesh=run_mesh,
    )


def _execute_phase_interactive_plan(
    rheology_n: int, anchor_target: int, plan: PhaseInteractivePlan
) -> None:
    rheology = _rheology_from_n(rheology_n)
    print(f"\n{'=' * 60}\n  RUN — {rheology.upper()}\n{'=' * 60}\n")

    if plan.run_vessel:
        assert plan.level is not None and plan.overwrite is not None and plan.n_vessels is not None
        vg = VesselGenerator(phase="kinematics")
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

    if plan.run_anchors:
        from src.data_gen.lib.anchor_generator import (
            AnchorGenerator,
            summarize_anchor_inventory,
        )

        anchor_output_dir = VesselGenerator(phase="kinematics").vessel_cfg.output_dir / _final_subdir_for_rheology(
            rheology
        )
        gen = AnchorGenerator(phase="kinematics", output_dir=anchor_output_dir)
        inv = summarize_anchor_inventory(gen.mesh_dir, gen.target_output_dir())
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
                    continuation_steps=None,
                )

    if plan.run_mesh:
        print("\n--- Mesh to graph ---")
        from src.data_gen.lib.mesh_to_graph import MeshToGraph

        # Process target graphs only; no intermediate continuation sweeps.
        final_subdir = _final_subdir_for_rheology(rheology)
        target_label = final_subdir
        print(f"\n⚙️ Converting Meshes -> Graphs for TARGET ({target_label})...")
        processor = MeshToGraph(phase="kinematics", n_subdir=final_subdir)
        processor.run()


def _parse_batch_args(argv: list[str]) -> Optional[argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="Kinematics/2 data pipeline: vessel meshes, optional COMSOL anchors, PyG graphs.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Non-interactive mode (requires --rheology or --both-rheologies; optional anchor flags).",
    )
    p.add_argument(
        "--both-rheologies",
        action="store_true",
        help="Run newtonian then carreau sequentially (independent cohorts; use --seed-newtonian/--seed-carreau).",
    )
    p.add_argument("--rheology", choices=("newtonian", "carreau"), help="Rheology target; omit when using --both-rheologies")
    p.add_argument("--level", type=int, choices=(0, 1), help="Geometry complexity")
    p.add_argument(
        "-n",
        "--num-vessels",
        type=int,
        metavar="N",
        help="Number of vessels to generate (both passes use this if per-rheology flags are omitted)",
    )
    p.add_argument(
        "--num-vessels-newtonian",
        type=int,
        default=None,
        metavar="N",
        help="With --both-rheologies: vessel count for newtonian pass (falls back to -n).",
    )
    p.add_argument(
        "--num-vessels-carreau",
        type=int,
        default=None,
        metavar="N",
        help="With --both-rheologies: vessel count for carreau pass (falls back to -n).",
    )
    p.add_argument("--overwrite", action="store_true", help="Start vessel indices at 0")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Gmsh RNG seed for vessel generation (single-rheology batch only; empty default = random).",
    )
    p.add_argument(
        "--seed-newtonian",
        type=int,
        default=None,
        metavar="INT",
        help="With --both-rheologies: Gmsh seed for newtonian pass (omit for random).",
    )
    p.add_argument(
        "--seed-carreau",
        type=int,
        default=None,
        metavar="INT",
        help="With --both-rheologies: Gmsh seed for carreau pass (omit for random).",
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
        help="COMSOL: target new .npz per phase if phase-specific flags omitted (omit with --skip-anchor)",
    )
    p.add_argument(
        "--anchor-max-new-newtonian",
        type=int,
        default=None,
        metavar="K",
        help="With --both-rheologies: anchor target for newtonian pass (falls back to --anchor-max-new).",
    )
    p.add_argument(
        "--anchor-max-new-carreau",
        type=int,
        default=None,
        metavar="K",
        help="With --both-rheologies: anchor target for carreau pass (falls back to --anchor-max-new).",
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
    if args.both_rheologies and args.rheology is not None:
        p.error("Do not pass --rheology with --both-rheologies")
    if args.both_rheologies and args.seed is not None:
        p.error("With --both-rheologies use --seed-newtonian and --seed-carreau (not --seed)")
    missing = []
    if not args.skip_vessel:
        if not args.both_rheologies and args.rheology is None:
            missing.append("--rheology or --both-rheologies")
        if args.level is None:
            missing.append("--level")
        if args.both_rheologies:
            ok_nv = args.num_vessels is not None or (
                args.num_vessels_newtonian is not None and args.num_vessels_carreau is not None
            )
            if not ok_nv:
                missing.append(
                    "-n / --num-vessels, or both --num-vessels-newtonian and --num-vessels-carreau"
                )
        elif args.num_vessels is None:
            missing.append("-n / --num-vessels")
    else:
        if not args.both_rheologies and args.rheology is None:
            missing.append("--rheology or --both-rheologies (needed for mesh step paths)")
    if missing:
        p.error(f"--batch mode missing: {', '.join(missing)}")
    if not args.skip_anchor:
        if not getattr(args, "both_rheologies", False):
            if args.anchor_max_new is None:
                p.error("--batch: specify --anchor-max-new or --skip-anchor")
        else:
            for key in ("newtonian", "carreau"):
                av = getattr(args, f"anchor_max_new_{key}", None)
                if av is None and args.anchor_max_new is None:
                    p.error(
                        "--both-rheologies: set --anchor-max-new, or both "
                        "--anchor-max-new-newtonian and --anchor-max-new-carreau"
                    )
    if getattr(args, "both_rheologies", False) and not args.skip_vessel:
        for key in ("newtonian", "carreau"):
            nv = getattr(args, f"num_vessels_{key}", None)
            if nv is None and args.num_vessels is None:
                p.error(
                    "--both-rheologies: set -n / --num-vessels, or both "
                    "--num-vessels-newtonian and --num-vessels-carreau"
                )
    return args


def _batch_num_vessels_for_rheology(rheology: str, args: argparse.Namespace) -> int:
    v = getattr(args, f"num_vessels_{rheology}", None)
    if v is not None:
        return int(v)
    assert args.num_vessels is not None
    return int(args.num_vessels)


def _batch_anchor_max_for_rheology(rheology: str, args: argparse.Namespace) -> int:
    v = getattr(args, f"anchor_max_new_{rheology}", None)
    if v is not None:
        return int(v)
    assert args.anchor_max_new is not None
    return int(args.anchor_max_new)


def _run_batch_for_phase(
    rheology: str,
    args: argparse.Namespace,
    *,
    vessel_seed: Optional[int],
    num_vessels: Optional[int] = None,
    anchor_max_new: Optional[int] = None,
) -> None:
    if not args.skip_vessel:
        assert num_vessels is not None
        vg = VesselGenerator(phase="kinematics")
        start_idx = 0 if args.overwrite else None
        print(
            f"--- Vessel generation: rheology={rheology} level={args.level} n={num_vessels} "
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

        anchor_output_dir = VesselGenerator(phase="kinematics").vessel_cfg.output_dir / _final_subdir_for_rheology(
            rheology
        )
        gen = AnchorGenerator(phase="kinematics", output_dir=anchor_output_dir)
        print(f"--- Anchor CFD: rheology={rheology} max_new={anchor_max_new} ---\n")
        with gen:
            gen.run_batch(
                max_new=anchor_max_new,
                max_json_to_scan=args.anchor_max_json_scan,
                shuffle_candidates=bool(args.anchor_shuffle),
                shuffle_seed=args.anchor_shuffle_seed,
                allow_overwrite=bool(args.anchor_overwrite),
                continuation_steps=None,
            )

    if not args.skip_mesh:
        print(f"--- Mesh to graph (rheology={rheology}) ---")
        from src.data_gen.lib.mesh_to_graph import MeshToGraph

        # Process target graphs only; no intermediate continuation sweeps.
        final_subdir = _final_subdir_for_rheology(rheology)
        target_label = final_subdir
        print(f"\n⚙️ Converting Meshes -> Graphs for TARGET ({target_label})...")
        processor = MeshToGraph(phase="kinematics", n_subdir=final_subdir)
        processor.run()


def run_batch_pipeline(args: argparse.Namespace) -> None:
    if getattr(args, "both_rheologies", False):
        rheologies = ("newtonian", "carreau")
        seeds = (args.seed_newtonian, args.seed_carreau)
    else:
        rheologies = (str(args.rheology),)
        seeds = (args.seed,)

    for i, rheology in enumerate(rheologies):
        if len(rheologies) > 1:
            print(f"\n========== Batch rheology {rheology} ==========\n")
        nv = _batch_num_vessels_for_rheology(rheology, args) if not args.skip_vessel else None
        am = _batch_anchor_max_for_rheology(rheology, args) if not args.skip_anchor else None
        _run_batch_for_phase(
            rheology,
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
