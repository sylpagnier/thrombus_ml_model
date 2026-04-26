"""
Biochem data generation: synthetic Gmsh cohort + Biochem graphs, and/or patient COMSOL extraction.

Synthetic track: ``VesselGeneratorPhase3`` -> ``MeshToGraphPhase3`` (``data/raw/biochem`` -> ``graphs_biochem``).

Patient track: ``PatientDataExtractor`` (``data/raw/biochem_patients`` + COMSOL exports -> ``graphs_biochem_patients``).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from typing import Optional

from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor
from src.data_gen.lib.mesh_to_graph_biochem import MeshToGraphPhase3
from src.data_gen.lib.vessel_generator import (
    VesselGeneratorPhase3,
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


def _prompt_optional_max_files() -> Optional[int]:
    while True:
        raw = input("Max .msh files to convert [all / blank]: ").strip()
        raw_l = raw.lower()
        if raw == "" or raw_l == "all":
            return None
        try:
            return int(raw)
        except ValueError:
            print("  Enter an integer, 'all', or leave blank.")


def run_interactive_pipeline() -> None:
    print("\n=== Biochem data generation (synthetic / patient extraction) ===\n")
    mode = _vg_prompt_int_choice("Track [1=synthetic Gmsh+graphs / 2=patient COMSOL extract / 3=both]", (1, 2, 3))

    run_syn = mode in (1, 3)
    run_pat = mode in (2, 3)

    if run_syn:
        print("\n--- Synthetic Biochem (Gmsh -> graphs_biochem) ---\n")
        run_vessel = _prompt_yes_no("Generate synthetic vessel meshes?", default=True)
        if run_vessel:
            level = _vg_prompt_int_choice("Geometry level", (0, 1))
            vg = VesselGeneratorPhase3()
            inv = summarize_vessel_mesh_inventory(vg.output_dir)
            n_on_disk = int(inv["count"])
            max_idx = int(inv["max_idx"])
            index_span = max_idx + 1 if max_idx >= 0 else 0
            print("\n--- Vessel mesh inventory (biochem) ---")
            print(f"  Output: {vg.output_dir}")
            print(f"  Index span: {index_span} | On disk: {n_on_disk}\n")

            overwrite = _vg_prompt_write_mode_vessel()
            default_n = 50 if n_on_disk > 0 else 200
            n_vessels = _vg_prompt_positive_int("How many vessels to generate", default_n)
            seed = _prompt_optional_int("RNG seed")
            num_workers = _prompt_optional_int("Worker processes")
            chunk_size = _prompt_optional_int("Chunk size")
            no_plot = _prompt_yes_no("Skip matplotlib preview?", default=True)

            if overwrite:
                start_idx = 0
            else:
                start_idx = int(inv["next_idx"])

            print("\n--- Running VesselGeneratorPhase3 ---\n")
            vg.run_pipeline(
                n=n_vessels,
                level=level,
                seed=seed,
                num_workers=num_workers,
                chunk_size=chunk_size,
                start_idx=start_idx,
            )

            if not no_plot:
                saved_indices = sorted(
                    int(p.stem.split("_")[-1])
                    for p in vg.output_dir.glob("vessel_*.msh")
                )[:9]
                if saved_indices:
                    vg.visualize_saved(saved_indices)

        run_m2g = _prompt_yes_no(
            "Run MeshToGraphPhase3 (synthetic .msh -> graphs_biochem .pt, replaces *.pt in output dir)?",
            default=True,
        )
        if run_m2g:
            max_files = _prompt_optional_max_files()
            print("\n--- MeshToGraphPhase3 ---\n")
            MeshToGraphPhase3().run(max_files=max_files)

    if run_pat:
        print("\n--- Patient / anchor extraction (biochem_patients) ---\n")
        print(
            "Expect COMSOL text exports + meshes under data/raw/biochem_patients "
            "(see PatientDataExtractor docstring)."
        )
        if not _prompt_yes_no("Run PatientDataExtractor now?", default=True):
            print("Skipped patient extraction.")
        else:
            print("\n--- PatientDataExtractor.run() ---\n")
            PatientDataExtractor(phase="biochem_patients").run()

    print("\n=== Biochem pipeline finished ===\n")


def _parse_batch_args(argv: list[str]) -> Optional[argparse.Namespace]:
    p = argparse.ArgumentParser(description="Biochem data pipeline (synthetic and/or patient extract).")
    p.add_argument("--batch", action="store_true", help="Non-interactive mode.")
    p.add_argument(
        "--track",
        choices=("synthetic", "patient", "both"),
        default="synthetic",
        help="Which track to run (default: synthetic).",
    )
    p.add_argument("--level", type=int, choices=(0, 1), default=0, help="Gmsh geometry level (synthetic).")
    p.add_argument("-n", "--num-vessels", type=int, default=None, metavar="N", help="Synthetic vessel count.")
    p.add_argument("--overwrite", action="store_true", help="Synthetic: start vessel indices at 0.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument(
        "--show-vessel-plot",
        action="store_true",
        help="Show matplotlib mesh preview after vessel generation (default: skip).",
    )
    p.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-vessel", action="store_true", help="Synthetic: skip Gmsh generation.")
    p.add_argument("--skip-mesh", action="store_true", help="Synthetic: skip MeshToGraphPhase3.")
    p.add_argument("--max-mesh-files", type=int, default=None, help="Synthetic: cap meshes for MeshToGraphPhase3.")
    p.add_argument("--skip-patient", action="store_true", help="Patient: skip PatientDataExtractor.")
    args = p.parse_args(argv)
    if not args.batch:
        return None
    if args.track in ("synthetic", "both"):
        if not args.skip_vessel and args.num_vessels is None:
            p.error("--batch synthetic/both requires -n / --num-vessels unless --skip-vessel")
    return args


def run_batch_pipeline(args: argparse.Namespace) -> None:
    if args.track in ("synthetic", "both"):
        if not args.skip_vessel:
            vg = VesselGeneratorPhase3()
            inv = summarize_vessel_mesh_inventory(vg.output_dir)
            start_idx = 0 if args.overwrite else int(inv["next_idx"])
            print(f"--- VesselGeneratorPhase3: n={args.num_vessels} level={args.level} start={start_idx} ---\n")
            vg.run_pipeline(
                n=int(args.num_vessels),
                level=int(args.level),
                seed=args.seed,
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
        if not args.skip_mesh:
            print("--- MeshToGraphPhase3 ---\n")
            MeshToGraphPhase3().run(max_files=args.max_mesh_files)

    if args.track in ("patient", "both") and not args.skip_patient:
        print("--- PatientDataExtractor ---\n")
        PatientDataExtractor(phase="biochem_patients").run()

    print("\n=== Biochem pipeline finished ===\n")


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    mp.freeze_support()

    batch = _parse_batch_args(argv)
    if batch is not None:
        run_batch_pipeline(batch)
        return
    if argv:
        print(
            "Unknown arguments (use --batch for non-interactive). "
            "Run with no arguments for interactive mode.",
            file=sys.stderr,
        )
        sys.exit(2)
    run_interactive_pipeline()


if __name__ == "__main__":
    main()
