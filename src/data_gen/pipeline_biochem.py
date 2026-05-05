"""
Biochem data generation pipeline with three explicit tracks:

1) Synthetic meshes + graphs:
   ``VesselGeneratorPhase3`` -> ``MeshToGraphPhase3``
   (``data/raw/biochem`` -> ``data/processed/graphs_biochem``)

2) Patient-candidate mesh generation for manual COMSOL CFD:
   ``VesselGeneratorPhase3(output_dir=data/raw/biochem_patients)``
   with ``unit='cm'`` for COMSOL CGS compatibility.

3) Anchor CFD extraction for patient candidates:
   ``PatientDataExtractor`` with explicit directories:
   ``data/raw/biochem_patients`` + ``data/processed/cfd_results_biochem_patients``
   -> ``data/processed/graphs_biochem_patients``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
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
from src.utils.paths import data_root


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


def _patient_paths() -> tuple[Path, Path, Path]:
    dr = data_root()
    return (
        dr / "raw/biochem_patients",
        dr / "processed/cfd_results_biochem_patients",
        dr / "processed/graphs_biochem_patients",
    )


def run_interactive_pipeline() -> None:
    print("\n=== Biochem data generation ===\n")
    mode = _vg_prompt_int_choice(
        "Track [1=synthetic meshes+graphs / 2=patient meshes for COMSOL / 3=extract patient anchor CFD]",
        (1, 2, 3),
    )

    if mode == 1:
        print("\n--- Synthetic Biochem (Gmsh -> graphs_biochem) ---\n")
        level = _vg_prompt_int_choice("Geometry level [0=easy / 1=curved / 2=high-thrombus]", (0, 1, 2))
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
        no_plot = _prompt_yes_no("Skip matplotlib preview?", default=True)
        start_idx = 0 if overwrite else int(inv["next_idx"])

        print("\n--- Running VesselGeneratorPhase3 ---\n")
        vg.run_pipeline(
            n=n_vessels,
            level=level,
            start_idx=start_idx,
        )

        if not no_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)

        max_files = _prompt_optional_max_files()
        print("\n--- MeshToGraphPhase3 ---\n")
        MeshToGraphPhase3().run(max_files=max_files)

    elif mode == 2:
        patient_raw_dir, patient_cfd_dir, _ = _patient_paths()
        print("\n--- Patient-candidate meshes for manual COMSOL CFD ---\n")
        print(f"  Mesh output (cm): {patient_raw_dir}")
        print(f"  Then export COMSOL CFD text to: {patient_cfd_dir}\n")

        level = _vg_prompt_int_choice("Geometry level [0=easy / 1=curved / 2=high-thrombus]", (0, 1, 2))
        vg = VesselGeneratorPhase3(output_dir=patient_raw_dir)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        n_on_disk = int(inv["count"])
        max_idx = int(inv["max_idx"])
        index_span = max_idx + 1 if max_idx >= 0 else 0
        print("--- Patient mesh inventory ---")
        print(f"  Output: {vg.output_dir}")
        print(f"  Index span: {index_span} | On disk: {n_on_disk}\n")

        overwrite = _vg_prompt_write_mode_vessel()
        default_n = 25 if n_on_disk > 0 else 100
        n_vessels = _vg_prompt_positive_int("How many patient-candidate meshes to generate", default_n)
        start_idx = 0 if overwrite else int(inv["next_idx"])

        print("\n--- Running VesselGeneratorPhase3 (unit=cm) ---\n")
        vg.run_pipeline(
            n=n_vessels,
            level=level,
            start_idx=start_idx,
            unit="cm",
        )

    else:
        patient_raw_dir, patient_cfd_dir, patient_graph_dir = _patient_paths()
        print("\n--- Extract patient anchor CFD (biochem_patients dirs) ---\n")
        print(
            "Expect COMSOL text exports + meshes under "
            f"{patient_raw_dir} and {patient_cfd_dir}."
        )
        if not _prompt_yes_no("Run PatientDataExtractor now?", default=True):
            print("Skipped patient extraction.")
        else:
            print("\n--- PatientDataExtractor.run() ---\n")
            PatientDataExtractor(
                phase="biochem",
                raw_dir=patient_raw_dir,
                label_dir=patient_cfd_dir,
                proc_dir=patient_graph_dir,
            ).run()

    print("\n=== Biochem pipeline finished ===\n")


def _parse_batch_args(argv: list[str]) -> Optional[argparse.Namespace]:
    p = argparse.ArgumentParser(description="Biochem data pipeline (synthetic, patient meshes, or anchor extract).")
    p.add_argument("--batch", action="store_true", help="Non-interactive mode.")
    p.add_argument(
        "--track",
        choices=("synthetic", "patient_meshes", "extract_anchor_cfd"),
        default="synthetic",
        help="Which pipeline track to run (default: synthetic).",
    )
    p.add_argument("--level", type=int, choices=(0, 1, 2), default=None, help="Geometry level override.")
    p.add_argument("-n", "--num-vessels", type=int, default=None, metavar="N", help="Synthetic vessel count.")
    p.add_argument("--overwrite", action="store_true", help="Synthetic: start vessel indices at 0.")
    p.add_argument(
        "--show-vessel-plot",
        action="store_true",
        help="Show matplotlib mesh preview after vessel generation (default: skip).",
    )
    p.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-mesh-files", type=int, default=None, help="Synthetic: cap meshes for MeshToGraphPhase3.")
    args = p.parse_args(argv)
    if not args.batch:
        return None
    if args.track in ("synthetic", "patient_meshes") and args.num_vessels is None:
        p.error("--batch synthetic/patient_meshes requires -n / --num-vessels")
    return args


def run_batch_pipeline(args: argparse.Namespace) -> None:
    patient_raw_dir, patient_cfd_dir, patient_graph_dir = _patient_paths()

    if args.track == "synthetic":
        level = 0 if args.level is None else int(args.level)
        vg = VesselGeneratorPhase3()
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        start_idx = 0 if args.overwrite else int(inv["next_idx"])
        print(f"--- VesselGeneratorPhase3 (synthetic): n={args.num_vessels} level={level} start={start_idx} ---\n")
        vg.run_pipeline(
            n=int(args.num_vessels),
            level=level,
            start_idx=start_idx,
        )
        if args.show_vessel_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)
        print("--- MeshToGraphPhase3 ---\n")
        MeshToGraphPhase3().run(max_files=args.max_mesh_files)

    elif args.track == "patient_meshes":
        level = 2 if args.level is None else int(args.level)
        vg = VesselGeneratorPhase3(output_dir=patient_raw_dir)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        start_idx = 0 if args.overwrite else int(inv["next_idx"])
        print(
            f"--- VesselGeneratorPhase3 (patient meshes, cm): n={args.num_vessels} "
            f"level={level} start={start_idx} ---\n"
        )
        vg.run_pipeline(
            n=int(args.num_vessels),
            level=level,
            start_idx=start_idx,
            unit="cm",
        )
        if args.show_vessel_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)

    else:
        print("--- PatientDataExtractor (anchor CFD extraction) ---\n")
        PatientDataExtractor(
            phase="biochem",
            raw_dir=patient_raw_dir,
            label_dir=patient_cfd_dir,
            proc_dir=patient_graph_dir,
        ).run()

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
