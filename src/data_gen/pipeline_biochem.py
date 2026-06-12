"""
Biochem data generation pipeline. The interactive menu exposes two top-level
tracks:

1) Synthetic data (meshes + graphs):
   ``VesselGeneratorPhase3`` -> ``MeshToGraphPhase3``
   (``data/raw/biochem`` -> ``data/processed/graphs_biochem``)

2) Anchor data (meshes, extraction, graphs in cm) -- has two sub-steps:

   2a) Generate anchor-candidate meshes for manual COMSOL CFD:
       ``VesselGeneratorPhase3(output_dir=data/raw/biochem_anchors)``
       with ``unit='cm'`` for COMSOL CGS compatibility.

   2b) Extract anchor CFD into graphs:
       ``PatientDataExtractor`` with explicit directories:
       ``data/raw/biochem_anchors`` + ``data/processed/cfd_results_biochem``
       -> ``data/processed/graphs_biochem_anchors``.

The batch CLI exposes these three steps directly via ``--track`` as
``synthetic`` / ``anchor_meshes`` / ``extract_anchor_cfd``.
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
    normalize_pathology_mode,
    summarize_vessel_mesh_inventory,
    _prompt_int_choice as _vg_prompt_int_choice,
    _prompt_positive_int as _vg_prompt_positive_int,
    _prompt_write_mode_vessel as _vg_prompt_write_mode_vessel,
)
from src.tools.prepare_biochem_anchors import scaffold_anchor_sidecars
from src.utils.paths import data_root


def _prompt_pathology_mode() -> Optional[str]:
    """Return ``max_stenosis``, ``max_aneurysm``, or ``None`` for random pathology sampling."""
    print(
        "\nPathology strength:\n"
        "  1 = random mix (default)\n"
        "  2 = max stenosis (~75% diameter occlusion at peak)\n"
        "  3 = max aneurysm (deepest configured expansion)\n"
    )
    while True:
        raw = input("Pathology strength [1/2/3] [1]: ").strip()
        if raw in ("", "1"):
            return None
        if raw == "2":
            return "max_stenosis"
        if raw == "3":
            return "max_aneurysm"
        print("  Enter 1, 2, or 3.")


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


def _anchor_paths() -> tuple[Path, Path, Path]:
    dr = data_root()
    return (
        dr / "raw/biochem_anchors",
        dr / "processed/cfd_results_biochem",
        dr / "processed/graphs_biochem_anchors",
    )


def _auto_scaffold_anchor_sidecars(anchor_raw_dir: Path) -> None:
    """Idempotently ensure every anchor mesh has a ``{"unit":"cm"}`` JSON sidecar.

    The anchor track is hard-coded to expect cm meshes (COMSOL CGS), and the
    runtime unit guard treats a missing sidecar as 'no opinion'. Writing a
    minimal sidecar here makes the unit assumption explicit on disk and gives
    ``assert_mesh_unit`` something concrete to validate against.
    """
    if not anchor_raw_dir.exists():
        return
    print("--- Anchor sidecar scaffold ---")
    created, updated, conflicts = scaffold_anchor_sidecars(
        anchor_raw_dir, unit="cm", force=False, dry_run=False
    )
    print(
        f"  sidecars: created {created}, augmented {updated}, conflicts {conflicts} "
        f"({anchor_raw_dir})"
    )
    if conflicts:
        print(
            "  Conflicting unit declarations were left untouched. To overwrite, run:\n"
            "    python -m src.tools.prepare_biochem_anchors --force"
        )


def run_interactive_pipeline() -> None:
    print("\n=== Biochem data generation ===\n")
    track = _vg_prompt_int_choice(
        "Track [1=synthetic data (meshes/graphs) / 2=anchor data (meshes, extraction, graphs in cm)]",
        (1, 2),
    )

    if track == 1:
        mode = 1
    else:
        sub = _vg_prompt_int_choice(
            "Anchor step [1=generate meshes/graphs / 2=extract data]",
            (1, 2),
        )
        mode = 2 if sub == 1 else 3

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
        pathology_mode = _prompt_pathology_mode()
        start_idx = 0 if overwrite else int(inv["next_idx"])

        print("\n--- Running VesselGeneratorPhase3 ---\n")
        vg.run_pipeline(
            n=n_vessels,
            level=level,
            start_idx=start_idx,
            pathology_mode=pathology_mode,
        )

        if not no_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)

        print("\n--- MeshToGraphPhase3 ---\n")
        MeshToGraphPhase3().run()

    elif mode == 2:
        anchor_raw_dir, anchor_cfd_dir, _ = _anchor_paths()
        print("\n--- Anchor-candidate meshes for manual COMSOL CFD ---\n")
        print(f"  Mesh output (cm): {anchor_raw_dir}")
        print(f"  Then export COMSOL CFD text to: {anchor_cfd_dir}\n")

        level = _vg_prompt_int_choice("Geometry level [0=easy / 1=curved / 2=high-thrombus]", (0, 1, 2))
        vg = VesselGeneratorPhase3(output_dir=anchor_raw_dir)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        n_on_disk = int(inv["count"])
        max_idx = int(inv["max_idx"])
        index_span = max_idx + 1 if max_idx >= 0 else 0
        print("--- Anchor mesh inventory ---")
        print(f"  Output: {vg.output_dir}")
        print(f"  Index span: {index_span} | On disk: {n_on_disk}\n")

        overwrite = _vg_prompt_write_mode_vessel()
        default_n = 25 if n_on_disk > 0 else 100
        n_vessels = _vg_prompt_positive_int("How many anchor-candidate meshes to generate", default_n)
        pathology_mode = _prompt_pathology_mode()
        start_idx = 0 if overwrite else int(inv["next_idx"])

        print("\n--- Running VesselGeneratorPhase3 (unit=cm) ---\n")
        vg.run_pipeline(
            n=n_vessels,
            level=level,
            start_idx=start_idx,
            unit="cm",
            pathology_mode=pathology_mode,
        )

    else:
        anchor_raw_dir, anchor_cfd_dir, anchor_graph_dir = _anchor_paths()
        print("\n--- Extract anchor CFD (biochem_anchors dirs) ---\n")
        print(
            "Expect COMSOL text exports + meshes under "
            f"{anchor_raw_dir} and {anchor_cfd_dir}."
        )
        if not _prompt_yes_no("Run PatientDataExtractor now?", default=True):
            print("Skipped anchor extraction.")
        else:
            _auto_scaffold_anchor_sidecars(anchor_raw_dir)
            print("\n--- PatientDataExtractor.run() ---\n")
            PatientDataExtractor(
                phase="biochem_anchors",
                raw_dir=anchor_raw_dir,
                label_dir=anchor_cfd_dir,
                proc_dir=anchor_graph_dir,
            ).run()

    print("\n=== Biochem pipeline finished ===\n")


def _parse_batch_args(argv: list[str]) -> Optional[argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="Biochem data pipeline (synthetic, anchor meshes, or anchor CFD extract)."
    )
    p.add_argument("--batch", action="store_true", help="Non-interactive mode.")
    p.add_argument(
        "--track",
        choices=("synthetic", "anchor_meshes", "extract_anchor_cfd"),
        default="synthetic",
        help="Which pipeline track to run (default: synthetic).",
    )
    p.add_argument("--level", type=int, choices=(0, 1, 2), default=None, help="Geometry level override.")
    p.add_argument(
        "-n",
        "--num-vessels",
        type=int,
        default=None,
        metavar="N",
        help="Vessel count for synthetic or anchor-mesh generation.",
    )
    p.add_argument("--overwrite", action="store_true", help="Synthetic: start vessel indices at 0.")
    p.add_argument(
        "--show-vessel-plot",
        action="store_true",
        help="Show matplotlib mesh preview after vessel generation (default: skip).",
    )
    p.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-mesh-files", type=int, default=None, help="Synthetic: cap meshes for MeshToGraphPhase3.")
    p.add_argument(
        "--pathology-mode",
        choices=("random", "max_stenosis", "max_aneurysm"),
        default="random",
        help="Pathology sampling: random (default), max_stenosis (~75%% occlusion), or max_aneurysm.",
    )
    args = p.parse_args(argv)
    if not args.batch:
        return None
    if args.track in ("synthetic", "anchor_meshes") and args.num_vessels is None:
        p.error("--batch synthetic/anchor_meshes requires -n / --num-vessels")
    return args


def run_batch_pipeline(args: argparse.Namespace) -> None:
    anchor_raw_dir, anchor_cfd_dir, anchor_graph_dir = _anchor_paths()

    if args.track == "synthetic":
        level = 0 if args.level is None else int(args.level)
        vg = VesselGeneratorPhase3()
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        start_idx = 0 if args.overwrite else int(inv["next_idx"])
        pathology_mode = normalize_pathology_mode(args.pathology_mode)
        print(
            f"--- VesselGeneratorPhase3 (synthetic): n={args.num_vessels} level={level} "
            f"start={start_idx} pathology={args.pathology_mode} ---\n"
        )
        vg.run_pipeline(
            n=int(args.num_vessels),
            level=level,
            start_idx=start_idx,
            pathology_mode=pathology_mode,
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

    elif args.track == "anchor_meshes":
        level = 2 if args.level is None else int(args.level)
        vg = VesselGeneratorPhase3(output_dir=anchor_raw_dir)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        start_idx = 0 if args.overwrite else int(inv["next_idx"])
        pathology_mode = normalize_pathology_mode(args.pathology_mode)
        print(
            f"--- VesselGeneratorPhase3 (anchor meshes, cm): n={args.num_vessels} "
            f"level={level} start={start_idx} pathology={args.pathology_mode} ---\n"
        )
        vg.run_pipeline(
            n=int(args.num_vessels),
            level=level,
            start_idx=start_idx,
            unit="cm",
            pathology_mode=pathology_mode,
        )
        if args.show_vessel_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)

    else:
        _auto_scaffold_anchor_sidecars(anchor_raw_dir)
        print("--- PatientDataExtractor (anchor CFD extraction) ---\n")
        PatientDataExtractor(
            phase="biochem_anchors",
            raw_dir=anchor_raw_dir,
            label_dir=anchor_cfd_dir,
            proc_dir=anchor_graph_dir,
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
