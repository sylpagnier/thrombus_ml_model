"""Interactive biochem COMSOL -> PyG graph extraction.

**Automated path (recommended):** run the study in COMSOL, save ``<stem>.mph`` next to the
mesh under ``data/raw/biochem_anchors/``, with COMSOL saves as
``comsol_models/phase2_nowound_XXX.mph`` (for ``patientXXX``), then::

    python -m src.tools.extract_biochem_comsol --stem patient007 --from-comsol

This samples the solved model via LiveLink (``mph``) and writes
``data/processed/cfd_results_biochem/*.txt`` before building graphs.

**Manual path:** export domain + boundary ``.txt`` yourself, then run without ``--from-comsol``.

PyCharm: **Run** module ``src.tools.extract_biochem_comsol`` (working directory = repo root).

CLI::

    python -m src.tools.extract_biochem_comsol --from-comsol
    python -m src.bin.main data extract-biochem -- --from-comsol --stem patient007
    python -m src.tools.extract_biochem_comsol --list-only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.data_gen.lib.biochem_comsol_auto_export import resolve_biochem_comsol_model_path
from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor
from src.data_gen.pipeline_biochem import _auto_scaffold_anchor_sidecars
from src.tools.prepare_biochem_anchors import enrich_anchor_meshes, stems_in_dir
from src.utils.paths import data_root

_BOUNDARY_SUFFIXES = ("_inlet", "_outlet", "_wall")


@dataclass(frozen=True)
class AnchorExtractStatus:
    stem: str
    has_mesh: bool
    has_domain_txt: bool
    has_inlet_txt: bool
    has_outlet_txt: bool
    has_wall_txt: bool
    has_biochem_graph: bool
    has_kine_graph: bool
    biochem_graph_mtime: float | None
    has_comsol_model: bool
    comsol_model_path: Path | None

    @property
    def export_count(self) -> int:
        return sum(
            (
                self.has_domain_txt,
                self.has_inlet_txt,
                self.has_outlet_txt,
                self.has_wall_txt,
            )
        )

    @property
    def exports_ready(self) -> bool:
        return self.export_count == 4

    @property
    def can_extract(self) -> bool:
        return self.has_mesh and self.has_domain_txt

    @property
    def can_pull_from_comsol(self) -> bool:
        return self.has_mesh and self.has_comsol_model

    @property
    def already_extracted(self) -> bool:
        return self.has_biochem_graph


def _domain_export_stems(label_dir: Path) -> list[str]:
    if not label_dir.is_dir():
        return []
    stems: list[str] = []
    for p in sorted(label_dir.glob("*.txt")):
        stem = p.stem
        if any(stem.endswith(suf) for suf in _BOUNDARY_SUFFIXES):
            continue
        stems.append(stem)
    return stems


def _collect_stems(raw_dir: Path, label_dir: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for stem in stems_in_dir(raw_dir) + _domain_export_stems(label_dir):
        if stem not in seen:
            seen.add(stem)
            out.append(stem)
    return sorted(out)


def _status_for_stem(
    stem: str,
    *,
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path,
    kine_dir: Path,
) -> AnchorExtractStatus:
    mesh = (raw_dir / f"{stem}.nas").exists() or (raw_dir / f"{stem}.msh").exists()
    biochem_pt = proc_dir / f"{stem}.pt"
    kine_pt = kine_dir / f"{stem}.pt"
    mtime = biochem_pt.stat().st_mtime if biochem_pt.is_file() else None
    mph_path = resolve_biochem_comsol_model_path(stem)
    return AnchorExtractStatus(
        stem=stem,
        has_mesh=mesh,
        has_domain_txt=(label_dir / f"{stem}.txt").is_file(),
        has_inlet_txt=(label_dir / f"{stem}_inlet.txt").is_file(),
        has_outlet_txt=(label_dir / f"{stem}_outlet.txt").is_file(),
        has_wall_txt=(label_dir / f"{stem}_wall.txt").is_file(),
        has_biochem_graph=biochem_pt.is_file(),
        has_kine_graph=kine_pt.is_file(),
        biochem_graph_mtime=mtime,
        comsol_model_path=mph_path,
        has_comsol_model=mph_path is not None,
    )


def _fmt_mtime(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _exports_label(s: AnchorExtractStatus) -> str:
    if s.exports_ready:
        return "OK (4/4)"
    if s.export_count == 0:
        return "missing"
    return f"partial ({s.export_count}/4)"


def _row_tag(s: AnchorExtractStatus) -> str:
    if s.already_extracted and s.can_extract:
        return "[extracted]"
    if s.can_extract and s.exports_ready:
        return "[ready]"
    if s.can_extract:
        return "[ready*]"
    if s.has_domain_txt and not s.has_mesh:
        return "[no mesh]"
    if s.has_mesh and not s.has_domain_txt and s.has_comsol_model:
        return "[mph ready]"
    if s.has_mesh and not s.has_domain_txt:
        return "[no export]"
    return "[incomplete]"


def print_status_table(
    statuses: list[AnchorExtractStatus],
    *,
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path,
) -> None:
    print(f"\n[i] Mesh dir:    {raw_dir}")
    print(f"[i] COMSOL txt:  {label_dir}")
    print(f"[i] Graph out:   {proc_dir}\n")
    if not statuses:
        print("[WARN] No anchor stems found (need .msh/.nas and/or domain .txt exports).")
        return
    print(
        f"{'#':>3}  {'stem':<18}  {'tag':<14}  {'mesh':<5}  {'exports':<14}  "
        f"{'biochem .pt':<20}  {'kine':<6}  {'COMSOL .mph':<22}"
    )
    print("-" * 110)
    for i, s in enumerate(statuses, start=1):
        mesh = "yes" if s.has_mesh else "no"
        graph = "yes" if s.has_biochem_graph else "no"
        if s.has_biochem_graph:
            graph = f"yes {_fmt_mtime(s.biochem_graph_mtime)}"
        kine = "yes" if s.has_kine_graph else "no"
        mph = s.comsol_model_path.name if s.comsol_model_path else "-"
        print(
            f"{i:>3}  {s.stem:<18}  {_row_tag(s):<14}  {mesh:<5}  "
            f"{_exports_label(s):<14}  {graph:<20}  {kine:<6}  {mph:<22}"
        )
    print(
        "\n[i] [ready] = mesh + domain txt. [mph ready] = mesh + phase2_nowound_XXX.mph for patientXXX. "
        "[extracted] = biochem graph exists."
    )
    print(
        "[i] patient007 -> comsol_models/phase2_nowound_007.mph. Mesh/export stem mismatch:\n"
        "      python -m src.tools.prepare_biochem_anchors --strip-prefix-underscore"
    )


def _prompt_yes_no(label: str, *, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y or n.")


def _resolve_choice(raw: str, statuses: list[AnchorExtractStatus]) -> AnchorExtractStatus | None:
    text = raw.strip()
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(statuses):
            return statuses[idx - 1]
        print(f"  Index out of range (1-{len(statuses)}).")
        return None
    lower = text.lower()
    for s in statuses:
        if s.stem.lower() == lower:
            return s
    print(f"  Unknown stem: {text}")
    return None


def _maybe_pull_comsol(
    stem: str,
    extractor: PatientDataExtractor,
    *,
    from_comsol: bool,
    model_path: Path | None,
    force: bool,
) -> bool:
    domain_txt = extractor.label_dir / f"{stem}.txt"
    if domain_txt.is_file() and not force:
        return True
    if not from_comsol:
        if not domain_txt.is_file():
            print(
                f"[ERR] {stem}: missing {domain_txt.name}. "
                "Use --from-comsol after saving <stem>.mph, or export txt manually."
            )
        return domain_txt.is_file()

    resolved = resolve_biochem_comsol_model_path(stem, model_path)
    if resolved is None:
        print(
            f"[ERR] {stem}: no .mph found. Save solved model to "
            f"{extractor.raw_dir / f'{stem}.mph'} or set BIOCHEM_COMSOL_MODEL."
        )
        return False

    print(f"[NEW] Pulling COMSOL fields from {resolved} ...")
    try:
        extractor.pull_comsol_exports(stem, model_path=resolved, force=force)
    except Exception as exc:
        print(f"[ERR] COMSOL pull failed for {stem}: {exc}")
        return False
    return domain_txt.is_file() or (extractor.label_dir / f"{stem}.txt").is_file()


def _run_extract(
    stem: str,
    extractor: PatientDataExtractor,
    *,
    force: bool,
    skip_enrich: bool,
    raw_dir: Path,
    from_comsol: bool,
    model_path: Path | None,
) -> bool:
    if not _maybe_pull_comsol(
        stem, extractor, from_comsol=from_comsol, model_path=model_path, force=force
    ):
        return False

    biochem_pt = extractor.proc_dir / f"{stem}.pt"
    if biochem_pt.is_file() and not force:
        print(
            f"[WARN] {stem}: graph already exists ({biochem_pt}). "
            "Use --force or answer y to overwrite."
        )
        if not _prompt_yes_no(f"Overwrite {stem}.pt?", default=False):
            print("[skip] extraction cancelled.")
            return False

    if not skip_enrich:
        enrich_anchor_meshes(raw_dir, overwrite=False, dry_run=False, stems=[stem])

    print(f"\n[NEW] Extracting {stem} ...")
    extractor.process_patient(stem)
    if biochem_pt.is_file():
        print(f"[OK] Wrote {biochem_pt}")
        kine_pt = extractor.kine_anchor_dir / f"{stem}.pt"
        print(f"[OK] Kinematics anchor (if steady step ran): {kine_pt}")
        return True
    print(f"[ERR] Extraction did not produce {biochem_pt} (see messages above).")
    return False


def _interactive_loop(
    statuses: list[AnchorExtractStatus],
    extractor: PatientDataExtractor,
    *,
    force: bool,
    skip_enrich: bool,
    raw_dir: Path,
    from_comsol: bool,
    model_path: Path | None,
) -> None:
    ready = [
        s
        for s in statuses
        if not s.already_extracted
        and (s.can_extract or (from_comsol and s.can_pull_from_comsol))
    ]
    print(f"\n[i] {len(ready)} stem(s) ready to extract (not yet graphed).")
    print("[i] Enter stem name, list index, 'l' to relist, 'q' to quit.\n")

    while True:
        raw = input("Extract which anchor? ").strip()
        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            break
        if raw.lower() in ("l", "list"):
            print_status_table(
                statuses,
                raw_dir=raw_dir,
                label_dir=extractor.label_dir,
                proc_dir=extractor.proc_dir,
            )
            continue

        picked = _resolve_choice(raw, statuses)
        if picked is None:
            continue

        can_run = picked.can_extract or (from_comsol and picked.can_pull_from_comsol)
        if not can_run:
            print(
                f"[ERR] {picked.stem}: need mesh in {raw_dir} and either domain .txt or a saved .mph "
                f"(--from-comsol)."
            )
            if picked.has_mesh and picked.export_count < 4:
                missing = []
                if not picked.has_inlet_txt:
                    missing.append("inlet")
                if not picked.has_outlet_txt:
                    missing.append("outlet")
                if not picked.has_wall_txt:
                    missing.append("wall")
                if missing:
                    print(
                        f"       Missing boundary exports: {', '.join(missing)} "
                        "(extraction may still fail BC mapping)."
                    )
            continue

        if not picked.exports_ready:
            print(
                f"[WARN] {picked.stem}: only {picked.export_count}/4 COMSOL txt files present; "
                "continuing anyway (domain txt is required)."
            )
            if not _prompt_yes_no("Continue?", default=False):
                continue

        ok = _run_extract(
            picked.stem,
            extractor,
            force=force,
            skip_enrich=skip_enrich,
            raw_dir=raw_dir,
            from_comsol=from_comsol,
            model_path=model_path,
        )
        if ok and not _prompt_yes_no("Extract another?", default=True):
            break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stem", type=str, default="", help="Extract this stem only (non-interactive pick).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .pt without prompting.")
    parser.add_argument("--list-only", action="store_true", help="Print status table and exit.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Anchor meshes (default: data/raw/biochem_anchors).")
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=None,
        help="COMSOL exports (default: data/processed/cfd_results_biochem).",
    )
    parser.add_argument(
        "--skip-sidecars",
        action="store_true",
        help="Skip automatic cm sidecar scaffold on raw meshes.",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip Gmsh sidecar enrichment before extract.",
    )
    parser.add_argument(
        "--from-comsol",
        action="store_true",
        help="Pull domain/boundary txt from a solved .mph via mph before graph extract.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit path to solved .mph (default: <stem>.mph in biochem_anchors or BIOCHEM_COMSOL_MODEL).",
    )
    args = parser.parse_args(argv)

    dr = data_root()
    raw_dir = args.raw_dir or (dr / "raw" / "biochem_anchors")
    label_dir = args.label_dir or (dr / "processed" / "cfd_results_biochem")

    if not raw_dir.is_dir() and not label_dir.is_dir():
        raise SystemExit(
            f"[ERR] Neither mesh dir nor export dir exists.\n"
            f"  meshes: {raw_dir}\n  exports: {label_dir}"
        )

    if not args.skip_sidecars and raw_dir.is_dir():
        _auto_scaffold_anchor_sidecars(raw_dir)

    extractor = PatientDataExtractor(phase="biochem_anchors", raw_dir=raw_dir, label_dir=label_dir)
    stems = _collect_stems(raw_dir, label_dir)
    statuses = [
        _status_for_stem(
            stem,
            raw_dir=raw_dir,
            label_dir=label_dir,
            proc_dir=extractor.proc_dir,
            kine_dir=extractor.kine_anchor_dir,
        )
        for stem in stems
    ]

    print_status_table(statuses, raw_dir=raw_dir, label_dir=label_dir, proc_dir=extractor.proc_dir)

    if args.list_only:
        return

    stem_arg = args.stem.strip()
    if stem_arg:
        picked = next((s for s in statuses if s.stem == stem_arg), None)
        if picked is None:
            picked = _status_for_stem(
                stem_arg,
                raw_dir=raw_dir,
                label_dir=label_dir,
                proc_dir=extractor.proc_dir,
                kine_dir=extractor.kine_anchor_dir,
            )
            statuses.append(picked)
        can_run = picked.can_extract or (args.from_comsol and picked.can_pull_from_comsol)
        if not can_run:
            raise SystemExit(f"[ERR] {stem_arg}: missing mesh and/or COMSOL source (see table above).")
        ok = _run_extract(
            picked.stem,
            extractor,
            force=args.force,
            skip_enrich=args.skip_enrich,
            raw_dir=raw_dir,
            from_comsol=args.from_comsol,
            model_path=args.model_path,
        )
        raise SystemExit(0 if ok else 1)

    _interactive_loop(
        statuses,
        extractor,
        force=args.force,
        skip_enrich=args.skip_enrich,
        raw_dir=raw_dir,
        from_comsol=args.from_comsol,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
