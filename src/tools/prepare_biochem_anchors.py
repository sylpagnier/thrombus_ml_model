"""Prepare an anchor mesh folder for ``PatientDataExtractor``.

Idempotent CLI that operates on a directory of ``.msh`` / ``.nas`` anchor
meshes (default: ``data/raw/biochem_anchors``) and performs two jobs:

1. **Sidecar scaffolding.** For every mesh stem that has no ``<stem>.json``
   sidecar, write a minimal one declaring ``{"unit": <unit>}``. If a sidecar
   already exists with a *different* ``unit`` value, the file is left alone
   and the conflict is reported so that ``assert_mesh_unit`` can do its job.
   If a sidecar exists with no ``unit`` field, add it (other fields are
   preserved). Use ``--force`` to overwrite a conflicting unit declaration.

2. **Stem-prefix underscore collapse** (optional, ``--strip-prefix-underscore``).
   Rename ``<alpha>_<digits>.<ext>`` to ``<alpha><digits>.<ext>`` for every
   mesh and its matching sidecar. This is the convention used by COMSOL
   text exports (``patient001.txt``), so collapsing the underscore lets
   ``PatientDataExtractor`` find the matching exports without renaming the
   12+ ``.txt`` files.

Both operations support ``--dry-run`` and print a clear summary. The tool is
safe to re-run: each step is a no-op if its target state already holds.

Examples::

    python -m src.tools.prepare_biochem_anchors --strip-prefix-underscore
    python -m src.tools.prepare_biochem_anchors --dir path/to/meshes --unit cm
    python -m src.tools.prepare_biochem_anchors --dry-run --strip-prefix-underscore
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from src.utils.paths import data_root
from src.utils.units import SUPPORTED_MESH_UNITS


_MESH_EXTS = (".msh", ".nas")
_PREFIX_UNDERSCORE_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<digits>\d+)$")


def _iter_mesh_files(root: Path) -> list[Path]:
    """All mesh files in ``root`` (non-recursive), sorted for determinism."""
    out: list[Path] = []
    for ext in _MESH_EXTS:
        out.extend(sorted(root.glob(f"*{ext}")))
    return out


def stems_in_dir(root: Path) -> list[str]:
    """Unique mesh stems in ``root`` (``.msh`` and ``.nas`` collapsed)."""
    seen: set[str] = set()
    stems: list[str] = []
    for p in _iter_mesh_files(root):
        if p.stem not in seen:
            seen.add(p.stem)
            stems.append(p.stem)
    return stems


def collapse_prefix_underscore(stem: str) -> str | None:
    """Return the collapsed stem (``patient_001`` -> ``patient001``) or ``None`` if unchanged."""
    m = _PREFIX_UNDERSCORE_RE.match(stem)
    if not m:
        return None
    new = f"{m.group('prefix')}{m.group('digits')}"
    return new if new != stem else None


# Backwards-compatible internal aliases (kept until call sites migrate).
_stems_in_dir = stems_in_dir
_collapse_prefix_underscore = collapse_prefix_underscore


def _files_for_stem(root: Path, stem: str) -> list[Path]:
    return [
        p
        for ext in (*_MESH_EXTS, ".json")
        for p in [root / f"{stem}{ext}"]
        if p.exists()
    ]


def _rename_collapse_underscore(
    root: Path, *, dry_run: bool
) -> tuple[int, int]:
    """Rename ``<alpha>_<digits>.*`` to ``<alpha><digits>.*`` for each stem in ``root``.

    Returns ``(renamed_files, skipped_stems_due_to_collision)``.
    """
    renamed = 0
    skipped = 0
    for stem in _stems_in_dir(root):
        new_stem = _collapse_prefix_underscore(stem)
        if new_stem is None:
            continue
        sources = _files_for_stem(root, stem)
        targets = [(p, root / f"{new_stem}{p.suffix}") for p in sources]
        collisions = [t for _, t in targets if t.exists()]
        if collisions:
            print(
                f"  [skip] {stem} -> {new_stem}: target already exists: "
                + ", ".join(str(c.name) for c in collisions)
            )
            skipped += 1
            continue
        for src, dst in targets:
            print(f"  [rename] {src.name} -> {dst.name}")
            if not dry_run:
                src.rename(dst)
            renamed += 1
    return renamed, skipped


def _read_sidecar(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  [warn] {path.name} is not valid JSON ({exc}); leaving alone.")
        return None


def _write_sidecar(path: Path, payload: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def scaffold_anchor_sidecars(
    root: Path,
    *,
    unit: str = "cm",
    force: bool = False,
    dry_run: bool = False,
    stems: Iterable[str] | None = None,
) -> tuple[int, int, int]:
    """Ensure every mesh stem in ``root`` has a sidecar declaring ``unit``.

    Idempotent. Existing sidecars matching ``unit`` are left untouched. A
    sidecar without a ``unit`` field is augmented in place. A sidecar declaring
    a *different* unit is reported as a conflict and only overwritten when
    ``force=True``.

    ``stems`` overrides the default disk scan. Pass the post-rename stems
    during a dry-run so the preview is accurate even though no files have
    moved yet.

    Returns ``(created, updated, conflicts)``.
    """
    created = 0
    updated = 0
    conflicts = 0
    iter_stems = list(stems) if stems is not None else _stems_in_dir(root)
    for stem in iter_stems:
        sidecar = root / f"{stem}.json"
        existing = _read_sidecar(sidecar)
        if existing is None and sidecar.exists():
            # Unparseable JSON: don't touch.
            conflicts += 1
            continue
        if existing is None:
            print(f"  [create] {sidecar.name} (unit={unit!r})")
            _write_sidecar(sidecar, {"unit": unit}, dry_run=dry_run)
            created += 1
            continue

        existing_unit = existing.get("unit")
        if existing_unit is None:
            new_payload = dict(existing)
            new_payload["unit"] = unit
            print(f"  [augment] {sidecar.name}: add unit={unit!r}")
            _write_sidecar(sidecar, new_payload, dry_run=dry_run)
            updated += 1
            continue

        if str(existing_unit).lower() == unit:
            continue

        if force:
            new_payload = dict(existing)
            new_payload["unit"] = unit
            print(
                f"  [overwrite] {sidecar.name}: unit={existing_unit!r} -> {unit!r} (--force)"
            )
            _write_sidecar(sidecar, new_payload, dry_run=dry_run)
            updated += 1
        else:
            print(
                f"  [conflict] {sidecar.name} declares unit={existing_unit!r}, "
                f"expected {unit!r}; pass --force to overwrite."
            )
            conflicts += 1
    return created, updated, conflicts


_scaffold_sidecars = scaffold_anchor_sidecars


def _default_anchor_dir() -> Path:
    return data_root() / "raw" / "biochem_anchors"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"Directory of anchor meshes (default: {_default_anchor_dir()}).",
    )
    p.add_argument(
        "--unit",
        type=str,
        default="cm",
        choices=SUPPORTED_MESH_UNITS,
        help="Length unit declared in each sidecar (default: cm).",
    )
    p.add_argument(
        "--strip-prefix-underscore",
        action="store_true",
        help="Rename ``<alpha>_<digits>.<ext>`` to ``<alpha><digits>.<ext>`` "
        "(matches COMSOL export stems like ``patient001.txt``).",
    )
    p.add_argument(
        "--no-sidecars",
        action="store_true",
        help="Skip the JSON sidecar scaffolding step.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing sidecar that declares a different unit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without modifying any files.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    target_dir = (args.dir if args.dir is not None else _default_anchor_dir()).resolve()
    if not target_dir.is_dir():
        print(f"ERROR: directory does not exist: {target_dir}", file=sys.stderr)
        return 2

    print(f"Anchor mesh dir: {target_dir}")
    print(f"Unit: {args.unit}    dry-run: {args.dry_run}")

    meshes = _iter_mesh_files(target_dir)
    if not meshes:
        print("No .msh / .nas files found; nothing to do.")
        return 0
    print(f"Discovered {len(meshes)} mesh file(s) across {len(_stems_in_dir(target_dir))} stem(s).")

    if args.strip_prefix_underscore:
        print("\n--- Renaming <alpha>_<digits>.* -> <alpha><digits>.* ---")
        renamed, skipped = _rename_collapse_underscore(target_dir, dry_run=args.dry_run)
        print(f"  renamed {renamed} file(s); {skipped} stem(s) skipped due to collision.")

    if not args.no_sidecars:
        print("\n--- Scaffolding sidecar JSONs ---")
        if args.dry_run and args.strip_prefix_underscore:
            stems_after_rename = [
                _collapse_prefix_underscore(s) or s for s in _stems_in_dir(target_dir)
            ]
        else:
            stems_after_rename = None
        created, updated, conflicts = _scaffold_sidecars(
            target_dir,
            unit=args.unit,
            force=args.force,
            dry_run=args.dry_run,
            stems=stems_after_rename,
        )
        print(f"  created {created} sidecar(s); updated {updated}; conflicts {conflicts}.")
        if conflicts and not args.force:
            print("Re-run with --force to overwrite conflicting sidecar units.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
