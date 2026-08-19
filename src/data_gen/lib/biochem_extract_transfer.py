"""One-folder transfer packs for biochem COMSOL extracts.

Canonical training paths stay split (``raw/biochem_anchors``, ``cfd_results_biochem``,
``graphs_biochem_anchors``). After a successful extract we also mirror the artifacts
into ``data/extract_transfer/<stem>/``.

Default pack is *lite* (graph + mesh sidecar + wound.txt). Do not copy ``.mph`` files for
graph work — they dominate Drive time and are unused on the analysis laptop.

On the COMSOL PC::

    python -m src.tools.extract_biochem_comsol --pack-transfer --zip-transfer --stem wound_patient001,wound_patient002

Upload ``data/extract_transfer.zip`` (or the ``extract_transfer`` folder). On this
laptop leave the Drive download in ``Downloads``, then::

    python -m src.tools.extract_biochem_comsol --install-bundles
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.data_gen.lib.biochem_comsol_auto_export import parse_biochem_extract_stem
from src.utils.paths import data_root, get_project_root

MANIFEST_NAME = "manifest.json"
# Drive/laptop copies: graph + small mesh + wound identity. Skip .mph, .nas, domain txt, kine.
_LITE_NAMES = frozenset(
    {"graph.pt", "graph_metadata.json", "mesh.msh", "mesh.json", "wound.txt"}
)

_BUNDLE_KEYS: tuple[tuple[str, str, bool], ...] = (
    # (bundle filename, dest relative to repo root with {stem}, required)
    ("graph.pt", "data/processed/graphs_biochem_anchors/{stem}.pt", True),
    ("graph_metadata.json", "data/processed/graphs_biochem_anchors/{stem}_metadata.json", False),
    ("mesh.msh", "data/raw/biochem_anchors/{stem}.msh", False),
    ("mesh.nas", "data/raw/biochem_anchors/{stem}.nas", False),
    ("mesh.json", "data/raw/biochem_anchors/{stem}.json", False),
    ("domain.txt", "data/processed/cfd_results_biochem/{stem}.txt", False),
    ("inlet.txt", "data/processed/cfd_results_biochem/{stem}_inlet.txt", False),
    ("outlet.txt", "data/processed/cfd_results_biochem/{stem}_outlet.txt", False),
    ("wall.txt", "data/processed/cfd_results_biochem/{stem}_wall.txt", False),
    ("wound.txt", "data/processed/cfd_results_biochem/{stem}_wound.txt", False),
    ("kine.pt", "data/processed/graphs_kinematics_anchors/carreau/{stem}.pt", False),
)


def extract_transfer_dir(*, root: Path | None = None) -> Path:
    """``data/extract_transfer`` — the single folder to copy between machines."""
    if root is None:
        return data_root() / "extract_transfer"
    return Path(root) / "data" / "extract_transfer"


def extract_transfer_zip_path(*, root: Path | None = None, transfer_dir: Path | None = None) -> Path:
    """``data/extract_transfer.zip`` next to the transfer folder."""
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    return base.with_suffix(".zip")


def default_downloads_dir() -> Path:
    """User Downloads folder (Windows ``~/Downloads``, with OneDrive fallback)."""
    candidates: list[Path] = [Path.home() / "Downloads"]
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / "Downloads")
    onedrive = os.environ.get("OneDrive")
    if onedrive:
        candidates.append(Path(onedrive) / "Downloads")
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


@dataclass(frozen=True)
class IncomingTransfer:
    """Resolved install source: a folder of stem bundles, or a zip to unpack."""

    transfer_dir: Path | None
    zip_path: Path | None
    label: str


def _is_bundle_dir(path: Path) -> bool:
    return path.is_dir() and (path / MANIFEST_NAME).is_file()


def normalize_bundle_root(path: Path) -> Path:
    """Unwrap ``extract_transfer/`` or a single nested folder down to the stem parent."""
    path = Path(path)
    if not path.is_dir():
        return path
    if list_transfer_bundles(transfer_dir=path):
        return path
    nested = path / "extract_transfer"
    if nested.is_dir() and list_transfer_bundles(transfer_dir=nested):
        return nested
    subdirs = [p for p in sorted(path.iterdir()) if p.is_dir() and p.name not in {"__MACOSX"}]
    hits = [p for p in subdirs if list_transfer_bundles(transfer_dir=p)]
    if len(hits) == 1:
        return hits[0]
    if len(subdirs) == 1:
        return normalize_bundle_root(subdirs[0])
    return path


def _newest(paths: Iterable[Path]) -> Path:
    ranked = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return ranked[0]


def discover_downloads_transfer(downloads_dir: Path) -> IncomingTransfer | None:
    """Find one transfer folder or ``extract_transfer*.zip`` in Downloads."""
    downloads_dir = Path(downloads_dir)
    if not downloads_dir.is_dir():
        return None

    named = downloads_dir / "extract_transfer"
    if named.is_dir():
        root = normalize_bundle_root(named)
        if list_transfer_bundles(transfer_dir=root):
            return IncomingTransfer(root, None, str(root))

    wrappers: list[Path] = []
    direct_bundles: list[Path] = []
    for child in downloads_dir.iterdir():
        if not child.is_dir() or child.name in {"__MACOSX"}:
            continue
        if list_transfer_bundles(transfer_dir=child) or list_transfer_bundles(
            transfer_dir=normalize_bundle_root(child)
        ):
            wrappers.append(normalize_bundle_root(child))
        elif _is_bundle_dir(child):
            direct_bundles.append(child)

    if wrappers:
        chosen = _newest(wrappers)
        return IncomingTransfer(chosen, None, str(chosen))
    if direct_bundles:
        return IncomingTransfer(downloads_dir, None, str(downloads_dir))

    zips = [
        p
        for p in downloads_dir.glob("extract_transfer*.zip")
        if p.is_file()
    ]
    if not zips:
        zips = [p for p in downloads_dir.glob("*extract_transfer*.zip") if p.is_file()]
    if zips:
        archive = _newest(zips)
        return IncomingTransfer(None, archive, str(archive))
    return None


def resolve_incoming_transfer(
    *,
    transfer_dir: Path | None = None,
    downloads_dir: Path | None = None,
    data_transfer_dir: Path | None = None,
) -> IncomingTransfer | None:
    """Prefer ``--transfer-dir``, then Downloads, then ``data/extract_transfer``."""
    if transfer_dir is not None:
        path = Path(transfer_dir)
        if path.is_file() and path.suffix.lower() == ".zip":
            return IncomingTransfer(None, path, str(path))
        if path.is_dir():
            root = normalize_bundle_root(path)
            return IncomingTransfer(root, None, str(root))
        return None

    found = discover_downloads_transfer(downloads_dir or default_downloads_dir())
    if found is not None:
        return found

    data_dir = Path(data_transfer_dir) if data_transfer_dir is not None else extract_transfer_dir()
    if list_transfer_bundles(transfer_dir=data_dir):
        return IncomingTransfer(data_dir, None, str(data_dir))
    archive = data_dir.with_suffix(".zip")
    if archive.is_file():
        return IncomingTransfer(None, archive, str(archive))
    return None


def unpack_incoming_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Unpack a Drive zip and return the folder that contains stem bundles."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), dest_dir)
    return normalize_bundle_root(dest_dir)


def bundle_dir_for_stem(stem: str, *, root: Path | None = None) -> Path:
    ref = parse_biochem_extract_stem(stem)
    canonical = ref.stem if ref is not None else stem
    return extract_transfer_dir(root=root) / canonical


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def stage_extract_transfer_bundle(
    stem: str,
    *,
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path,
    kine_dir: Path | None = None,
    root: Path | None = None,
    lite: bool = True,
) -> Path | None:
    """Mirror one extracted stem into ``data/extract_transfer/<stem>/``. Returns bundle dir or None.

    ``lite=True`` (default) is the Drive pack: graph + mesh sidecar + wound.txt,
    no ``.mph`` / ``.nas`` / domain txt.
    """
    ref = parse_biochem_extract_stem(stem)
    canonical = ref.stem if ref is not None else stem
    graph_src = Path(proc_dir) / f"{canonical}.pt"
    if not graph_src.is_file():
        return None

    bundle = bundle_dir_for_stem(canonical, root=root)
    bundle.mkdir(parents=True, exist_ok=True)

    sources = {
        "graph.pt": graph_src,
        "graph_metadata.json": Path(proc_dir) / f"{canonical}_metadata.json",
        "mesh.msh": Path(raw_dir) / f"{canonical}.msh",
        "mesh.nas": Path(raw_dir) / f"{canonical}.nas",
        "mesh.json": Path(raw_dir) / f"{canonical}.json",
        "domain.txt": Path(label_dir) / f"{canonical}.txt",
        "inlet.txt": Path(label_dir) / f"{canonical}_inlet.txt",
        "outlet.txt": Path(label_dir) / f"{canonical}_outlet.txt",
        "wall.txt": Path(label_dir) / f"{canonical}_wall.txt",
        "wound.txt": Path(label_dir) / f"{canonical}_wound.txt",
        "kine.pt": (Path(kine_dir) / f"{canonical}.pt") if kine_dir is not None else None,
    }
    want = set(sources)
    if lite:
        want = set(_LITE_NAMES)
        msh = sources["mesh.msh"]
        if msh is None or not Path(msh).is_file():
            want.add("mesh.nas")
    packed: list[str] = []
    for name, src in sources.items():
        if src is None or name not in want:
            continue
        if _copy_if_exists(src, bundle / name):
            packed.append(name)

    keep = set(packed) | {MANIFEST_NAME}
    for leftover in bundle.iterdir():
        if leftover.is_file() and leftover.name not in keep:
            leftover.unlink()

    files = {
        name: rel.format(stem=canonical)
        for name, rel, _required in _BUNDLE_KEYS
        if name in packed
    }
    manifest = {
        "stem": canonical,
        "variant": ref.variant if ref is not None else "unknown",
        "lite": lite,
        "files": files,
    }
    (bundle / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return bundle


def list_transfer_bundles(*, root: Path | None = None, transfer_dir: Path | None = None) -> list[Path]:
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    if not base.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / MANIFEST_NAME).is_file():
            out.append(child)
    return out


def install_extract_transfer_bundle(
    bundle_dir: Path,
    *,
    root: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Copy one transfer folder into canonical ``data/`` paths. Returns ``{bundle_name: dest}``."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {bundle_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stem = str(manifest.get("stem") or bundle_dir.name)
    repo = Path(root) if root is not None else get_project_root()
    written: dict[str, str] = {}
    files = manifest.get("files") or {}
    for name, rel, required in _BUNDLE_KEYS:
        src = bundle_dir / name
        dest_rel = files.get(name) or rel.format(stem=stem)
        dest = repo / dest_rel
        if not src.is_file():
            if required:
                raise FileNotFoundError(f"{bundle_dir.name}: missing required {name}")
            continue
        if dest.is_file() and not force:
            written[name] = f"skip {dest_rel}"
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written[name] = dest_rel
    return written


def install_all_extract_transfer_bundles(
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
    stems: Iterable[str] | None = None,
    force: bool = False,
) -> list[tuple[str, dict[str, str]]]:
    """Install every (or selected) bundle under the transfer folder."""
    want: set[str] | None = None
    if stems is not None:
        want = set()
        for raw in stems:
            ref = parse_biochem_extract_stem(raw)
            want.add(ref.stem if ref is not None else raw)
    results: list[tuple[str, dict[str, str]]] = []
    for bundle in list_transfer_bundles(root=root, transfer_dir=transfer_dir):
        if want is not None and bundle.name not in want:
            continue
        results.append(
            (
                bundle.name,
                install_extract_transfer_bundle(bundle, root=root, force=force),
            )
        )
    return results


def install_incoming_extract_transfer(
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
    downloads_dir: Path | None = None,
    data_transfer_dir: Path | None = None,
    stems: Iterable[str] | None = None,
    force: bool = False,
) -> tuple[IncomingTransfer | None, list[tuple[str, dict[str, str]]]]:
    """Discover a Downloads folder/zip (or explicit path), then install stem bundles."""
    incoming = resolve_incoming_transfer(
        transfer_dir=transfer_dir,
        downloads_dir=downloads_dir,
        data_transfer_dir=data_transfer_dir,
    )
    if incoming is None:
        return None, []
    if incoming.zip_path is not None:
        with tempfile.TemporaryDirectory(prefix="extract_transfer_") as tmp:
            bundle_root = unpack_incoming_zip(incoming.zip_path, Path(tmp))
            results = install_all_extract_transfer_bundles(
                root=root,
                transfer_dir=bundle_root,
                stems=stems,
                force=force,
            )
            return incoming, results
    results = install_all_extract_transfer_bundles(
        root=root,
        transfer_dir=incoming.transfer_dir,
        stems=stems,
        force=force,
    )
    return incoming, results


def zip_extract_transfer_dir(
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
    dest_zip: Path | None = None,
) -> Path:
    """Write one zip of ``data/extract_transfer`` for a single Drive upload."""
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    if not base.is_dir():
        raise FileNotFoundError(f"No transfer folder at {base}")
    if dest_zip is None:
        dest_zip = base.with_suffix(".zip")
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists():
        dest_zip.unlink()
    archive = shutil.make_archive(
        str(dest_zip.with_suffix("")),
        "zip",
        root_dir=base.parent,
        base_dir=base.name,
    )
    return Path(archive)


def unpack_extract_transfer_zip(
    zip_path: Path | None = None,
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
) -> Path:
    """Unpack ``extract_transfer.zip`` so ``data/extract_transfer/<stem>/`` exists."""
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    archive = Path(zip_path) if zip_path is not None else base.with_suffix(".zip")
    if not archive.is_file():
        raise FileNotFoundError(f"No transfer zip at {archive}")
    shutil.unpack_archive(archive, base.parent)
    return base
