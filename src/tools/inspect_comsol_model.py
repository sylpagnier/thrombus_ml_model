"""
Live COMSOL model inspector (restored + modernized).

Examples:
    python -m src.tools.inspect_comsol_model --list-models
    python -m src.tools.inspect_comsol_model --model comsol_models/phase1_template.mph
    python -m src.tools.inspect_comsol_model --all-models --show-properties
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from src.utils.paths import comsol_models_dir


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        return [str(v) for v in list(value)]
    except Exception:
        try:
            return [str(v) for v in value]
        except Exception:
            return []


def _discover_models() -> list[Path]:
    model_dir = comsol_models_dir()
    if not model_dir.exists():
        return []
    return sorted(p for p in model_dir.glob("*.mph") if p.is_file())


def _resolve_model_path(model_arg: str | None) -> Path:
    models = _discover_models()
    if model_arg:
        p = Path(model_arg)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    if not models:
        raise FileNotFoundError(f"No .mph files found in {comsol_models_dir()}")
    return models[0]


def _print_parameters(model) -> None:
    print("\n=== PARAMETERS (global) ===")
    root = _safe_call(lambda: model.java.param(), None)
    names = _to_list(_safe_call(lambda: root.varnames(), [])) if root is not None else []
    print(f"Count: {len(names)}")
    for name in names:
        expr = _safe_call(lambda: str(root.get(name)), "<unavailable>")
        unit = _safe_call(lambda: str(root.evaluateUnit(name)), "")
        desc = _safe_call(lambda: str(root.descr(name)), "")
        unit_s = f" [{unit}]" if unit else ""
        desc_s = f" :: {desc}" if desc else ""
        print(f"- {name} = {expr}{unit_s}{desc_s}")


def _print_variables(model) -> None:
    print("\n=== VARIABLES (global + component) ===")
    g_root = _safe_call(lambda: model.java.variable(), None)
    g_ids = _to_list(_safe_call(lambda: g_root.tags(), [])) if g_root is not None else []
    print(f"Global variable groups: {len(g_ids)}")
    for group_id in g_ids:
        group = _safe_call(lambda: g_root(group_id), None)
        names = _to_list(_safe_call(lambda: group.varnames(), [])) if group is not None else []
        print(f"  - {group_id}: {len(names)} vars")

    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        v_root = _safe_call(lambda: comp.variable(), None)
        v_ids = _to_list(_safe_call(lambda: v_root.tags(), [])) if v_root is not None else []
        print(f"Component {comp_id} variable groups: {len(v_ids)}")
        for group_id in v_ids:
            group = _safe_call(lambda: v_root(group_id), None)
            names = _to_list(_safe_call(lambda: group.varnames(), [])) if group is not None else []
            print(f"  - {group_id}: {len(names)} vars")


def _print_functions(func_root, scope_name: str) -> None:
    print(f"\n=== FUNCTIONS ({scope_name}) ===")
    f_ids = _to_list(_safe_call(lambda: func_root.tags(), [])) if func_root is not None else []
    print(f"Count: {len(f_ids)}")
    for fid in f_ids:
        f_node = _safe_call(lambda: func_root.get(fid), None)
        if f_node is None:
            continue
        f_name = _safe_call(lambda: str(f_node.name()), fid)
        f_type = _safe_call(lambda: str(f_node.getType()), "<unknown>")
        expr = _safe_call(lambda: str(f_node.getString("expr")), "")
        if expr:
            print(f"- {f_name} [{f_type}] :: {expr}")
        else:
            print(f"- {f_name} [{f_type}]")


def _iter_feature_properties(feat) -> Iterable[tuple[str, str]]:
    props = _to_list(_safe_call(lambda: feat.properties(), []))
    for key in props:
        val = _safe_call(lambda: str(feat.getString(key)), "")
        if val and val != "[]":
            yield key, val


def _print_model_structure(model, show_properties: bool = False) -> None:
    print("\n=== MODEL STRUCTURE ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    if not comp_ids:
        print("No components found.")
        return

    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue

        print(f"\nComponent: {comp_id}")
        mesh_ids = _to_list(_safe_call(lambda: comp.mesh().tags(), []))
        print(f"  Meshes: {len(mesh_ids)}")
        for mesh_id in mesh_ids:
            mesh = _safe_call(lambda: comp.mesh(mesh_id), None)
            feat_ids = _to_list(_safe_call(lambda: mesh.feature().tags(), [])) if mesh is not None else []
            print(f"    - {mesh_id}: {len(feat_ids)} features")

        phys_ids = _to_list(_safe_call(lambda: comp.physics().tags(), []))
        print(f"  Physics interfaces: {len(phys_ids)}")
        for phys_id in phys_ids:
            phys = _safe_call(lambda: comp.physics(phys_id), None)
            phys_type = _safe_call(lambda: str(phys.getType()), "<unknown>") if phys is not None else "<unknown>"
            print(f"    - {phys_id} ({phys_type})")
            feat_ids = _to_list(_safe_call(lambda: phys.feature().tags(), []))
            for feat_id in feat_ids:
                feat = _safe_call(lambda: phys.feature(feat_id), None)
                if feat is None:
                    continue
                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>")
                print(f"      * {feat_id} ({feat_type})")
                if show_properties:
                    for k, v in _iter_feature_properties(feat):
                        print(f"        -> {k}: {v}")

        mat_ids = _to_list(_safe_call(lambda: comp.material().tags(), []))
        print(f"  Materials: {len(mat_ids)}")
        for mat_id in mat_ids:
            print(f"    - {mat_id}")

        _print_functions(_safe_call(lambda: comp.func(), None), f"Component {comp_id}")


def _print_studies(model) -> None:
    print("\n=== STUDIES ===")
    s_root = _safe_call(lambda: model.java.study(), None)
    s_ids = _to_list(_safe_call(lambda: s_root.tags(), [])) if s_root is not None else []
    for sid in s_ids:
        study = _safe_call(lambda: s_root.get(sid), None)
        if study is None:
            continue
        print(f"- {sid}")
        feat_ids = _to_list(_safe_call(lambda: study.feature().tags(), []))
        for fid in feat_ids:
            feat = _safe_call(lambda: study.feature(fid), None)
            ftype = _safe_call(lambda: str(feat.getType()), "<unknown>") if feat is not None else "<unknown>"
            print(f"  * {fid} ({ftype})")


def inspect_model(model_path: Path, show_properties: bool = False) -> None:
    try:
        import mph
    except Exception as exc:
        raise RuntimeError(
            "Failed to import `mph`. Install package `mph` and ensure COMSOL + LiveLink are available."
        ) from exc

    print(f"\nInspecting COMSOL model: {model_path}")
    client = mph.start()
    model = client.load(str(model_path))
    try:
        _print_model_structure(model, show_properties=show_properties)
        _print_parameters(model)
        _print_variables(model)
        _print_functions(_safe_call(lambda: model.java.func(), None), "Global")
        _print_studies(model)
    finally:
        _safe_call(lambda: model.remove())
        _safe_call(lambda: client.clear())
        _safe_call(lambda: client.disconnect())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect COMSOL .mph internals via mph/LiveLink.")
    parser.add_argument("--model", type=str, default=None, help="Path to .mph model (defaults to first in comsol_models/).")
    parser.add_argument("--all-models", action="store_true", help="Inspect all .mph files under comsol_models/.")
    parser.add_argument("--list-models", action="store_true", help="List discoverable .mph models and exit.")
    parser.add_argument(
        "--show-properties",
        action="store_true",
        help="Print physics feature property key/value pairs (verbose).",
    )
    args = parser.parse_args()

    models = _discover_models()
    if args.list_models:
        if not models:
            print(f"No .mph files found in {comsol_models_dir()}")
            return
        print("Discovered models:")
        for m in models:
            print(f"- {m}")
        return

    if args.all_models:
        if not models:
            raise FileNotFoundError(f"No .mph files found in {comsol_models_dir()}")
        for m in models:
            inspect_model(m, show_properties=args.show_properties)
        return

    selected = _resolve_model_path(args.model)
    inspect_model(selected, show_properties=args.show_properties)


if __name__ == "__main__":
    main()
