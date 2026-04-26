"""
Live COMSOL model inspector (restored + modernized).

Examples:
    python -m src.tools.inspect_comsol_model --list-models
    python -m src.tools.inspect_comsol_model --model comsol_models/kinematics_template.mph
    python -m src.tools.inspect_comsol_model --all-models --show-properties
    python -m src.tools.inspect_comsol_model   # interactive model picker
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


def _prompt_int_choice(label: str, min_value: int, max_value: int) -> int:
    """Read an integer from stdin until it falls in [min_value, max_value]."""
    while True:
        raw = input(f"{label} [{min_value}-{max_value}]: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("  Enter an integer.")
            continue
        if min_value <= value <= max_value:
            return value
        print(f"  Must be between {min_value} and {max_value}.")


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    """Read a yes/no answer; empty input returns default."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{label} {suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y/yes or n/no.")


def _prompt_model_choice(models: list[Path]) -> Path:
    if not models:
        raise FileNotFoundError(f"No .mph files found in {comsol_models_dir()}")
    print("\nDiscovered COMSOL models:")
    for idx, model_path in enumerate(models, start=1):
        print(f"  {idx}. {model_path}")
    choice = _prompt_int_choice("Select model number", 1, len(models))
    return models[choice - 1]


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
    
    # 1. Global Variables
    g_root = _safe_call(lambda: model.java.variable(), None)
    g_ids = _to_list(_safe_call(lambda: g_root.tags(), [])) if g_root is not None else []
    print(f"Global variable groups: {len(g_ids)}")
    for group_id in g_ids:
        # FIX: Use .get(group_id) instead of (group_id) for the Java API
        group = _safe_call(lambda: g_root.get(group_id), None)
        names = _to_list(_safe_call(lambda: group.varnames(), [])) if group is not None else []
        print(f"  - {group_id}: {len(names)} vars")
        
        # FIX: Loop through and print the actual variable names and expressions
        for name in names:
            expr = _safe_call(lambda: str(group.get(name)), "<unavailable>")
            desc = _safe_call(lambda: str(group.descr(name)), "")
            desc_s = f" :: {desc}" if desc else ""
            print(f"    * {name} = {expr}{desc_s}")

    # 2. Component Variables
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        v_root = _safe_call(lambda: comp.variable(), None)
        v_ids = _to_list(_safe_call(lambda: v_root.tags(), [])) if v_root is not None else []
        print(f"Component {comp_id} variable groups: {len(v_ids)}")
        for group_id in v_ids:
            # FIX: Use .get(group_id) here too
            group = _safe_call(lambda: v_root.get(group_id), None)
            names = _to_list(_safe_call(lambda: group.varnames(), [])) if group is not None else []
            print(f"  - {group_id}: {len(names)} vars")
            
            # FIX: Loop through and print
            for name in names:
                expr = _safe_call(lambda: str(group.get(name)), "<unavailable>")
                desc = _safe_call(lambda: str(group.descr(name)), "")
                desc_s = f" :: {desc}" if desc else ""
                print(f"    * {name} = {expr}{desc_s}")


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
        # Use robust extraction so array-valued fields are not truncated.
        val = _get_feature_property_value(feat, key)
        if val and val != "[]":
            yield key, val


def _get_feature_property_value(node, key: str) -> str:
    """Best-effort property extraction across common COMSOL value types."""
    # Prioritize array extraction to avoid truncating multi-element fields.
    arr_val = _safe_call(lambda: node.getStringArray(key), None)
    if arr_val is not None:
        val_list = [str(x) for x in arr_val]
        if len(val_list) > 1:
            return "[" + ", ".join(val_list) + "]"
        if len(val_list) == 1:
            return val_list[0]

    candidates = (
        lambda: str(node.getString(key)),
        lambda: str(node.get(key)),
        lambda: str(node.getDouble(key)),
        lambda: str(node.getInt(key)),
        lambda: str(node.getBoolean(key)),
    )
    for getter in candidates:
        value = _safe_call(getter, None)
        if value is None:
            continue
        value_s = str(value).strip()
        if value_s and value_s != "[]":
            return value_s
    return ""


def _iter_equation_like_properties(feat) -> Iterable[tuple[str, str]]:
    """Extract properties likely to contain governing-equation content."""
    eq_tokens = (
        "equ",
        "pde",
        "weak",
        "rhs",
        "source",
        "flux",
        "constraint",
        "reaction",
    )
    props = _to_list(_safe_call(lambda: feat.properties(), []))
    for key in props:
        key_lower = key.lower()
        if not any(token in key_lower for token in eq_tokens):
            continue
        value = _get_feature_property_value(feat, key)
        if value:
            yield key, value


def _selection_entities(selection) -> list[str]:
    """Return best-effort entity id lists for common geometric dimensions."""
    out: list[str] = []
    for dim in (0, 1, 2, 3):
        ids = _safe_call(lambda d=dim: selection.entities(d), None)
        id_list = _to_list(ids)
        if id_list:
            out.append(f"dim {dim}: [{', '.join(id_list)}]")
    return out


def _format_feature_selection(feat) -> str:
    """Human-readable selection summary for a physics feature."""
    sel = _safe_call(lambda: feat.selection(), None)
    if sel is None:
        return "selection unavailable"

    named = _safe_call(lambda: str(sel.named()), "")
    if named and named != "null":
        return f"named selection: {named}"

    geom_dim = _safe_call(lambda: str(sel.geomdim()), "")
    geom_name = _safe_call(lambda: str(sel.geom()), "")
    entities = _selection_entities(sel)
    if entities:
        prefix = "selection"
        if geom_name or geom_dim:
            suffix = f" ({geom_name}, dim={geom_dim})" if geom_name else f" (dim={geom_dim})"
            prefix += suffix
        return f"{prefix}: " + "; ".join(entities)

    # Fallback for "all" style selections.
    is_all = _safe_call(lambda: bool(sel.all()), False)
    if is_all:
        return "applies to all entities in selection scope"

    return "no explicit entities reported"


def _print_applied_boundary_conditions(model) -> None:
    print("\n=== APPLIED BOUNDARY CONDITIONS / TARGET SELECTIONS ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    if not comp_ids:
        print("No components found.")
        return

    total_conditions = 0
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        print(f"\nComponent: {comp_id}")
        phys_ids = _to_list(_safe_call(lambda: comp.physics().tags(), []))
        if not phys_ids:
            print("  No physics interfaces.")
            continue

        for phys_id in phys_ids:
            phys = _safe_call(lambda: comp.physics(phys_id), None)
            if phys is None:
                continue
            phys_type = _safe_call(lambda: str(phys.getType()), "<unknown>")
            print(f"  Physics: {phys_id} ({phys_type})")

            feat_ids = _to_list(_safe_call(lambda: phys.feature().tags(), []))
            if not feat_ids:
                print("    No physics features.")
                continue

            for feat_id in feat_ids:
                feat = _safe_call(lambda: phys.feature(feat_id), None)
                if feat is None:
                    continue
                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>")
                feat_name = _safe_call(lambda: str(feat.name()), feat_id)
                sel_summary = _format_feature_selection(feat)
                total_conditions += 1
                print(f"    - {feat_id} ({feat_type}) :: {feat_name}")
                print(f"      applies on -> {sel_summary}")

    print(f"\nTotal physics conditions/features reported: {total_conditions}")


def _print_equation_forms(model) -> None:
    print("\n=== FULL EQUATION FORMS (best effort) ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    if not comp_ids:
        print("No components found.")
        return

    total_equation_fields = 0
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        print(f"\nComponent: {comp_id}")
        phys_ids = _to_list(_safe_call(lambda: comp.physics().tags(), []))
        if not phys_ids:
            print("  No physics interfaces.")
            continue

        for phys_id in phys_ids:
            phys = _safe_call(lambda: comp.physics(phys_id), None)
            if phys is None:
                continue

            phys_type = _safe_call(lambda: str(phys.getType()), "<unknown>")
            print(f"  Physics: {phys_id} ({phys_type})")

            # Interface-level equation-like properties.
            interface_pairs = list(_iter_equation_like_properties(phys))
            if interface_pairs:
                print("    Interface equation properties:")
                for key, value in interface_pairs:
                    total_equation_fields += 1
                    print(f"      - {key}: {value}")

            feat_ids = _to_list(_safe_call(lambda: phys.feature().tags(), []))
            if not feat_ids:
                print("    No physics features.")
                continue

            for feat_id in feat_ids:
                feat = _safe_call(lambda: phys.feature(feat_id), None)
                if feat is None:
                    continue
                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>")
                equation_pairs = list(_iter_equation_like_properties(feat))
                if not equation_pairs:
                    continue
                print(f"    Feature: {feat_id} ({feat_type})")
                for key, value in equation_pairs:
                    total_equation_fields += 1
                    print(f"      - {key}: {value}")

    print(f"\nExtracted equation-related fields: {total_equation_fields}")
    if total_equation_fields == 0:
        print("No equation-like properties were discovered with current API calls.")


def _print_materials_content(model) -> None:
    print("\n=== MATERIALS CONTENT (detailed) ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    if not comp_ids:
        print("No components found.")
        return

    total_material_fields = 0
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue

        print(f"\nComponent: {comp_id}")
        mat_ids = _to_list(_safe_call(lambda: comp.material().tags(), []))
        print(f"  Materials: {len(mat_ids)}")
        if not mat_ids:
            continue

        for mat_id in mat_ids:
            mat = _safe_call(lambda: comp.material(mat_id), None)
            if mat is None:
                print(f"  - {mat_id} (unavailable)")
                continue
            mat_name = _safe_call(lambda: str(mat.name()), mat_id)
            mat_type = _safe_call(lambda: str(mat.getType()), "<unknown>")
            print(f"  - {mat_id}: {mat_name} ({mat_type})")

            # COMSOL materials commonly store properties under property groups.
            pg_root = _safe_call(lambda: mat.propertyGroup(), None)
            pg_ids = _to_list(_safe_call(lambda: pg_root.tags(), [])) if pg_root is not None else []
            if not pg_ids:
                print("      (no material property groups found)")
                continue

            for pg_id in pg_ids:
                pg = _safe_call(lambda: mat.propertyGroup(pg_id), None)
                if pg is None:
                    continue
                pg_name = _safe_call(lambda: str(pg.name()), pg_id)
                print(f"      * Property group: {pg_id} ({pg_name})")
                prop_keys = _to_list(_safe_call(lambda: pg.properties(), []))
                if not prop_keys:
                    print("          - (no properties listed)")
                    continue

                for key in prop_keys:
                    value = _get_feature_property_value(pg, key)
                    if not value:
                        continue
                    total_material_fields += 1
                    print(f"          - {key}: {value}")

    print(f"\nExtracted material fields: {total_material_fields}")
    if total_material_fields == 0:
        print("No material properties were discovered with current API calls.")


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
        _print_applied_boundary_conditions(model)
        _print_equation_forms(model)
        _print_materials_content(model)
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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .mph model (if omitted, interactive model picker is used).",
    )
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

    if args.model:
        selected = _resolve_model_path(args.model)
        inspect_model(selected, show_properties=args.show_properties)
        return

    selected = _prompt_model_choice(models)
    show_properties = args.show_properties or _prompt_yes_no(
        "Also print verbose physics feature properties?",
        default=False,
    )
    inspect_model(selected, show_properties=show_properties)


if __name__ == "__main__":
    main()
