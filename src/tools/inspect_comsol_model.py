"""
Live COMSOL model inspector (Optimized for PINN/AI Context).

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

    g_root = _safe_call(lambda: model.java.variable(), None)
    g_ids = _to_list(_safe_call(lambda: g_root.tags(), [])) if g_root is not None else []
    print(f"Global variable groups: {len(g_ids)}")
    for group_id in g_ids:
        group = _safe_call(lambda: g_root.get(group_id), None)
        names = _to_list(_safe_call(lambda: group.varnames(), [])) if group is not None else []
        print(f"  - {group_id}: {len(names)} vars (Scope: Global)")

        for name in names:
            expr = _safe_call(lambda: str(group.get(name)), "<unavailable>")
            desc = _safe_call(lambda: str(group.descr(name)), "")
            desc_s = f" :: {desc}" if desc else ""
            print(f"    * {name} = {expr}{desc_s}")

    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        v_root = _safe_call(lambda: comp.variable(), None)
        v_ids = _to_list(_safe_call(lambda: v_root.tags(), [])) if v_root is not None else []
        print(f"Component {comp_id} variable groups: {len(v_ids)}")
        for group_id in v_ids:
            group = _safe_call(lambda: v_root.get(group_id), None)
            names = _to_list(_safe_call(lambda: group.varnames(), [])) if group is not None else []

            sel = _safe_call(lambda: group.selection(), None)
            scope_str = "Global/Entire Geometry"
            if sel is not None:
                is_all = _safe_call(lambda: bool(sel.all()), False)
                if not is_all:
                    entities = _selection_entities(sel)
                    if entities:
                        scope_str = "Scope: " + "; ".join(entities)
                    else:
                        scope_str = "Scope: empty selection"

            print(f"  - {group_id}: {len(names)} vars ({scope_str})")

            for name in names:
                expr = _safe_call(lambda: str(group.get(name)), "<unavailable>")
                desc = _safe_call(lambda: str(group.descr(name)), "")
                desc_s = f" :: {desc}" if desc else ""
                print(f"    * {name} = {expr}{desc_s}")


def _print_named_selections(model) -> None:
    print("\n=== NAMED/EXPLICIT SELECTIONS ===")
    s_root = _safe_call(lambda: model.java.selection(), None)
    s_ids = _to_list(_safe_call(lambda: s_root.tags(), [])) if s_root is not None else []

    filtered_sids = [sid for sid in s_ids if "nastran" not in sid.lower() and "imp" not in sid.lower()]
    print(f"Count: {len(filtered_sids)} (Ignored {len(s_ids) - len(filtered_sids)} auto-imports)")

    for sid in filtered_sids:
        sel = _safe_call(lambda: s_root.get(sid), None)
        if sel is None:
            continue

        s_type = _safe_call(lambda: str(sel.getType()), "<unknown>")
        s_name = _safe_call(lambda: str(sel.name()), sid)
        entities = _selection_entities(sel)
        is_all = _safe_call(lambda: bool(sel.all()), False)

        print(f"- {sid}: {s_name} [{s_type}]")
        if entities:
            print(f"  entities -> {'; '.join(entities)}")
        elif is_all:
            print("  entities -> all entities in selection scope")
        else:
            print("  entities -> no explicit entities reported")


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

        # Explicitly extract the callable name and arguments used in the equations
        func_call = _safe_call(lambda: str(f_node.getString("funcname")), "")
        args_arr = _safe_call(lambda: f_node.getStringArray("args"), [])
        args_str = ", ".join([str(a) for a in args_arr]) if args_arr else ""

        call_sig = f"{func_call}({args_str})" if func_call else f_name

        if f_type == "Analytic":
            expr = _safe_call(lambda: str(f_node.getString("expr")), "")
            print(f"- {f_name} [{f_type}] :: {call_sig} = {expr}")

        elif f_type == "Step":
            loc = _safe_call(lambda: str(f_node.getString("location")), "")
            f_val = _safe_call(lambda: str(f_node.getString("from")), "")
            t_val = _safe_call(lambda: str(f_node.getString("to")), "")
            print(f"- {f_name} [{f_type}] :: {call_sig} -> Step at {loc} (from {f_val} to {t_val})")

        elif f_type == "Piecewise":
            print(f"- {f_name} [{f_type}] :: {call_sig}")
            funcs_mat = _safe_call(lambda: f_node.getStringMatrix("funcs"), [])
            if funcs_mat:
                for row in funcs_mat:
                    row_strs = [str(x) for x in row]
                    print(f"    * interval/expr: {row_strs}")
        elif f_type == "Interpolation":
            print(f"- {f_name} [{f_type}] :: {call_sig}")
            table_data = _safe_call(lambda: f_node.getStringMatrix("table"), [])
            if table_data:
                # Limit output to avoid context bloat if table is very large.
                if len(table_data) > 20:
                    print(f"    * [Table contains {len(table_data)} rows - showing first 5]")
                    for row in table_data[:5]:
                        print(f"    * {row}")
                    print("    * ...")
                else:
                    for row in table_data:
                        print(f"    * {row}")
        else:
            expr = _safe_call(lambda: str(f_node.getString("expr")), "")
            if expr:
                print(f"- {f_name} [{f_type}] :: {call_sig} = {expr}")
            else:
                print(f"- {f_name} [{f_type}] :: {call_sig}")


def _iter_feature_properties(feat) -> Iterable[tuple[str, str]]:
    props = _to_list(_safe_call(lambda: feat.properties(), []))

    # Aggressively filter out GUI noise and unrelated properties
    ignore_substrings = [
        "minput_", "showPhysicsSymbols", "StudyStep", "constraint",
        "pairContrib", "CompensateFor", "editModelInputs", "coordinateSystem"
    ]

    for key in props:
        if any(sub in key for sub in ignore_substrings):
            continue

        val = _get_feature_property_value(feat, key)
        if val and val != "[]":
            yield key, val


def _get_feature_property_value(node, key: str) -> str:
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
    eq_tokens = (
        "equ", "pde", "weak", "rhs", "source", "flux", "reaction",
        "init", "diff", "u0", "p0", "c0"
    )
    props = _to_list(_safe_call(lambda: feat.properties(), []))
    for key in props:
        if not any(token in key.lower() for token in eq_tokens):
            continue
        value = _get_feature_property_value(feat, key)
        if value:
            yield key, value


def _selection_entities(selection) -> list[str]:
    out: list[str] = []
    for dim in (0, 1, 2, 3):
        ids = _safe_call(lambda d=dim: selection.entities(d), None)
        id_list = _to_list(ids)
        if id_list:
            out.append(f"dim {dim}: [{', '.join(id_list)}]")
    return out


def _format_feature_selection(feat) -> str:
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

    is_all = _safe_call(lambda: bool(sel.all()), False)
    if is_all:
        return "applies to all entities in selection scope"

    return "no explicit entities reported"


def _print_applied_boundary_conditions(model) -> None:
    print("\n=== APPLIED BOUNDARY CONDITIONS / TARGET SELECTIONS ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    if not comp_ids:
        return

    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        print(f"\nComponent: {comp_id}")
        phys_ids = _to_list(_safe_call(lambda: comp.physics().tags(), []))
        for phys_id in phys_ids:
            phys = _safe_call(lambda: comp.physics(phys_id), None)
            if phys is None or not _safe_call(lambda: phys.isActive(), True):
                continue
            phys_type = _safe_call(lambda: str(phys.getType()), "<unknown>")
            print(f"  Physics: {phys_id} ({phys_type})")

            feat_ids = _to_list(_safe_call(lambda: phys.feature().tags(), []))
            for feat_id in feat_ids:
                feat = _safe_call(lambda: phys.feature(feat_id), None)
                if feat is None or not _safe_call(lambda: feat.isActive(), True):
                    continue
                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>")
                feat_name = _safe_call(lambda: str(feat.name()), feat_id)
                sel_summary = _format_feature_selection(feat)
                print(f"    - {feat_id} ({feat_type}) :: {feat_name}")
                print(f"      applies on -> {sel_summary}")


def _print_equation_forms(model) -> None:
    print("\n=== FULL EQUATION FORMS ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        phys_ids = _to_list(_safe_call(lambda: comp.physics().tags(), []))
        for phys_id in phys_ids:
            phys = _safe_call(lambda: comp.physics(phys_id), None)
            if phys is None or not _safe_call(lambda: phys.isActive(), True):
                continue

            phys_type = _safe_call(lambda: str(phys.getType()), "<unknown>")
            print(f"\n  Physics: {phys_id} ({phys_type})")

            feat_ids = _to_list(_safe_call(lambda: phys.feature().tags(), []))
            for feat_id in feat_ids:
                feat = _safe_call(lambda: phys.feature(feat_id), None)
                if feat is None or not _safe_call(lambda: feat.isActive(), True):
                    continue

                equation_pairs = list(_iter_equation_like_properties(feat))
                if not equation_pairs:
                    continue

                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>")
                print(f"    Feature: {feat_id} ({feat_type})")
                for key, value in equation_pairs:
                    print(f"      - {key}: {value}")


def _print_materials_content(model) -> None:
    print("\n=== MATERIALS CONTENT ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue

        mat_ids = _to_list(_safe_call(lambda: comp.material().tags(), []))
        for mat_id in mat_ids:
            mat = _safe_call(lambda: comp.material(mat_id), None)
            if mat is None:
                continue
            mat_name = _safe_call(lambda: str(mat.name()), mat_id)
            mat_type = _safe_call(lambda: str(mat.getType()), "<unknown>")
            print(f"\n  - {mat_id}: {mat_name} ({mat_type})")

            pg_root = _safe_call(lambda: mat.propertyGroup(), None)
            pg_ids = _to_list(_safe_call(lambda: pg_root.tags(), [])) if pg_root is not None else []
            for pg_id in pg_ids:
                pg = _safe_call(lambda: mat.propertyGroup(pg_id), None)
                if pg is None:
                    continue
                pg_name = _safe_call(lambda: str(pg.name()), pg_id)
                print(f"      * Property group: {pg_id} ({pg_name})")

                for key in _to_list(_safe_call(lambda: pg.properties(), [])):
                    value = _get_feature_property_value(pg, key)
                    if value:
                        print(f"          - {key}: {value}")


def _print_model_structure(model, show_properties: bool = False) -> None:
    print("\n=== MODEL STRUCTURE ===")
    comp_ids = _to_list(_safe_call(lambda: model.java.component().tags(), []))

    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue

        print(f"\nComponent: {comp_id}")
        phys_ids = _to_list(_safe_call(lambda: comp.physics().tags(), []))
        print(f"  Physics interfaces: {len(phys_ids)}")

        for phys_id in phys_ids:
            phys = _safe_call(lambda: comp.physics(phys_id), None)
            if phys is not None and not _safe_call(lambda: phys.isActive(), True):
                continue
            phys_type = _safe_call(lambda: str(phys.getType()), "<unknown>") if phys is not None else "<unknown>"
            print(f"    - {phys_id} ({phys_type})")

            feat_ids = _to_list(_safe_call(lambda: phys.feature().tags(), []))
            for feat_id in feat_ids:
                feat = _safe_call(lambda: phys.feature(feat_id), None)
                if feat is None or not _safe_call(lambda: feat.isActive(), True):
                    continue
                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>")
                print(f"      * {feat_id} ({feat_type})")

                if show_properties:
                    for k, v in _iter_feature_properties(feat):
                        print(f"        -> {k}: {v}")

        _print_functions(_safe_call(lambda: comp.func(), None), f"Component {comp_id}")


def _print_biochem_extract_readiness(model) -> None:
    """Pre-flight: Export nodes + datasets used by extract_biochem_comsol."""
    from src.data_gen.lib.biochem_comsol_datasets import (
        list_comsol_datasets,
        resolve_boundary_datasets,
        resolve_solution_dataset,
    )
    from src.data_gen.lib.biochem_comsol_mph_export import (
        discover_export_tags,
        list_result_export_tags,
        resolve_export_tags,
    )

    print("\n=== BIOCHEM EXTRACT READINESS ===")
    print("[i] Default pull: Results > Export nodes (set BIOCHEM_COMSOL_USE_MPH_EXPORTS=0 for Interp fallback).")

    exports = list_result_export_tags(model.java)
    print(f"\nExport nodes ({len(exports)}):")
    for tag in exports:
        print(f"  - {tag}")
    discovered = discover_export_tags(model.java)
    need = resolve_export_tags(model.java)
    print("\nResolved Export tags (env overrides win; else auto from labels/dset1/edg*):")
    for role, tag in need.items():
        hit = tag if tag in exports else next((e for e in exports if e.lower() == tag.lower()), None)
        status = "OK" if hit else "MISSING"
        auto = discovered.get(role)
        extra = ""
        if auto and auto != tag:
            extra = f" (auto={auto})"
        elif auto and status == "OK":
            extra = " (auto)"
        print(f"  - {role}: {tag} -> {status}{extra}")

    rows = list_comsol_datasets(model.java)
    print(f"\nResult datasets ({len(rows)}):")
    for row in rows:
        sol = row.get("solution") or ""
        extra = f", sol={sol}" if sol else ""
        print(f"  - {row['tag']}: {row.get('label') or row['tag']}{extra}")

    try:
        dom = resolve_solution_dataset(model.java, "sol1")
        print(f"\nDomain dataset for sol1: {dom}")
    except Exception as exc:
        print(f"\n[WARN] Domain dataset for sol1: {exc}")

    try:
        bmap = resolve_boundary_datasets(model.java)
        print(f"Boundary datasets: {bmap}")
    except Exception as exc:
        print(f"[WARN] Boundary datasets: {exc}")

    print(
        "\n[i] Template selections: box1=inlet, box2=outlet, dif1=wall; "
        "vars is_inlet=sel1(x,y) (Interp may fail; Export nodes preferred)."
    )
    print("[i] Material mu: mu_b*(mu1(Mat)+mu2(FI)) must match sol_data expressions in Export.")


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
        _print_named_selections(model)
        _print_biochem_extract_readiness(model)
        _print_functions(_safe_call(lambda: model.java.func(), None), "Global")
        # _print_studies(model)  # Removed: Not relevant for PINN context mapping
    finally:
        _safe_call(lambda: model.remove())
        _safe_call(lambda: client.clear())
        _safe_call(lambda: client.disconnect())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect COMSOL .mph internals via mph/LiveLink.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--show-properties", action="store_true")
    args = parser.parse_args()

    models = _discover_models()
    if args.list_models:
        for m in models:
            print(f"- {m}")
        return

    if args.all_models:
        for m in models:
            inspect_model(m, show_properties=args.show_properties)
        return

    if args.model:
        inspect_model(_resolve_model_path(args.model), show_properties=args.show_properties)
        return

    selected = _prompt_model_choice(models)
    show_properties = args.show_properties or _prompt_yes_no(
        "Also print verbose physics feature properties?", default=False
    )
    inspect_model(selected, show_properties=show_properties)


if __name__ == "__main__":
    main()
