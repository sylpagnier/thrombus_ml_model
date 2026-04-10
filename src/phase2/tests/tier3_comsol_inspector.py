import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import VesselConfig, BiochemConfig
from src.utils.paths import get_project_root


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _to_py_list(java_arr):
    if java_arr is None:
        return []
    try:
        return [str(x) for x in list(java_arr)]
    except Exception:
        try:
            return [str(x) for x in java_arr]
        except Exception:
            return []


def _discover_models():
    model_dir = get_project_root() / "comsol_models"
    if not model_dir.exists():
        return []
    return sorted([p for p in model_dir.glob("*.mph") if p.is_file()])


def _select_model(models):
    if not models:
        raise FileNotFoundError("No .mph files found in comsol_models.")
    if len(models) == 1:
        return models[0]
    default_idx = 0
    for i, m in enumerate(models):
        if m.stem.lower() == "phase2_patient001":
            default_idx = i
            break
    print("\nAvailable COMSOL models:")
    for i, m in enumerate(models):
        print(f"  [ {i} ] {m.name}")
    raw = input(f"\nSelect model index [0-{len(models) - 1}] (default {default_idx}): ").strip()
    if raw == "":
        return models[default_idx]
    idx = int(raw)
    if idx < 0 or idx >= len(models):
        raise ValueError(f"Index out of range: {idx}")
    return models[idx]


def _print_model_structure(model):
    print("\n=== MODEL STRUCTURE ===")
    comp_ids = _to_py_list(_safe_call(lambda: model.java.component().tags(), []))
    if not comp_ids:
        print("No components found.")
        return
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        print(f"\nComponent: {comp_id}")
        mesh_ids = _to_py_list(_safe_call(lambda: comp.mesh().tags(), []))
        print(f"  Meshes: {len(mesh_ids)}")
        for mesh_id in mesh_ids:
            mesh = _safe_call(lambda: comp.mesh(mesh_id), None)
            feat_ids = _to_py_list(_safe_call(lambda: mesh.feature().tags(), [])) if mesh is not None else []
            print(f"    - {mesh_id}: {len(feat_ids)} mesh features")
            for feat_id in feat_ids:
                feat = _safe_call(lambda: mesh.feature(feat_id), None)
                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>") if feat is not None else "<unknown>"
                print(f"      * {feat_id} ({feat_type})")
        phys_ids = _to_py_list(_safe_call(lambda: comp.physics().tags(), []))
        print(f"  Physics interfaces: {len(phys_ids)}")
        for phys_id in phys_ids:
            phys = _safe_call(lambda: comp.physics(phys_id), None)
            phys_type = _safe_call(lambda: str(phys.getType()), "<unknown>") if phys is not None else "<unknown>"
            print(f"    - {phys_id} ({phys_type})")
            feat_ids = _to_py_list(_safe_call(lambda: phys.feature().tags(), []))
            for feat_id in feat_ids:
                feat = _safe_call(lambda: phys.feature(feat_id), None)
                if feat is None:
                    continue
                feat_type = _safe_call(lambda: str(feat.getType()), "<unknown>")
                print(f"      * Feature: {feat_id} ({feat_type})")
                props = _to_py_list(_safe_call(lambda: feat.properties(), []))
                for prop in props:
                    val = _safe_call(lambda: str(feat.getString(prop)), "")
                    if val and val != "[]":
                        print(f"        -> {prop}: {val}")
        mat_ids = _to_py_list(_safe_call(lambda: comp.material().tags(), []))
        print(f"  Materials: {len(mat_ids)}")
        for mat_id in mat_ids:
            print(f"    - {mat_id}")
            mat = _safe_call(lambda: comp.material(mat_id), None)
            if mat is None:
                continue
            pg_ids = _to_py_list(_safe_call(lambda: mat.propertyGroup().tags(), []))
            for pg_id in pg_ids:
                pg = _safe_call(lambda: mat.propertyGroup(pg_id), None)
                if pg is None:
                    continue
                params = _to_py_list(_safe_call(lambda: pg.varnames(), []))
                for param in params:
                    val = _safe_call(lambda: str(pg.getString(param)), "")
                    print(f"      * {param}: {val}")
        geom_ids = _to_py_list(_safe_call(lambda: comp.geom().tags(), []))
        for geom_id in geom_ids:
            geom = _safe_call(lambda: comp.geom(geom_id), None)
            feat_ids = _to_py_list(_safe_call(lambda: geom.feature().tags(), [])) if geom is not None else []
            print(f"  Geometry ({geom_id}): {len(feat_ids)} features")
            for feat_id in feat_ids:
                feat_type = _safe_call(lambda: str(geom.feature(feat_id).getType()), "")
                print(f"    - {feat_id} ({feat_type})")
        c_func_root = _safe_call(lambda: comp.func(), None)
        c_func_ids = _to_py_list(_safe_call(lambda: c_func_root.tags(), [])) if c_func_root is not None else []
        print(f"  Component Functions: {len(c_func_ids)}")
        _print_functions_advanced(c_func_root, f"Component {comp_id}")


def _print_parameters(model):
    print("\n=== PARAMETERS (global) ===")
    root = _safe_call(lambda: model.java.param(), None)
    names = _to_py_list(_safe_call(lambda: root.varnames(), [])) if root is not None else []
    print(f"Count: {len(names)}")
    for name in names:
        expr = _safe_call(lambda: str(root.get(name)), "<unavailable>")
        unit = _safe_call(lambda: str(root.evaluateUnit(name)), "")
        desc = _safe_call(lambda: str(root.descr(name)), "")
        u = f" [{unit}]" if unit else ""
        d = f" :: {desc}" if desc else ""
        print(f"- {name} = {expr}{u}{d}")


def _print_variables(model):
    print("\n=== VARIABLES (global + component) ===")
    gv_root = _safe_call(lambda: model.java.variable(), None)
    gv_ids = _to_py_list(_safe_call(lambda: gv_root.tags(), [])) if gv_root is not None else []
    print(f"Global variable groups: {len(gv_ids)}")
    for group_id in gv_ids:
        grp = _safe_call(lambda: gv_root(group_id), None)
        names = _to_py_list(_safe_call(lambda: grp.varnames(), [])) if grp is not None else []
        print(f"  - {group_id}: {len(names)} vars")
        for n in names:
            expr = _safe_call(lambda: str(grp.get(n)), "<unavailable>")
            unit = _safe_call(lambda: str(grp.evaluateUnit(n)), "")
            u = f" [{unit}]" if unit else ""
            print(f"      {n} = {expr}{u}")

    comp_ids = _to_py_list(_safe_call(lambda: model.java.component().tags(), []))
    for comp_id in comp_ids:
        comp = _safe_call(lambda: model.java.component(comp_id), None)
        if comp is None:
            continue
        v_root = _safe_call(lambda: comp.variable(), None)
        v_ids = _to_py_list(_safe_call(lambda: v_root.tags(), [])) if v_root is not None else []
        print(f"Component {comp_id} variable groups: {len(v_ids)}")
        for var_group_id in v_ids:
            grp = _safe_call(lambda: v_root(var_group_id), None)
            names = _to_py_list(_safe_call(lambda: grp.varnames(), [])) if grp is not None else []
            print(f"  - {var_group_id}: {len(names)} vars")
            for n in names:
                expr = _safe_call(lambda: str(grp.get(n)), "<unavailable>")
                unit = _safe_call(lambda: str(grp.evaluateUnit(n)), "")
                u = f" [{unit}]" if unit else ""
                print(f"      {n} = {expr}{u}")


def _print_functions_advanced(func_root, scope_name):
    func_ids = _to_py_list(_safe_call(lambda: func_root.tags(), [])) if func_root is not None else []
    print(f"\n=== FUNCTIONS ({scope_name}) ===")
    print(f"Count: {len(func_ids)}")
    for f_id in func_ids:
        f_node = _safe_call(lambda: func_root.get(f_id), None)
        if not f_node:
            continue

        f_name = _safe_call(lambda: str(f_node.name()), f_id)
        f_type = _safe_call(lambda: str(f_node.getType()), "<unknown>")

        if f_type == "Analytic":
            expr = _safe_call(lambda: str(f_node.getString("expr")), "<no expr>")
            print(f"- {f_name} [{f_type}] :: expr = {expr}")
        elif f_type == "Step":
            loc = _safe_call(lambda: str(f_node.getString("location")), "<no loc>")
            val_from = _safe_call(lambda: str(f_node.getString("from")), "")
            val_to = _safe_call(lambda: str(f_node.getString("to")), "")
            print(f"- {f_name} [{f_type}] :: loc={loc}, from={val_from}, to={val_to}")
        elif f_type == "Interpolation":
            source = _safe_call(lambda: str(f_node.getString("source")), "")
            print(f"- {f_name} [{f_type}] :: source={source}")
        else:
            print(f"- {f_name} [{f_type}]")


def _print_functions(model):
    _print_functions_advanced(_safe_call(lambda: model.java.func(), None), "Global")


def _print_studies(model):
    print("\n=== STUDIES & SOLVERS ===")
    std_root = _safe_call(lambda: model.java.study(), None)
    std_ids = _to_py_list(_safe_call(lambda: std_root.tags(), [])) if std_root is not None else []
    for std_id in std_ids:
        std = _safe_call(lambda: std_root.get(std_id), None)
        if std is None:
            continue
        print(f"- Study: {std_id}")
        step_ids = _to_py_list(_safe_call(lambda: std.feature().tags(), []))
        for step_id in step_ids:
            step = _safe_call(lambda: std.feature(step_id), None)
            if step is None:
                continue
            s_type = _safe_call(lambda: str(step.getType()), "<unknown>")
            print(f"  * Step: {step_id} ({s_type})")
            if s_type == "Transient":
                tlist = _safe_call(lambda: str(step.getString("tlist")), "<not found>")
                print(f"    -> Times: {tlist}")


def inspect_live_models():
    try:
        import mph
    except Exception as exc:
        raise RuntimeError("Failed to import `mph`. Install package `mph` and ensure COMSOL is available.") from exc

    models = _discover_models()
    if not models:
        raise FileNotFoundError("No .mph files found in comsol_models.")

    print("\n=== LIVE COMSOL API AUDIT ===")
    selected = _select_model(models)
    print(f"\nAnalyzing COMSOL model: {selected}")
    client = mph.start()
    model = client.load(str(selected))
    _print_model_structure(model)
    _print_parameters(model)
    _print_variables(model)
    _print_functions(model)
    _print_studies(model)
    _safe_call(lambda: model.remove())
    _safe_call(lambda: client.clear())
    _safe_call(lambda: client.disconnect())


def get_boundary_mask(boundary_file, tree, num_nodes, tolerance=1e-6):
    mask = np.zeros(num_nodes, dtype=bool)
    if not boundary_file.exists():
        print(f"Warning: Boundary file missing: {boundary_file}")
        return mask
    bnd_df = pd.read_csv(boundary_file, comment="%", sep=r"\s+", header=None)
    bnd_coords = np.unique(bnd_df.iloc[:, -2:].values, axis=0)
    distances, indices = tree.query(bnd_coords)
    valid_matches = indices[distances < tolerance]
    mask[valid_matches] = True
    return mask


def _available_stems(data_dir):
    stems = []
    for p in sorted(Path(data_dir).glob("*.txt")):
        if p.stem.endswith("_inlet") or p.stem.endswith("_outlet") or p.stem.endswith("_wall"):
            continue
        stems.append(p.stem)
    return stems


def _pick_stem_interactively(data_dir):
    stems = _available_stems(data_dir)
    if len(stems) == 0:
        print(f"No domain .txt files found in {data_dir}")
        return None
    print("\nAvailable patient stems:")
    for idx, stem in enumerate(stems):
        print(f"  [ {idx} ] {stem}")
    while True:
        user_input = input(f"\nSelect index [0-{len(stems) - 1}] or q to quit: ").strip()
        if user_input.lower() in ["q", "quit", "exit"]:
            return None
        try:
            idx = int(user_input)
            if 0 <= idx < len(stems):
                return stems[idx]
        except ValueError:
            pass
        print("Invalid input.")


def _all_stems(data_dir):
    return _available_stems(data_dir)


def inspect_patient(stem, data_dir):
    domain_file = Path(data_dir) / f"{stem}.txt"
    inlet_file = Path(data_dir) / f"{stem}_inlet.txt"
    outlet_file = Path(data_dir) / f"{stem}_outlet.txt"
    wall_file = Path(data_dir) / f"{stem}_wall.txt"
    if not domain_file.exists():
        print(f"CRITICAL ERROR: Main file not found at {domain_file}")
        return

    df_full = pd.read_csv(domain_file, comment="%", sep=r"\s+", header=None)
    if df_full.shape[1] < 20:
        raise ValueError(f"Unexpected COMSOL export format in {domain_file.name}: only {df_full.shape[1]} columns.")
    df = df_full.iloc[:, 2:20].copy()
    df.columns = ["x", "y", "u", "v", "p", "mu_eff", "rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi", "M", "Mas", "Mat"]
    tree = cKDTree(df[["x", "y"]].values)
    mask_inlet = get_boundary_mask(inlet_file, tree, len(df))
    mask_outlet = get_boundary_mask(outlet_file, tree, len(df))
    mask_wall = get_boundary_mask(wall_file, tree, len(df))
    mask_fluid = ~(mask_inlet | mask_outlet | mask_wall)
    print("\n" + "=" * 45)
    print(f"   GROUND-TRUTH SELECTION: {stem.upper()}")
    print("=" * 45)
    print(f"Total Unique Nodes: {len(df)}")
    print("-" * 45)
    print(f"Inlet Nodes:       {mask_inlet.sum()}")
    print(f"Outlet Nodes:      {mask_outlet.sum()}")
    print(f"Wall Nodes:        {mask_wall.sum()}")
    print(f"Interior Fluid:    {mask_fluid.sum()}")
    print("=" * 45)


def inspect_units(stem, data_dir, sample_rows=50000, timestep_samples=3):
    domain_file = Path(data_dir) / f"{stem}.txt"
    if not domain_file.exists():
        print(f"CRITICAL ERROR: Main file not found at {domain_file}")
        return
    unit_tokens = []
    with open(domain_file, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(50):
            line = f.readline()
            if not line or not line.startswith("%"):
                continue
            unit_tokens.extend(re.findall(r"\[([^\]]+)\]", line))
    if unit_tokens:
        print("\nHeader unit tokens found:", sorted(set(unit_tokens)))
    else:
        print("\nHeader unit tokens found: none (falling back to magnitude-based inference)")

    with open(domain_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_line = ""
    for line in lines:
        if line.startswith("% x") and "@ t=" in line:
            header_line = line
            break

    times = []
    if header_line:
        for match in re.finditer(r"t=([0-9.]+)", header_line):
            t_val = float(match.group(1))
            if t_val not in times:
                times.append(t_val)

    df_full = pd.read_csv(domain_file, comment="%", sep=r"\s+", header=None, nrows=sample_rows)
    if df_full.shape[1] < 20:
        print(f"Cannot audit units: expected >=20 columns, got {df_full.shape[1]}")
        return

    vars_per_step = 18
    inferred_steps = max(1, int((df_full.shape[1] - 2) / vars_per_step))
    if not times:
        # Fallback when header has no explicit t= tokens.
        times = [float(i) for i in range(inferred_steps)]
    else:
        inferred_steps = min(inferred_steps, len(times))

    if inferred_steps <= 1:
        chosen_idx = [0]
    else:
        k = max(1, int(timestep_samples))
        chosen_idx = sorted(set(np.linspace(0, inferred_steps - 1, num=min(k, inferred_steps), dtype=int).tolist()))

    step_dfs = []
    for i in chosen_idx:
        start_col = 2 + (i * vars_per_step)
        end_col = start_col + vars_per_step
        if end_col > df_full.shape[1]:
            continue
        df_step = df_full.iloc[:, start_col:end_col].copy()
        df_step.columns = [
            "x", "y", "u", "v", "p", "mu_eff", "rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi", "M", "Mas", "Mat"
        ]
        step_dfs.append((i, times[i] if i < len(times) else float(i), df_step))

    if not step_dfs:
        print("Cannot audit units: no valid timestep blocks parsed.")
        return
    bio = BiochemConfig(tier="tier3")
    ref_cgs = {
        "rp": bio.c_RP0 / 1e6, "ap": (0.05 * bio.c_RP0) / 1e6,
        "apr": bio.APRcrit * 1e3, "aps": bio.APScrit * 1e3, "PT": bio.c_pT0 * 1e3,
        "th": bio.Tcrit * 1e3, "at": bio.cAT0 * 1e3, "fg": bio.c_Fg0 * 1e3, "fi": bio.c_Fg0 * 1e3,
        "M": bio.Minf / 1e4, "Mas": bio.Minf / 1e4, "Mat": bio.Minf / 1e4,
    }
    species = ["rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi", "M", "Mas", "Mat"]
    timestep_labels = ", ".join([f"#{i} (t={t:g})" for i, t, _ in step_dfs])
    print(f"\nSpecies magnitude audit (sampled timestep blocks: {timestep_labels})")
    print(f"{'col':<5} {'p95+ max':>12} {'ref(CGS)':>12} {'ratio':>9}  {'likely unit'}")
    for col in species:
        p95_per_step = []
        for _, _, df in step_dfs:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            pos = vals[vals > 0.0]
            p95_per_step.append(float(np.nanpercentile(pos, 95)) if pos.size else 0.0)
        p95 = max(p95_per_step) if p95_per_step else 0.0
        ref = max(ref_cgs[col], 1e-18)
        ratio = p95 / ref if p95 > 0 else 0.0
        if col in ("rp", "ap"):
            unit = "plt/ml (CGS)" if 0.1 <= ratio <= 10 else "check platelets"
        elif col in ("M", "Mas", "Mat"):
            unit = "plt/cm^2 (CGS)" if 0.1 <= ratio <= 10 else "check surface"
        else:
            unit = "uM (CGS)" if 0.1 <= ratio <= 10 else "check uM vs mol/m^3"
        print(f"{col:<5} {p95:12.4g} {ref:12.4g} {ratio:9.3g}  {unit}")
    print("\nIf PT/AT/FG are near CGS uM baselines, extractor must do uM->mol/m^3 (x1e-3) before ND/log.")


def main():
    # Single comprehensive workflow:
    # 1) Live API audit across all .mph models in comsol_models
    # 2) Export-based checks across tier3_patients and tier3 stems
    inspect_live_models()

    print("\n=== EXPORT-BASED AUDIT ===")
    tiers = ["tier3_patients", "tier3"]
    for tier in tiers:
        data_dir = get_project_root() / VesselConfig(tier=tier).output_dir
        stems = _all_stems(data_dir)
        if not stems:
            print(f"\n{tier}: no export stems found in {data_dir}")
            continue
        for stem in stems:
            print(f"\n--- Export audit: {tier} / {stem} ---")
            inspect_units(stem, data_dir, sample_rows=50000)
            inspect_patient(stem, data_dir)


if __name__ == "__main__":
    main()
