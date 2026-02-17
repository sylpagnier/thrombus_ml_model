import mph
from pathlib import Path

# --- 1. Setup Paths ---
template_path = 'comsol_models/phase1_template.mph'
# Adjust "root" logic if your script is in src/anchor_cfds/ vs project root
# Assuming this script is run from project root or similar to your previous one:
current_path = Path(__file__).resolve()
# Heuristic to find project root if running from src/
if 'src' in str(current_path):
    project_root = current_path.parent.parent.parent.parent
else:
    project_root = current_path.parent.parent

abs_template = project_root / template_path

print(f"Loading: {abs_template}")
if not abs_template.exists():
    raise FileNotFoundError(f"Cannot find: {abs_template}")

client = mph.start()
model = client.load(str(abs_template))

print("\n=== RAW JAVA LAYER INSPECTION ===")
print("This inspects the internal COMSOL tags directly, bypassing mph wrappers.")

# 1. Inspect Components
# model.java.component() returns the container for components
comp_tags = model.java.component().tags()

if not comp_tags:
    print("Warning: No components found in model!")

for c_tag in comp_tags:
    print(f"\n[Component]: {c_tag}")
    comp_node = model.java.component(c_tag)

    # 2. Inspect Meshes inside this Component
    # Note: We skip 'geometries' since that caused your crash and you likely need 'meshes' anyway
    print(f"  Looking for Meshes in {c_tag}...")
    try:
        mesh_tags = comp_node.mesh().tags()
        if not mesh_tags:
            print("    (No meshes found)")

        for m_tag in mesh_tags:
            print(f"    [Mesh]: {m_tag}")

            # 3. Inspect Features inside the Mesh (Look for your Import!)
            mesh_node = comp_node.mesh(m_tag)
            feature_tags = mesh_node.feature().tags()

            for f_tag in feature_tags:
                # We get the 'type' to confirm it's an import feature
                f_type = mesh_node.feature(f_tag).getType()
                print(f"      -> Feature: {f_tag} | Type: {f_type}")

    except Exception as e:
        print(f"    ! Error inspecting meshes: {e}")

    # 4. Inspect Geometries (Safe Mode)
    print(f"  Looking for Geometries in {c_tag}...")
    try:
        geom_tags = comp_node.geom().tags()
        if not geom_tags:
            print("    (No geometries found)")
        for g_tag in geom_tags:
            print(f"    [Geometry]: {g_tag}")
            # List features inside geometry
            g_features = comp_node.geom(g_tag).feature().tags()
            for gf_tag in g_features:
                print(f"      -> Feature: {gf_tag}")
    except Exception as e:
        print(f"    ! Error inspecting geometries: {e}")

print("\n=================================")