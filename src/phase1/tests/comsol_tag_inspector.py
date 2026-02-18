import mph
from pathlib import Path
from src.utils.paths import get_project_root
from src.config import VesselConfig


def inspect_comsol_tags():
    root = get_project_root()
    cfg = VesselConfig()

    # Resolve template path dynamically from Config
    if Path(cfg.template_path).is_absolute():
        abs_template = Path(cfg.template_path)
    else:
        abs_template = root / cfg.template_path

    print(f"Checking Template: {abs_template}")
    if not abs_template.exists():
        raise FileNotFoundError(f"Cannot find template at: {abs_template}")

    client = mph.start()
    model = client.load(str(abs_template))

    print("\n=== COMSOL TAG INSPECTION ===")

    # 1. Inspect Components
    comp_tags = model.java.component().tags()
    if not comp_tags:
        print("No components found!")
        return

    for c_tag in comp_tags:
        print(f"\nComponent: {c_tag}")
        comp_node = model.java.component(c_tag)

        # 2. Inspect Meshes
        print(f"   ├── Looking for Meshes...")
        mesh_tags = comp_node.mesh().tags()

        if not mesh_tags:
            print("   │   No meshes found.")

        for m_tag in mesh_tags:
            print(f"   │   ├── Mesh Tag: {m_tag}")
            mesh_node = comp_node.mesh(m_tag)
            feature_tags = mesh_node.feature().tags()

            # 3. Validation for AnchorGenerator compatibility
            found_import = False
            for f_tag in feature_tags:
                f_type = mesh_node.feature(f_tag).getType()
                print(f"   │   │   ├── Feature: {f_tag} (Type: {f_type})")

                if f_type == 'Import':
                    found_import = True

            # Specific check for anchor_generator logic
            if found_import:
                print("   │   │    'Import' feature found (Compatible with AnchorGenerator)")
            else:
                print("   │   │    WARNING: No 'Import' feature found! AnchorGenerator will fail.")
                print("   │   │      (It expects an 'Import' node to inject the mesh)")

    client.clear()
    print("\nDone.")


if __name__ == "__main__":
    inspect_comsol_tags()