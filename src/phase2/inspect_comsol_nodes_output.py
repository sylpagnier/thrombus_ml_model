import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree

# --- 1. SET UP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(project_root, 'data', 'processed', 'cfd_results_tier3_patients')

# Define the patient stem so we can dynamically load all 4 files
patient_stem = 'patient001'

domain_file = os.path.join(data_dir, f'{patient_stem}.txt')
inlet_file = os.path.join(data_dir, f'{patient_stem}_inlet.txt')
outlet_file = os.path.join(data_dir, f'{patient_stem}_outlet.txt')
wall_file = os.path.join(data_dir, f'{patient_stem}_wall.txt')

if not os.path.exists(domain_file):
    print(f"CRITICAL ERROR: Main file not found at {domain_file}")
    exit()

# --- 2. LOAD MAIN DOMAIN DATA ---
# Removed the mask columns from the expected COMSOL export since we do it spatially now
col_names = [
    'x_orig', 'y_orig', 'x', 'y', 'u', 'v', 'p', 'mu_eff',
    'rp', 'ap', 'apr', 'aps', 'PT', 'th', 'at', 'fg', 'fi',
    'M', 'Mas', 'Mat'
]
df = pd.read_csv(domain_file, comment='%', sep=r'\s+', header=None, names=col_names)

# Extract main coordinates for the KDTree
domain_coords = df[['x', 'y']].values
tree = cKDTree(domain_coords)


# --- 3. HELPER FUNCTION TO TAG BOUNDARIES ---
def get_boundary_mask(boundary_file, tree, num_nodes, tolerance=1e-6):
    mask = np.zeros(num_nodes, dtype=bool)
    if not os.path.exists(boundary_file):
        print(f"Warning: Boundary file missing: {boundary_file}")
        return mask

    # Load boundary coords
    bnd_df = pd.read_csv(boundary_file, comment='%', sep=r'\s+', header=None)

    # Grab the last two columns (x, y) and drop duplicates in case 'Time: All' was used
    bnd_coords = bnd_df.iloc[:, -2:].values
    bnd_coords = np.unique(bnd_coords, axis=0)

    # Query the KDTree for exact spatial matches
    distances, indices = tree.query(bnd_coords)

    # Tag nodes that fall within a strict tolerance
    valid_matches = indices[distances < tolerance]
    mask[valid_matches] = True
    return mask


# --- 4. EXTRACT MASKS ---
mask_inlet = get_boundary_mask(inlet_file, tree, len(df))
mask_outlet = get_boundary_mask(outlet_file, tree, len(df))
mask_wall = get_boundary_mask(wall_file, tree, len(df))

# Combine to find interior fluid nodes
mask_fluid = ~(mask_inlet | mask_outlet | mask_wall)

# Add them to dataframe for diagnostic checking
df['is_inlet'] = mask_inlet
df['is_outlet'] = mask_outlet
df['is_wall'] = mask_wall

# --- 5. VERIFICATION OUTPUT ---
print("\n" + "=" * 45)
print(f"   GROUND-TRUTH SELECTION: {patient_stem.upper()}")
print("=" * 45)
print(f"Total Unique Nodes: {len(df)}")
print("-" * 45)
print(f"Inlet Nodes:       {mask_inlet.sum()}")
print(f"Outlet Nodes:      {mask_outlet.sum()}")
print(f"Wall Nodes:        {mask_wall.sum()}")
print(f"Interior Fluid:    {mask_fluid.sum()}")
print("=" * 45)

if mask_inlet.sum() == 0:
    print("❌ ERROR: Still 0 Inlet nodes. Check your COMSOL Edge exports.")
else:
    print("✅ SUCCESS: Boundary nodes successfully extracted from spatial mapping.")