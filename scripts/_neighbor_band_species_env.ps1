# Shared env: GT-flow neighbor-band species teacher + physics trigger eval.
# Dot-source from go_neighbor_band_species_trigger.ps1.

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$env:BIOCHEM_PRESET = "passive_transport"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_GT_KINE_VEL = "1"
$env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
$env:BIOCHEM_DETACH_MACRO_STATE = "0"
$env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
$env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0"
$env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"

# Neighbor shell = wall + GT clot seeds + 1-hop (clot-phi mask recipe).
$env:BIOCHEM_DATA_BIO_MASK_MODE = "neighbor"
$env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
$env:BIOCHEM_ADR_MASK_MODE = "match_data_bio"
$env:BIOCHEM_ADR_EXCLUDE_WALL = "1"

# Trigger-relevant species only (FI + Mat); cascade channels still roll in forward.
$env:BIOCHEM_DATA_BIO_SPECIES_SCOPE = "fi_mat"

# Species val on same mask (skip heavy mu viz).
$env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
$env:BIOCHEM_PASSIVE_SPECIES_VAL_ONLY = "1"

# Physics trigger baseline (explicit Mat/FI gelation; no learned gate).
$env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
$env:CLOT_PHI_PHYSICS_GELATION_GATE = "0"
