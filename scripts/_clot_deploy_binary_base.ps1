# Shared CAVO deploy ladder: binary clot shell (phi BCE only), hard mu projection, GT flow @ t_in.
# Dot-source from S0 / S1 / G1 launchers.

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$env:CLOT_FORECAST_MODE = "one_step"
$env:CLOT_PHI_VEL_SOURCE = "gt"
$env:CLOT_PHI_ROLLOUT = "0"
$env:CLOT_PHI_TIME_STRIDE = "1"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "0"

# Binary shell: phi BCE + fixed mu_solid blend (no continuous mu regression).
$env:CLOT_PHI_FIXED_MU_FROM_PHI = "1"
$env:CLOT_PHI_HYBRID = "0"
$env:CLOT_PHI_MU_LOG_LAMBDA = "0"
$env:CLOT_PHI_MU_SOLID_SI = "0.10"
$env:CLOT_PHI_SHAPE_USE_T_OUT = "1"

# Physics band + deploy support projection.
$env:CLOT_PHI_DGAMMA_SLICE = "1"
$env:CLOT_PHI_DGAMMA_REF_TIME = "0"
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"
$env:CLOT_PHI_HARD_SUPPORT_PROJECTION = "1"
$env:CLOT_PHI_SUPPORT_BAND = "physics"
$env:CLOT_FORECAST_MASK = "deploy_band"

# Full-mesh shape aux (off-band bulk suppression).
$env:CLOT_PHI_MESH_AUX_LAMBDA = "0.65"
$env:CLOT_PHI_MESH_BULK_LAMBDA = "0.50"

# No temporal carry until G2.
Remove-Item Env:CLOT_FORECAST_MU_CARRY -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_PHI -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_LOG_MU -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_FORECAST_INPUT_MU -ErrorAction SilentlyContinue

$env:CLOT_PHI_VAL_ANCHOR = "patient007"
$env:CLOT_PHI_SOFT_LABELS = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_POS_WEIGHT_CAP = "8"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_SPECIES_FEATURES = "0"
$env:CLOT_PHI_JOINT_BIO = "0"
