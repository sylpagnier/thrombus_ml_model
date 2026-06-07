# Shared R2a+ one-step phi-only base env (dot-source from sweep launchers).

$env:CLOT_FORECAST_MODE = "one_step"
$env:CLOT_FORECAST_PAIR_STRIDE = "1"
$env:CLOT_FORECAST_INPUT_MU = "1"
$env:CLOT_PHI_FIXED_MU_FROM_PHI = "1"
$env:CLOT_PHI_HYBRID = "0"
$env:CLOT_PHI_ROLLOUT = "0"
$env:CLOT_PHI_MU_SOLID_SI = "0.10"
$env:CLOT_PHI_MU_LOG_LAMBDA = "0"
$env:CLOT_PHI_SHAPE_USE_T_OUT = "1"
Remove-Item Env:CLOT_PHI_CARRY_PHI -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_CARRY_LOG_MU -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_MESH_AUX_LAMBDA -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_MESH_BULK_LAMBDA -ErrorAction SilentlyContinue

$env:BIOCHEM_MLP_NEIGHBOR_SEED = "pred_clot"
$env:BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI = "0"
$env:BIOCHEM_MLP_MU_MAP_PHI_THRESH = "0.5"
$env:BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE = "0"

$env:CLOT_PHI_MODEL = "mlp"
$env:CLOT_PHI_HIDDEN = "32"
$env:CLOT_PHI_MLP_DEPTH = "2"
$env:CLOT_PHI_DROPOUT = "0.15"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_PHI_SPECIES_FEATURES = "0"
$env:CLOT_PHI_JOINT_BIO = "0"
$env:CLOT_PHI_PHYSICS_BLEND = "0"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_TIME_STRIDE = "1"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "0"
