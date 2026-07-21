# Lane A deploy clot-phi base env: pred-kine dump anchors + forecast one-step head.
# Dot-source from go_gnode12_lane_a_deploy_clot.ps1 and autonomy launchers.

. (Join-Path $PSScriptRoot "_clot_forecast_r2a_plus_base.ps1")

$env:CLOT_PHI_ANCHOR_DIR = "outputs/biochem/gnode10_sweep/anchors_gnode12_predkine_uvp"
# Dump wrote pred [u,v,p] into y ch 0:2; vel=gt reads y (deploy-faithful flow on dump files).
$env:CLOT_PHI_VEL_SOURCE = "gt"
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"
$env:CLOT_PHI_TIME_STRIDE = "1"
$env:CLOT_PHI_TIME_STRIDE_AUTO = "0"
$env:CLOT_PHI_DGAMMA_SLICE = "1"
$env:CLOT_PHI_DGAMMA_REF_TIME = "0"
$env:CLOT_PHI_HARD_SUPPORT_PROJECTION = "1"
$env:CLOT_PHI_SUPPORT_BAND = "physics"
$env:CLOT_PHI_MASK_MODE = "neighbor"
$env:CLOT_PHI_VAL_ANCHOR = "patient007"
$env:CLOT_PHI_SOFT_LABELS = "1"
$env:CLOT_PHI_BALANCED = "1"
$env:CLOT_PHI_POS_WEIGHT_CAP = "8"
$env:CLOT_PHI_DICE_LAMBDA = "0.2"
$env:CLOT_PHI_LR = "1e-3"
$env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
Remove-Item Env:CLOT_PHI_JOINT_BIO -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_PHYSICS_BLEND -ErrorAction SilentlyContinue
Remove-Item Env:CLOT_PHI_HYBRID -ErrorAction SilentlyContinue
