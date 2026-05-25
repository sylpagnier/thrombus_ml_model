# K11e: localized clot gate (best-practice bundle after K11d diagnostic).
#
# - Fixed GT clot label (absolute μ floor only; no ratio×μ_inf)
# - Forward: adjacent apply, no trigger multiply at inference (TRIGGER_APPLY=0)
# - BCE support: adjacent ∪ GT clot ∪ COMSOL trigger (sync)
# - Mech prior: COMSOL-aligned d(spf.sr,x) hybrid (adjacent p95 norm; no top-quantile zeroing)
# - Checkpoint: K11 localization score (gate + clot_frac), not all-truth logMAE alone
#
# Prereq: outputs/kinematics/kinematics_best.pth
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k11e_clot_gate.ps1" -Fresh

param([switch] $Fresh, [switch] $OomSafe = $true)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
if (-not (Test-Path (Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"))) {
    Write-Host "Missing kinematics_best.pth" -ForegroundColor Red
    exit 1
}

$env:KINEMATICS_USE_HARD_BCS = "1"
$env:KINEMATICS_USE_WIDTH_PRIORS = "1"
$env:BIOCHEM_MU_K11_CLOT_GATE = "1"
Remove-Item Env:BIOCHEM_MU_K10E_SIMPLE,Env:BIOCHEM_MU_K10D_SIMPLE,Env:BIOCHEM_K10G_ORACLE_CLOTS -ErrorAction SilentlyContinue
$env:BIOCHEM_MU_IC_STEADY_KIN = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
Remove-Item Env:BIOCHEM_USE_WALL_DELTA_HEAD,Env:BIOCHEM_MU_ADDITIVE_DELTA -ErrorAction SilentlyContinue

$env:BIOCHEM_K11_D_PEAK_ND = "0.008"
$env:BIOCHEM_K11_SIGMA_ND = "0.008"
$env:BIOCHEM_K11_SDF_MAX_ND = "0.04"
$env:BIOCHEM_K11_MU_CLOT_SI = "0.10"
$env:BIOCHEM_K11_APPLY_MODE = "adjacent"
$env:BIOCHEM_K11_WALL_PROX_LAMBDA_ND = "0.008"
$env:BIOCHEM_K11_CLOT_GROWTH = "0"
$env:BIOCHEM_K11_CLOT_MU_SI_MIN = "0.055"
$env:BIOCHEM_K11_BCE_ON_RAW = "1"
$env:BIOCHEM_K11_BCE_SUPPORT = "adjacent_trigger"
$env:BIOCHEM_K11_BCE_TRIG_THRESH = "0.25"
$env:BIOCHEM_K11_CLOT_POS_WEIGHT = "8.0"
$env:BIOCHEM_K11_CLOT_NEG_WEIGHT = "4.0"
$env:BIOCHEM_K11_WALL_NEG_WEIGHT = "6.0"
$env:BIOCHEM_K11_CLOT_LOGIT_BIAS = "-0.5"
$env:BIOCHEM_K11_LOGIT_TEMP = "1.0"
$env:BIOCHEM_VIZ_CLOT_MU_SI_THRESH = "0.055"

# Triggers: used in loss weighting / suppress only; not multiplied into forward p_clot.
$env:BIOCHEM_K11_TRIGGER_WEIGHT = "2.0"
$env:BIOCHEM_K11_TRIGGER_APPLY = "0"
$env:BIOCHEM_K11_TRIGGER_TIME = "sync"
$env:BIOCHEM_K11_TRIGGER_W_BIO = "0.45"
$env:BIOCHEM_K11_TRIGGER_W_MECH = "0.55"
$env:BIOCHEM_K11_TRIGGER_COMBINE = "or"
Remove-Item Env:BIOCHEM_K11_MECH_QUANTILE -ErrorAction SilentlyContinue
$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_DGAMMA_DX_THRESH = "800"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"
$env:BIOCHEM_PRIOR_NORM_MODE = "adjacent_p95"
$env:BIOCHEM_K11_TRIGGER_MIN_WEIGHT = "0.10"
$env:BIOCHEM_K11_TRIGGER_SUPPRESS_WEIGHT = "1.5"
$env:BIOCHEM_K11_TRIGGER_SUPPRESS_THRESH = "0.25"
$env:BIOCHEM_K11_GATE_TARGET_MEAN = "0.006"
$env:BIOCHEM_K11_GATE_TARGET_WEIGHT = "3.0"
$env:BIOCHEM_K11_CKPT_SCORE = "1"

$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_ODE = "0"
$env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
$env:BIOCHEM_TRAIN_BIO_DECODER = "0"
$env:BIOCHEM_TRAIN_KIN_LORA = "0"
$env:BIOCHEM_LOSS_ISOLATE = "K11"
$env:BIOCHEM_K11_CLOT_BCE_WEIGHT = "1.0"
$env:BIOCHEM_K11_CLOT_MU_HUBER_WEIGHT = "1.0"
$env:BIOCHEM_K11_WALL_FP_WEIGHT = "6.0"
$env:BIOCHEM_K11_DATA_KINE_WEIGHT = "0.20"
Remove-Item Env:BIOCHEM_MU_LOG_WALL_WEIGHT,Env:BIOCHEM_LOSS_DATA_ONLY,Env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT -ErrorAction SilentlyContinue
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "3"
$env:BIOCHEM_TEACHER_EPOCHS = "12"
$env:BIOCHEM_EPOCHS = "12"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "12"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_STOCK_DEFAULTS = "0"
$env:BIOCHEM_RUN_NOTE = "K11e_localized_best_practice"

if ($OomSafe) {
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
}

$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_SKIP_PRETRAIN = "0"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
$env:BIOCHEM_AE_EPOCHS = "14"
$env:BIOCHEM_ODE_RXN_EPOCHS = "12"
Remove-Item Env:BIOCHEM_INIT_FROM_BEST -ErrorAction SilentlyContinue

if ($Fresh) { Remove-Item (Join-Path $RepoRoot "outputs\biochem\*.pth") -ErrorAction SilentlyContinue }

python -m src.training.train_biochem_corrector --new --epochs 12 --save-best --run-name K11e_localized_best_practice

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Training OK. Viz (K11 p_clot row + forward_policy from ckpt):" -ForegroundColor Cyan
    Write-Host '  python -m src.evaluation.visualize_pipeline --teacher-only --biochem-checkpoint outputs/biochem/biochem_teacher_best_high_mu.pth --anchor patient007'
    Write-Host ""
    Write-Host "Prior vs COMSOL diagnostic:" -ForegroundColor DarkGray
    Write-Host '  python scripts/diagnose_k11_clot_prior.py --anchor patient007 --checkpoint outputs/biochem/biochem_teacher_best_high_mu.pth'
}

exit $LASTEXITCODE
