# Teacher-only viscosity V4 runner with two targeted profiles.
#
# Profiles:
#   global_plus        - short 4GB-safe baseline probe (existing).
#   high_mu_only       - short high-mu isolate probe (existing).
#   global_long_stable - long (~10h) stability-oriented global objective.
#   tail_bridge_long   - long tail-emphasis joint objective (5GB+ suggested).
#   carreau_tail_split_4g - new physics split-head run for 4GB.
#   carreau_tail_split_5g - new physics split-head run for 5GB.
#   carreau_tail_stageA_diag_4g - stage-A diagnostic (tail learning behavior).
#   carreau_tail_stageAB_5g - two-stage tail->global recovery (5GB).
#   carreau_tail_stageAB_wall_4g - 4GB follow-up: reintroduce wall in Stage B.
#   walltail_arch_v1_4g - new wall-aware split architecture (4GB).
#   walltail_arch_v1_5g - new wall-aware split architecture (5GB).
#   walltail_arch_v2_long_4g - smoother staged transition, long 4GB wall-tail run.
#   walltail_arch_v2_long_5g - smoother staged transition, long 5GB wall-tail run.
#   sweep_arch_4g - unattended 2-leg architecture sweep for 4GB.
#   sweep_arch_5g - unattended 2-leg architecture sweep for 5GB.
#
# Usage:
#   .\scripts\run_biochem_teacher_visc_v4.ps1 -Profile global_plus
#   .\scripts\run_biochem_teacher_visc_v4.ps1 -Profile high_mu_only
#   .\scripts\run_biochem_teacher_visc_v4.ps1 -ListProfiles
#
param(
    [ValidateSet("global_plus", "high_mu_only", "global_long_stable", "tail_bridge_long", "carreau_tail_split_4g", "carreau_tail_split_5g", "carreau_tail_stageA_diag_4g", "carreau_tail_stageAB_5g", "carreau_tail_stageAB_wall_4g", "walltail_arch_v1_4g", "walltail_arch_v1_5g", "walltail_arch_v2_long_4g", "walltail_arch_v2_long_5g", "sweep_arch_4g", "sweep_arch_5g")]
    [string] $Profile = "global_plus",
    [switch] $ListProfiles,
    [switch] $WideArch,
    [switch] $ForcePretrain,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($ListProfiles) {
    Write-Host "V4 teacher viscosity profiles:" -ForegroundColor Cyan
    Write-Host "  global_plus        - short joint step-2 objective, 4GB-safe arch by default"
    Write-Host "  high_mu_only       - short MU_LOG_HIGH isolate, 4GB-safe arch by default"
    Write-Host "  global_long_stable - long run, lower LR + TF floor to reduce late collapse"
    Write-Host "  tail_bridge_long   - long run, high-tail emphasis without hard isolate (5GB+)"
    Write-Host "  carreau_tail_split_4g - split bulk/tail heads + trigger gate, wall=0 (4GB-safe)"
    Write-Host "  carreau_tail_split_5g - split bulk/tail heads + trigger gate, wall=0 (5GB)"
    Write-Host "  carreau_tail_stageA_diag_4g - stage-A loss diagnostic (tail bug-check)"
    Write-Host "  carreau_tail_stageAB_5g - staged curriculum (A tail-heavy -> B global recovery)"
    Write-Host "  carreau_tail_stageAB_wall_4g - staged curriculum with wall reintroduced in Stage B (4GB)"
    Write-Host "  walltail_arch_v1_4g - wall-aware residual branch + split-head staged objective (4GB)"
    Write-Host "  walltail_arch_v1_5g - wall-aware residual branch + split-head staged objective (5GB)"
    Write-Host "  walltail_arch_v2_long_4g - long wall-tail run with smooth Stage-A->B transition (4GB)"
    Write-Host "  walltail_arch_v2_long_5g - long wall-tail run with smooth Stage-A->B transition (5GB)"
    Write-Host "  sweep_arch_4g - sequential sweep: carreau_tail_stageAB_wall_4g -> walltail_arch_v2_long_4g"
    Write-Host "  sweep_arch_5g - sequential sweep: walltail_arch_v1_5g -> walltail_arch_v2_long_5g"
    Write-Host ""
    Write-Host "Use -WideArch on >=6-8GB GPUs to try latent=320 variants."
    exit 0
}

function Invoke-TeacherProfileChild {
    param(
        [Parameter(Mandatory = $true)]
        [string] $ChildProfile
    )
    Write-Host ""
    Write-Host "=== Sweep leg start: $ChildProfile ===" -ForegroundColor Magenta
    $childArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-Profile", $ChildProfile
    )
    if ($ForcePretrain) {
        $childArgs += "-ForcePretrain"
    }
    & powershell @childArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Sweep leg '$ChildProfile' failed with exit code $LASTEXITCODE"
    }
    Write-Host "=== Sweep leg complete: $ChildProfile ===" -ForegroundColor Magenta
}

# Clear stale env from prior experiments.
Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

# Shared stable base from recent SAFEVAL/V3 runs.
$env:BIOCHEM_STOCK_DEFAULTS = "1"
$env:BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
$env:BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "24"
$env:BIOCHEM_TEACHER_VAL_EVERY = "4"
$env:BIOCHEM_VAL_TIME_STRIDE = "20"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
$env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
$env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_MU_SI_MULTI_STEP = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
$env:BIOCHEM_USE_WALL_DELTA_HEAD = "0"
$env:BIOCHEM_MU_WALL_MASK_MIX = "0.80"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:BIOCHEM_TEACHER_LR = "0.002"
$env:BIOCHEM_MU_PATH_LR_MULT = "1.0"
$env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "0"
$env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.0"
$env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.0"
$env:BIOCHEM_MU_WALL_LR_MULT = "1.0"
$env:BIOCHEM_MU_STAGE_TRANSITION_EPOCHS = "0"

switch ($Profile) {
    "sweep_arch_4g" {
        Invoke-TeacherProfileChild -ChildProfile "carreau_tail_stageAB_wall_4g"
        Invoke-TeacherProfileChild -ChildProfile "walltail_arch_v2_long_4g"
        Write-Host ""
        Write-Host "Sweep finished: sweep_arch_4g" -ForegroundColor Green
        exit 0
    }
    "sweep_arch_5g" {
        Invoke-TeacherProfileChild -ChildProfile "walltail_arch_v1_5g"
        Invoke-TeacherProfileChild -ChildProfile "walltail_arch_v2_long_5g"
        Write-Host ""
        Write-Host "Sweep finished: sweep_arch_5g" -ForegroundColor Green
        exit 0
    }
    "global_plus" {
        # 4GB-safe by default; optional wide arch for larger GPUs.
        $env:BIOCHEM_LATENT_DIM = if ($WideArch) { "320" } else { "256" }
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"

        # Balanced weights with slight tail emphasis (without wall over-push).
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "2.0"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "1.6"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "1.4"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "10"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "6"
        $env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE = "0.50"
        $env:BIOCHEM_RUN_NOTE = if ($WideArch) { "VISC_V4_GLOBAL_PLUS_WIDE" } else { "VISC_V4_GLOBAL_PLUS_SAFE" }
    }
    "high_mu_only" {
        # High-mu-focused 4GB-safe default; optional wide arch for larger GPUs.
        $env:BIOCHEM_LATENT_DIM = if ($WideArch) { "320" } else { "256" }
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = if ($WideArch) { "4" } else { "2" }

        # Isolate the high-mu subset objective to learn clot-tail behavior.
        $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG_HIGH"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "3.0"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "0"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.1"
        $env:BIOCHEM_RUN_NOTE = if ($WideArch) { "VISC_V4_HIGH_MU_ONLY_WIDE" } else { "VISC_V4_HIGH_MU_ONLY_SAFE" }
    }
    "global_long_stable" {
        # Long stability run for 4GB cards: keep proven arch, reduce optimizer aggressiveness.
        $env:BIOCHEM_LATENT_DIM = "256"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        $env:BIOCHEM_TEACHER_EPOCHS = "64"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
        $env:BIOCHEM_TEACHER_LR = "0.0010"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.65"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.10"

        # Keep balanced objective close to the best stable family.
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "1.5"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "1.4"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "1.2"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "14"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "14"
        $env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE = "0.49"
        $env:BIOCHEM_RUN_NOTE = "VISC_V5_GLOBAL_LONG_STABLE"
    }
    "tail_bridge_long" {
        # 5GB+ long run: bridge high-tail gains into full objective (no hard isolate).
        $env:BIOCHEM_LATENT_DIM = "320"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"
        $env:BIOCHEM_TEACHER_EPOCHS = "64"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
        $env:BIOCHEM_TEACHER_LR = "0.0008"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.50"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.15"

        # Tail-heavy but still anchored by all/wall/si terms.
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.2"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.8"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "1.6"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "2.8"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "16"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "24"
        $env:BIOCHEM_RUN_NOTE = "VISC_V5_TAIL_BRIDGE_LONG"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "carreau_tail_split_4g" {
        # New-physics architecture: Carreau baseline + trigger-gated split residual heads.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.12"
        $env:BIOCHEM_LATENT_DIM = "256"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        $env:BIOCHEM_TEACHER_EPOCHS = "64"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
        $env:BIOCHEM_TEACHER_LR = "0.0009"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.60"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.10"
        # Remove wall objective; focus on all + high tail.
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.2"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "3.2"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "20"
        # Anti-collapse + Pareto checkpointing.
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.8"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.35"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.4"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.03"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.035"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.04"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.01"
        $env:BIOCHEM_RUN_NOTE = "VISC_V6_CARREAU_TAIL_SPLIT_4G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "carreau_tail_split_5g" {
        # Same new-physics design on wider model for 5GB machine.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.10"
        $env:BIOCHEM_LATENT_DIM = "320"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"
        $env:BIOCHEM_TEACHER_EPOCHS = "64"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
        $env:BIOCHEM_TEACHER_LR = "0.0007"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.45"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.15"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.2"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.2"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "3.0"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "24"
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.8"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.35"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.4"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.03"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.04"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.04"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.01"
        $env:BIOCHEM_RUN_NOTE = "VISC_V6_CARREAU_TAIL_SPLIT_5G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "carreau_tail_stageA_diag_4g" {
        # Diagnostic run: prove tail losses/pathways can move at all (bug-check profile).
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.10"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_START = "0.16"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_END = "0.08"
        $env:BIOCHEM_LATENT_DIM = "256"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        $env:BIOCHEM_TEACHER_EPOCHS = "36"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
        $env:BIOCHEM_TEACHER_LR = "0.0010"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.80"
        $env:BIOCHEM_MU_BULK_LR_MULT = "0.55"
        $env:BIOCHEM_MU_TAIL_LR_MULT = "1.60"
        $env:BIOCHEM_MU_GATE_LR_MULT = "1.35"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.12"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.5"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.05"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "4.0"
        $env:BIOCHEM_MU_LOG_BOUNDARY_WEIGHT = "1.6"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "8"
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.20"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.25"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.10"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.02"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.05"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.05"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.001"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.01"
        $env:BIOCHEM_RUN_NOTE = "VISC_V7_STAGEA_DIAG_4G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "carreau_tail_stageAB_5g" {
        # Two-stage run: Stage A tail-heavy, Stage B recover global while preserving tail.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.12"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_START = "0.18"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_END = "0.07"
        $env:BIOCHEM_LATENT_DIM = "320"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"
        $env:BIOCHEM_TEACHER_EPOCHS = "64"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
        $env:BIOCHEM_TEACHER_LR = "0.0008"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.60"
        $env:BIOCHEM_MU_BULK_LR_MULT = "0.60"
        $env:BIOCHEM_MU_TAIL_LR_MULT = "1.45"
        $env:BIOCHEM_MU_GATE_LR_MULT = "1.25"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.15"
        # Stage A weights
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.8"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.10"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "3.2"
        $env:BIOCHEM_MU_LOG_BOUNDARY_WEIGHT = "1.2"
        # Stage B switch + weights
        $env:BIOCHEM_MU_STAGE_SWITCH_EPOCH = "24"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT_STAGE_B = "1.6"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT_STAGE_B = "0.35"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT_STAGE_B = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT_STAGE_B = "1.8"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "20"
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.25"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.25"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.12"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.02"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.035"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.05"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.01"
        $env:BIOCHEM_RUN_NOTE = "VISC_V7_STAGEAB_5G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "carreau_tail_stageAB_wall_4g" {
        # 4GB follow-up: keep split-tail architecture, then reintroduce wall in Stage B.
        # Goal: preserve tail gains from Stage A while recovering all/wall in Stage B.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.11"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_START = "0.17"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_END = "0.08"
        $env:BIOCHEM_LATENT_DIM = "256"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        $env:BIOCHEM_TEACHER_EPOCHS = "48"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
        $env:BIOCHEM_TEACHER_LR = "0.0009"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.75"
        $env:BIOCHEM_MU_BULK_LR_MULT = "0.65"
        $env:BIOCHEM_MU_TAIL_LR_MULT = "1.50"
        $env:BIOCHEM_MU_GATE_LR_MULT = "1.20"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.12"
        # Stage A (tail-focused, wall off)
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.7"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.08"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "3.4"
        $env:BIOCHEM_MU_LOG_BOUNDARY_WEIGHT = "1.2"
        # Stage B switch (bring global + wall back, reduce tail pressure)
        $env:BIOCHEM_MU_STAGE_SWITCH_EPOCH = "18"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT_STAGE_B = "1.5"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT_STAGE_B = "0.30"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT_STAGE_B = "0.8"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT_STAGE_B = "1.9"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "14"
        # Keep anti-collapse light; avoid V6 over-constraint.
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.20"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.25"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.10"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.02"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.03"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.05"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.008"
        $env:BIOCHEM_RUN_NOTE = "VISC_V8_STAGEAB_WALL_4G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "walltail_arch_v1_4g" {
        # New architecture probe (4GB): split bulk/tail + dedicated near-wall residual head.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.11"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_START = "0.16"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_END = "0.07"
        $env:BIOCHEM_MU_WALL_GATE_TEMP = "0.16"
        $env:BIOCHEM_MU_WALL_GATE_CENTER = "0.52"
        $env:BIOCHEM_MU_WALL_DELTA_GAIN = "0.75"
        $env:BIOCHEM_LATENT_DIM = "256"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        $env:BIOCHEM_TEACHER_EPOCHS = "52"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
        $env:BIOCHEM_TEACHER_LR = "0.00085"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.72"
        $env:BIOCHEM_MU_BULK_LR_MULT = "0.65"
        $env:BIOCHEM_MU_TAIL_LR_MULT = "1.45"
        $env:BIOCHEM_MU_GATE_LR_MULT = "1.20"
        $env:BIOCHEM_MU_WALL_LR_MULT = "1.55"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.12"
        # Stage A: tail+boundary discovery without explicit wall penalty.
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.9"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.10"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "3.0"
        $env:BIOCHEM_MU_LOG_BOUNDARY_WEIGHT = "1.4"
        # Stage B: recover global + wall while retaining high-tail.
        $env:BIOCHEM_MU_STAGE_SWITCH_EPOCH = "18"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT_STAGE_B = "1.6"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT_STAGE_B = "0.30"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT_STAGE_B = "1.1"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT_STAGE_B = "2.0"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "12"
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.20"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.25"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.10"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.02"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.03"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.05"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.008"
        $env:BIOCHEM_RUN_NOTE = "VISC_V9_WALLTAIL_ARCH_V1_4G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "walltail_arch_v1_5g" {
        # New architecture probe (5GB): wider split-head with explicit near-wall residual branch.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.12"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_START = "0.18"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_END = "0.07"
        $env:BIOCHEM_MU_WALL_GATE_TEMP = "0.15"
        $env:BIOCHEM_MU_WALL_GATE_CENTER = "0.50"
        $env:BIOCHEM_MU_WALL_DELTA_GAIN = "0.85"
        $env:BIOCHEM_LATENT_DIM = "320"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"
        $env:BIOCHEM_TEACHER_EPOCHS = "64"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
        $env:BIOCHEM_TEACHER_LR = "0.00078"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.62"
        $env:BIOCHEM_MU_BULK_LR_MULT = "0.60"
        $env:BIOCHEM_MU_TAIL_LR_MULT = "1.40"
        $env:BIOCHEM_MU_GATE_LR_MULT = "1.20"
        $env:BIOCHEM_MU_WALL_LR_MULT = "1.70"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.15"
        # Stage A
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.12"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "2.8"
        $env:BIOCHEM_MU_LOG_BOUNDARY_WEIGHT = "1.3"
        # Stage B
        $env:BIOCHEM_MU_STAGE_SWITCH_EPOCH = "24"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT_STAGE_B = "1.8"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT_STAGE_B = "0.35"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT_STAGE_B = "1.2"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT_STAGE_B = "1.8"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "0"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "14"
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.20"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.25"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.10"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.02"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.03"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.05"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.008"
        $env:BIOCHEM_RUN_NOTE = "VISC_V9_WALLTAIL_ARCH_V1_5G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "walltail_arch_v2_long_4g" {
        # 4GB long run (~10h): smooth objective transition + stronger wall branch supervision.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.11"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_START = "0.16"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_END = "0.07"
        $env:BIOCHEM_MU_WALL_GATE_TEMP = "0.14"
        $env:BIOCHEM_MU_WALL_GATE_CENTER = "0.48"
        $env:BIOCHEM_MU_WALL_MASK_MIX = "0.90"
        $env:BIOCHEM_MU_WALL_DELTA_GAIN = "0.95"
        $env:BIOCHEM_LATENT_DIM = "256"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        $env:BIOCHEM_TEACHER_EPOCHS = "66"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
        $env:BIOCHEM_TEACHER_LR = "0.00072"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.58"
        $env:BIOCHEM_MU_BULK_LR_MULT = "0.62"
        $env:BIOCHEM_MU_TAIL_LR_MULT = "1.32"
        $env:BIOCHEM_MU_GATE_LR_MULT = "1.15"
        $env:BIOCHEM_MU_WALL_LR_MULT = "1.85"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.14"
        # Stage A: stabilize global+tail before wall pressure.
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.12"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "2.6"
        $env:BIOCHEM_MU_LOG_BOUNDARY_WEIGHT = "1.2"
        # Stage B: introduce wall smoothly.
        $env:BIOCHEM_MU_STAGE_SWITCH_EPOCH = "18"
        $env:BIOCHEM_MU_STAGE_TRANSITION_EPOCHS = "12"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT_STAGE_B = "1.9"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT_STAGE_B = "0.32"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT_STAGE_B = "1.6"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT_STAGE_B = "1.8"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "12"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "16"
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.16"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.22"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.08"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.015"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.025"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.045"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.007"
        $env:BIOCHEM_RUN_NOTE = "VISC_V10_WALLTAIL_ARCH_V2_LONG_4G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
    "walltail_arch_v2_long_5g" {
        # 5GB long run (~10h): wider model + smooth Stage-A->B, with stronger wall branch.
        $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
        $env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP = "0.12"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_START = "0.18"
        $env:BIOCHEM_MU_TRIGGER_GATE_TEMP_END = "0.07"
        $env:BIOCHEM_MU_WALL_GATE_TEMP = "0.13"
        $env:BIOCHEM_MU_WALL_GATE_CENTER = "0.46"
        $env:BIOCHEM_MU_WALL_MASK_MIX = "0.95"
        $env:BIOCHEM_MU_WALL_DELTA_GAIN = "1.00"
        $env:BIOCHEM_LATENT_DIM = "320"
        $env:BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"
        $env:BIOCHEM_TEACHER_EPOCHS = "64"
        $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
        $env:BIOCHEM_TEACHER_LR = "0.00068"
        $env:BIOCHEM_MU_PATH_LR_MULT = "0.56"
        $env:BIOCHEM_MU_BULK_LR_MULT = "0.60"
        $env:BIOCHEM_MU_TAIL_LR_MULT = "1.28"
        $env:BIOCHEM_MU_GATE_LR_MULT = "1.12"
        $env:BIOCHEM_MU_WALL_LR_MULT = "1.95"
        $env:BIOCHEM_TEACHER_FORCE_MIN = "0.16"
        # Stage A
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.1"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.14"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "2.4"
        $env:BIOCHEM_MU_LOG_BOUNDARY_WEIGHT = "1.2"
        # Stage B
        $env:BIOCHEM_MU_STAGE_SWITCH_EPOCH = "21"
        $env:BIOCHEM_MU_STAGE_TRANSITION_EPOCHS = "12"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT_STAGE_B = "2.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT_STAGE_B = "0.34"
        $env:BIOCHEM_MU_LOG_WALL_WEIGHT_STAGE_B = "1.7"
        $env:BIOCHEM_MU_LOG_HIGH_WEIGHT_STAGE_B = "1.7"
        $env:BIOCHEM_MU_LOG_WALL_RAMP_EPOCHS = "12"
        $env:BIOCHEM_MU_LOG_HIGH_RAMP_EPOCHS = "16"
        $env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT = "0.16"
        $env:BIOCHEM_TRIGGER_GATE_MIN_HIGH = "0.22"
        $env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT = "0.08"
        $env:BIOCHEM_TRIGGER_LEARNED_MIN_HIGH = "0.015"
        $env:BIOCHEM_TEACHER_PARETO_CHECKPOINT = "1"
        $env:BIOCHEM_TEACHER_PARETO_ALL_TOL = "0.025"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_TOL = "0.045"
        $env:BIOCHEM_TEACHER_PARETO_ALL_GAIN_MIN = "0.0015"
        $env:BIOCHEM_TEACHER_PARETO_HIGH_GAIN_MIN = "0.007"
        $env:BIOCHEM_RUN_NOTE = "VISC_V10_WALLTAIL_ARCH_V2_LONG_5G"
        Remove-Item Env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE -ErrorAction SilentlyContinue
    }
}

Write-Host "Teacher viscosity V4 run | profile=$Profile" -ForegroundColor Cyan
Write-Host "Warm-start: $UseWarmStart | Corrector: OFF | epochs=$env:BIOCHEM_TEACHER_EPOCHS" -ForegroundColor Cyan
Write-Host "Arch: latent=$env:BIOCHEM_LATENT_DIM prior_dim=$env:BIOCHEM_BIO_ENCODER_PRIOR_DIM delta_head=$env:BIOCHEM_USE_DELTA_MU_HEAD" -ForegroundColor Cyan
Write-Host "OOM guard: TBPTT=$env:BIOCHEM_TBPTT_MAX_WINDOW RK4=$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS CUDA_ALLOC_CONF=$env:PYTORCH_CUDA_ALLOC_CONF" -ForegroundColor DarkGray
Write-Host "Optimizer: teacher_lr=$env:BIOCHEM_TEACHER_LR mu_path_lr_mult=$env:BIOCHEM_MU_PATH_LR_MULT TFmin=$env:BIOCHEM_TEACHER_FORCE_MIN val_every=$env:BIOCHEM_TEACHER_VAL_EVERY" -ForegroundColor DarkGray
Write-Host "Weights: MuLog=$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT MuSI=$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT Wall=$env:BIOCHEM_MU_LOG_WALL_WEIGHT High=$env:BIOCHEM_MU_LOG_HIGH_WEIGHT" -ForegroundColor Cyan
Write-Host "New physics: split_mu_head=$env:BIOCHEM_USE_SPLIT_MU_HEAD wall_delta_head=$env:BIOCHEM_USE_WALL_DELTA_HEAD pareto_ckpt=$env:BIOCHEM_TEACHER_PARETO_CHECKPOINT trigger_floor=$env:BIOCHEM_TRIGGER_GATE_FLOOR_WEIGHT/$env:BIOCHEM_TRIGGER_LEARNED_FLOOR_WEIGHT" -ForegroundColor DarkGray
if ($env:BIOCHEM_LOSS_ISOLATE) {
    Write-Host "Isolate objective: $env:BIOCHEM_LOSS_ISOLATE" -ForegroundColor Yellow
}
if ($UseWarmStart) {
    Write-Host "Warm-start checkpoint: $WarmStart" -ForegroundColor Yellow
}

python -m src.training.train_biochem_corrector --new @ExtraArgs
if ($LASTEXITCODE -ne 0) {
    throw "train_biochem_corrector failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Check outputs/reports/training/biochem/<timestamp>/metrics.jsonl" -ForegroundColor Green
Write-Host "Track val subsets: mu_log_mae(all/wall/high), mu_pearson, and train L_Back."
