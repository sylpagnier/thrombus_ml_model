# Laptop B (Quadro P2200 5GB) - teacher architecture sweep.
# Goal: test wider teacher architectures and prior-channel capacity on the same teacher recipe.
# Keep optimization settings fixed (MU_LOG isolate, low TF), vary architecture knobs only.
#
# Usage (repo root, venv active):
#   .\scripts\run_biochem_teacher_architecture_laptop_b.ps1
#   .\scripts\run_biochem_teacher_architecture_laptop_b.ps1 -ListLegs
#   .\scripts\run_biochem_teacher_architecture_laptop_b.ps1 -Legs @("B0","B2") -DryRun
#
# Prereq: outputs\biochem\biochem_post_pretrain.pth
# Summary: outputs/reports/training/biochem/laptop_b_teacher_architecture_summary.txt

param(
    [string[]] $Legs = @("B0", "B1", "B2", "B3", "B4"),
    [switch] $ListLegs,
    [switch] $DryRun,
    [switch] $ForcePretrain,
    [int] $OomSafe = 1,
    [int] $TeacherEpochs = 8,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_biochem_teacher_complexity_common.ps1")

$LegCatalog = @{
    B0 = "Baseline arch: latent=256, prior_dim=2, delta_mu_head=1"
    B1 = "Wider latent: latent=320, prior_dim=2, delta_mu_head=1"
    B2 = "Wider + richer priors: latent=320, prior_dim=4, delta_mu_head=1"
    B3 = "Richer priors only: latent=256, prior_dim=4, delta_mu_head=1"
    B4 = "Delta-head ablation at larger latent: latent=320, prior_dim=2, delta_mu_head=0"
}

if ($ListLegs) {
    Write-Host "Laptop B - teacher architecture sweep legs:" -ForegroundColor Cyan
    foreach ($k in ($LegCatalog.Keys | Sort-Object)) { Write-Host "  $k  $($LegCatalog[$k])" }
    Write-Host ""
    Write-Host "Default order: B0,B1,B2,B3,B4"
    Write-Host "Pair with: .\scripts\run_biochem_teacher_architecture_laptop_a.ps1"
    exit 0
}

$WarmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $WarmStart)

$Detach = if ($OomSafe -ne 0) { "1" } else { "0" }
$Rk4 = if ($OomSafe -ne 0) { "10" } else { "12" }

$Base = @{
    BIOCHEM_STOCK_DEFAULTS = "1"
    BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    BIOCHEM_STOP_AFTER_TEACHER = "1"
    BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    BIOCHEM_LOSS_DATA_ONLY = "1"
    BIOCHEM_PRESET = ""
    BIOCHEM_COMPLEXITY_STEP = "2"
    BIOCHEM_TEACHER_MU_RATIO_MAX = "20.0"
    BIOCHEM_TEACHER_FORCE_MIN = "0.0"
    BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
    BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    BIOCHEM_TEACHER_VAL_EVERY = "2"
    BIOCHEM_VAL_TIME_STRIDE = "10"
    BIOCHEM_TBPTT_MAX_WINDOW = "6"
    BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    BIOCHEM_DETACH_MACRO_STATE = $Detach
    BIOCHEM_ADJOINT_RK4_SUBSTEPS = $Rk4
    BIOCHEM_DATALOADER_WORKERS = "0"
    BIOCHEM_TEACHER_SKIP_VAL = "0"
    BIOCHEM_DEBUG = "0"
    BIOCHEM_LOSS_ISOLATE = "MU_LOG"
    BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
    BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    BIOCHEM_MU_SI_MULTI_STEP = "1"
    BIOCHEM_DATA_ONLY_PHYS_TEMP = "0"
    BIOCHEM_TRAIN_MU_ENCODER = "1"
    BIOCHEM_USE_MU_PATH_GROUP = "1"
    BIOCHEM_FI_GATE_START_WEIGHT = "0.0"
}

$EarlyWindow = @{
    BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
}

$LegDefs = @{
    B0 = @{
        Label = "B0_arch_baseline_lat256_prior2_delta1"
        BIOCHEM_LATENT_DIM = "256"
        BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        BIOCHEM_USE_DELTA_MU_HEAD = "1"
    }
    B1 = @{
        Label = "B1_arch_wide_lat320_prior2_delta1"
        BIOCHEM_LATENT_DIM = "320"
        BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        BIOCHEM_USE_DELTA_MU_HEAD = "1"
    }
    B2 = @{
        Label = "B2_arch_wide_lat320_prior4_delta1"
        BIOCHEM_LATENT_DIM = "320"
        BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"
        BIOCHEM_USE_DELTA_MU_HEAD = "1"
    }
    B3 = @{
        Label = "B3_arch_lat256_prior4_delta1"
        BIOCHEM_LATENT_DIM = "256"
        BIOCHEM_BIO_ENCODER_PRIOR_DIM = "4"
        BIOCHEM_USE_DELTA_MU_HEAD = "1"
    }
    B4 = @{
        Label = "B4_arch_wide_lat320_prior2_delta0"
        BIOCHEM_LATENT_DIM = "320"
        BIOCHEM_BIO_ENCODER_PRIOR_DIM = "2"
        BIOCHEM_USE_DELTA_MU_HEAD = "0"
    }
}

$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\laptop_b_teacher_architecture_summary.txt"
$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) { New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null }

Write-Host "Laptop B teacher architecture sweep | legs=$($Legs -join ',') | OOM-safe=$OomSafe" -ForegroundColor Cyan
Write-Host "  Fixed recipe: MU_LOG isolate, TBPTT=6, val_every=2, stride=10, teacher_ep=$TeacherEpochs" -ForegroundColor DarkGray
if ($UseWarmStart) { Write-Host "Warm-start: $WarmStart" -ForegroundColor Yellow }
else { Write-Host "No warm-start - first leg runs pretrain (longer)." -ForegroundColor Red }

"laptop B architecture started $(Get-Date -Format o) legs=$($Legs -join ',') oom_safe=$OomSafe ep=$TeacherEpochs" |
    Set-Content -Path $SummaryPath -Encoding utf8

$legIndex = 0
$legTotal = $Legs.Count
foreach ($leg in $Legs) {
    $key = $leg.ToUpper()
    if (-not $LegDefs.ContainsKey($key)) {
        Write-Warning "Unknown leg '$leg'; skip. Use -ListLegs."
        continue
    }
    $legIndex++
    Invoke-BiochemTeacherLeg -LegKey $key -LegDef $LegDefs[$key] -Base $Base -EarlyWindow $EarlyWindow `
        -UseWarmStart $UseWarmStart -LegIndex $legIndex -LegTotal $legTotal -SummaryPath $SummaryPath `
        -ExtraArgs $ExtraArgs -DryRun:$DryRun
}

Write-Host ""
Write-Host "Laptop B architecture sweep done. Summary: $SummaryPath" -ForegroundColor Green
Write-Host "Compare val mu_log_mae (all|wall|high-mu) and r across B0-B4."
