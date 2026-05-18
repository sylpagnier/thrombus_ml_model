# μ formulation study — teacher-only, MU_LOG-first, before step-2 multitask / corona.
# Goal: improve and understand μ closure on standard held-out val (patient007).
#
# Usage (repo root):
#   .\scripts\run_biochem_mu_formulation_study.ps1 -ListLegs
#   .\scripts\run_biochem_mu_formulation_study.ps1 -Phase A -Leg A0
#   .\scripts\run_biochem_mu_formulation_study.ps1 -Phase B -Leg B1 -NewRun
#
# Docs: src/docs/BIOCHEM_TRAINING_PROGRESS.md (μ formulation study plan)

param(
    [ValidateSet("A", "B", "C", "D")]
    [string] $Phase = "A",
    [string] $Leg = "A0",
    [switch] $ListLegs,
    [switch] $NewRun,
    [int] $OomSafe = 1,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LegCatalog = @{
    "A0" = "Reproduce MU_LOG+mu-path on full anchors (patient007 val); 12 ep"
    "A1" = "A0 + DETACH_MACRO=0, TBPTT=6 (needs VRAM; set -OomSafe 0 if 5GB GPU)"
    "A2" = "A0 + TEACHER_MU_RATIO_MAX=80"
    "B0" = "Ablation reference: full mu-path stack"
    "B1" = "Ablation: no delta_mu head"
    "B2" = "Ablation: freeze mu_encoder"
    "B3" = "Ablation: no delta, no mu_encoder (explicit+learned_clot only)"
    "B4" = "Ablation: joint W_MuLog=2 + W_MuSI=4 (no isolate) — tail probe"
    "C0" = "Temporal stress: TBPTT=8, 16 ep, TF warmup 4"
    "C1" = "Temporal stress: DETACH=0, TBPTT=6"
    "C2" = "AR stress: TEACHER_FORCE_MIN=0.2"
    "D1" = "Coupling: step-2 DATA_ONLY, W_MuLog=2, no isolate (adds L_Data_Kine)"
    "D2" = "Coupling: D1 + W_MuSI=4"
    "D3" = "Coupling: MU_LOG isolate + DATA_ONLY_PHYS_TEMP"
}

if ($ListLegs) {
    Write-Host "μ formulation study legs:" -ForegroundColor Cyan
    foreach ($k in ($LegCatalog.Keys | Sort-Object)) {
        Write-Host "  $k  $($LegCatalog[$k])"
    }
    Write-Host ""
    Write-Host "Phases: A=reproduce, B=ablate, C=temporal, D=coupling (run D only after A pass)"
    exit 0
}

if (-not $LegCatalog.ContainsKey($Leg)) {
    throw "Unknown leg '$Leg'. Use -ListLegs."
}

Write-Host "μ formulation study: Phase=$Phase Leg=$Leg" -ForegroundColor Cyan
Write-Host "  $($LegCatalog[$Leg])" -ForegroundColor DarkGray

Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$warmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$useWarm = Test-Path $warmStart

# --- shared μ-study defaults ---
$env:BIOCHEM_RUN_NOTE = "mu_study_P${Phase}_${Leg}"
$env:BIOCHEM_STOCK_DEFAULTS = "1"
$env:BIOCHEM_PRESET = ""
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
$env:BIOCHEM_MU_SI_MULTI_STEP = "1"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_DELTA_MU_LOG_CLIP = "2.0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
$env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "20.0"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_SKIP_VAL = "0"
$env:BIOCHEM_TEACHER_VAL_EVERY = "2"
$env:BIOCHEM_TEACHER_EPOCHS = "12"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "4"
$env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_DEBUG = "0"
$env:BIOCHEM_LOW_ANCHOR_MODE = "0"
# Full anchor load — do NOT cap vessels (patient007 val)
Remove-Item Env:BIOCHEM_MAX_LOAD_VESSELS -ErrorAction SilentlyContinue
Remove-Item Env:BIOCHEM_MAX_LOAD_SHUFFLE -ErrorAction SilentlyContinue

if ($useWarm) {
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
} else {
    $env:BIOCHEM_SKIP_PRETRAIN = "0"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
}

if ($OomSafe -ne 0) {
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
} else {
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "12"
}

switch ($Leg) {
    "A0" { }
    "A1" {
        if ($OomSafe -ne 0) {
            Write-Host "A1 requests DETACH=0; pass -OomSafe 0 on GPUs with headroom." -ForegroundColor Yellow
        }
        $env:BIOCHEM_DETACH_MACRO_STATE = "0"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "12"
    }
    "A2" { $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0" }
    "B0" { $env:BIOCHEM_TEACHER_EPOCHS = "8" }
    "B1" {
        $env:BIOCHEM_USE_DELTA_MU_HEAD = "0"
        $env:BIOCHEM_TEACHER_EPOCHS = "8"
    }
    "B2" {
        $env:BIOCHEM_TRAIN_MU_ENCODER = "0"
        $env:BIOCHEM_TEACHER_EPOCHS = "8"
    }
    "B3" {
        $env:BIOCHEM_USE_DELTA_MU_HEAD = "0"
        $env:BIOCHEM_TRAIN_MU_ENCODER = "0"
        $env:BIOCHEM_TEACHER_EPOCHS = "8"
    }
    "B4" {
        Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
        $env:BIOCHEM_LOSS_DATA_ONLY = "1"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "4.0"
        $env:BIOCHEM_TEACHER_EPOCHS = "8"
    }
    "C0" {
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "8"
        $env:BIOCHEM_TEACHER_EPOCHS = "16"
        $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
    }
    "C1" {
        $env:BIOCHEM_DETACH_MACRO_STATE = "0"
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "12"
    }
    "C2" { $env:BIOCHEM_TEACHER_FORCE_MIN = "0.2" }
    "D1" {
        Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
        $env:BIOCHEM_LOSS_DATA_ONLY = "1"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    }
    "D2" {
        Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
        $env:BIOCHEM_LOSS_DATA_ONLY = "1"
        $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
        $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "4.0"
    }
    "D3" {
        $env:BIOCHEM_DATA_ONLY_PHYS_TEMP = "1"
        # w_pt from stock/teacher defaults when NO_TEACHER_DEFAULTS=1 may be 0 — set explicitly if needed
    }
}

Write-Host ""
Write-Host "Key env:" -ForegroundColor Cyan
Write-Host "  LOSS_ISOLATE=$($env:BIOCHEM_LOSS_ISOLATE)  W_MuLog=$($env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT)  W_MuSI=$($env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT)"
Write-Host "  TBPTT=$($env:BIOCHEM_TBPTT_MAX_WINDOW)  DETACH=$($env:BIOCHEM_DETACH_MACRO_STATE)  MU_RATIO_MAX=$($env:BIOCHEM_TEACHER_MU_RATIO_MAX)"
Write-Host "  epochs=$($env:BIOCHEM_TEACHER_EPOCHS)  TF_MIN=$($env:BIOCHEM_TEACHER_FORCE_MIN)  delta_head=$($env:BIOCHEM_USE_DELTA_MU_HEAD)  mu_enc=$($env:BIOCHEM_TRAIN_MU_ENCODER)"
Write-Host ""

$trainArgs = @()
if ($NewRun) { $trainArgs += "--new" }
$trainArgs += $ExtraArgs

python -m src.training.train_biochem_corrector @trainArgs

Write-Host ""
Write-Host "Done. Check val mu_log_mae (all | wall | high-mu) in console and outputs/reports/training/biochem/metrics.jsonl" -ForegroundColor Green
