# M5 block pass (M5.3-M5.6): wall/high finetune -> bridge -> K10 explore -> lock clot teacher.
# Prereq: M5.1 passive_mu_unlock_best.pth (and ideally passive_xy_locked from I.3).
#
# Target ~8h on RTX 500 class GPU (12+12+18+18+18 ep + gates). GT_KINE_VEL=1 throughout.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_m5_block_pass.ps1"
#   powershell ... -Turbo          # ~3-4h (8+8+12+10+10 ep)
#   powershell ... -SkipFinetune   # if M5.3 already ran

param(
    [string] $UnlockCkpt = "outputs/biochem/biochem_teacher_passive_mu_unlock_best.pth",
    [int] $FinetuneEpochs = 12,
    [int] $BridgeEpochs = 12,
    [int] $K10WideEpochs = 18,
    [int] $K10NarrowEpochs = 18,
    [int] $K10BiasEpochs = 18,
    [string] $DestLocked = "outputs/biochem/biochem_teacher_passive_m5_clot_locked.pth",
    [string] $ManifestPath = "outputs/biochem/passive_m5_clot_locked_manifest.json",
    [switch] $SkipFinetune,
    [switch] $SkipAudit,
    [switch] $Turbo
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

if ($Turbo) {
    $FinetuneEpochs = 8
    $BridgeEpochs = 8
    $K10WideEpochs = 12
    $K10NarrowEpochs = 12
    $K10BiasEpochs = 12
}

$unlockPath = Join-Path $RepoRoot $UnlockCkpt
if (-not (Test-Path $unlockPath)) {
    Write-Host "[ERR] Missing $UnlockCkpt (finish M5.1 mu-unlock probe first)" -ForegroundColor Red
    exit 1
}

$OutRoot = Join-Path $RepoRoot "outputs\biochem\m5_block"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$started = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

function Save-K10LegCkpt {
    param([string] $LegTag)
    $bestSrc = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
    $lastSrc = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
    if (-not (Test-Path $bestSrc)) { return $null }
    $destBest = Join-Path $OutRoot "biochem_teacher_${LegTag}_best.pth"
    Copy-Item $bestSrc $destBest -Force
    if (Test-Path $lastSrc) {
        Copy-Item $lastSrc (Join-Path $OutRoot "biochem_teacher_${LegTag}_last.pth") -Force
    }
    return $destBest.Replace("\", "/")
}
Write-Host "[NEW] M5 block pass M5.3-M5.6 (~8h tier) finetune=$FinetuneEpochs bridge=$BridgeEpochs k10=$K10WideEpochs/$K10NarrowEpochs/$K10BiasEpochs" -ForegroundColor Cyan
Write-Host "[i]  Init unlock=$UnlockCkpt | GT_KINE_VEL=1 | goal=localized clot viz on biochem teacher" -ForegroundColor Cyan

$auditArgs = @{}
if ($SkipAudit) { $auditArgs["SkipAudit"] = $true }
$chainInit = $UnlockCkpt

# --- M5.3 wall/high finetune ---
if (-not $SkipFinetune) {
    Write-Host "[NEW] M5.3 mu-unlock finetune ($FinetuneEpochs ep)" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "go_passive_mu_unlock_finetune.ps1") `
        -InitCkpt $UnlockCkpt -Epochs $FinetuneEpochs @auditArgs
    $ftGate = $LASTEXITCODE
    if ($ftGate -eq 0) {
        Write-Host "[OK] M5.3 finetune gate passed; chain init=biochem_teacher_last.pth" -ForegroundColor Green
        $chainInit = "outputs/biochem/biochem_teacher_last.pth"
    } else {
        Write-Host "[WARN] M5.3 finetune gate failed; chain init stays unlock_best" -ForegroundColor Yellow
        $chainInit = $UnlockCkpt
    }
} else {
    Write-Host "[skip] M5.3 finetune (SkipFinetune)" -ForegroundColor Yellow
}

# --- M5.4 step-2 bridge (species + masked ADR + modest mu aux) ---
Write-Host "[NEW] M5.4 step-2 bridge ($BridgeEpochs ep) init=$chainInit" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "go_passive_step2_bridge.ps1") `
    -InitCkpt $chainInit `
    -Epochs $BridgeEpochs `
    -RunNote "passive_m5_bridge" `
    -GradScaleOnCap `
    @auditArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] M5.4 bridge gate failed; K10 legs use bridge last anyway" -ForegroundColor Yellow
}
$bridgeLast = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
if (-not (Test-Path $bridgeLast)) {
    Write-Host "[ERR] Missing biochem_teacher_last.pth after bridge" -ForegroundColor Red
    exit 1
}
$k10Init = "outputs/biochem/biochem_teacher_last.pth"

# --- M5.6a K10f wide from bridge ---
Write-Host "[NEW] M5.6a K10f wide ($K10WideEpochs ep)" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "go_m5_k10e_from_passive.ps1") `
    -InitCkpt $k10Init `
    -Epochs $K10WideEpochs `
    -Variant wide `
    -RunNote "m5_k10f_wide_from_passive" `
    @auditArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$wideBest = Save-K10LegCkpt "m5_k10f_wide"
$wideLast = "outputs/biochem/biochem_teacher_last.pth"

# --- M5.6b K10e narrow from bridge (A/B band) ---
Write-Host "[NEW] M5.6b K10e narrow ($K10NarrowEpochs ep)" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "go_m5_k10e_from_passive.ps1") `
    -InitCkpt $k10Init `
    -Epochs $K10NarrowEpochs `
    -Variant narrow `
    -RunNote "m5_k10e_narrow_from_passive" `
    @auditArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$narrowBest = Save-K10LegCkpt "m5_k10e_narrow"

# --- M5.6c K10g bias finetune from wide winner ---
Write-Host "[NEW] M5.6c K10g bias ($K10BiasEpochs ep) from wide last" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "go_m5_k10e_from_passive.ps1") `
    -InitCkpt $wideLast `
    -Epochs $K10BiasEpochs `
    -Variant bias `
    -RunNote "m5_k10g_bias_from_passive" `
    @auditArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$biasBest = Save-K10LegCkpt "m5_k10g_bias"

# --- M5.5 audit + lock best K10 leg ---
Write-Host "[NEW] M5.5 gate + lock best K10 ckpt" -ForegroundColor Cyan
$summaryPath = Join-Path $OutRoot "summary.json"
$gateRc = Invoke-PythonRc scripts/check_m5_block_pass.py --summary $summaryPath
$summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
$bestNote = $summary.best_k10.run_note
$bestDir = $summary.best_k10.run_dir
Write-Host "[i]  Best K10 leg: $bestNote ($bestDir)" -ForegroundColor Cyan

$legTag = switch ($bestNote) {
    "m5_k10f_wide_from_passive"   { "m5_k10f_wide" }
    "m5_k10e_narrow_from_passive" { "m5_k10e_narrow" }
    "m5_k10g_bias_from_passive"   { "m5_k10g_bias" }
    default                       { $null }
}
$srcBest = if ($legTag) { Join-Path $OutRoot "biochem_teacher_${legTag}_best.pth" } else { $null }
if (-not $srcBest -or -not (Test-Path $srcBest)) {
    Write-Host "[WARN] Missing archived best for $bestNote; falling back to biochem_teacher_best_high_mu.pth" -ForegroundColor Yellow
    $srcBest = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
}

$dst = Join-Path $RepoRoot $DestLocked
Copy-Item $srcBest $dst -Force
Copy-Item $dst (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

$manifest = @{
    locked_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    started_utc   = $started
    run_note      = $bestNote
    source_ckpt   = $srcBest.Replace("\", "/")
    dest_ckpt     = $DestLocked.Replace("\", "/")
    unlock_init   = $UnlockCkpt.Replace("\", "/")
    m5_summary    = $summaryPath.Replace("\", "/")
    viz_cmd       = "python -m src.evaluation.visualize_pipeline --teacher-only --biochem-checkpoint $DestLocked --anchor patient007"
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $RepoRoot $ManifestPath) -Encoding utf8

Write-Host "[save] Locked M5 clot teacher -> $DestLocked (from $bestNote)" -ForegroundColor Green
Write-Host "[i]  M5.5 viz (interactive): $($manifest.viz_cmd)" -ForegroundColor Cyan
Write-Host "[i]  Oracle sanity: powershell -File .\scripts\go_k10g_oracle_clots_viz.ps1 (set ckpt to locked path first)" -ForegroundColor Cyan

if ($gateRc -eq 0) {
    Write-Host "[OK] M5 block pass complete" -ForegroundColor Green
} else {
    Write-Host "[WARN] M5 block finished but gate not fully met; inspect $summaryPath" -ForegroundColor Yellow
}
exit $gateRc
