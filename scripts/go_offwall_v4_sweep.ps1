# Off-wall clot v4 sweep: targeted split-saturation and nucleation modifications.
#
# Legs:
#   WC_v4_offwall_sat15       - scale_offwall=15 (lower off-wall saturation constraint)
#   WC_v4_offwall_sat30       - scale_offwall=30
#   WC_v4_offwall_sat50       - scale_offwall=50
#   WC_v4_offwall_nuc4_sat15  - scale_offwall=15 + aggressive 4-hop nucleation front
#
# All legs init from outputs/biochem/biochem_gnn/species/best.pth (promoted widenet).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_offwall_v4_sweep.ps1 -Fresh
#   powershell ... -Fast -Fresh                         # smoke test (1 ep each)
#   powershell ... -EvalOnly                            # re-eval existing ckpts
#

param(
    [string[]] $Legs = @(
        "WC_v4_offwall_sat15",
        "WC_v4_offwall_sat30",
        "WC_v4_offwall_sat50",
        "WC_v4_offwall_nuc4_sat15"
    ),
    [int] $Epochs      = 20,
    [int] $EarlyStop   = 12,
    [int] $MaxWindows  = 32,
    [string] $ValAnchor = "patient007",
    [switch] $Fast,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipSummary
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# Fast preset: 1 epoch each, 4 windows (smoke test)
if ($Fast) {
    $Epochs      = 1
    $EarlyStop   = 1
    $MaxWindows  = 4
}

# Normalize comma-separated leg strings
$legList = @()
foreach ($l in $Legs) {
    if ($l.Contains(",")) {
        $legList += $l.Split(",") | ForEach-Object { $_.Trim() }
    } else {
        $legList += $l.Trim()
    }
}
$Legs = @($legList | Where-Object { $_ })

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/mat_growth_ladder"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Off-wall clot v4 sweep" -ForegroundColor Cyan
Write-Host "  Legs        : $($Legs -join ', ')" -ForegroundColor Cyan
Write-Host "  Epochs      : $Epochs  (early_stop=$EarlyStop)" -ForegroundColor Cyan
Write-Host "  MaxWindows  : $MaxWindows  Fresh=$Fresh  EvalOnly=$EvalOnly" -ForegroundColor Cyan
Write-Host "  Init ckpt   : species/best.pth (promoted widenet)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ── Training loop ────────────────────────────────────────────────────────────
foreach ($leg in $Legs) {
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "[NEW] Training leg: $leg" -ForegroundColor Cyan

    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg",        $leg,
        "-Epochs",     "$Epochs",
        "-EarlyStop",  "$EarlyStop",
        "-MaxWindows", "$MaxWindows",
        "-ValAnchor",  $ValAnchor
    )
    if ($Fresh)    { $legArgs += "-Fresh" }
    if ($EvalOnly) { $legArgs += "-EvalOnly" }

    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Leg $leg failed (exit=$LASTEXITCODE)"
    }

    Write-Host "[OK] leg=$leg done" -ForegroundColor Green
    Write-Host ""
}

# ── Summary ──────────────────────────────────────────────────────────────────
if (-not $SkipSummary) {
    Write-Host "[i] Generating ladder summary (all legs with compare.json)..." -ForegroundColor DarkGray

    Invoke-PythonRcCheck -Label "off-wall v4 summary" -PyArgs @(
        "scripts/summarize_mat_growth_ladder.py",
        "--run-root", "outputs/biochem/biochem_gnn/mat_growth_ladder",
        "--val-anchor", $ValAnchor
    )
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host " [OK] go_offwall_v4_sweep done" -ForegroundColor Green
Write-Host "  Results -> $RunRoot/WC_v4_*/species/best.json" -ForegroundColor Green
Write-Host ""
Write-Host "  Key metrics to compare (vs WC_v3_widenet baseline):" -ForegroundColor Green
Write-Host "    deploy_clot_offwall_relaxed_f1   (target: >> 0.374)" -ForegroundColor Green
Write-Host "    deploy_clot_offwall_n_pred        (target: >> 1.1, GT=20.1)" -ForegroundColor Green
Write-Host "    deploy_clot_score                 (must not drop vs 0.979)" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
