# Off-wall clot v6 sweep: architectural pivots for off-wall growth.
#
# Legs:
#   WC_v6_closed_loop_eval   - V6 closed loop baseline (align F1)
#   WC_v6_skiphop_multiscale - V6 skiphop multiscale skip connections (Direction 2)
#   WC_v6_blind_loss         - V6 closed loop + midside blind loss (Direction 3)
#   WC_v6_sdf_gating         - V6 closed loop + SDF weighted FP gating (Direction 4)
#   WC_v6_latent_dropout     - V6 closed loop + Latent Dropout 0.5 (Direction 5)
#   WC_v6_spatial_heads      - V6 spatially gated heads + isolated offwall loss scaling (Direction 6)
#
# All legs init from outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_offwall_v6_sweep.ps1 -Fresh
#   powershell ... -Fast -Fresh                         # smoke test (1 ep each)
#   powershell ... -EvalOnly                            # re-eval existing ckpts
#

param(
    [string[]] $Legs = @(
        "WC_v6_closed_loop_eval",
        "WC_v6_skiphop_multiscale",
        "WC_v6_blind_loss",
        "WC_v6_sdf_gating",
        "WC_v6_latent_dropout",
        "WC_v6_spatial_heads"
    ),
    [int] $Epochs      = 15,
    [int] $EarlyStop   = 10,
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
Write-Host " Off-wall clot v6 sweep" -ForegroundColor Cyan
Write-Host "  Legs        : $($Legs -join ', ')" -ForegroundColor Cyan
Write-Host "  Epochs      : $Epochs  (early_stop=$EarlyStop)" -ForegroundColor Cyan
Write-Host "  MaxWindows  : $MaxWindows  Fresh=$Fresh  EvalOnly=$EvalOnly" -ForegroundColor Cyan
Write-Host "  Init ckpt   : outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth" -ForegroundColor Cyan
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

    Invoke-PythonRcCheck -Label "off-wall v6 summary" -PyArgs @(
        "scripts/summarize_mat_growth_ladder.py",
        "--run-root", "outputs/biochem/biochem_gnn/mat_growth_ladder",
        "--val-anchor", $ValAnchor
    )
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host " [OK] go_offwall_v6_sweep done" -ForegroundColor Green
Write-Host "  Results -> $RunRoot/WC_v6_*/species/best.json" -ForegroundColor Green
Write-Host ""
Write-Host "  Key metrics to compare:" -ForegroundColor Green
Write-Host "    deploy_clot_offwall_relaxed_f1" -ForegroundColor Green
Write-Host "    deploy_clot_offwall_n_pred" -ForegroundColor Green
Write-Host "    deploy_clot_score" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
