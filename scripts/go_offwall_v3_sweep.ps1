# Off-wall clot v3 sweep: all 6 legs with full off-wall supervision.
#
# All legs set CLOT_PHI_PHYSICS_WALL_MAT_ONLY=0 and CLOT_V2_NUCLEATION_HOPS>=3
# so off-wall gradients are enabled during training. Each leg isolates one
# physically-motivated architectural change versus the v3 baseline.
#
# Legs:
#   WC_v3_baseline        - v2_baseline recipe + full off-wall unlock (control)
#   WC_v3_widenet         - wider GNN band (Hop 5) + recall-biased loss
#   WC_v3_focal_offwall   - strong focal loss (gamma=5) for rare off-wall class
#   WC_v3_neighbor_offwall - autocatalytic neighbor commit gate (chain reaction)
#   WC_v3_widenet_focal   - kitchen-sink: wide + focal + neighbor + nuc_hops=4
#   WC_v3_convection_offwall - convection-aware upwind feature
#
# All legs init from WC_v2_dilation (only prior ckpt with off-wall signal).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_offwall_v3_sweep.ps1 -Fresh
#   powershell ... -Fast -Fresh                         # smoke test (1 ep each)
#   powershell ... -Legs WC_v3_baseline,WC_v3_focal_offwall -Fresh  # subset
#   powershell ... -EvalOnly                            # re-eval existing ckpts
#   powershell ... -Diag                               # run diagnostic first
#
# Expected runtime: ~2-3h total (20 epochs, early-stop 12, single GPU)

param(
    [string[]] $Legs = @(
        "WC_v3_baseline",
        "WC_v3_widenet",
        "WC_v3_focal_offwall",
        "WC_v3_neighbor_offwall",
        "WC_v3_widenet_focal",
        "WC_v3_convection_offwall"
    ),
    [int] $Epochs      = 20,
    [int] $EarlyStop   = 12,
    [int] $MaxWindows  = 32,
    [string] $ValAnchor = "patient007",
    [switch] $Fast,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipSummary,
    [switch] $Diag
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# Fast preset: 1 epoch each, 4 windows (pure smoke test < 5 minutes total)
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
Write-Host " Off-wall clot v3 sweep" -ForegroundColor Cyan
Write-Host "  Legs        : $($Legs -join ', ')" -ForegroundColor Cyan
Write-Host "  Epochs      : $Epochs  (early_stop=$EarlyStop)" -ForegroundColor Cyan
Write-Host "  MaxWindows  : $MaxWindows  Fresh=$Fresh  EvalOnly=$EvalOnly" -ForegroundColor Cyan
Write-Host "  Init ckpt   : WC_v2_dilation/species/best.pth (off-wall trained)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ── Optional diagnostic pass ────────────────────────────────────────────────
if ($Diag) {
    Write-Host "[DIAG] Running off-wall underprediction diagnostics on WC_v2_dilation..." -ForegroundColor Yellow
    $diagJson = Join-Path $RunRoot "diag_offwall_underpred.json"
    Invoke-PythonRcCheck -Label "diag_offwall_underpred" -PyArgs @(
        "scripts/_diag_offwall_underpred.py",
        "--anchor", $ValAnchor,
        "--out", $diagJson
    )
    Write-Host "[DIAG] Results saved -> $diagJson" -ForegroundColor Green
    Write-Host ""
}

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

    Invoke-PythonRcCheck -Label "off-wall v3 summary" -PyArgs @(
        "scripts/summarize_mat_growth_ladder.py",
        "--run-root", "outputs/biochem/biochem_gnn/mat_growth_ladder",
        "--val-anchor", $ValAnchor
    )
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host " [OK] go_offwall_v3_sweep done" -ForegroundColor Green
Write-Host "  Results -> $RunRoot/WC_v3_*/species/best.json" -ForegroundColor Green
Write-Host ""
Write-Host "  Key metrics to compare (vs WC_v2_dilation baseline):" -ForegroundColor Green
Write-Host "    deploy_clot_offwall_relaxed_f1   (target: >> 0.368)" -ForegroundColor Green
Write-Host "    deploy_clot_offwall_n_pred        (target: >> 1.1, GT=20.1)" -ForegroundColor Green
Write-Host "    deploy_clot_score                 (must not drop vs 0.930)" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
