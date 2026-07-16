# Full-budget canonical baseline pick: W (stagnation flow) vs WC (dynamic coupled flow).
#
# Goal: choose the new Mat deploy canonical with FP-aware ranking and a tie-break
# preference for WC when clot F1 is close (dynamic flow may generalize better as clots grow).
#
# Recipe: mat_growth_simple precision-first, 40 ep / ES 25 / all windows (~3-4 h/leg).
# Budget cap 10 h (2 legs). Use -Fresh for a clean head-to-head.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_w_wc_canonical.ps1 -Fresh
#   powershell ... -File .\scripts\go_mat_w_wc_canonical.ps1 -EvalOnly
#   powershell ... -File .\scripts\go_mat_w_wc_canonical.ps1 -Fresh -Promote -Viz

param(
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $Promote,
    [switch] $Viz,
    [string] $Legs = "W_mat_flow_stagnation,WC_mat_flow_dynamic",
    [int] $Epochs = 40,
    [int] $EarlyStop = 25,
    [int] $MaxWindows = 0,
    [int] $TargetHours = 10,
    [string] $ValAnchor = "patient007",
    [float] $TieF1Eps = 0.008
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$legList = @($Legs.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($legList.Count -lt 1) { throw "need at least one leg in -Legs" }

$RunRoot = "outputs/biochem/biochem_gnn/mat_w_wc_canonical"
$SummaryJson = "$RunRoot/mat_w_wc_canonical_summary.json"
$RunRecord = "$RunRoot/run_record_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$budgetMin = $TargetHours * 60
$estMinPerLeg = [math]::Round($Epochs * 4.5, 0)
$estTotalH = [math]::Round(($legList.Count * $estMinPerLeg) / 60, 1)

Write-Host "[NEW] mat W vs WC canonical baseline ($($legList.Count) legs)" -ForegroundColor Cyan
Write-Host "[i] recipe=mat_growth_simple; $Epochs ep / ES $EarlyStop / max_windows=$MaxWindows" -ForegroundColor DarkGray
Write-Host "[i] legs: $($legList -join ', '); est ~${estTotalH}h (cap ${TargetHours}h)" -ForegroundColor DarkGray
Write-Host "[i] winner: over/gt<=0.02; rank f1, then min p90FP,medFP,medFN, then score; prefer WC within f1 eps $TieF1Eps" -ForegroundColor DarkGray

if ($Fresh) {
    foreach ($leg in $legList) {
        $legDir = "outputs/biochem/biochem_gnn/mat_growth_ladder/$leg"
        Remove-Item -Force "$legDir/species/best.pth" -ErrorAction SilentlyContinue
        Remove-Item -Force "$legDir/species/best.json" -ErrorAction SilentlyContinue
        Remove-Item -Force "$legDir/species/train_log.jsonl" -ErrorAction SilentlyContinue
        Remove-Item -Force "$legDir/compare.json" -ErrorAction SilentlyContinue
    }
}

if (-not $EvalOnly) {
    Invoke-PythonRcCheck -Label "mat w/wc canonical pytest gate" -PyArgs @(
        "-m", "pytest",
        "src/tests/test_mat_growth_simple_scope.py",
        "src/tests/test_species_flow_feats.py",
        "src/tests/test_clot_timeline_metrics.py",
        "src/tests/test_summarize_mat_pick_winner.py",
        "-q"
    )
}

$started = Get-Date
$completed = @()

foreach ($leg in $legList) {
    $totMin = [int]((Get-Date) - $started).TotalMinutes
    if ($totMin -ge $budgetMin) {
        Write-Host "[WARN] cumulative runtime >= ${TargetHours}h; skipping remaining legs" -ForegroundColor Yellow
        break
    }

    $legStart = Get-Date
    Write-Host "[leg] $leg starting at $($legStart.ToString('HH:mm:ss')) (cumulative ${totMin} min)" -ForegroundColor Cyan
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", $leg,
        "-Epochs", "$Epochs",
        "-EarlyStop", "$EarlyStop",
        "-MaxWindows", "$MaxWindows",
        "-ValAnchor", $ValAnchor
    )
    if ($Fresh) { $legArgs += "-Fresh" }
    if ($EvalOnly) { $legArgs += "-EvalOnly" }

    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) { throw "$leg failed (exit=$LASTEXITCODE)" }

    $legMin = [int]((Get-Date) - $legStart).TotalMinutes
    $completed += $leg
    Write-Host "[leg] $leg done in $legMin min" -ForegroundColor DarkGray
}

$summaryLegs = if ($completed.Count -gt 0) { $completed } else { $legList }
if ($summaryLegs.Count -lt $legList.Count) {
    Write-Host "[WARN] summary uses completed legs only ($($summaryLegs -join ', '))" -ForegroundColor Yellow
}

Invoke-PythonRcCheck -Label "mat w/wc canonical summary + winner" -PyArgs @(
    "scripts/summarize_mat_only_full.py",
    "--legs", ($summaryLegs -join ","),
    "--out", $SummaryJson,
    "--pick-winner",
    "--max-overpaint-per-gt", "0.02",
    "--rank-by", "deploy_clot_f1,clot_fp_p90,clot_fp_median,clot_fn_median,deploy_clot_score",
    "--minimize-metrics", "clot_fp_p90,clot_fp_median,clot_fn_median",
    "--prefer-leg", "WC_mat_flow_dynamic",
    "--tie-eps", "$TieF1Eps"
)

$elapsedMin = [int]((Get-Date) - $started).TotalMinutes
$record = @{
    started_utc = $started.ToUniversalTime().ToString("o")
    elapsed_min = $elapsedMin
    legs_requested = $legList
    legs_completed = $completed
    epochs = $Epochs
    early_stop = $EarlyStop
    max_windows = $MaxWindows
    val_anchor = $ValAnchor
    target_hours = $TargetHours
    tie_f1_eps = $TieF1Eps
    fresh = [bool]$Fresh
    eval_only = [bool]$EvalOnly
    summary_json = $SummaryJson
    promote_rule = "over/gt<=0.02; rank f1,min p90FP,medFP,medFN,score; prefer WC within f1 eps"
}
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $RunRecord

Write-Host "[OK] done in $elapsedMin min; summary -> $SummaryJson" -ForegroundColor Green
Write-Host "[save] run record -> $RunRecord" -ForegroundColor DarkGray

if ($Promote) {
    Invoke-PythonRcCheck -Label "promote mat canonical deploy" -PyArgs @(
        "scripts/promote_mat_canonical_deploy.py",
        "--summary", $SummaryJson
    )
}

if ($Viz) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "go_viz_mat_w_wc_canonical.ps1") -Legs ($completed -join ",")
    if ($LASTEXITCODE -ne 0) { throw "viz failed (exit=$LASTEXITCODE)" }
}
