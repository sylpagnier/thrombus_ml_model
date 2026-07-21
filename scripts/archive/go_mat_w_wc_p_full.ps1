# Fair full-budget head-to-head: W (stagnation flow) vs WC (dynamic flow) vs P (plain Mat control).
#
# All legs share precision-first mat_growth_simple recipe, 40 ep / ES 25 / all windows (~3-4 h/leg).
# Budget cap 12 h; pytest gate ensures deploy-faithful eval metadata restore.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_w_wc_p_full.ps1 -Fresh
#   powershell ... -File .\scripts\go_mat_w_wc_p_full.ps1 -EvalOnly
#   powershell ... -File .\scripts\go_mat_w_wc_p_full.ps1 -Fresh -Legs W_mat_flow_stagnation,WC_mat_flow_dynamic

param(
    [switch] $Fresh,
    [switch] $EvalOnly,
    [string] $Legs = "",
    [int] $Epochs = 40,
    [int] $EarlyStop = 25,
    [int] $MaxWindows = 0,
    [int] $TargetHours = 12,
    [string] $ValAnchor = "patient007"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if ($Legs.Trim()) {
    $legList = @($Legs.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
} else {
    $legList = @(
        "W_mat_flow_stagnation",
        "WC_mat_flow_dynamic",
        "P_mat_plain"
    )
}

$RunRoot = "outputs/biochem/biochem_gnn/mat_w_wc_p_full"
$SummaryJson = "$RunRoot/mat_w_wc_p_full_summary.json"
$RunRecord = "$RunRoot/run_record_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$budgetMin = $TargetHours * 60
$estMinPerLeg = [math]::Round($Epochs * 4.5, 0)
$estTotalH = [math]::Round(($legList.Count * $estMinPerLeg) / 60, 1)

Write-Host "[NEW] mat W/WC/P full-budget compare ($($legList.Count) legs)" -ForegroundColor Cyan
Write-Host "[i] recipe=mat_growth_simple (precision-first); $Epochs ep / ES $EarlyStop / max_windows=$MaxWindows" -ForegroundColor DarkGray
Write-Host "[i] legs: $($legList -join ', '); est ~${estTotalH}h (cap ${TargetHours}h)" -ForegroundColor DarkGray
Write-Host "[i] promote rule: over/gt <= 0.02 then rank clot_score, clot_f1" -ForegroundColor DarkGray

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
    Invoke-PythonRcCheck -Label "mat w/wc/p pytest gate" -PyArgs @(
        "-m", "pytest",
        "src/tests/test_mat_growth_simple_scope.py",
        "src/tests/test_species_flow_feats.py",
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
Invoke-PythonRcCheck -Label "mat w/wc/p summary + winner" -PyArgs @(
    "scripts/summarize_mat_only_full.py",
    "--legs", ($summaryLegs -join ","),
    "--out", $SummaryJson,
    "--pick-winner",
    "--max-overpaint-per-gt", "0.02",
    "--rank-by", "deploy_clot_score,deploy_clot_f1"
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
    fresh = [bool]$Fresh
    eval_only = [bool]$EvalOnly
    summary_json = $SummaryJson
    promote_rule = "over/gt <= 0.02; rank deploy_clot_score then deploy_clot_f1"
}
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $RunRecord

Write-Host "[OK] done in $elapsedMin min; summary -> $SummaryJson" -ForegroundColor Green
Write-Host "[save] run record -> $RunRecord" -ForegroundColor DarkGray
