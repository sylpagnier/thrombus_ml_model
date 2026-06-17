# Quick A/B: baseline vs global-time Fourier context (<= ~30 min target).
#
# A: no explicit time context (current baseline input path)
# B: global time context with tau=t/30000 + Fourier features
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_time_ab_30m.ps1

param(
    [int] $Epochs = 3,
    [int] $EarlyStop = 2,
    [int] $MaxWindows = 24,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/time_ab_30m"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$InitWarm = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$BetaCkpt = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
if (-not (Test-Path $InitWarm)) { throw "missing init ckpt: $InitWarm" }
if (-not (Test-Path $BetaCkpt)) { throw "missing beta ckpt: $BetaCkpt" }

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

function Train-Leg([string]$LegKey, [string]$Label, [bool]$TimeContext) {
    $legDir = Join-Path $RunRoot $Label
    $speciesDir = Join-Path $legDir "species"
    $evalDir = Join-Path $legDir "eval"
    New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
    $speciesOut = Join-Path $speciesDir "best.pth"
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $manifest = Join-Path $RepoRoot "data/reference/biochem_gnn_time_ab_$($LegKey.ToLower()).json"

    if ($Fresh -and (Test-Path $speciesOut)) { Remove-Item $speciesOut -Force }
    if ($Fresh -and (Test-Path $evalOut)) { Remove-Item $evalOut -Force }

    if ($TimeContext) {
        $env:SPECIES_CONTINUOUS_TIME_CONTEXT = "1"
        $env:SPECIES_CONTINUOUS_TIME_REF_S = "3000"
        $env:SPECIES_CONTINUOUS_TIME_FOURIER_FREQS = "8"
    } else {
        $env:SPECIES_CONTINUOUS_TIME_CONTEXT = "0"
        Remove-Item Env:\SPECIES_CONTINUOUS_TIME_REF_S -ErrorAction SilentlyContinue
        Remove-Item Env:\SPECIES_CONTINUOUS_TIME_FOURIER_FREQS -ErrorAction SilentlyContinue
    }

    Write-Host "[run] [$LegKey] train (time_ctx=$($env:SPECIES_CONTINUOUS_TIME_CONTEXT))" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$LegKey] train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--anchors", $Anchors,
        "--val-anchor", "patient007",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--unroll", "10",
        "--arch", "sage",
        "--init-s26", $InitWarm,
        "--out", $speciesOut
    )

    $payload = @{
        name = "biochem_gnn_time_ab_$($LegKey.ToLower())"
        version = 1
        baseline = @{
            species_gnn_ckpt = (RelPath $speciesOut)
            viscosity_beta = (RelPath $BetaCkpt)
            kinematics_ckpt = "outputs/kinematics/kinematics_best.pth"
            train_val_anchor = "patient007"
            flow_modes = "kinematics"
            gamma_mode = "max"
            deploy_horizon = "full"
            clot_score = "guiding"
            pushforward_arch = "sage"
            gate_mode = "global_sigmoid"
            species_scope = "fi_mat"
            loao_auto = "0"
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($manifest, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)
    Write-Host "[i] [$LegKey] manifest -> $manifest" -ForegroundColor DarkGray

    Write-Host "[run] [$LegKey] eval deploy_frozen @t53" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$LegKey] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $manifest,
        "--modes", "deploy_frozen",
        "--times", "53",
        "--anchors", $Anchors,
        "--out", $evalOut
    )
    return $evalOut
}

Write-Host "[i] time A/B quick run root: $RunRoot" -ForegroundColor DarkGray
$evalA = Train-Leg -LegKey "A" -Label "baseline_no_time" -TimeContext:$false
$evalB = Train-Leg -LegKey "B" -Label "global_time_fourier" -TimeContext:$true

$SummaryJson = Join-Path $RunRoot "time_ab_summary.json"
$SummaryMd = Join-Path $RunRoot "time_ab_verdict.md"
$aRows = ((Get-Content $evalA -Raw | ConvertFrom-Json).rows | Where-Object { $_.mode -eq "deploy_frozen" -and -not $_.error })
$bRows = ((Get-Content $evalB -Raw | ConvertFrom-Json).rows | Where-Object { $_.mode -eq "deploy_frozen" -and -not $_.error })
function Leg-Metrics($rows) {
    $p007 = $rows | Where-Object { $_.anchor -eq "patient007" } | Select-Object -First 1
    if ($null -eq $p007) { $p007 = $rows | Select-Object -First 1 }
    $hold = $rows | Where-Object { $_.anchor -ne "patient007" }
    $holdMean = 0.0
    if ($hold.Count -gt 0) { $holdMean = ($hold | Measure-Object -Property clot_f1_main -Average).Average }
    return @{
        deploy_clot_score = [double]$p007.clot_score_main
        clot_f1_main = [double]$p007.clot_f1_main
        holdout_mean_clot_f1_main = [double]$holdMean
    }
}
$legs = @{
    A = Leg-Metrics $aRows
    B = Leg-Metrics $bRows
}
$winner = if ($legs.B.deploy_clot_score -gt $legs.A.deploy_clot_score) { "B" } else { "A" }
$summary = @{
    legs = $legs
    winner = $winner
    winner_key = "deploy_clot_score"
}
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($SummaryJson, ($summary | ConvertTo-Json -Depth 6), $utf8NoBom)
$md = @(
    "# Time Context A/B (30m)",
    "",
    "| Leg | deploy_clot_score | p007 clot_f1 | holdout mean clot_f1 |",
    "|-----|-------------------|--------------|-----------------------|",
    ("| A | {0:N3} | {1:N3} | {2:N3} |" -f $legs.A.deploy_clot_score, $legs.A.clot_f1_main, $legs.A.holdout_mean_clot_f1_main),
    ("| B | {0:N3} | {1:N3} | {2:N3} |" -f $legs.B.deploy_clot_score, $legs.B.clot_f1_main, $legs.B.holdout_mean_clot_f1_main),
    "",
    "Winner: $winner (deploy_clot_score)"
)
[System.IO.File]::WriteAllText($SummaryMd, ($md -join "`n") + "`n", $utf8NoBom)

Write-Host "[OK] summary -> $SummaryJson" -ForegroundColor Green
Write-Host "[OK] verdict -> $SummaryMd" -ForegroundColor Green
