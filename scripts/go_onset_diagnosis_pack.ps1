# Onset diagnosis pack (eval-only): toggles temporal/commit mechanics.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_onset_diagnosis_pack.ps1

param(
    [string] $Manifest = "outputs/biochem/clot_baseline/manifest.json",
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [string] $Times = "0,10,20,27,35,44,53,62,80,100,120",
    [double] $OnsetThreshold = 0.2
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/onset_diagnosis_pack"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$savedEnv = @{
    SPECIES_CONTINUOUS_TEMPORAL_GATE = $env:SPECIES_CONTINUOUS_TEMPORAL_GATE
    CLOT_TRIGGER_COMMIT_THRESH = $env:CLOT_TRIGGER_COMMIT_THRESH
    SPECIES_CONTINUOUS_MAT_COMMIT_THRESH = $env:SPECIES_CONTINUOUS_MAT_COMMIT_THRESH
}

function Set-DiagEnv {
    param([string]$TemporalGate, [string]$ClotCommit, [string]$MatCommit)
    if ($TemporalGate -eq "") { Remove-Item Env:\SPECIES_CONTINUOUS_TEMPORAL_GATE -ErrorAction SilentlyContinue } else { $env:SPECIES_CONTINUOUS_TEMPORAL_GATE = $TemporalGate }
    if ($ClotCommit -eq "") { Remove-Item Env:\CLOT_TRIGGER_COMMIT_THRESH -ErrorAction SilentlyContinue } else { $env:CLOT_TRIGGER_COMMIT_THRESH = $ClotCommit }
    if ($MatCommit -eq "") { Remove-Item Env:\SPECIES_CONTINUOUS_MAT_COMMIT_THRESH -ErrorAction SilentlyContinue } else { $env:SPECIES_CONTINUOUS_MAT_COMMIT_THRESH = $MatCommit }
}

function Run-Cond {
    param(
        [string]$Name,
        [string]$Modes = "deploy_frozen",
        [string]$TemporalGate = "",
        [string]$ClotCommit = "",
        [string]$MatCommit = ""
    )
    Set-DiagEnv -TemporalGate $TemporalGate -ClotCommit $ClotCommit -MatCommit $MatCommit
    $outDir = Join-Path $RunRoot $Name
    New-Item -ItemType Directory -Force -Path (Join-Path $outDir "eval") | Out-Null
    $out = Join-Path $outDir "eval/deploy_ab_eval.json"
    Write-Host "[run] $Name (modes=$Modes tg=$TemporalGate clot_commit=$ClotCommit mat_commit=$MatCommit)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label $Name -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $Manifest,
        "--anchors", $Anchors,
        "--times", $Times,
        "--modes", $Modes,
        "--out", $out
    )
}

try {
    Run-Cond -Name "baseline" -Modes "deploy_frozen"
    Run-Cond -Name "no_temporal_gate" -Modes "deploy_frozen" -TemporalGate "0"
    Run-Cond -Name "low_commit_thresh" -Modes "deploy_frozen" -ClotCommit "0.40" -MatCommit "8e-5"
    Run-Cond -Name "high_commit_thresh" -Modes "deploy_frozen" -ClotCommit "0.60" -MatCommit "2e-4"
    Run-Cond -Name "legacy_oracle" -Modes "legacy_oracle"

    $null = Invoke-PythonRcCheck -Label "onset summary" -PyArgs @(
        "scripts/summarize_onset_diagnosis.py",
        "--run-root", $RunRoot,
        "--onset-threshold", "$OnsetThreshold"
    )
    Write-Host "[OK] report -> $RunRoot/onset_diagnosis_report.md" -ForegroundColor Green
} finally {
    foreach ($k in $savedEnv.Keys) {
        $v = $savedEnv[$k]
        if ([string]::IsNullOrEmpty($v)) { Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
        else { Set-Item "Env:\$k" $v }
    }
}
