# G1 + G2 rule ladder (multistep-winner env). Run after S0/S1 accepted.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_ladder_g1g2.ps1"

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [switch] $StopOnFail,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")
. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

$rungs = @("g1", "g2")
$passed = @()
$failed = @()

foreach ($rung in $rungs) {
    Write-Host ""
    Write-Host "[NEW] ladder rung $($rung.ToUpper()) eval + gate" -ForegroundColor Cyan
    $evalArgs = @(
        "scripts/eval_clot_prior_rule_ladder.py",
        "--stage", $rung,
        "--anchor-dir", $AnchorDir
    )
    Invoke-PythonRcCheck @evalArgs -Label "rule ladder $rung"

    $gateArgs = @(
        "scripts/check_clot_deploy_rung.py",
        "--rung", $rung,
        "--mode", "rule"
    )
    $gateRc = Invoke-PythonRc @gateArgs
    if ($gateRc -eq 0) {
        Write-Host "[OK]  rung $($rung.ToUpper()) PASS" -ForegroundColor Green
        $passed += $rung.ToUpper()
    } else {
        Write-Host "[FAIL] rung $($rung.ToUpper()) did not pass gate" -ForegroundColor Red
        $failed += $rung.ToUpper()
        if ($StopOnFail) {
            break
        }
    }
}

Write-Host ""
Write-Host "[i]  ladder summary passed=$($passed -join ',') failed=$($failed -join ',')" -ForegroundColor DarkGray

if ($failed.Count -gt 0) { exit 1 }
