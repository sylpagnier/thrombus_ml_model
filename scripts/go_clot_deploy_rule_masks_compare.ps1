param(
    [string] $Anchor = "patient007",
    [int] $TimeIndex = -1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$tArg = if ($TimeIndex -ge 0) { @("--time-index", "$TimeIndex") } else { @() }
Invoke-PythonRcCheck @(
    "-m", "src.evaluation.viz_clot_deploy_rule_masks",
    "--anchor", $Anchor,
    "--out", "outputs/biochem/viz/clot_deploy/clot_rule_masks_compare_${Anchor}.png"
) + $tArg -Label "deploy rule mask compare"
