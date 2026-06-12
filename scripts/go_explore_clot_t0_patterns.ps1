# Exploratory probe: t=0 flow/geometry vs GT clot @ t_final inside wall+2 ceiling.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_explore_clot_t0_patterns.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_explore_clot_t0_patterns.ps1" -Anchor patient007 -Detail

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Anchor = "",
    [switch] $Detail
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_PHI_CEILING_HOPS = "2"
$env:CLOT_PHI_DGAMMA_SLICE = "1"
$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

$pyArgs = @("scripts/explore_clot_t0_patterns.py", "--anchor-dir", $AnchorDir)
if ($Anchor) { $pyArgs += @("--anchor", $Anchor) }
if ($Detail) { $pyArgs += "--detail" }

Invoke-PythonRc @pyArgs -Label "clot t0 pattern probe"
