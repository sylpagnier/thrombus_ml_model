# Side-by-side S0 viz: ceiling wall+2 hops vs ceiling + sdf_nd rank cap.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_prior_rule_rank_compare_viz.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_prior_rule_rank_compare_viz.ps1" -SdfMax 0.035

param(
    [string] $Anchor = "patient007",
    [double] $SdfMax = 0.040
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")

Invoke-PythonRcCheck "scripts/viz_clot_prior_rule_rank_compare.py" "--anchor" $Anchor "--sdf-max" "$SdfMax" -Label "rank pool compare viz"
