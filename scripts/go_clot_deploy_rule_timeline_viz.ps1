# Timeline viz for sweep-winner prior rule (S0/S1/G1/G2 ladder stages).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_rule_timeline_viz.ps1"
#   powershell ... -Stage s1 -Anchor patient007 -Keyframes 8
#   powershell ... -AllStages

param(
    [ValidateSet("s0", "s1", "g1", "g2")]
    [string] $Stage = "s1",
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [int] $Keyframes = 8,
    [double] $ScatterSize = 5,
    [switch] $AllStages
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_deploy_binary_base.ps1")
. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
$env:BIOCHEM_PRIOR_COMSOL_ALIGNED = "1"
$env:BIOCHEM_PRIOR_NORM_MASK = "adjacent"

$vizDir = Join-Path $RepoRoot "outputs/biochem/viz/clot_deploy"
New-Item -ItemType Directory -Force -Path $vizDir | Out-Null

$stages = if ($AllStages) { @("s0", "s1", "g1", "g2") } else { @($Stage) }

foreach ($st in $stages) {
    $png = Join-Path $vizDir "prior_rule_${Anchor}_${st}_timeline.png"
    $jsonl = Join-Path $vizDir "prior_rule_${Anchor}_${st}_timeline.jsonl"
    $pyArgs = @(
        "scripts/viz_clot_prior_rule_timeline.py",
        "--anchor", $Anchor,
        "--anchor-dir", $AnchorDir,
        "--stage", $st,
        "--keyframes", "$Keyframes",
        "--scatter-size", "$ScatterSize",
        "--out", $png,
        "--summary-json", $jsonl
    )
    Write-Host "[NEW] prior rule timeline stage=$($st.ToUpper()) anchor=$Anchor keyframes=$Keyframes" -ForegroundColor Cyan
    Invoke-PythonRcCheck @pyArgs -Label "prior rule timeline $st"
    Write-Host "[OK]  timeline -> $png" -ForegroundColor Green
    Write-Host "[OK]  metrics  -> $jsonl" -ForegroundColor Green
}
