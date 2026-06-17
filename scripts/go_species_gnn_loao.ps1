# Legacy wrapper -> go_clot_deploy_gnn.ps1 (loao + eval)
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_gnn_loao.ps1"

param(
    [string] $Holdouts = "",
    [int] $Epochs = 40,
    [int] $EarlyStop = 18,
    [string] $InitCkpt = "",
    [string] $OutRoot = "",
    [switch] $SkipTrain,
    [switch] $SkipEval,
    [switch] $SkipPredict
)

$ErrorActionPreference = "Stop"
Write-Host "[i] go_species_gnn_loao.ps1 -> go_clot_deploy_gnn.ps1" -ForegroundColor DarkGray

if (-not $SkipTrain) {
    $loaoArgs = @(
        "-File", (Join-Path $PSScriptRoot "go_clot_deploy_gnn.ps1"),
        "-Step", "loao",
        "-LoaoEpochs", "$Epochs",
        "-LoaoEarlyStop", "$EarlyStop"
    )
    & powershell -NoProfile -ExecutionPolicy Bypass @loaoArgs
}

if (-not $SkipEval) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "go_clot_deploy_gnn.ps1") -Step eval
}

if (-not $SkipPredict) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
    Set-Location $RepoRoot
    . (Join-Path $PSScriptRoot "_python_rc.ps1")
    $manifest = "data/reference/clot_deploy_gnn_staging.json"
    Invoke-PythonRcCheck -Label "predict p004" -PyArgs @(
        "-m", "src.inference.predict_species_gnn_deploy",
        "--graph", "data/processed/graphs_biochem_anchors/patient004.pt",
        "--flow", "kinematics",
        "--loao",
        "--manifest", $manifest
    )
}
