param(
    [int]    $Epochs       = 15,
    [int]    $EarlyStop    = 10,
    [string] $TrainAnchors = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor    = "patient020",
    [string] $HoldoutAnchors = "patient020",
    [string] $RunRoot      = "outputs/biochem/eda/flow_source_ab",
    [string] $ArmFilter    = "",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SummaryOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

Write-Host "[NEW] flow_source_ab: $Epochs ep / ES $EarlyStop" -ForegroundColor Cyan
Write-Host "[i] train=$TrainAnchors val=$ValAnchor holdout=$HoldoutAnchors" -ForegroundColor DarkGray

$pyArgs = @(
    "scripts/run_flow_source_ab.py",
    "--epochs", "$Epochs",
    "--early-stop", "$EarlyStop",
    "--train-anchors", $TrainAnchors,
    "--val-anchor", $ValAnchor,
    "--holdout-anchors", $HoldoutAnchors,
    "--run-root", $RunRoot
)
if ($ArmFilter) { $pyArgs += @("--arm-filter", $ArmFilter) }
if ($Fresh) { $pyArgs += "--fresh" }
if ($EvalOnly) { $pyArgs += "--eval-only" }
if ($SummaryOnly) { $pyArgs += "--summary-only" }

$null = Invoke-PythonRcCheck -Label "flow_source_ab" -PyArgs $pyArgs
