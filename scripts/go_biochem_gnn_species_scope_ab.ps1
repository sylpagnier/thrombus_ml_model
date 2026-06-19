# Species scope A/B: does adding thrombin improve deploy clot metrics?
#
# A: fi_mat (FI + Mat)           -- current baseline scope
# B: fi_mat_thrombin (+ thrombin)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_species_scope_ab.ps1
#   powershell ... -Epochs 6 -Fresh

param(
    [int] $Epochs = 8,
    [int] $EarlyStop = 5,
    [int] $MaxWindows = 40,
    [double] $Lr = 3e-4,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/species_scope_ab"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$InitWarm = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$BetaCkpt = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
if (-not (Test-Path $InitWarm)) { throw "missing init ckpt: $InitWarm" }
if (-not (Test-Path $BetaCkpt)) { throw "missing beta ckpt: $BetaCkpt" }

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

function Train-Leg([string]$LegKey, [string]$Label, [string]$Scope) {
    $legDir = Join-Path $RunRoot $Label
    $speciesDir = Join-Path $legDir "species"
    $evalDir = Join-Path $legDir "eval"
    New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
    $speciesOut = Join-Path $speciesDir "best.pth"
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $manifest = Join-Path $RepoRoot "data/reference/biochem_gnn_species_scope_ab_$($LegKey.ToLower()).json"

    if ($Fresh) {
        if (Test-Path $speciesOut) { Remove-Item $speciesOut -Force }
        if (Test-Path $evalOut) { Remove-Item $evalOut -Force }
    }

    $env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE = $Scope
    $env:SPECIES_PUSHFORWARD_ARCH = "sage"
    python -c "from src.biochem_gnn.config import apply_train_recipe_env; apply_train_recipe_env()" | Out-Null
    $env:SPECIES_TRAIN_VEL_SOURCE = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE = "resting"

    Write-Host "[run] [$LegKey] train scope=$Scope ($Epochs ep)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$LegKey] train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--anchors", $Anchors,
        "--val-anchor", "patient007",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--unroll", "10",
        "--lr", "$Lr",
        "--arch", "sage",
        "--init", $InitWarm,
        "--out", $speciesOut
    )

    $payload = @{
        name = "biochem_gnn_species_scope_ab_$($LegKey.ToLower())"
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
            species_scope = $Scope
            loao_auto = "0"
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($manifest, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)
    Write-Host "[i] [$LegKey] manifest -> $manifest" -ForegroundColor DarkGray

    $env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE = $Scope
    Write-Host "[run] [$LegKey] eval deploy_frozen" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$LegKey] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $manifest,
        "--modes", "deploy_frozen",
        "--times", "53,200",
        "--anchors", $Anchors,
        "--out", $evalOut
    )
    return $evalOut
}

Write-Host "[i] species scope A/B root: $RunRoot" -ForegroundColor DarkGray
$evalA = Train-Leg -LegKey "A" -Label "fi_mat" -Scope "fi_mat"
$evalB = Train-Leg -LegKey "B" -Label "fi_mat_thrombin" -Scope "fi_mat_thrombin"

$SummaryJson = Join-Path $RunRoot "species_scope_ab_summary.json"
$SummaryMd = Join-Path $RunRoot "species_scope_ab_report.md"
$null = Invoke-PythonRcCheck -Label "species scope summary" -PyArgs @(
    "scripts/summarize_species_scope_ab.py",
    "--eval-a", $evalA,
    "--eval-b", $evalB,
    "--out-json", $SummaryJson,
    "--out-md", $SummaryMd
)
Write-Host "[OK] summary -> $SummaryJson" -ForegroundColor Green
Write-Host "[OK] report  -> $SummaryMd" -ForegroundColor Green
