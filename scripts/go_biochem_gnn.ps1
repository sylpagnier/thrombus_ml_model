# Biochem GNN baseline — canonical deploy stack train / LOAO / promote / eval / gate.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn.ps1 -Step all
#   powershell ... -Step species -Fresh -AllAnchors
#   powershell ... -Step promote -Gate -Viz
#
# Steps: species | viscosity | loao | eval | promote | gate | viz | all
# Aliases: global -> species, beta -> viscosity

param(
    [ValidateSet("species", "viscosity", "loao", "eval", "promote", "gate", "viz", "all", "global", "beta")]
    [string] $Step = "all",
    [string] $ValAnchor = "patient007",
    [switch] $AllAnchors,
    [string] $Anchors = "",
    [int] $Epochs = 50,
    [int] $EarlyStop = 35,
    [int] $LoaoEpochs = 40,
    [int] $LoaoEarlyStop = 18,
    [switch] $Fresh,
    [switch] $SkipTrain,
    [switch] $Gate,
    [switch] $Viz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:SPECIES_SNAPSHOT_WALL_HOPS = "3"
$env:CLOT_PHI_CEILING_HOPS = "3"

if ($Step -eq "global") { $Step = "species" }
if ($Step -eq "beta") { $Step = "viscosity" }

$SpeciesCkpt = "outputs/biochem/biochem_gnn/species/best.pth"
$BetaCkpt = "outputs/biochem/biochem_gnn/viscosity/beta.pth"
$LoaoRoot = "outputs/biochem/biochem_gnn/loao"
$InitWarm = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$StagingManifest = "data/reference/biochem_gnn_staging.json"
$ReferenceManifest = "data/reference/biochem_gnn_baseline.json"
$StagingEval = "outputs/biochem/biochem_gnn/staging/loao_eval_gt.json"

function Invoke-TrainSpecies {
    if ($SkipTrain) { return }
    $pyArgs = @(
        "-m", "src.training.train_biochem_gnn",
        "--step", "species",
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--init", $InitWarm,
        "--species-out", $SpeciesCkpt
    )
    if ($AllAnchors) { $pyArgs += "--all-anchors" }
    elseif ($Anchors.Trim()) { $pyArgs += @("--anchors", $Anchors) }
    else { $pyArgs += "--all-anchors" }
    if ($Fresh) { $pyArgs += "--fresh" }
    Invoke-PythonRcCheck -Label "biochem_gnn species" -PyArgs $pyArgs
}

function Invoke-TrainViscosity {
    if ($SkipTrain) { return }
    Invoke-PythonRcCheck -Label "biochem_gnn viscosity" -PyArgs @(
        "-m", "src.training.train_biochem_gnn",
        "--step", "viscosity",
        "--species-out", $SpeciesCkpt,
        "--beta-out", $BetaCkpt
    )
}

function Invoke-TrainLoao {
    if ($SkipTrain) { return }
    Invoke-PythonRcCheck -Label "biochem_gnn loao" -PyArgs @(
        "-m", "src.training.train_biochem_gnn",
        "--step", "loao",
        "--loao-epochs", "$LoaoEpochs",
        "--loao-early-stop", "$LoaoEarlyStop",
        "--species-out", $SpeciesCkpt,
        "--loao-root", $LoaoRoot
    )
}

function Invoke-EvalLoao {
    $stagingDir = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/staging"
    New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null
    python -c @"
import json
from pathlib import Path
from src.biochem_gnn.config import default_manifest_payload, rel_path, global_ckpt_path, beta_ckpt_path, loao_root_path
m = default_manifest_payload()
m['species_gnn_ckpt'] = rel_path(global_ckpt_path())
m['viscosity_beta'] = rel_path(beta_ckpt_path())
m['loao_dir'] = rel_path(loao_root_path())
m['phase'] = 'biochem_gnn_staging'
p = Path('$StagingManifest')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(m, indent=2), encoding='utf-8')
print('[OK] staging manifest', p)
"@
    Invoke-PythonRcCheck -Label "deploy eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $StagingManifest,
        "--modes", "deploy_frozen",
        "--times", "27,53",
        "--out", $StagingEval
    )
}

function Invoke-Promote {
    Invoke-PythonRcCheck -Label "promote biochem gnn" -PyArgs @(
        "scripts/promote_biochem_gnn.py",
        "--src-manifest", $StagingManifest,
        "--loao-eval", $StagingEval
    )
}

function Invoke-Gate {
    Invoke-PythonRcCheck -Label "biochem gnn gate" -PyArgs @("scripts/check_biochem_gnn_gate.py")
}

function Invoke-Viz {
    Invoke-PythonRcCheck -Label "biochem gnn viz" -PyArgs @(
        "scripts/viz_species_gnn_deploy.py",
        "--anchor", "patient007",
        "--flow", "kinematics",
        "--manifest", $ReferenceManifest
    )
}

switch ($Step) {
    "species" { Invoke-TrainSpecies }
    "viscosity" { Invoke-TrainViscosity }
    "loao" { Invoke-TrainLoao }
    "eval" { Invoke-EvalLoao }
    "promote" { Invoke-Promote; if ($Gate) { Invoke-Gate }; if ($Viz) { Invoke-Viz } }
    "gate" { Invoke-Gate }
    "viz" { Invoke-Viz }
    "all" {
        Invoke-TrainSpecies
        Invoke-TrainViscosity
        Invoke-TrainLoao
        Invoke-EvalLoao
        Invoke-Promote
        if ($Gate) { Invoke-Gate }
        if ($Viz) { Invoke-Viz }
    }
}

Write-Host "[OK] biochem_gnn step=$Step" -ForegroundColor Green
Write-Host "[i] species=$SpeciesCkpt viscosity=$BetaCkpt loao=$LoaoRoot" -ForegroundColor DarkGray
Write-Host "[i] reference=$ReferenceManifest" -ForegroundColor DarkGray
