# Star 5 (T5): retrain deploy-faithful species teacher (pred kine + FI/Mat on all anchors),
# dump pred-flow species cache, then T4-style trigger eval + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t5.ps1"
#   powershell ... -Fresh
#   powershell ... -SkipTrain -SkipEval   # dump + viz only
#
# Time (CUDA): train ~1-2h (12ep) + dump ~30-45min + eval live ~30-60min LOAO

param(
    [switch] $Fresh,
    [int] $TeacherEpochs = 12,
    [string] $SpeciesScope = "fi_mat",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Checkpoint = "outputs/biochem/clot_trigger/t1/clot_trigger_t1_best.pth",
    [string] $OutRoot = "outputs/biochem/clot_trigger/t5_deploy_teacher",
    [string] $DumpDir = "outputs/biochem/anchors_teacher_species_predkine",
    [int] $DumpStride = 6,
    [int] $DumpMinSteps = 8,
    [string] $Val = "patient007",
    [string] $Anchor = "patient007",
    [string] $Anchor2 = "patient002",
    [string] $VizDir = "outputs/biochem/viz/clot_trigger",
    [int] $ProgressStep = 5,
    [switch] $SkipTrain,
    [switch] $SkipDump,
    [switch] $SkipEval,
    [switch] $SkipViz,
    [switch] $EvalDumpedOnly,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_trigger_t5_deploy_env.ps1")

$env:PYTHONUNBUFFERED = "1"

$DeployTeacher = Join-Path $OutRoot "biochem_teacher_deploy.pth"
$TrainLast = "outputs/biochem/biochem_teacher_last.pth"

if ($Fresh) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $RepoRoot $OutRoot)
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $RepoRoot $DumpDir)
}
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null

if (-not (Test-Path (Join-Path $RepoRoot $KineCkpt))) {
    Write-Host "[ERR] missing kinematics ckpt: $KineCkpt" -ForegroundColor Red
    exit 1
}

if (-not $SkipTrain) {
    Write-Host "[NEW] T5 train deploy teacher (pred kine, scope=$SpeciesScope, ep=$TeacherEpochs)" -ForegroundColor Cyan
    Set-ClotTriggerT5DeployTrainEnv -RunNote "clot_trigger_t5_deploy" -TeacherEpochs $TeacherEpochs `
        -SpeciesScope $SpeciesScope -KineCkpt $KineCkpt
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    Invoke-PythonRcCheck -Label "t5 train teacher" -PyArgs @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--skip-pretrain", "--init-from-best",
        "--epochs", "$TeacherEpochs", "--save-best",
        "--run-name", "clot_trigger_t5_deploy"
    )
    if (-not (Test-Path (Join-Path $RepoRoot $TrainLast))) {
        Write-Host "[ERR] missing $TrainLast after train" -ForegroundColor Red
        exit 1
    }
    Invoke-PythonRcCheck -Label "t5 promote teacher" -PyArgs @(
        "scripts/promote_clot_trigger_t5_teacher.py",
        "--src", $TrainLast,
        "--note", "clot_trigger_t5_deploy ep=$TeacherEpochs scope=$SpeciesScope"
    )
} elseif (-not (Test-Path (Join-Path $RepoRoot $DeployTeacher))) {
    Write-Host "[ERR] missing deploy teacher: $DeployTeacher (run without -SkipTrain)" -ForegroundColor Red
    exit 1
}

if (-not $SkipDump) {
    Write-Host "[NEW] T5 dump pred-flow species -> $DumpDir" -ForegroundColor Cyan
    $dumpArgs = @(
        "scripts/dump_teacher_species_to_anchors.py",
        "--teacher", $DeployTeacher,
        "--out-dir", $DumpDir,
        "--device", "cuda",
        "--time-stride", "$DumpStride",
        "--min-steps", "$DumpMinSteps",
        "--write-kine-macro",
        "--force"
    )
    Invoke-PythonRcCheck -Label "t5 dump predkine species" -PyArgs $dumpArgs
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if (-not $SkipEval) {
    if (-not $EvalDumpedOnly) {
        Write-Host "[NEW] T5 live eval (deploy teacher + pred kine)" -ForegroundColor Cyan
        $liveOut = Join-Path $OutRoot "t5_deploy_live.json"
        $liveArgs = @(
            "scripts/eval_clot_trigger_t3_full_stack.py",
            "--species-source", "live",
            "--star", "t5",
            "--checkpoint", $Checkpoint,
            "--teacher", $DeployTeacher,
            "--kine-ckpt", $KineCkpt,
            "--out", $liveOut,
            "--val", $Val,
            "--progress-step", "$ProgressStep"
        )
        if ($Quiet) { $liveArgs += "--quiet" }
        Invoke-PythonRcCheck -Label "t5 eval live" -PyArgs $liveArgs
    }
    Write-Host "[NEW] T5 fast eval (predkine species dump)" -ForegroundColor Cyan
    $dumpOut = Join-Path $OutRoot "t5_deploy_dumped.json"
    $fastArgs = @(
        "scripts/eval_clot_trigger_t3_full_stack.py",
        "--species-source", "dumped",
        "--star", "t5",
        "--anchor-dir", $DumpDir,
        "--checkpoint", $Checkpoint,
        "--kine-ckpt", $KineCkpt,
        "--out", $dumpOut,
        "--val", $Val,
        "--progress-step", "$ProgressStep"
    )
    if ($Quiet) { $fastArgs += "--quiet" }
    Invoke-PythonRcCheck -Label "t5 eval dumped" -PyArgs $fastArgs
}

if (-not $SkipViz) {
    foreach ($anc in @($Anchor, $Anchor2)) {
        if (-not $anc) { continue }
        Write-Host "[NEW] T5 viz $anc (live deploy teacher)" -ForegroundColor Cyan
        Invoke-PythonRcCheck -Label "t5 viz $anc" -PyArgs @(
            "scripts/viz_clot_trigger_t3.py",
            "--star", "t5",
            "--species-source", "live",
            "--anchor", $anc,
            "--checkpoint", $Checkpoint,
            "--teacher", $DeployTeacher,
            "--kine-ckpt", $KineCkpt,
            "--out", "$VizDir/t5_$anc.png"
        )
    }
}

Write-Host ""
Write-Host "[OK] T5 done -> $OutRoot" -ForegroundColor Green
Write-Host "[i]  next: go_clot_trigger_t6.ps1 (coupled kine)" -ForegroundColor DarkGray
