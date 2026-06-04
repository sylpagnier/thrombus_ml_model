# Rung 10 mini ladder: predicted Stage-A kine end-to-end (no GT u,v,p answer key in dump/clot).
#
# 1) Optional K5 teacher finetune (GT_KINE_VEL=0, already default in sweep)
# 2) Dump K5 species + predicted [u,v,p] on June time grid (mu_eff labels unchanged)
# 3) Preflight gt+ gate
# 4) Clot-phi 35ep on dumped anchors (vel from file = predicted u,v,p)
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_kine_loop.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_kine_loop.ps1 -SkipDump
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_kine_loop.ps1 -RolloutKine -KineTf 0.3
#
# Compare to go_gnode10_finish (GT u,v,p in anchors, clot vel=gt): p007 ~0.629.

param(
    [string] $TeacherCkpt = "",
    [string] $JuneAnchorDir = "outputs\biochem\gnode_8h_ladder\anchors_stride_72",
    [string] $OutAnchorDir = "outputs\biochem\gnode10_sweep\anchors_june_times_k5_predkine_uvp",
    [string] $ClotLeg = "gnode10_k5_predkine_uvp_clotphi",
    [int] $FinetuneEpochs = 0,
    [int] $ClotEpochs = 35,
    [double] $MinGtPosFrac = 0.55,
    [double] $KineTf = 0.0,
    [switch] $RolloutKine,
    [switch] $SkipFinetune,
    [switch] $SkipDump,
    [switch] $SkipClot,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode10_env.ps1")

function Resolve-Gnode10K5Ckpt {
    param([string] $UserPath)
    if ($UserPath -and (Test-Path (Join-Path $RepoRoot $UserPath))) { return $UserPath }
    foreach ($rel in @(
            "outputs\biochem\gnode10_sweep\K5_kine15_final\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\promoted\biochem_teacher_best_high_mu.pth"
        )) {
        if (Test-Path (Join-Path $RepoRoot $rel)) { return $rel }
    }
    return $null
}

$TeacherCkpt = Resolve-Gnode10K5Ckpt -UserPath $TeacherCkpt
if (-not $TeacherCkpt) {
    Write-Host "[ERR] K5 teacher missing. Run go_gnode10_sweep.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot $JuneAnchorDir))) {
    Write-Host "[ERR] June anchors missing: $JuneAnchorDir" -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] GNODE 10 kine loop | teacher=$TeacherCkpt" -ForegroundColor Cyan
Write-Host "[i]  June times -> $OutAnchorDir | clot=$ClotLeg | rollout_kine=$RolloutKine kine_tf=$KineTf" -ForegroundColor DarkGray

if ($FinetuneEpochs -gt 0 -and -not $SkipFinetune) {
    $ftDir = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\K5_kine15_kine_loop_ft"
    New-Item -ItemType Directory -Force -Path $ftDir | Out-Null
    Clear-Gnode10BiochemEnv
    Set-Gnode10PredictedKineBaseEnv -RunNote "gnode10_k5_kine_loop_ft" -Epochs $FinetuneEpochs -OomSafe
    Apply-Gnode10LegOverrides -Leg @{
        Title = "K5 kine loop finetune"
        TeacherForceMin = 0.5
        KineWeight = 0.15
        TrainKinLora = $true
        TbpttWindow = 5
    }
    $env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $ftDir
    Copy-Item -Force (Join-Path $RepoRoot $TeacherCkpt) (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")
    Write-Host "[NEW] K5 finetune ${FinetuneEpochs}ep (GT_KINE_VEL=0)" -ForegroundColor Cyan
    Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $FinetuneEpochs --save-best --run-name gnode10_k5_kine_loop_ft
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $ftBest = Join-Path $ftDir "biochem_teacher_best_high_mu.pth"
    if (Test-Path $ftBest) { $TeacherCkpt = $ftBest.Replace($RepoRoot + "\", "").Replace($RepoRoot + "/", "") }
}

$env:BIOCHEM_GT_KINE_VEL = "0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "1"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "6"

if (-not $SkipDump) {
    Write-Host "[NEW] dump species + pred [u,v,p] (DEQ kine)" -ForegroundColor Cyan
    Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
        --teacher $TeacherCkpt `
        --src-dir $JuneAnchorDir `
        --out-dir $OutAnchorDir `
        --device cuda `
        --no-subsample `
        --write-kine-macro `
        --force
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} elseif (-not (Test-Path (Join-Path $RepoRoot $OutAnchorDir))) {
    Write-Host "[ERR] -SkipDump but missing $OutAnchorDir" -ForegroundColor Red
    exit 1
}

$preflightLeg = "${ClotLeg}_preflight"
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
    -AnchorDir $OutAnchorDir -LegName $preflightLeg -Epochs 1 -SkipViz -SkipEval
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$preflightLog = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$preflightLeg\clot_phi_train_log.jsonl"
$gtPlus = $null
if (Test-Path $preflightLog) {
    $row = (Get-Content $preflightLog -Tail 1) | ConvertFrom-Json
    $gtPlus = [double]$row.val.gt_pos_frac
}
if ($null -eq $gtPlus -or $gtPlus -lt $MinGtPosFrac) {
    Write-Host "[ERR] preflight gt+=$gtPlus (need >= $MinGtPosFrac)" -ForegroundColor Red
    exit 1
}
Write-Host "[OK]  preflight gt+=$([math]::Round($gtPlus, 3))" -ForegroundColor Green

if ($SkipClot) { exit 0 }

# Clot-phi: default reads [u,v,p] from anchor file (now predicted). -RolloutKine = live DEQ each step (6b-style).
$env:CLOT_PHI_ANCHOR_DIR = $OutAnchorDir
if ($RolloutKine) {
    $env:CLOT_PHI_ROLLOUT = "1"
    $env:CLOT_PHI_ROLLOUT_DETACH = "1"
    $env:CLOT_PHI_VEL_SOURCE = "kinematics"
    $env:CLOT_PHI_CARRY_PHI = "1"
    $env:CLOT_PHI_CARRY_LOG_MU = "1"
    $env:CLOT_PHI_KINE_TF = "$KineTf"
    $env:CLOT_PHI_KINE_CKPT = "outputs/kinematics/kinematics_best.pth"
    $env:CLOT_PHI_SPECIES_FEATURES = "1"
} else {
    Remove-Item Env:CLOT_PHI_ROLLOUT -ErrorAction SilentlyContinue
    $env:CLOT_PHI_VEL_SOURCE = "gt"
    Remove-Item Env:CLOT_PHI_KINE_TF -ErrorAction SilentlyContinue
}

$clotArgs = @(
    "-AnchorDir", $OutAnchorDir,
    "-LegName", $ClotLeg,
    "-Epochs", "$ClotEpochs"
)
if ($SkipViz) { $clotArgs += "-SkipViz" }

Write-Host "[NEW] clot-phi ${ClotEpochs}ep vel=$(if ($RolloutKine) { 'kinematics+rollout' } else { 'file_pred_uvp' })" -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" @clotArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ckpt = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$ClotLeg\clot_phi_best.pth"
$evalJson = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\multi_anchor_$ClotLeg.jsonl"
try {
    Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalJson --anchor-dir $OutAnchorDir
} finally {
    Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:CLOT_PHI_ROLLOUT -ErrorAction SilentlyContinue
    Remove-Item Env:CLOT_PHI_VEL_SOURCE -ErrorAction SilentlyContinue
}

Write-Host "[OK]  Kine loop complete." -ForegroundColor Green
Write-Host "[i]  anchors: $OutAnchorDir" -ForegroundColor DarkGray
Write-Host "[i]  clot-phi: $ckpt" -ForegroundColor DarkGray
Write-Host "[i]  eval: $evalJson (vs finish baseline p007 ~0.629, min >= 0.26)" -ForegroundColor DarkGray
