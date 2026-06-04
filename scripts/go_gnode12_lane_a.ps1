# Rung 12 Lane A: optional mu_ratio uncap finetune -> dump (pred kine) -> clot-phi.
#
# Compare clot-phi to gnode10_kine_loop (p007 ~0.522, min >= 0.26). Mu unlock is teacher-only
# with BIOCHEM_PASSIVE_MU_UNLOCK=1 and mu_ratio_max>1 so val mu_log_mae can move off ~1.44.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode12_lane_a.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode12_lane_a.ps1 -SkipMuUnlock
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode12_lane_a.ps1 -MuRatioMax 80 -MuUnlockEpochs 8
#
# After run:
#   python scripts/check_gnode12_lane_a_gate.py --eval-json outputs/biochem/gnode10_sweep/multi_anchor_gnode12_lane_a_clotphi.jsonl

param(
    [string] $TeacherCkpt = "",
    [string] $JuneAnchorDir = "outputs\biochem\gnode_8h_ladder\anchors_stride_72",
    [string] $OutAnchorDir = "outputs\biochem\gnode10_sweep\anchors_gnode12_predkine_uvp",
    [string] $ClotLeg = "gnode12_lane_a_clotphi",
    [int] $MuUnlockEpochs = 6,
    [double] $MuRatioMax = 20,
    [int] $ClotEpochs = 35,
    [double] $MinGtPosFrac = 0.55,
    [double] $MinClotMinF1 = 0.26,
    [switch] $SkipMuUnlock,
    [switch] $SkipDump,
    [switch] $SkipClot,
    [switch] $SkipViz,
    [switch] $SkipGate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode12_env.ps1")

$TeacherPath = Resolve-Gnode12TeacherCkpt -UserPath $TeacherCkpt
if (-not $TeacherPath) {
    Write-Host "[ERR] Teacher ckpt missing. Run go_gnode11_finish.ps1 or pass -TeacherCkpt." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot $JuneAnchorDir))) {
    Write-Host "[ERR] June anchors missing: $JuneAnchorDir" -ForegroundColor Red
    exit 1
}

$MuUnlockDir = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\gnode12_mu_unlock"
$EvalJson = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\multi_anchor_$ClotLeg.jsonl"

Write-Host "[NEW] GNODE 12 Lane A (dump + clot-phi)" -ForegroundColor Cyan
Write-Host "[i]  teacher=$TeacherPath" -ForegroundColor DarkGray
Write-Host "[i]  mu_ratio_max=$MuRatioMax | mu_unlock_ep=$MuUnlockEpochs (skip=$SkipMuUnlock)" -ForegroundColor DarkGray
Write-Host "[i]  dump -> $OutAnchorDir | clot=$ClotLeg" -ForegroundColor DarkGray

if ($MuUnlockEpochs -gt 0 -and -not $SkipMuUnlock) {
    New-Item -ItemType Directory -Force -Path $MuUnlockDir | Out-Null
    $env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $MuUnlockDir
    Set-Gnode12MuUnlockEnv -Epochs $MuUnlockEpochs -MuRatioMax "$MuRatioMax"
    Copy-Item -Force $TeacherPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")
    Write-Host "[NEW] mu unlock ${MuUnlockEpochs}ep (pred kine, mu_ratio_max=$MuRatioMax)" -ForegroundColor Cyan
    Write-Host "[i]  mu unlock training (~${MuUnlockEpochs} min on GPU; console may be quiet until val lines)" -ForegroundColor DarkGray
    $rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $MuUnlockEpochs --save-best --run-name gnode12_mu_unlock
    if ($rc -ne 0) {
        Write-Host "[ERR] mu unlock training failed (exit=$rc). See outputs/reports/training/biochem/*/run.jsonl" -ForegroundColor Red
        exit $rc
    }
    Write-Host "[OK]  mu unlock training finished (exit=0)" -ForegroundColor Green
    foreach ($rel in @(
            "biochem_teacher_passive_mu_unlock_best.pth",
            "biochem_teacher_best_high_mu.pth"
        )) {
        $cand = Join-Path $MuUnlockDir $rel
        if (Test-Path $cand) {
            $TeacherPath = Resolve-Gnode12RepoPath -Path $cand
            break
        }
    }
    if (-not (Test-Path $TeacherPath)) {
        Write-Host "[ERR] mu unlock ckpt missing under $MuUnlockDir" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK]  mu unlock ckpt -> $TeacherPath" -ForegroundColor Green
}

Set-Gnode12DumpRolloutEnv -MuRatioMax "$MuRatioMax"
$TeacherRel = Get-Gnode12PathRelativeToRepo -FullPath $TeacherPath -RepoRoot $RepoRoot

if (-not $SkipDump) {
    Write-Host "[NEW] dump species + pred [u,v,p] (mu_ratio_max=$MuRatioMax)" -ForegroundColor Cyan
    Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
        --teacher $TeacherRel `
        --src-dir $JuneAnchorDir `
        --out-dir $OutAnchorDir `
        --device cuda `
        --no-subsample `
        --write-kine-macro `
        --mu-ratio-max $MuRatioMax `
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

if ($SkipClot) {
    if (-not $SkipGate) {
        Write-Host "[WARN] clot skipped; lane gate needs eval json from a full run" -ForegroundColor Yellow
    }
    exit 0
}

$env:CLOT_PHI_ANCHOR_DIR = $OutAnchorDir
Remove-Item Env:CLOT_PHI_ROLLOUT -ErrorAction SilentlyContinue
$env:CLOT_PHI_VEL_SOURCE = "gt"

$clotArgs = @(
    "-AnchorDir", $OutAnchorDir,
    "-LegName", $ClotLeg,
    "-Epochs", "$ClotEpochs"
)
if ($SkipViz) { $clotArgs += "-SkipViz" }

Write-Host "[NEW] clot-phi ${ClotEpochs}ep (vel=file pred u,v,p)" -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" @clotArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ckpt = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$ClotLeg\clot_phi_best.pth"
try {
    Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $EvalJson --anchor-dir $OutAnchorDir
} finally {
    Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue
}

if (-not $SkipGate) {
    Write-Host "[NEW] lane A gate check" -ForegroundColor Cyan
    $gateRc = Invoke-PythonRc scripts/check_gnode12_lane_a_gate.py `
        --eval-json $EvalJson `
        --min-clot-min-f1 $MinClotMinF1 `
        --min-gt-pos-frac $MinGtPosFrac
    if ($gateRc -ne 0) { exit $gateRc }
}

Write-Host "[OK]  GNODE 12 Lane A complete." -ForegroundColor Green
Write-Host "[i]  anchors: $OutAnchorDir" -ForegroundColor DarkGray
Write-Host "[i]  clot-phi: $ckpt" -ForegroundColor DarkGray
Write-Host "[i]  eval: $EvalJson (baseline: gnode10_kine_loop p007 ~0.522)" -ForegroundColor DarkGray
