# Rung 10 finish: K5 species on June time grid (gt+ parity) + clot-phi gate.
#
# Teacher (K5) already trained by go_gnode10_sweep.ps1. This re-rolls species on the
# canonical June subsampled anchors (no re-stride from full T=54) then runs clot-phi.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_finish.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_finish.ps1 -SkipClot
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_finish.ps1 -SkipDump   # resume after dump OK

param(
    [string] $TeacherCkpt = "",
    [string] $JuneAnchorDir = "outputs\biochem\gnode_8h_ladder\anchors_stride_72",
    [string] $OutAnchorDir = "outputs\biochem\gnode10_sweep\anchors_june_times_k5_predkine",
    [string] $ClotLeg = "gnode10_k5_june_times_clotphi",
    [int] $ClotEpochs = 35,
    [double] $MinGtPosFrac = 0.55,
    [switch] $SkipClot,
    [switch] $SkipDump,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

if (-not $TeacherCkpt) {
    $candidates = @(
        "outputs\biochem\gnode10_sweep\K5_kine15_final\biochem_teacher_best_high_mu.pth",
        "outputs\biochem\gnode10_sweep\promoted\biochem_teacher_best_high_mu.pth",
        "outputs\biochem\gnode10_sweep\K5_kine15\biochem_teacher_best_high_mu.pth"
    )
    foreach ($rel in $candidates) {
        if (Test-Path (Join-Path $RepoRoot $rel)) {
            $TeacherCkpt = $rel
            break
        }
    }
}
if (-not $TeacherCkpt -or -not (Test-Path (Join-Path $RepoRoot $TeacherCkpt))) {
    Write-Host "[ERR] K5 teacher ckpt missing. Run go_gnode10_sweep.ps1 first or pass -TeacherCkpt." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $RepoRoot $JuneAnchorDir))) {
    Write-Host "[ERR] June anchor cache missing: $JuneAnchorDir" -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] GNODE 10 finish | teacher=$TeacherCkpt" -ForegroundColor Cyan
Write-Host "[i]  src=$JuneAnchorDir (no resubsample) -> $OutAnchorDir" -ForegroundColor DarkGray

$env:BIOCHEM_GT_KINE_VEL = "0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "1"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "6"

if (-not $SkipDump) {
    Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
        --teacher $TeacherCkpt `
        --src-dir $JuneAnchorDir `
        --out-dir $OutAnchorDir `
        --device cuda `
        --no-subsample `
        --force
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} elseif (-not (Test-Path (Join-Path $RepoRoot $OutAnchorDir))) {
    Write-Host "[ERR] -SkipDump but anchor dir missing: $OutAnchorDir" -ForegroundColor Red
    exit 1
}

$preflightLeg = "${ClotLeg}_preflight"
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
    -AnchorDir $OutAnchorDir -LegName $preflightLeg -Epochs 1 -SkipViz -SkipEval
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$preflightLog = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$preflightLeg\clot_phi_train_log.jsonl"
$gtPlus = $null
if (Test-Path $preflightLog) {
    $lastLine = Get-Content $preflightLog -Tail 1
    if ($lastLine) {
        $row = $lastLine | ConvertFrom-Json
        $gtPlus = [double]$row.val.gt_pos_frac
    }
}
if ($null -eq $gtPlus) {
    Write-Host "[ERR] preflight log missing gt+ at $preflightLog" -ForegroundColor Red
    exit 1
}
Write-Host "[i]  preflight gt+=$([math]::Round($gtPlus, 3)) (gate >= $MinGtPosFrac)" -ForegroundColor DarkGray
if ($gtPlus -lt $MinGtPosFrac) {
    Write-Host "[ERR] dump gt+ too low ($gtPlus < $MinGtPosFrac); fix anchor cache before 35ep clot-phi." -ForegroundColor Red
    exit 1
}
Write-Host "[OK]  preflight gt+ gate PASS" -ForegroundColor Green

if ($SkipClot) {
    Write-Host "[OK]  Preflight done. Run clot-phi without -SkipClot when gt+ looks good." -ForegroundColor Green
    exit 0
}

if ($SkipViz) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
        -AnchorDir $OutAnchorDir -LegName $ClotLeg -Epochs $ClotEpochs -SkipViz
} else {
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
        -AnchorDir $OutAnchorDir -LegName $ClotLeg -Epochs $ClotEpochs
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$evalJson = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\multi_anchor_$ClotLeg.jsonl"
$ckpt = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$ClotLeg\clot_phi_best.pth"
$env:CLOT_PHI_ANCHOR_DIR = $OutAnchorDir
try {
    Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalJson --anchor-dir $OutAnchorDir
} finally {
    Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue
}

$promoted = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep\promoted"
New-Item -ItemType Directory -Force -Path $promoted | Out-Null
if (Test-Path $ckpt) {
    Copy-Item -Force $ckpt (Join-Path $promoted "clot_phi_best.pth")
    Copy-Item -Force (Join-Path $RepoRoot $TeacherCkpt) (Join-Path $promoted "biochem_teacher_best_high_mu.pth")
}

Write-Host "[OK]  GNODE 10 finish complete." -ForegroundColor Green
Write-Host "[i]  anchors: $OutAnchorDir" -ForegroundColor DarkGray
Write-Host "[i]  clot-phi: $ckpt" -ForegroundColor DarkGray
Write-Host "[i]  eval: $evalJson (compare min_f1 to 0.26; p007 target ~0.63 on good dump)" -ForegroundColor DarkGray
