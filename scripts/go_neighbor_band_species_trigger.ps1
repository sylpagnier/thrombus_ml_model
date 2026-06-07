# Neighbor-band species teacher (GT flow) + physics trigger eval baseline.
#
# Trains biochem teacher on FI/Mat only inside the clot-phi neighbor shell,
# then evaluates explicit Mat/FI gelation trigger (mu_ratio_max=4).
#
# Baseline ladder (easiest first):
#   1) This script: learned species + explicit physics trigger (no extra train for clot)
#   2) Optional -ClotPhi: kinematic clot-phi MLP on dumped species (learned gate)
#
# Example:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_neighbor_band_species_trigger.ps1 -Fresh
#   powershell ... -SpeciesScope all -SkipTriggerEval   # ablate fi_mat scope

param(
    [switch] $Fresh,
    [int] $TeacherEpochs = 12,
    [string] $SpeciesScope = "fi_mat",
    [switch] $SkipTriggerEval,
    [switch] $ClotPhi,
    [int] $ClotEpochs = 20,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_neighbor_band_species_env.ps1")
. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

$OutRoot = "outputs\biochem\neighbor_band_species"
if ($Fresh) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $OutRoot
}
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$env:BIOCHEM_RUN_NOTE = "neighbor_band_species"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "2"
$env:BIOCHEM_DATA_BIO_SPECIES_SCOPE = $SpeciesScope

$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_RESUME = "0"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_INIT_FROM_BEST = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"

Write-Host "[NEW] Neighbor-band species teacher (scope=$SpeciesScope, ep=$TeacherEpochs)" -ForegroundColor Cyan

python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $TeacherEpochs --save-best --run-name neighbor_band_species
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Teacher = "outputs\biochem\biochem_teacher_last.pth"
if (-not (Test-Path $Teacher)) {
    Write-Host "[ERR] Missing teacher ckpt: $Teacher" -ForegroundColor Red
    exit 1
}

if (-not $SkipViz) {
    Invoke-BiochemTeacherSnapshot -Checkpoint $Teacher -Anchor patient007 -Label "neighbor_band_species"
}

if (-not $SkipTriggerEval) {
    Write-Host "[NEW] Physics trigger eval (explicit Mat/FI, mu_ratio=4)" -ForegroundColor Cyan
    $evalOut = Join-Path $OutRoot "trigger_eval.json"
    $EvalCkpt = "outputs\biochem\biochem_teacher_best_high_mu.pth"
    if (-not (Test-Path $EvalCkpt)) { $EvalCkpt = $Teacher }
    python scripts/eval_neighbor_band_trigger.py `
        --checkpoint $EvalCkpt `
        --compare-gt-species `
        --final-only `
        --out $evalOut
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Trigger eval gate not met (species may still be OK)." -ForegroundColor Yellow
    }
}

if ($ClotPhi) {
    $AnchorDir = Join-Path $OutRoot "anchors_stride36"
    Write-Host "[NEW] Dump species -> $AnchorDir" -ForegroundColor Cyan
    python scripts/dump_teacher_species_to_anchors.py `
        --teacher $Teacher `
        --out-dir $AnchorDir `
        --device cuda `
        --time-stride 36 `
        --min-steps 4 `
        --force
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "[NEW] Learned gate baseline: kinematic clot-phi MLP ($ClotEpochs ep)" -ForegroundColor Cyan
    $clotArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", ".\scripts\go_clot_phi_from_anchor_dir.ps1",
        "-AnchorDir", $AnchorDir,
        "-LegName", "neighbor_band_kinematic",
        "-Epochs", $ClotEpochs
    )
    if ($SkipViz) { $clotArgs += "-SkipViz" }
    & powershell @clotArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[OK] Done -> $OutRoot" -ForegroundColor Green
exit 0
