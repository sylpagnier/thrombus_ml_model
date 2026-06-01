# M3 analytical ADR alignment block: Phase B ramp -> cal probe -> formulation ladders -> lock -> gate.
#
# Prereq: outputs/biochem/biochem_teacher_passive_species_locked.pth (I.1 promote) or align_locked.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_m3_block_pass.ps1" -Probe
#   ... -Turbo                    # shorter full block (~3-5h)
#   ... -SkipNarrowing -SkipSweep # cal + promote only (~2h)
#   ... -SkipPhaseB               # use existing phaseB_ramp1_last.pth

param(
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_species_locked.pth",
    [string] $AlignRunNote = "m3_align_transport_union",
    [string] $DestLocked = "outputs/biochem/biochem_teacher_passive_m3_locked.pth",
    [string] $ManifestPath = "outputs/biochem/passive_m3_locked_manifest.json",
    [switch] $Probe,
    [switch] $Turbo,
    [switch] $SkipAudit,
    [switch] $SkipPhaseB,
    [switch] $SkipNarrowing,
    [switch] $SkipSweep,
    [switch] $SkipLock
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

if ($Probe) {
    $Turbo = $true
    $SkipAudit = $true
    $SkipNarrowing = $true
    $SkipSweep = $true
    $SkipLock = $true
}

$Ramp1Epochs = if ($Probe) { 2 } elseif ($Turbo) { 3 } else { 3 }
$Ramp2Epochs = if ($Probe) { 2 } elseif ($Turbo) { 4 } else { 6 }
$AlignEpochs = if ($Probe) { 3 } elseif ($Turbo) { 6 } else { 12 }
$NarrowEpochs = if ($Turbo) { 2 } else { 3 }
$SweepEpochs = if ($Turbo) { 3 } else { 6 }

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init: $initPath (finish I.1 X promote first)" -ForegroundColor Red
    exit 1
}

$best = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
Copy-Item $initPath $best -Force
$mode = if ($Probe) { "PROBE (trends only, ~30-45m)" } elseif ($Turbo) { "TURBO full block (~3-5h)" } else { "FULL block (~8-12h)" }
Write-Host "[NEW] M3 block pass [$mode] init=$InitCkpt" -ForegroundColor Cyan
Write-Host "[i]  ramp1=$Ramp1Epochs ramp2=$Ramp2Epochs align=$AlignEpochs narrow=$NarrowEpochs sweep=$SweepEpochs" -ForegroundColor Cyan
if ($Probe) {
    Write-Host "[i]  Probe: no lock/promote gate; read run.jsonl + check_m3_align_gate (may WARN on 3ep)" -ForegroundColor Cyan
}

if (-not $SkipAudit) {
    $auditRc = Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --all-formulations
    if ($auditRc -ne 0) {
        Write-Host "[WARN] GT formulation audit exit $auditRc (continuing)" -ForegroundColor Yellow
    }
}

if (-not $SkipPhaseB) {
    & (Join-Path $PSScriptRoot "go_phaseB_xy_passive.ps1") -Ramp1Epochs $Ramp1Epochs -Ramp2Epochs $Ramp2Epochs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    $r1 = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_phaseB_ramp1_last.pth"
    if (-not (Test-Path $r1)) {
        Write-Host "[ERR] -SkipPhaseB but missing $r1" -ForegroundColor Red
        exit 1
    }
    Copy-Item $r1 $best -Force
    Write-Host "[i] Phase B skipped; seeded best from phaseB_ramp1_last" -ForegroundColor Cyan
}

$alignNote = if ($Probe) { "${AlignRunNote}_probe" } else { $AlignRunNote }
& (Join-Path $PSScriptRoot "go_m3_align_probe.ps1") -Epochs $AlignEpochs -RunNote $alignNote -SkipAudit
if ($LASTEXITCODE -ne 0 -and -not $Probe) { exit $LASTEXITCODE }
if ($Probe -and $LASTEXITCODE -ne 0) {
    Write-Host "[WARN] align probe train exit $LASTEXITCODE (continuing for logs)" -ForegroundColor Yellow
}

if (-not $SkipNarrowing) {
    & (Join-Path $PSScriptRoot "go_m3_narrowing_90m.ps1") -Epochs $NarrowEpochs -SkipAudit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] M3 narrowing had failures (see narrow_log.jsonl)" -ForegroundColor Yellow
    }
    Invoke-PythonRc scripts/summarize_m3_narrowing.py | Out-Host
}

if (-not $SkipSweep) {
    & (Join-Path $PSScriptRoot "go_m3_adr_alignment_sweep.ps1") -Epochs $SweepEpochs -SkipAudit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] M3 alignment sweep had failures (see m3_log.jsonl)" -ForegroundColor Yellow
    }
}

if (-not $SkipLock) {
    $last = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
    if (-not (Test-Path $last)) {
        Write-Host "[ERR] Missing $last after calibration" -ForegroundColor Red
        exit 1
    }
    & (Join-Path $PSScriptRoot "go_passive_lock_align_ckpt.ps1") `
        -SourceCkpt "outputs/biochem/biochem_teacher_last.pth" `
        -DestCkpt $DestLocked `
        -RunNote $AlignRunNote `
        -ManifestPath $ManifestPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Probe) {
    $gateRc = Invoke-PythonRc scripts/check_m3_align_gate.py --run-note $alignNote -Quiet
    if ($gateRc -eq 0) {
        Write-Host "[OK] M3 probe: align gate passed (lucky on short run)" -ForegroundColor Green
    } else {
        Write-Host "[WARN] M3 probe: align gate not passed (expected on 3ep); inspect run.jsonl trends" -ForegroundColor Yellow
    }
    Write-Host "[i]  python scripts/check_m3_align_gate.py --run-note $alignNote" -ForegroundColor Cyan
    Write-Host "[i]  Full wrap: go_m3_block_pass.ps1 -Turbo or default (no -Probe)" -ForegroundColor Cyan
    exit 0
}

$finalRc = Invoke-PythonRc scripts/check_m3_block_pass.py --run-note $AlignRunNote --locked-ckpt $DestLocked
exit $finalRc
