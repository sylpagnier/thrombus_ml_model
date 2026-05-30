# Phase A quick iteration: X (data-only species) then Y (isolated analytical terms).
# Short legs only; use check_phase_a_gate.py after each batch.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_phase_a_xy_iterate.ps1"
#   ... -XOnly -EpochsX 5 -EpochsY 3
#   ... -SkipX -SkipY

param(
    [switch] $XOnly,
    [switch] $YOnly,
    [switch] $SkipX,
    [switch] $SkipY,
    [int] $EpochsX = 5,
    [int] $EpochsY = 3,
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")
. (Join-Path $PSScriptRoot "_passive_x_block_env.ps1")
. (Join-Path $PSScriptRoot "_passive_phase_a_env.ps1")

$LogRoot = Join-Path $RepoRoot "outputs\biochem\phase_a_xy"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$LogPath = Join-Path $LogRoot "phase_a_log.jsonl"

function Write-PhaseALog {
    param([string] $Step, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        step = $Step
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $LogPath -Encoding utf8
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } else { "Cyan" })
}

function Invoke-PhaseATrain {
    param([string] $RunNote, [int] $Epochs)
    if (Test-Path (Join-Path $RepoRoot $InitCkpt)) {
        Copy-Item (Join-Path $RepoRoot $InitCkpt) (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
    }
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    Write-PhaseALog $RunNote "START" @{ epochs = $Epochs }
    $rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $Epochs --save-best --run-name $RunNote
    if ($rc -ne 0) {
        Write-PhaseALog $RunNote "FAIL" @{ exit = $rc }
        return $false
    }
    $gateRc = 0
    if ($env:BIOCHEM_LOSS_ISOLATE -eq "PASSIVE" -or $env:BIOCHEM_LOSS_ISOLATE -eq "DATA_BIO") {
        $gateRc = Invoke-PythonRc scripts/check_passive_x_species_gate.py --run-note $RunNote -Quiet
        $xRc = Invoke-PythonRc scripts/check_phase_a_gate.py --mode x --run-note $RunNote -Quiet
        Write-PhaseALog ($RunNote + "_gate_flow") $(if ($gateRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $gateRc }
        Write-PhaseALog ($RunNote + "_gate_x") $(if ($xRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $xRc }
    } else {
        $term = $env:BIOCHEM_LOSS_ISOLATE
        $yRc = Invoke-PythonRc scripts/check_phase_a_gate.py --mode y --run-note $RunNote --term $term -Quiet
        Write-PhaseALog ($RunNote + "_gate_y") $(if ($yRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $yRc; term = $term }
    }
    Write-PhaseALog $RunNote "DONE" @{}
    return $true
}

# --- Phase A.X: compare recipes (5ep) ---
if (-not $SkipX -and -not $YOnly) {
    $xLegs = @(
        @{ note = "phaseA_X_cb_lr1e3"; lr = "1e-3"; mask = "clot_band"; iso = "PASSIVE"; fi = "3.0"; mat = "2.0" },
        @{ note = "phaseA_X_cb_lr5e4"; lr = "5e-4"; mask = "clot_band"; iso = "PASSIVE"; fi = "3.0"; mat = "2.0" },
        @{ note = "phaseA_X_global_lr1e3"; lr = "1e-3"; mask = "global"; iso = "PASSIVE"; fi = "3.0"; mat = "2.0" },
        @{ note = "phaseA_X_data_bio"; lr = "1e-3"; mask = "clot_band"; iso = "DATA_BIO"; fi = "3.0"; mat = "2.0" },
        @{ note = "phaseA_X_fi2mat2"; lr = "1e-3"; mask = "clot_band"; iso = "PASSIVE"; fi = "2.0"; mat = "2.0" }
    )
    foreach ($leg in $xLegs) {
        Set-PassiveXLegEnv -RunNote $leg.note -Epochs $EpochsX -InitCkpt $InitCkpt `
            -LossIsolate $leg.iso -TeacherLr $leg.lr -BioMask $leg.mask `
            -FiWeight $leg.fi -MatWeight $leg.mat
        Invoke-PhaseATrain -RunNote $leg.note -Epochs $EpochsX | Out-Null
    }
    foreach ($seed in @(101, 202)) {
        $note = "phaseA_X_cb_lr1e3_seed$seed"
        Set-PassiveXLegEnv -RunNote $note -Epochs $EpochsX -InitCkpt $InitCkpt -LossIsolate "PASSIVE" -TeacherLr "1e-3"
        $env:PYTHONHASHSEED = "$seed"
        Invoke-PhaseATrain -RunNote $note -Epochs $EpochsX | Out-Null
        $env:PYTHONHASHSEED = "420"
    }
}

# --- Phase A.Y: isolated analytical terms (3ep, Y harness) ---
if (-not $SkipY -and -not $XOnly) {
    $yTerms = @("ADR_F", "ADR_S", "W_PHY", "W_BIO", "BIO_IO")
    foreach ($term in $yTerms) {
        $note = "phaseA_Y_${term}_clip10"
        Set-PhaseAYHarnessEnv -RunNote $note -LossIsolate $term -TeacherLr "1e-3"
        $ep = if ($term -eq "W_PHY") { 4 } else { $EpochsY }
        Invoke-PhaseATrain -RunNote $note -Epochs $ep | Out-Null
    }
    foreach ($lr in @("3e-4", "1e-3")) {
        $note = "phaseA_Y_ADR_S_lr" + ($lr -replace '\.', '')
        Set-PhaseAYHarnessEnv -RunNote $note -LossIsolate "ADR_S" -TeacherLr $lr
        Invoke-PhaseATrain -RunNote $note -Epochs $EpochsY | Out-Null
    }
}

Write-Host "[OK] Phase A iterate log -> $LogPath" -ForegroundColor Green
Write-Host "[i] Review: Get-Content outputs\biochem\phase_a_xy\phase_a_log.jsonl -Tail 30" -ForegroundColor Cyan
