# M3 alignment sweep: test ADR/data co-training fixes (masked ADR, wall split, fast transient).
#
# Prereq: outputs/biochem/biochem_teacher_phaseB_ramp1_last.pth (or run phase B ramp1 once).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_m3_adr_alignment_sweep.ps1"
#   ... -Epochs 6 -SkipAudit

param(
    [int] $Epochs = 6,
    [switch] $SkipAudit,
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_phaseB_ramp1_last.pth"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_m3_alignment_env.ps1")

$LogRoot = Join-Path $RepoRoot "outputs\biochem\m3_alignment_sweep"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$LogPath = Join-Path $LogRoot "m3_log.jsonl"

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath (run go_phaseB_xy_passive.ps1 ramp1 first)" -ForegroundColor Red
    exit 1
}

function Write-M3Log {
    param([string] $Step, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{ ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"); step = $Step; status = $Status }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $LogPath -Encoding utf8
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } else { "Cyan" })
}

function Invoke-M3Leg {
    param(
        [string] $Note,
        [hashtable] $EnvOverrides = @{},
        [string] $TfMin = "1.0"
    )

    Clear-M3AlignmentOverrides
    Set-M3AlignmentBaseEnv -RunNote $Note -TfMin $TfMin
    foreach ($k in $EnvOverrides.Keys) {
        Set-Item -Path "Env:\$k" -Value $EnvOverrides[$k]
    }

    Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

    Write-M3Log $Note "START" @{ epochs = $Epochs }
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"

    $rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $Epochs --save-best --run-name $Note
    if ($rc -ne 0) {
        Write-M3Log $Note "FAIL" @{ exit = $rc }
        return $false
    }

    $gateRc = Invoke-PythonRc scripts/check_m3_alignment_gate.py --run-note $Note -Quiet
    $st = if ($gateRc -eq 0) { "OK" } else { "WARN" }
    Write-M3Log ($Note + "_gate") $st @{ exit = $gateRc }
    Write-M3Log $Note "DONE" @{}
    return ($gateRc -eq 0)
}

if (-not $SkipAudit) {
    Write-Host "[NEW] GT residual audit (patient007, TF=1)" -ForegroundColor Cyan
    $auditRc = Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007
    Write-M3Log "m3_gt_audit" $(if ($auditRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $auditRc }
}

$legs = @(
    @{ note = "m3_A0_baseline"; env = @{ BIOCHEM_ADR_MASK_MODE = "global" } },
    @{ note = "m3_A1_mask_match"; env = @{ BIOCHEM_ADR_MASK_MODE = "match_data_bio" } },
    @{ note = "m3_A2_mask_nowall"; env = @{ BIOCHEM_ADR_MASK_MODE = "match_data_bio"; BIOCHEM_ADR_EXCLUDE_WALL = "1" } },
    @{ note = "m3_A3_fast_transient"; env = @{ BIOCHEM_ADR_MASK_MODE = "global"; BIOCHEM_ADR_FAST_TRANSIENT = "1" } },
    @{
        note = "m3_A4_mask_nowall_wallbp"
        env = @{
            BIOCHEM_ADR_MASK_MODE = "match_data_bio"
            BIOCHEM_ADR_EXCLUDE_WALL = "1"
            BIOCHEM_PASSIVE_WALL_BACKPROP = "1"
            BIOCHEM_PASSIVE_WALL_WEIGHT = "1e-3"
        }
    },
    @{
        note = "m3_A5_combo"
        env = @{
            BIOCHEM_ADR_MASK_MODE = "match_data_bio"
            BIOCHEM_ADR_EXCLUDE_WALL = "1"
            BIOCHEM_ADR_FAST_TRANSIENT = "1"
            BIOCHEM_PASSIVE_WALL_BACKPROP = "1"
            BIOCHEM_PASSIVE_WALL_WEIGHT = "1e-3"
        }
    },
    @{
        note = "m3_A6_combo_tf05"
        env = @{
            BIOCHEM_ADR_MASK_MODE = "match_data_bio"
            BIOCHEM_ADR_EXCLUDE_WALL = "1"
            BIOCHEM_ADR_FAST_TRANSIENT = "1"
            BIOCHEM_PASSIVE_WALL_BACKPROP = "1"
            BIOCHEM_PASSIVE_WALL_WEIGHT = "1e-3"
        }
        tf = "0.5"
    }
)

$pass = 0
foreach ($leg in $legs) {
    $tf = if ($leg.tf) { $leg.tf } else { "1.0" }
    if (Invoke-M3Leg -Note $leg.note -EnvOverrides $leg.env -TfMin $tf) { $pass++ }
}

Write-Host "[OK] M3 sweep complete: $pass / $($legs.Count) legs passed alignment gate" -ForegroundColor Green
Write-Host "[i] Log: $LogPath" -ForegroundColor Cyan
Write-Host "[i] Review: Get-Content outputs\biochem\m3_alignment_sweep\m3_log.jsonl -Tail 20" -ForegroundColor Cyan
