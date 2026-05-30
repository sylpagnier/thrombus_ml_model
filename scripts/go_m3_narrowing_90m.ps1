# M3 narrowing experiment (~90 min): formulation / scale / scope ladder (3ep per leg).
#
# Prereq: outputs/biochem/biochem_teacher_phaseB_ramp1_last.pth
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_m3_narrowing_90m.ps1"
#   ... -Epochs 3 -SkipAudit

param(
    [int] $Epochs = 3,
    [switch] $SkipAudit,
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_phaseB_ramp1_last.pth"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_m3_narrowing_env.ps1")

$LogRoot = Join-Path $RepoRoot "outputs\biochem\m3_narrowing_90m"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$LogPath = Join-Path $LogRoot "narrow_log.jsonl"

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath (run go_phaseB_xy_passive.ps1 -Ramp1Epochs 3 -Ramp2Epochs 0)" -ForegroundColor Red
    exit 1
}

function Write-NarrowLog {
    param([string] $Step, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{ ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"); step = $Step; status = $Status }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $LogPath -Encoding utf8
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } else { "Cyan" })
}

function Invoke-NarrowLeg {
    param([string] $Note, [hashtable] $EnvOverrides = @{})

    Clear-M3NarrowingOverrides
    Set-M3NarrowingBaseEnv -RunNote $Note -Epochs $Epochs
    foreach ($k in $EnvOverrides.Keys) {
        Set-Item -Path "Env:\$k" -Value $EnvOverrides[$k]
    }

    Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force

    Write-NarrowLog $Note "START" @{ epochs = $Epochs }
    $rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $Epochs --save-best --run-name $Note
    if ($rc -ne 0) {
        Write-NarrowLog ($Note + "_train") "FAIL" @{ exit = $rc }
        return $false
    }

    $gateRc = Invoke-PythonRc scripts/check_m3_narrowing_gate.py --run-note $Note -Quiet
    $st = if ($gateRc -eq 0) { "OK" } else { "WARN" }
    Write-NarrowLog ($Note + "_gate") $st @{ exit = $gateRc }
    Write-NarrowLog $Note "DONE" @{}
    return ($gateRc -eq 0)
}

Write-Host "[NEW] M3 narrowing ladder ($Epochs ep/leg, ~90m budget)" -ForegroundColor Cyan
Write-Host "[i] Log: $LogPath" -ForegroundColor Cyan

if (-not $SkipAudit) {
    Write-Host "[NEW] GT formulation audit (patient007)" -ForegroundColor Cyan
    $auditRc = Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --all-formulations
    Write-NarrowLog "gt_formulation_audit" $(if ($auditRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $auditRc }
}

# Simple -> complex (each leg resets from ramp1 ckpt)
$legs = @(
    @{
        note = "m3n_E0_data_only"
        env  = @{
            BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
            BIOCHEM_ADR_MASK_MODE         = "global"
        }
    },
    @{
        note = "m3n_E1_global_w1e4"
        env  = @{
            BIOCHEM_ADR_MASK_MODE        = "global"
            BIOCHEM_ADR_EXCLUDE_WALL     = "0"
            BIOCHEM_PASSIVE_ADR_WEIGHT   = "1e-4"
        }
    },
    @{
        note = "m3n_E2_match_nowall"
        env  = @{
            BIOCHEM_ADR_MASK_MODE      = "match_data_bio"
            BIOCHEM_ADR_EXCLUDE_WALL   = "1"
            BIOCHEM_PASSIVE_ADR_WEIGHT = "1e-3"
        }
    },
    @{
        note = "m3n_E3_log_residual"
        env  = @{
            BIOCHEM_ADR_RESIDUAL_MODE = "log"
        }
    },
    @{
        note = "m3n_E4_relative_nd"
        env  = @{
            BIOCHEM_ADR_RESIDUAL_MODE = "relative_nd"
        }
    },
    @{
        note = "m3n_E5_transport_only"
        env  = @{
            BIOCHEM_ADR_RESIDUAL_MODE = "transport_only"
        }
    },
    @{
        note = "m3n_E6_fi_only"
        env  = @{
            BIOCHEM_ADR_SPECIES_SCOPE = "fi"
        }
    },
    @{
        note = "m3n_E7_w1e4_match"
        env  = @{
            BIOCHEM_PASSIVE_ADR_WEIGHT = "1e-4"
        }
    },
    @{
        note = "m3n_E8_wall_backprop"
        env  = @{
            BIOCHEM_PASSIVE_WALL_BACKPROP = "1"
            BIOCHEM_PASSIVE_WALL_WEIGHT   = "1e-3"
        }
    },
    @{
        note = "m3n_E9_fast_transient"
        env  = @{
            BIOCHEM_ADR_FAST_TRANSIENT = "1"
        }
    }
)

$pass = 0
foreach ($leg in $legs) {
    if (Invoke-NarrowLeg -Note $leg.note -EnvOverrides $leg.env) { $pass++ }
}

Write-Host "[OK] M3 narrowing complete: $pass / $($legs.Count) legs passed gate" -ForegroundColor Green
Write-Host "[i] Summarize: python scripts/summarize_m3_narrowing.py" -ForegroundColor Cyan
