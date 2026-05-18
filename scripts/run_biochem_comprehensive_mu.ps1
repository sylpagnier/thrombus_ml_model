# EXPERIMENTAL / UNVALIDATED — comprehensive_mu preset (corona bundle + long schedules).
# Not shown to beat MU_LOG μ-path teacher-only runs. See BIOCHEM_TRAINING_PROGRESS.md.
#
# From repo root:
#   .\scripts\run_biochem_comprehensive_mu.ps1
#
# Logs: outputs/reports/training/biochem/<timestamp>/
# Check val mu_log_mae (all / wall / high-mu_gt), train L_MuLog_aux, anchor batch fraction.

param(
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_PRESET = "comprehensive_mu"
# Preset sets STOP_AFTER_TEACHER=0, teacher+corrector, mu best practices.
# Optional overrides (uncomment to tune):
# $env:BIOCHEM_TEACHER_SKIP_VAL = "0"
# $env:BIOCHEM_TBPTT_MAX_WINDOW = "8"
# $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "24"

Write-Host "Repo: $RepoRoot"
Write-Host "BIOCHEM_PRESET=$env:BIOCHEM_PRESET (comprehensive_mu diagnostic)"
python -m src.training.train_biochem_corrector @ExtraArgs
