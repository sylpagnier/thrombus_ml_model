# Computer B sweep (Windows/PowerShell): loss dynamics and rheology.
#
# Usage:
#   .\scripts\run_biochem_sweep_comp_b.ps1
#   .\scripts\run_biochem_sweep_comp_b.ps1 -Epochs 8 -DryRun
#   .\scripts\run_biochem_sweep_comp_b.ps1 -FullMatrix
#
param(
    [int] $Epochs = 8,
    [switch] $DryRun,
    [switch] $FullMatrix,
    [double] $EstimatedMinutesPerLeg = 75.0,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_TRAIN_MODE = "new"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_PRESET = ""
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
$env:BIOCHEM_EPOCHS = "$Epochs"

if ($FullMatrix) {
    $Legs = @()
    foreach ($scale in @(1.0, 10.0, 50.0)) {
        foreach ($cap in @(100.0, 500.0, 1000.0)) {
            foreach ($adj in @(0, 1)) {
                foreach ($tbptt in @(5, 10)) {
                    $Legs += @{
                        scale = $scale
                        cap = $cap
                        adj = $adj
                        tbptt = $tbptt
                    }
                }
            }
        }
    }
} else {
    # Budgeted ~10h profile on one machine (8 representative legs).
    $Legs = @(
        @{ scale = 1.0;  cap = 100.0;  adj = 0; tbptt = 5  },
        @{ scale = 1.0;  cap = 500.0;  adj = 1; tbptt = 10 },
        @{ scale = 10.0; cap = 100.0;  adj = 1; tbptt = 10 },
        @{ scale = 10.0; cap = 500.0;  adj = 0; tbptt = 5  },
        @{ scale = 10.0; cap = 1000.0; adj = 1; tbptt = 5  },
        @{ scale = 50.0; cap = 500.0;  adj = 1; tbptt = 5  },
        @{ scale = 50.0; cap = 1000.0; adj = 0; tbptt = 10 },
        @{ scale = 50.0; cap = 100.0;  adj = 0; tbptt = 10 }
    )
}

$estHours = [math]::Round(($Legs.Count * $EstimatedMinutesPerLeg) / 60.0, 1)
Write-Host "Computer B sweep started: MuScale x RheologyCap x Adjoint x TBPTT" -ForegroundColor Cyan
Write-Host "Profile: $(if ($FullMatrix) {'full-matrix'} else {'budgeted-10h'}) | legs=$($Legs.Count) | epochs=$Epochs | est_hours~$estHours" -ForegroundColor DarkGray

foreach ($leg in $Legs) {
    $scale = [double]$leg.scale
    $cap = [double]$leg.cap
    $adj = [int]$leg.adj
    $tbptt = [int]$leg.tbptt
    $runName = "compB_M${scale}_C${cap}_A${adj}_T${tbptt}"
    Write-Host ""
    Write-Host "▶ $runName" -ForegroundColor Yellow

    $env:BIOCHEM_MU_LOSS_SCALE = "$scale"
    $env:BIOCHEM_RHEOLOGY_CAP = "$cap"
    $env:BIOCHEM_ODEINT_USE_ADJOINT = "$adj"
    $env:BIOCHEM_TBPTT_WINDOW = "$tbptt"
    $env:BIOCHEM_RUN_NOTE = $runName

    $cmd = @("-m", "src.training.train_biochem_corrector", "--new", "--run-name", $runName, "--epochs", "$Epochs", "--save-best")
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        continue
    }

    python @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $runName (exit=$LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "Computer B sweep complete." -ForegroundColor Green
