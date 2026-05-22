# Computer A sweep (Windows/PowerShell): spatial architecture and encoding.
# Mirrors the user's Bash matrix while using this repo's env conventions.
#
# Usage:
#   .\scripts\run_biochem_sweep_comp_a.ps1
#   .\scripts\run_biochem_sweep_comp_a.ps1 -Epochs 8 -DryRun
#   .\scripts\run_biochem_sweep_comp_a.ps1 -FullMatrix
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
    foreach ($layers in @(4, 6)) {
        foreach ($siren in @(0, 1)) {
            foreach ($bands in @(4, 8, 16)) {
                foreach ($rank in @(0, 8, 16)) {
                    if ($siren -eq 1 -and $bands -ne 4) { continue }
                    $Legs += @{
                        layers = $layers
                        siren = $siren
                        bands = $bands
                        rank = $rank
                    }
                }
            }
        }
    }
} else {
    # Budgeted ~10h profile on one machine (typically 8 legs x ~75m/leg).
    $Legs = @(
        @{ layers = 4; siren = 0; bands = 8;  rank = 0  },
        @{ layers = 4; siren = 0; bands = 8;  rank = 8  },
        @{ layers = 4; siren = 0; bands = 16; rank = 8  },
        @{ layers = 4; siren = 1; bands = 4;  rank = 8  },
        @{ layers = 6; siren = 0; bands = 8;  rank = 0  },
        @{ layers = 6; siren = 0; bands = 8;  rank = 8  },
        @{ layers = 6; siren = 0; bands = 16; rank = 8  },
        @{ layers = 6; siren = 1; bands = 4;  rank = 8  }
    )
}

$estHours = [math]::Round(($Legs.Count * $EstimatedMinutesPerLeg) / 60.0, 1)
Write-Host "Computer A sweep started: Layers x SIREN x Fourier x LoRA" -ForegroundColor Cyan
Write-Host "Profile: $(if ($FullMatrix) {'full-matrix'} else {'budgeted-10h'}) | legs=$($Legs.Count) | epochs=$Epochs | est_hours~$estHours" -ForegroundColor DarkGray

foreach ($leg in $Legs) {
    $layers = [int]$leg.layers
    $siren = [int]$leg.siren
    $bands = [int]$leg.bands
    $rank = [int]$leg.rank

    $runName = "compA_L${layers}_S${siren}_B${bands}_R${rank}"
    Write-Host ""
    Write-Host "▶ $runName" -ForegroundColor Yellow

    $env:BIOCHEM_GNODE_LAYERS = "$layers"
    $env:BIOCHEM_USE_SIREN = "$siren"
    $env:BIOCHEM_FOURIER_BANDS = "$bands"
    $env:BIOCHEM_LORA_RANK = "$rank"
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
Write-Host "Computer A sweep complete." -ForegroundColor Green
