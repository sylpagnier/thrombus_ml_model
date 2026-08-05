param(
    [int]    $Epochs         = 14,
    [int]    $EarlyStop      = 5,
    [double] $Lr             = 3e-5,
    [string] $HoldoutAnchor  = "patient020",
    [string] $RunRoot        = "outputs/biochem/eda/wall_gen_prec_loao",
    [string] $InitCkpt       = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth",
    [string] $Leg            = "WG_prec_loao",
    [double] $MassSprayAbort = 2.5,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipViz,
    [switch] $NoInit,
    [switch] $FreezeBackbone
)

# Clot-rich LOAO with tight mass/FP (best small-cohort recipe). Holdout patient020.
# Abort if mass sprays. Prefer --InitCkpt pointing at best of prec_iter/mirror/ft.
#   .\scripts\go_wg_prec_loao.ps1 -Epochs 14 -EarlyStop 5 -Fresh

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
$ArmDir = Join-Path $OutDir $Leg
$ArmCkpt = Join-Path $ArmDir "best.pth"
$ArmHold = Join-Path $ArmDir "eval_holdout_cold.json"
$ArmLog = Join-Path $ArmDir "train_log.jsonl"
New-Item -ItemType Directory -Force -Path $ArmDir | Out-Null

$trainCsv = (python -c @"
from src.biochem_gnn.mat_growth_simple import wall_gen_clot_rich_train_anchors
print(','.join(wall_gen_clot_rich_train_anchors(holdout='$HoldoutAnchor')))
"@).Trim()
if (-not $trainCsv -or $LASTEXITCODE -ne 0) {
    Write-Host "[ERR] failed to resolve clot-rich train anchors (holdout=$HoldoutAnchor)" -ForegroundColor Red
    exit 1
}
$trainList = @($trainCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })

$forbidden = @("patient002", "patient023", $HoldoutAnchor)
foreach ($bad in $forbidden) {
    if ($trainList -contains $bad) {
        Write-Host "[ERR] train list must not include $bad" -ForegroundColor Red
        Write-Host "[i] train=$trainCsv" -ForegroundColor DarkGray
        exit 1
    }
}
if ($trainList.Count -lt 8) {
    Write-Host "[ERR] expected >=8 clot-rich train vessels, got $($trainList.Count)" -ForegroundColor Red
    exit 1
}

$InitPath = Join-Path $RepoRoot $InitCkpt
if (-not $NoInit -and -not (Test-Path $InitPath)) {
    Write-Host "[ERR] Missing warm-start ckpt: $InitCkpt" -ForegroundColor Red
    exit 1
}

$featfixTrainLeak = @("patient005", "patient006", "patient010", "patient023", "patient002")
if (-not $NoInit -and ($featfixTrainLeak -contains $HoldoutAnchor)) {
    Write-Host "[ERR] holdout=$HoldoutAnchor was in featfix train set; warm-start would leak." -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] wg_prec_loao ($Leg): $Epochs ep / ES $EarlyStop / lr=$Lr" -ForegroundColor Cyan
Write-Host "[i] goal=clot-rich LOAO with tight mass/FP; raise cold $HoldoutAnchor" -ForegroundColor DarkGray
Write-Host "[i] train($($trainList.Count))=$trainCsv" -ForegroundColor DarkGray
Write-Host "[i] val/holdout=$HoldoutAnchor" -ForegroundColor DarkGray
Write-Host "[i] abort if mass > $MassSprayAbort or all epochs mass-rejected" -ForegroundColor DarkGray
if ($NoInit) {
    Write-Host "[i] init=random" -ForegroundColor DarkGray
} else {
    Write-Host "[i] init=$InitCkpt" -ForegroundColor DarkGray
}

if ($Fresh) {
    Remove-Item -Force $ArmCkpt, $ArmHold, $ArmLog -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $ArmDir "best.json") -ErrorAction SilentlyContinue
}

if ((Test-Path $ArmHold) -and -not $Fresh -and -not $EvalOnly) {
    Write-Host "[skip] $Leg already completed; pass -Fresh to rerun" -ForegroundColor DarkGray
    exit 0
}

if (-not $EvalOnly) {
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--recipe", "mat_growth_simple",
        "--leg", $Leg,
        "--out", $ArmCkpt,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--lr", "$Lr",
        "--anchors", $trainCsv,
        "--val-anchor", $HoldoutAnchor,
        "--exclude-val-from-train",
        "--drop-xy",
        "--deploy-freq", "1"
    )
    if ($NoInit) {
        $trainArgs += "--no-init"
    } else {
        $trainArgs += @("--init", $InitCkpt, "--init-mode", "full")
    }
    if ($FreezeBackbone -or $Leg -match "freeze") {
        $trainArgs += "--freeze-backbone"
        Write-Host "[i] freeze_backbone=1 (heads-only FT)" -ForegroundColor DarkGray
    }

    # Run train in a nested process so we can abort on spray without killing the launcher shell first.
    $trainLog = Join-Path $ArmDir "launcher_train.log"
    $trainErr = Join-Path $ArmDir "launcher_train.err.log"
    $pyExe = (Get-Command python).Source
    $trainProc = Start-Process -FilePath $pyExe -ArgumentList (@("-u") + $trainArgs) `
        -WorkingDirectory $RepoRoot -PassThru -NoNewWindow `
        -RedirectStandardOutput $trainLog -RedirectStandardError $trainErr

    Write-Host ("[i] train pid={0}" -f $trainProc.Id) -ForegroundColor DarkGray
    $sprayed = $false
    while (-not $trainProc.HasExited) {
        Start-Sleep -Seconds 45
        if (Test-Path $ArmLog) {
            $last = Get-Content $ArmLog | Select-Object -Last 1 | ConvertFrom-Json
            $mass = [double]$last.deploy_clot_mass_ratio
            $ep = [int]$last.epoch
            $mode = [string]$last.select_mode
            Write-Host ("[i] loao ep={0} score={1:N4} mass={2:N2} mode={3}" -f `
                $ep, [double]$last.deploy_clot_score, $mass, $mode) -ForegroundColor DarkGray
            if ($ep -ge 1 -and ($mass -gt $MassSprayAbort)) {
                Write-Host "[WARN] loao spray mass=$mass at ep=$ep; aborting train" -ForegroundColor Yellow
                $sprayed = $true
                Stop-Process -Id $trainProc.Id -Force -ErrorAction SilentlyContinue
                Start-Sleep 2
                Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -eq "python.exe" -and $_.CommandLine -match "WG_prec_loao"
                } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                break
            }
            if ($ep -ge 2 -and ($mode -match "mass_reject")) {
                $lines = Get-Content $ArmLog
                if ($lines.Count -ge 2) {
                    $prev = $lines[-2] | ConvertFrom-Json
                    $prevMode = [string]$prev.select_mode
                    if ($prevMode -match "mass_reject") {
                        Write-Host "[WARN] loao mass_reject streak; aborting train" -ForegroundColor Yellow
                        $sprayed = $true
                        Stop-Process -Id $trainProc.Id -Force -ErrorAction SilentlyContinue
                        Start-Sleep 2
                        Get-CimInstance Win32_Process | Where-Object {
                            $_.Name -eq "python.exe" -and $_.CommandLine -match "WG_prec_loao"
                        } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                        break
                    }
                }
            }
        }
    }
    if (-not $sprayed) {
        $trainProc.WaitForExit()
        if ($trainProc.ExitCode -ne 0) {
            Write-Host ("[ERR] train exited {0}" -f $trainProc.ExitCode) -ForegroundColor Red
            Get-Content $trainErr -ErrorAction SilentlyContinue | Select-Object -Last 30
            exit $trainProc.ExitCode
        }
    }

    if (-not (Test-Path $ArmCkpt)) {
        Write-Host "[ERR] $Leg failed to produce checkpoint (spray / all mass-rejected?)" -ForegroundColor Red
        exit 1
    }
    if ($sprayed) {
        Write-Host "[WARN] aborted for spray; cold-evaling best.pth if present" -ForegroundColor Yellow
    }
} elseif (-not (Test-Path $ArmCkpt)) {
    Write-Host "[ERR] $Leg missing ckpt for -EvalOnly" -ForegroundColor Red
    exit 1
}

$null = Invoke-PythonRcCheck -Label "$Leg cold eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $ArmCkpt,
    "--no-baseline",
    "--anchors", $HoldoutAnchor,
    "--out", $ArmHold
)

if (-not $SkipViz) {
    $vizDir = Join-Path $RepoRoot "outputs/biochem/viz/mat_growth"
    New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
    $vizOut = Join-Path $vizDir "clot_ladder_prec_loao_$HoldoutAnchor.png"
    $null = Invoke-PythonRcCheck -Label "$Leg ladder" -PyArgs @(
        "scripts/viz_mat_growth_clot_ladder.py",
        "--anchor", $HoldoutAnchor,
        "--ckpt", $ArmCkpt,
        "--arm-label", "prec_loao",
        "--leg", $Leg,
        "--flow", "kinematics",
        "--out", $vizOut
    )
}

Write-Host "[OK] $Leg complete" -ForegroundColor Green
Write-Host "[i] ckpt=$ArmCkpt" -ForegroundColor DarkGray
Write-Host "[i] eval=$ArmHold" -ForegroundColor DarkGray
