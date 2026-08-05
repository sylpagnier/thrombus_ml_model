param(
    [int]    $BudgetMinutes  = 120,
    [int]    $MidEpochs      = 14,
    [int]    $MidES          = 5,
    [int]    $FtEpochs       = 16,
    [int]    $FtES           = 5,
    [double] $Lr             = 5e-5,
    [string] $HoldoutAnchor  = "patient020",
    [double] $MassSprayAbort = 2.5,
    [switch] $SkipMid,
    [switch] $SkipFt,
    [switch] $SkipViz
)

# Remaining-budget pivot after full N+ (WG_prec_sites) sprayed:
#   1) WG_prec_mid: 6-vessel controlled expand, init prec_iter
#   2) if mid sprays or finishes: WG_prec_ft tight small-cohort FT
#   .\scripts\go_wg_prec_pivot.ps1 -BudgetMinutes 120

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$started = Get-Date
$deadline = $started.AddMinutes($BudgetMinutes)
$midTrain = "patient005,patient006,patient010,patient001,patient007,patient012"
$smallTrain = "patient005,patient006,patient010"
$precInit = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth"

function Remaining-Minutes {
    return [math]::Round(($deadline - (Get-Date)).TotalMinutes, 1)
}

function Read-ColdScore([string]$evalPath) {
    if (-not (Test-Path $evalPath)) { return $null }
    $py = @"
import json
d = json.load(open(r'$evalPath', encoding='utf-8'))
per = ((d.get('simple') or {}).get('per_anchor') or {}).get('$HoldoutAnchor') or {}
print('%.6f,%.6f,%.6f' % (
    float(per.get('deploy_clot_score') or 0),
    float(per.get('deploy_clot_f1') or 0),
    float(per.get('deploy_clot_mass_ratio') or 0),
))
"@
    $out = (python -c $py).Trim()
    if (-not $out) { return $null }
    $p = $out.Split(",")
    return @{ score = [double]$p[0]; f1 = [double]$p[1]; mass = [double]$p[2] }
}

function Mass-Ok([double]$mass) {
    return ($mass -gt 0.0 -and $mass -le $MassSprayAbort)
}

function Watch-And-Abort-Spray([string]$logPath, [string]$legTag) {
    # Poll while train alive; abort if last epoch mass sprays.
    while ($true) {
        $alive = Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match ("train_species_pushforward_continuous.*" + [regex]::Escape($legTag))
        }
        if (-not $alive) { return $false }
        if ((Test-Path $logPath) -and (Remaining-Minutes) -lt 5) {
            Write-Host "[WARN] budget nearly gone; stop $legTag" -ForegroundColor Yellow
            $alive | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Start-Sleep 2
            Get-CimInstance Win32_Process | Where-Object {
                $_.Name -eq 'python.exe' -and $_.CommandLine -match $legTag
            } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            return $true
        }
        if (Test-Path $logPath) {
            $last = Get-Content $logPath | Select-Object -Last 1 | ConvertFrom-Json
            $mass = [double]$last.deploy_clot_mass_ratio
            $ep = [int]$last.epoch
            Write-Host ("[i] {0} ep={1} score={2:N4} mass={3:N2} rem={4}m" -f $legTag, $ep, [double]$last.deploy_clot_score, $mass, (Remaining-Minutes)) -ForegroundColor DarkGray
            if ($ep -ge 2 -and -not (Mass-Ok $mass)) {
                Write-Host "[WARN] $legTag spray mass=$mass; abort" -ForegroundColor Yellow
                $alive | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                Start-Sleep 2
                Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -eq 'python.exe' -and $_.CommandLine -match $legTag
                } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                return $true
            }
        }
        if ((Get-Date) -ge $deadline) {
            Write-Host "[WARN] deadline; stop $legTag" -ForegroundColor Yellow
            $alive | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            return $true
        }
        Start-Sleep 60
    }
}

Write-Host "[NEW] wg_prec_pivot: budget=${BudgetMinutes}m deadline=$($deadline.ToString('HH:mm:ss'))" -ForegroundColor Cyan
Write-Host "[i] full N+ sprayed; try mid cohort then tight FT" -ForegroundColor DarkGray

if (-not (Test-Path (Join-Path $RepoRoot $precInit))) {
    Write-Host "[ERR] missing prec_iter ckpt: $precInit" -ForegroundColor Red
    exit 1
}

$midSprayed = $false

# ---- Mid cohort ----
if (-not $SkipMid -and (Remaining-Minutes) -gt 40) {
    Write-Host "[NEW] stage mid: WG_prec_mid train=$midTrain" -ForegroundColor Cyan
    $job = Start-Job -ScriptBlock {
        param($RepoRoot, $MidEpochs, $MidES, $Lr, $HoldoutAnchor, $midTrain, $precInit, $SkipViz)
        Set-Location $RepoRoot
        $args = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $RepoRoot "scripts\go_wg_prec_iter.ps1"),
            "-Epochs", "$MidEpochs", "-EarlyStop", "$MidES", "-Lr", "$Lr",
            "-TrainAnchors", $midTrain, "-HoldoutAnchor", $HoldoutAnchor,
            "-RunRoot", "outputs/biochem/eda/wall_gen_prec_mid",
            "-InitCkpt", $precInit, "-Leg", "WG_prec_mid", "-Fresh"
        )
        if ($SkipViz) { $args += "-SkipViz" }
        & powershell.exe @args
        return $LASTEXITCODE
    } -ArgumentList $RepoRoot, $MidEpochs, $MidES, $Lr, $HoldoutAnchor, $midTrain, $precInit, [bool]$SkipViz

    Start-Sleep 15
    $midLog = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_mid/WG_prec_mid/train_log.jsonl"
    $midSprayed = Watch-And-Abort-Spray $midLog "WG_prec_mid"
    Wait-Job $job -Timeout 10 | Out-Null
    if ($job.State -eq "Running") {
        # train already killed; wait for launcher to exit
        Wait-Job $job -Timeout 120 | Out-Null
        if ($job.State -eq "Running") { Stop-Job $job -Force; Remove-Job $job -Force }
    } else {
        Receive-Job $job | Out-Null
        Remove-Job $job -Force -ErrorAction SilentlyContinue
    }

    $midEval = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_mid/WG_prec_mid/eval_holdout_cold.json"
    $midCkpt = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_mid/WG_prec_mid/best.pth"
    if ((Test-Path $midCkpt) -and -not (Test-Path $midEval) -and -not $midSprayed) {
        if ($SkipViz) {
            & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") -EvalOnly -SkipViz `
                -RunRoot "outputs/biochem/eda/wall_gen_prec_mid" -Leg "WG_prec_mid" `
                -HoldoutAnchor $HoldoutAnchor -InitCkpt $precInit
        } else {
            & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") -EvalOnly `
                -RunRoot "outputs/biochem/eda/wall_gen_prec_mid" -Leg "WG_prec_mid" `
                -HoldoutAnchor $HoldoutAnchor -InitCkpt $precInit
        }
    }
    $midCold = Read-ColdScore $midEval
    if ($null -ne $midCold) {
        Write-Host ("[OK] prec_mid cold score={0:N4} f1={1:N4} mass={2:N2}" -f $midCold.score, $midCold.f1, $midCold.mass)
        if (-not (Mass-Ok $midCold.mass)) { $midSprayed = $true }
    }
} else {
    Write-Host "[skip] mid" -ForegroundColor DarkGray
}

# ---- Tight FT ----
$needFt = (-not $SkipFt) -and ((Remaining-Minutes) -gt 35) -and ($midSprayed -or $SkipMid -or $true)
# Prefer FT if mid sprayed OR always run if budget remains (localization ceiling push).
if ((-not $SkipFt) -and (Remaining-Minutes) -gt 35) {
    Write-Host "[NEW] stage ft: WG_prec_ft (tight mass/FP, small cohort)" -ForegroundColor Cyan
    if ($SkipViz) {
        & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") `
            -Epochs $FtEpochs -EarlyStop $FtES -Lr $Lr `
            -TrainAnchors $smallTrain -HoldoutAnchor $HoldoutAnchor `
            -RunRoot "outputs/biochem/eda/wall_gen_prec_ft" `
            -InitCkpt $precInit -Leg "WG_prec_ft" -Fresh -SkipViz
    } else {
        & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") `
            -Epochs $FtEpochs -EarlyStop $FtES -Lr $Lr `
            -TrainAnchors $smallTrain -HoldoutAnchor $HoldoutAnchor `
            -RunRoot "outputs/biochem/eda/wall_gen_prec_ft" `
            -InitCkpt $precInit -Leg "WG_prec_ft" -Fresh
    }
    $ftEval = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_ft/WG_prec_ft/eval_holdout_cold.json"
    $ftCold = Read-ColdScore $ftEval
    if ($null -ne $ftCold) {
        Write-Host ("[OK] prec_ft cold score={0:N4} f1={1:N4} mass={2:N2}" -f $ftCold.score, $ftCold.f1, $ftCold.mass)
    }
} else {
    Write-Host "[skip] ft (budget/flags)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "====== prec_pivot summary ======" -ForegroundColor Cyan
foreach ($pair in @(
    @("prec_iter", "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/eval_holdout_cold.json"),
    @("prec_mirror", "outputs/biochem/eda/wall_gen_prec_mirror/WG_prec_mirror/eval_holdout_cold.json"),
    @("prec_mid", "outputs/biochem/eda/wall_gen_prec_mid/WG_prec_mid/eval_holdout_cold.json"),
    @("prec_ft", "outputs/biochem/eda/wall_gen_prec_ft/WG_prec_ft/eval_holdout_cold.json")
)) {
    $c = Read-ColdScore (Join-Path $RepoRoot $pair[1])
    if ($null -ne $c) {
        Write-Host ("{0,-12} score={1:N4} f1={2:N4} mass={3:N2}" -f $pair[0], $c.score, $c.f1, $c.mass)
    }
}
Write-Host ("elapsed_min={0:N1} rem={1:N1}" -f ((Get-Date) - $started).TotalMinutes, (Remaining-Minutes))
Write-Host "[OK] wg_prec_pivot done" -ForegroundColor Green
