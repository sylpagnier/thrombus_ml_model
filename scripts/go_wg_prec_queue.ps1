param(
    [int]    $BudgetMinutes  = 300,
    [int]    $PrecEarlyStop  = 8,
    [int]    $MirrorEpochs   = 22,
    [int]    $MirrorES       = 6,
    [int]    $SitesEpochs    = 16,
    [int]    $SitesES        = 5,
    [double] $Lr             = 1e-4,
    [string] $HoldoutAnchor  = "patient020",
    [double] $Featfix03Score = 0.329,
    [double] $MassSprayAbort = 2.5,
    [switch] $SkipMirror,
    [switch] $SkipSites,
    [switch] $ForceSites,
    [switch] $SkipViz
)

# Autonomous 5h wall-gen queue after prec_iter:
#   1) finish/eval WG_prec_iter (or stop if flat/spray)
#   2) WG_prec_mirror (more shapes, same small cohort + prec loss)
#   3) WG_prec_sites if mass stayed healthy and score not regressing
#   .\scripts\go_wg_prec_queue.ps1 -BudgetMinutes 300

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$started = Get-Date
$deadline = $started.AddMinutes($BudgetMinutes)

function Remaining-Minutes {
    return [math]::Round(($deadline - (Get-Date)).TotalMinutes, 1)
}

function Read-ColdScore([string]$evalPath) {
    if (-not (Test-Path $evalPath)) { return $null }
    $py = @"
import json
p = r'$evalPath'
d = json.load(open(p, encoding='utf-8'))
per = ((d.get('simple') or {}).get('per_anchor') or {}).get('$HoldoutAnchor') or {}
score = float(per.get('deploy_clot_score') or 0)
f1 = float(per.get('deploy_clot_f1') or 0)
mass = float(per.get('deploy_clot_mass_ratio') or 0)
print('%.6f,%.6f,%.6f' % (score, f1, mass))
"@
    $out = (python -c $py).Trim()
    if (-not $out) { return $null }
    $parts = $out.Split(",")
    return @{ score = [double]$parts[0]; f1 = [double]$parts[1]; mass = [double]$parts[2] }
}

function Read-TrainBest([string]$logPath) {
    if (-not (Test-Path $logPath)) { return $null }
    $best = $null
    Get-Content $logPath | ForEach-Object {
        $j = $_ | ConvertFrom-Json
        $sel = [double]($j.select_score)
        $mass = [double]($j.deploy_clot_mass_ratio)
        if ($null -eq $best -or $sel -gt [double]$best.select_score) {
            $best = $j
        }
        $script:lastEp = $j
    }
    return @{ best = $best; last = $script:lastEp }
}

function Mass-Healthy([double]$mass) {
    return ($mass -gt 0.0 -and $mass -le $MassSprayAbort)
}

Write-Host "[NEW] wg_prec_queue: budget=${BudgetMinutes}m deadline=$($deadline.ToString('HH:mm:ss'))" -ForegroundColor Cyan
Write-Host "[i] gate=deploy-faithful cold $HoldoutAnchor; no GT UV; featfix03 bar~$Featfix03Score" -ForegroundColor DarkGray

# ---------- Stage 0: wait / finish prec_iter ----------
$precDir = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter"
$precCkpt = Join-Path $precDir "best.pth"
$precEval = Join-Path $precDir "eval_holdout_cold.json"
$precLog = Join-Path $precDir "train_log.jsonl"

Write-Host "[i] stage0: wait for WG_prec_iter (or eval existing best)" -ForegroundColor Cyan
$flatStreak = 0
$bestSelSeen = -1e9
while ((Get-Date) -lt $deadline) {
    $alive = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'train_species_pushforward_continuous.*WG_prec_iter'
    }
    if (Test-Path $precLog) {
        $stats = Read-TrainBest $precLog
        if ($null -ne $stats.best) {
            $sel = [double]$stats.best.select_score
            $ep = [int]$stats.best.epoch
            $lastEp = [int]$stats.last.epoch
            $lastMass = [double]$stats.last.deploy_clot_mass_ratio
            $lastScore = [double]$stats.last.deploy_clot_score
            $flatStreak = [math]::Max(0, $lastEp - $ep)
            if ($sel -gt $bestSelSeen) { $bestSelSeen = $sel }
            Write-Host ("[i] prec_iter last_ep={0} score={1:N4} mass={2:N2} best_sel={3:N4}@ep{4} stale={5} rem={6}m" -f `
                $lastEp, $lastScore, $lastMass, $sel, $ep, $flatStreak, (Remaining-Minutes)) -ForegroundColor DarkGray

            # Stop early if no longer useful: long flat + mass creeping, or spray.
            if (-not (Mass-Healthy $lastMass)) {
                Write-Host "[WARN] prec_iter mass spray; stopping train to preserve budget" -ForegroundColor Yellow
                Get-CimInstance Win32_Process | Where-Object {
                    $_.CommandLine -match 'train_species_pushforward_continuous.*WG_prec_iter'
                } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                # Orphan python children after killing pwsh wrappers.
                Start-Sleep 2
                Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'WG_prec_iter'
                } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                break
            }
            if ($flatStreak -ge 5 -and $lastEp -ge 12 -and $sel -lt ($Featfix03Score + 0.05)) {
                Write-Host "[i] prec_iter flat (stale>=5); stop early as useful-test ended" -ForegroundColor Yellow
                Get-CimInstance Win32_Process | Where-Object {
                    $_.CommandLine -match 'train_species_pushforward_continuous.*WG_prec_iter'
                } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                Start-Sleep 2
                Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'WG_prec_iter'
                } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                break
            }
        }
    }
    if (-not $alive) {
        Write-Host "[OK] prec_iter train not running" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 45
}

if ((Test-Path $precCkpt) -and -not (Test-Path $precEval)) {
    Write-Host "[i] cold-eval prec_iter best.pth" -ForegroundColor Cyan
    if ($SkipViz) {
        & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") -EvalOnly -SkipViz
    } else {
        & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") -EvalOnly
    }
}

$precCold = Read-ColdScore $precEval
if ($null -eq $precCold -and (Test-Path $precCkpt)) {
    Write-Host "[WARN] prec_iter eval missing; running cold eval now" -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") -EvalOnly -SkipViz
    $precCold = Read-ColdScore $precEval
}
if ($null -ne $precCold) {
    Write-Host ("[OK] prec_iter cold p020 score={0:N4} f1={1:N4} mass={2:N2}" -f $precCold.score, $precCold.f1, $precCold.mass) -ForegroundColor Green
} else {
    Write-Host "[WARN] no prec_iter cold metrics; continuing with featfix03 bar" -ForegroundColor Yellow
    $precCold = @{ score = $Featfix03Score; f1 = 0.0; mass = 1.2 }
}

$barScore = [math]::Max($Featfix03Score, [double]$precCold.score)
$mirrorOk = $false
$sitesOk = $false

# ---------- Stage 1: Mirror-Y ----------
if (-not $SkipMirror -and (Remaining-Minutes) -gt 40) {
    Write-Host "[NEW] stage1: WG_prec_mirror (rem=$(Remaining-Minutes)m)" -ForegroundColor Cyan
    if ($SkipViz) {
        & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") `
            -Epochs $MirrorEpochs -EarlyStop $MirrorES -Lr $Lr `
            -HoldoutAnchor $HoldoutAnchor `
            -RunRoot "outputs/biochem/eda/wall_gen_prec_mirror" `
            -Leg "WG_prec_mirror" -Fresh -SkipViz
    } else {
        & (Join-Path $PSScriptRoot "go_wg_prec_iter.ps1") `
            -Epochs $MirrorEpochs -EarlyStop $MirrorES -Lr $Lr `
            -HoldoutAnchor $HoldoutAnchor `
            -RunRoot "outputs/biochem/eda/wall_gen_prec_mirror" `
            -Leg "WG_prec_mirror" -Fresh
    }
    $mirrorEval = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_mirror/WG_prec_mirror/eval_holdout_cold.json"
    $mirrorCold = Read-ColdScore $mirrorEval
    if ($null -ne $mirrorCold) {
        Write-Host ("[OK] prec_mirror cold score={0:N4} f1={1:N4} mass={2:N2}" -f $mirrorCold.score, $mirrorCold.f1, $mirrorCold.mass) -ForegroundColor Green
        $mirrorOk = (Mass-Healthy $mirrorCold.mass) -and ($mirrorCold.score -ge ($Featfix03Score - 0.02))
    }
} else {
    Write-Host "[skip] mirror (SkipMirror or low remaining=$(Remaining-Minutes)m)" -ForegroundColor DarkGray
}

# ---------- Stage 2: controlled N+ with prec loss ----------
$runSites = $ForceSites -or (
    -not $SkipSites -and
    (Remaining-Minutes) -gt 50 -and
    (Mass-Healthy ([double]$precCold.mass)) -and
    (
        $ForceSites -or $mirrorOk -or ([double]$precCold.score -ge $Featfix03Score)
    )
)
if ($runSites) {
    Write-Host "[NEW] stage2: WG_prec_sites (rem=$(Remaining-Minutes)m)" -ForegroundColor Cyan
    if ($SkipViz) {
        & (Join-Path $PSScriptRoot "go_wg_prec_sites.ps1") `
            -Epochs $SitesEpochs -EarlyStop $SitesES -Lr $Lr `
            -HoldoutAnchor $HoldoutAnchor -Fresh -SkipViz
    } else {
        & (Join-Path $PSScriptRoot "go_wg_prec_sites.ps1") `
            -Epochs $SitesEpochs -EarlyStop $SitesES -Lr $Lr `
            -HoldoutAnchor $HoldoutAnchor -Fresh
    }
    $sitesEval = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_sites/WG_prec_sites/eval_holdout_cold.json"
    $sitesCold = Read-ColdScore $sitesEval
    if ($null -ne $sitesCold) {
        Write-Host ("[OK] prec_sites cold score={0:N4} f1={1:N4} mass={2:N2}" -f $sitesCold.score, $sitesCold.f1, $sitesCold.mass) -ForegroundColor Green
        $sitesOk = Mass-Healthy $sitesCold.mass
    }
} else {
    Write-Host "[skip] sites (mass/score gate or low budget rem=$(Remaining-Minutes)m)" -ForegroundColor DarkGray
}

# ---------- Summary ----------
Write-Host ""
Write-Host "====== prec_queue summary ======" -ForegroundColor Cyan
Write-Host ("featfix03 bar score ~ {0:N3}" -f $Featfix03Score)
if ($null -ne $precCold) {
    Write-Host ("prec_iter     score={0:N4} f1={1:N4} mass={2:N2}" -f $precCold.score, $precCold.f1, $precCold.mass)
}
$mEval = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_mirror/WG_prec_mirror/eval_holdout_cold.json"
$mCold = Read-ColdScore $mEval
if ($null -ne $mCold) {
    Write-Host ("prec_mirror   score={0:N4} f1={1:N4} mass={2:N2} gate_ok={3}" -f $mCold.score, $mCold.f1, $mCold.mass, $mirrorOk)
}
$sEval = Join-Path $RepoRoot "outputs/biochem/eda/wall_gen_prec_sites/WG_prec_sites/eval_holdout_cold.json"
$sCold = Read-ColdScore $sEval
if ($null -ne $sCold) {
    Write-Host ("prec_sites    score={0:N4} f1={1:N4} mass={2:N2} gate_ok={3}" -f $sCold.score, $sCold.f1, $sCold.mass, $sitesOk)
}
Write-Host ("elapsed_min={0:N1} remaining_min={1:N1}" -f ((Get-Date) - $started).TotalMinutes, (Remaining-Minutes))
Write-Host "[OK] wg_prec_queue done" -ForegroundColor Green
