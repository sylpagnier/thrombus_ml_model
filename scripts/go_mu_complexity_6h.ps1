# ~6h overnight: comprehensive GNODE stacks (pred GINO-DEQ kine + GNODE biochem + teacher -> corrector/synth).
# Few legs (not 12 ablations). Each leg resets to the same init ckpt. Rank by val mu_log_mae.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mu_complexity_6h.ps1
#   powershell ... -InitCkpt outputs\biochem\gnode10_sweep\gnode12_lane_a_promoted\biochem_teacher_best_high_mu.pth
#   powershell ... -DryRun
#   powershell ... -Legs smoke,FULL_step2,FULL_step2p5,FULL_step3,FULL_overnight
#   (comma-separated OK) or: -Legs @('FULL_step2','FULL_step2p5')
#
# Morning:
#   python scripts/summarize_mu_complexity_6h.py
#   Get-Content outputs\biochem\sweep_mu_complexity_6h\manifest.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Sort-Object val_mu_log_mae | Format-Table leg_id, tier, val_mu_log_mae, pseudo_w, corrector_val_n, run_note

param(
    [string] $InitCkpt = "",
    [int] $TeacherEpochs = 0,
    [int] $CorrectorEpochs = 0,
    [string[]] $Legs = @(),
    [switch] $SkipPytest,
    [switch] $SkipGate,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_mu_complexity_sweep_env.ps1")

$SweepDir = Join-Path $RepoRoot "outputs\biochem\sweep_mu_complexity_6h"
$LogPath = Join-Path $SweepDir "sweep_log.jsonl"
$ManifestPath = Join-Path $SweepDir "manifest.jsonl"
New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null

# ~6h @ ~35-95 min/leg (RTX 500 class): 12+18 teacher+corrector + pseudo mix
$LegCatalog = [ordered]@{
    smoke = @{
        Tier = "0-smoke-full"
        TeacherEpochs = 2
        CorrectorEpochs = 2
        Minutes = 8
        Hypothesis = "Plumbing: pred kine + step-2 bridge + 2+2ep teacher+corrector + pseudo"
    }
    FULL_step2 = @{
        Tier = "2-full"
        TeacherEpochs = 12
        CorrectorEpochs = 18
        Minutes = 95
        Hypothesis = "Step-2 bridge data-only + pseudo bank (Phase II.0 style, deploy kine)"
    }
    FULL_step2p5 = @{
        Tier = "2.5-full"
        TeacherEpochs = 12
        CorrectorEpochs = 18
        Minutes = 95
        Hypothesis = "Step-2 + COMSOL temporal loss (L_PhysTemp) + corrector/synth"
    }
    FULL_step3 = @{
        Tier = "3-full"
        TeacherEpochs = 10
        CorrectorEpochs = 14
        Minutes = 85
        Hypothesis = "Kendall multitask backward (LOSS_DATA_ONLY=0) + corrector/synth; may regress"
    }
    FULL_overnight = @{
        Tier = "2-overnight-full"
        TeacherEpochs = 14
        CorrectorEpochs = 20
        Minutes = 110
        Hypothesis = "overnight_step2 preset slice + pseudo (long TBPTT/phys-temp flags)"
    }
}

$DefaultOrder = @(
    "smoke",
    "FULL_step2",
    "FULL_step2p5",
    "FULL_step3"
)

# PowerShell often passes "-Legs a,b,c" as one string; split for convenience.
if ($Legs.Count -eq 1 -and ($Legs[0] -match ',')) {
    $Legs = @($Legs[0] -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

if ($Legs.Count -gt 0) {
    $RunOrder = @($Legs)
} else {
    $RunOrder = $DefaultOrder
}

function Write-SweepLog {
    param([string] $Step, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        step = $Step
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    $line = ($row | ConvertTo-Json -Compress) + "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::AppendAllText($LogPath, $line, $utf8NoBom)
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } elseif ($Status -eq "WARN") { "Yellow" } else { "Cyan" })
}

$initPath = Resolve-MuComplexityInitCkpt -UserPath $InitCkpt
if (-not $initPath) {
    Write-Host "[ERR] Init ckpt missing. Pass -InitCkpt or promote Lane A / finish Track 2 teacher." -ForegroundColor Red
    exit 1
}

$estMin = 0
foreach ($id in $RunOrder) {
    if ($LegCatalog.Contains($id)) { $estMin += [int]$LegCatalog[$id].Minutes }
}

Write-Host "[NEW] Mu comprehensive sweep (~${estMin} min planned)" -ForegroundColor Cyan
Write-Host "[i]  init=$initPath" -ForegroundColor DarkGray
Write-Host "[i]  out=$SweepDir" -ForegroundColor DarkGray
Write-Host "[i]  stack: GINO-DEQ pred kine + GNODE Phase3 | teacher -> corrector+synth+pseudo" -ForegroundColor DarkGray
Write-Host "[i]  metric: val mu_log_mae (lower better); reset init each leg" -ForegroundColor DarkGray

if ($DryRun) {
    foreach ($id in $RunOrder) {
        $cat = $LegCatalog[$id]
        if (-not $cat) {
            Write-Host "[ERR] unknown leg: $id" -ForegroundColor Red
            continue
        }
        $te = if ($TeacherEpochs -gt 0) { $TeacherEpochs } else { $cat.TeacherEpochs }
        $ce = if ($CorrectorEpochs -gt 0) { $CorrectorEpochs } else { $cat.CorrectorEpochs }
        Write-Host ("  {0,-16} {1,-16} {2,2}+{3,2}ep ~{4,3}min  {5}" -f $id, $cat.Tier, $te, $ce, $cat.Minutes, $cat.Hypothesis)
    }
    exit 0
}

if (-not $SkipPytest) {
    Write-SweepLog "pytest_biochem" "START" @{}
    $rc = Invoke-PythonRc -m pytest src/tests/test_biochem_passive_transport.py src/tests/test_biochem_physics.py -q --tb=line
    if ($rc -ne 0) {
        Write-SweepLog "pytest_biochem" "FAIL" @{ exit = $rc }
        exit $rc
    }
    Write-SweepLog "pytest_biochem" "OK" @{}
}

foreach ($legId in $RunOrder) {
    if (-not $LegCatalog.Contains($legId)) {
        Write-Host "[ERR] Unknown leg id: $legId" -ForegroundColor Red
        exit 1
    }
    $cat = $LegCatalog[$legId]
    $te = if ($TeacherEpochs -gt 0) { $TeacherEpochs } else { [int]$cat.TeacherEpochs }
    $ce = if ($CorrectorEpochs -gt 0) { $CorrectorEpochs } else { [int]$cat.CorrectorEpochs }
    $note = "mu6h_" + $legId
    $legDir = Join-Path $SweepDir $legId
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null

    Write-SweepLog $note "START" @{
        leg_id = $legId
        tier = $cat.Tier
        teacher_epochs = $te
        corrector_epochs = $ce
        hypothesis = $cat.Hypothesis
    }

    $env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $legDir
    Apply-MuComprehensiveLegEnv -LegId $legId -RunNote $note -TeacherEpochs $te -CorrectorEpochs $ce

    Copy-Item -Force $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")

    Write-Host "[NEW] leg=$legId | ${te}ep teacher + ${ce}ep corrector | pred kine | tier=$($cat.Tier)" -ForegroundColor Cyan

    $trainLog = Join-Path $legDir "train.log"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $ce --save-best --run-name $note 2>&1 | Tee-Object -FilePath $trainLog
    $trainRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap

    $summRc = Invoke-PythonRc scripts/summarize_mu_complexity_6h.py --run-note $note --manifest-append $ManifestPath `
        --leg-id $legId --tier $($cat.Tier) --hypothesis $($cat.Hypothesis) --train-exit $trainRc

    if ($trainRc -ne 0) {
        Write-SweepLog $note "FAIL" @{ exit = $trainRc; log = $trainLog }
        Write-Host "[WARN] Leg $legId failed; continuing sweep" -ForegroundColor Yellow
        continue
    }

    foreach ($name in @(
            "biochem_teacher_best_high_mu.pth",
            "biochem_teacher_last.pth",
            "biochem_best_high_mu.pth",
            "biochem_latest_checkpoint.pth"
        )) {
        $src = Join-Path $RepoRoot "outputs\biochem\$name"
        if (Test-Path $src) {
            Copy-Item -Force $src (Join-Path $legDir $name)
        }
    }

    if (-not $SkipGate -and $legId -ne "smoke") {
        $gateRc = Invoke-PythonRc scripts/check_gnode11_finish_gate.py --min-corrector-val 3 --min-pseudo-w 0.01
        Write-SweepLog ($note + "_gate") $(if ($gateRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $gateRc }
    }

    Write-SweepLog $note $(if ($summRc -eq 0) { "OK" } else { "WARN" }) @{ log = $trainLog; manifest = $ManifestPath }
}

Write-SweepLog "summarize" "START" @{}
$summPath = Join-Path $SweepDir "summary.json"
$summRc = Invoke-PythonRc scripts/summarize_mu_complexity_6h.py --manifest $ManifestPath --out $summPath
Write-SweepLog "summarize" $(if ($summRc -eq 0) { "OK" } else { "WARN" }) @{ out = $summPath }

Write-Host "[OK]  Mu comprehensive sweep complete." -ForegroundColor Green
Write-Host "[i]  python scripts/summarize_mu_complexity_6h.py --manifest $ManifestPath" -ForegroundColor Cyan
exit 0
