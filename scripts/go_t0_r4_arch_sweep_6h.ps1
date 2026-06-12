# T0 Rung4 architecture overnight sweep (~8h): s4/s5/S-star + species/arch diagnostics.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_r4_arch_sweep_6h.ps1"
#   powershell ... -DryRun
#   powershell ... -SkipDiagnostics
#   powershell ... -Legs s_star_full,s4_delta_gnn
#
# Morning:
#   python scripts/summarize_t0_r4_arch_sweep_6h.py
#   python scripts/diagnose_t0_r4_sweep_postflight.py

param(
    [string] $ValAnchor = "patient007",
    [string] $EvalAnchor = "patient007",
    [int] $Epochs = 0,
    [double] $TargetHours = 0,
    [string[]] $Legs = @(),
    [switch] $SkipViz,
    [switch] $SkipDiagnostics,
    [switch] $SkipPytest,
    [switch] $SkipCompleted,
    [switch] $DryRun,
    [string] $TeacherCkpt = "outputs/biochem/biochem_teacher_last.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }

$SweepDir = Join-Path $RepoRoot "outputs\biochem\sweep_t0_r4_arch_6h"
$ManifestPath = Join-Path $SweepDir "manifest.jsonl"
$LogPath = Join-Path $SweepDir "sweep_log.jsonl"
$DiagDir = Join-Path $SweepDir "diagnostics"
New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagDir | Out-Null

$LegCatalog = [ordered]@{
    smoke_s4 = @{ Minutes = 6;  DiagMinutes = 2; Hypothesis = "Plumbing: 4ep gate GNN" }
    ref_s0 = @{ Minutes = 4;  DiagMinutes = 3; Hypothesis = "Deploy s0 baseline eval + species timeline" }
    s4_gate_gnn = @{ Minutes = 24; DiagMinutes = 4; Hypothesis = "2L band GNN gate residual (current s4)" }
    s4_delta_gnn = @{ Minutes = 28; DiagMinutes = 4; Hypothesis = "GNN FI/Mat delta on s0 (bypass rank barrier)" }
    s4_gate_fpstrong = @{ Minutes = 24; DiagMinutes = 4; Hypothesis = "Gate GNN + stronger FP suppression" }
    s4_gate_commit = @{ Minutes = 28; DiagMinutes = 4; Hypothesis = "Gate GNN + commit BCE on FN/FP" }
    s4_risk_gnn = @{ Minutes = 26; DiagMinutes = 4; Hypothesis = "GNN risk reweight before s0 top-8%" }
    s5_mlp_fimat = @{ Minutes = 28; DiagMinutes = 4; Hypothesis = "MLP FI/Mat delta on s0 hotspots" }
    s5_gnn_fimat = @{ Minutes = 30; DiagMinutes = 4; Hypothesis = "Band GNN FI/Mat delta in E(t)" }
    s5_gru_fimat = @{ Minutes = 32; DiagMinutes = 4; Hypothesis = "GRU temporal smooth on FI/Mat delta" }
    s_star_gate = @{ Minutes = 28; DiagMinutes = 4; Hypothesis = "S* gate only (where/when)" }
    s_star_species = @{ Minutes = 28; DiagMinutes = 4; Hypothesis = "S* species only (magnitude on s0 gate)" }
    s_star_dyn = @{ Minutes = 26; DiagMinutes = 4; Hypothesis = "S* dynamics only (temporal smooth)" }
    s_star_gate_species = @{ Minutes = 32; DiagMinutes = 4; Hypothesis = "S* gate + species" }
    s_star_full = @{ Minutes = 35; DiagMinutes = 5; Hypothesis = "S* full: gate + species + dyn" }
    s_star_small_ml = @{ Minutes = 30; DiagMinutes = 4; Hypothesis = "Small ML on rules (tiny gate+species)" }
}

$DefaultOrder = @(
    "smoke_s4", "ref_s0", "s4_gate_gnn", "s4_delta_gnn", "s4_gate_commit",
    "s4_gate_fpstrong", "s4_risk_gnn", "s5_mlp_fimat", "s5_gnn_fimat", "s5_gru_fimat",
    "s_star_gate", "s_star_species", "s_star_dyn", "s_star_gate_species",
    "s_star_full", "s_star_small_ml"
)

$PreflightMinutes = 12
$PostflightMinutes = 4
$VizMinutesPerLeg = 3

if ($Legs.Count -eq 1 -and ($Legs[0] -match ',')) {
    $Legs = @($Legs[0] -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
$RunOrder = if ($Legs.Count -gt 0) { @($Legs) } else { $DefaultOrder }

$trainLegIds = @($RunOrder | Where-Object { $_ -ne "ref_s0" })
if ($TargetHours -gt 0 -and $trainLegIds.Count -gt 0) {
    $estMin = 0
    foreach ($lid in $trainLegIds) {
        if ($LegCatalog.Contains($lid)) {
            $estMin += [int]$LegCatalog[$lid].Minutes
            if (-not $SkipDiagnostics) { $estMin += [int]$LegCatalog[$lid].DiagMinutes }
        }
    }
    if (-not $SkipDiagnostics) { $estMin += $PreflightMinutes + $PostflightMinutes }
    if (-not $SkipViz) { $estMin += $VizMinutesPerLeg * $trainLegIds.Count }
    $scaled = [int][Math]::Round(32.0 * ($TargetHours * 60.0) / [double]$estMin)
    if ($Epochs -le 0) { $Epochs = [Math]::Max(6, [Math]::Min(40, $scaled)) }
}

function Write-SweepLog {
    param([string] $Leg, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        leg_id = $Leg
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    $line = ($row | ConvertTo-Json -Compress) + "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::AppendAllText($LogPath, $line, $utf8NoBom)
    Write-Host "[$Status] $Leg" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } elseif ($Status -eq "WARN") { "Yellow" } else { "Cyan" })
}

function Invoke-LegDiagnostic {
    param([string] $LegId, [string] $LegDir)
    if ($SkipDiagnostics) { return 0 }
    if ($LegId -eq "ref_s0") {
        Write-Host "[NEW] ref s0 species diagnostic ($EvalAnchor)" -ForegroundColor Cyan
        $diagArgs = @(
            "scripts/diagnose_t0_r4_sweep_preflight.py",
            "--anchor", $EvalAnchor,
            "--out", (Join-Path $LegDir "diagnostic_${EvalAnchor}.json")
        )
        return (Invoke-PythonRcCheck -Label "ref s0 species diag" -PyArgs $diagArgs)
    }
    if (-not (Test-Path (Join-Path $LegDir "best.pth"))) {
        Write-Host "[skip] no ckpt for diagnostic $LegId" -ForegroundColor Yellow
        return 0
    }
    Write-Host "[NEW] leg diagnostic $LegId ($EvalAnchor)" -ForegroundColor Cyan
    $diagArgs = @(
        "scripts/diagnose_t0_r4_sweep_leg.py",
        "--anchor", $EvalAnchor,
        "--leg-dir", $LegDir
    )
    return (Invoke-PythonRcCheck -Label "diag $LegId" -PyArgs $diagArgs)
}

$estTotal = $PreflightMinutes + $PostflightMinutes
foreach ($lid in $RunOrder) {
    if ($LegCatalog.Contains($lid)) {
        $estTotal += [int]$LegCatalog[$lid].Minutes
        if (-not $SkipDiagnostics) { $estTotal += [int]$LegCatalog[$lid].DiagMinutes }
        if (-not $SkipViz -and $lid -ne "ref_s0") { $estTotal += $VizMinutesPerLeg }
    }
}

Write-Host ""
Write-Host "[NEW] T0 Rung4 arch sweep (~$estTotal min / ~$([math]::Round($estTotal/60,1)) h, val=$ValAnchor)" -ForegroundColor Cyan
Write-Host "[i] sweep_dir=$SweepDir" -ForegroundColor DarkGray
Write-Host "[i] legs=$($RunOrder -join ', ')" -ForegroundColor DarkGray
Write-Host "[i] diagnostics=$(if ($SkipDiagnostics) { 'OFF' } else { 'ON' })" -ForegroundColor DarkGray
if ($Epochs -gt 0) { Write-Host "[i] epochs override=$Epochs" -ForegroundColor DarkGray }

if ($DryRun) {
    Write-Host "[i] preflight ~${PreflightMinutes}min | postflight ~${PostflightMinutes}min" -ForegroundColor DarkGray
    foreach ($lid in $RunOrder) {
        if (-not $LegCatalog.Contains($lid)) {
            Write-Host "[ERR] unknown leg: $lid" -ForegroundColor Red
            continue
        }
        $cat = $LegCatalog[$lid]
        $tag = if ($lid -eq "ref_s0") { "eval+diag" } else { "train+eval+diag" }
        $dm = if ($SkipDiagnostics) { 0 } else { $cat.DiagMinutes }
        Write-Host ("  {0,-22} ~{1,3}min + {2}min diag  [{3}]  {4}" -f $lid, $cat.Minutes, $dm, $tag, $cat.Hypothesis)
    }
    exit 0
}

if (-not $SkipPytest) {
    Write-SweepLog "pytest_t0_sweep" "START" @{}
    $rc = Invoke-PythonRc -m pytest src/tests/test_t0_r4_sweep.py -q --tb=line
    if ($rc -ne 0) {
        Write-SweepLog "pytest_t0_sweep" "FAIL" @{ exit = $rc }
        exit $rc
    }
    Write-SweepLog "pytest_t0_sweep" "OK" @{}
}

if (-not $SkipDiagnostics) {
    Write-SweepLog "preflight" "START" @{}
    Write-Host "[NEW] Pre-sweep diagnostics (oracle ceilings, rank barrier, s0 species)" -ForegroundColor Cyan
    $pfArgs = @(
        "scripts/diagnose_t0_r4_sweep_preflight.py",
        "--anchor", $EvalAnchor,
        "--out", (Join-Path $SweepDir "preflight.json")
    )
    $pfRc = Invoke-PythonRcCheck -Label "sweep preflight" -PyArgs $pfArgs

    Write-Host "[NEW] Carreau x gelation oracle (GT u,v + GT species)" -ForegroundColor Cyan
    $gelLog = Join-Path $DiagDir "carreau_gelation_${EvalAnchor}.log"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python scripts/diagnose_t0_carreau_gelation.py --anchor $EvalAnchor --times "0,27,53" 2>&1 | Tee-Object -FilePath $gelLog
    $gelRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    Write-SweepLog "preflight" $(if ($pfRc -eq 0) { "OK" } else { "WARN" }) @{ gelation_exit = $gelRc }
}

foreach ($legId in $RunOrder) {
    if (-not $LegCatalog.Contains($legId)) {
        Write-Host "[ERR] Unknown leg: $legId" -ForegroundColor Red
        exit 1
    }
    $cat = $LegCatalog[$legId]
    $legDir = Join-Path $SweepDir $legId
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null
    $ckptRel = "outputs/biochem/sweep_t0_r4_arch_6h/$legId/best.pth"

    Write-SweepLog $legId "START" @{ hypothesis = $cat.Hypothesis }

    if ($legId -eq "ref_s0") {
        Write-Host "[NEW] ref s0 eval ($EvalAnchor)" -ForegroundColor Cyan
        $evalArgs = @(
            "scripts/eval_t0_rung4_step.py",
            "--anchor", $EvalAnchor,
            "--times", "0,7,15,22,27,40,53",
            "--step", "s0"
        )
        if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
        $evalRc = Invoke-PythonRcCheck -Label "ref s0 eval" -PyArgs $evalArgs
        $s0Json = Join-Path $RepoRoot "outputs/biochem/clot_trigger/t0_rung4_s0_${EvalAnchor}.json"
        if (Test-Path $s0Json) {
            Copy-Item -Force $s0Json (Join-Path $legDir "eval_$EvalAnchor.json")
        }
        $diagRc = Invoke-LegDiagnostic -LegId $legId -LegDir $legDir
        $summArgs = @(
            "scripts/summarize_t0_r4_arch_sweep_6h.py",
            "--leg-id", "ref_s0",
            "--train-exit", "$evalRc",
            "--hypothesis", $cat.Hypothesis,
            "--manifest-append", $ManifestPath
        )
        Invoke-PythonRcCheck -Label "summarize ref_s0" -PyArgs $summArgs | Out-Null
        Write-SweepLog $legId $(if ($evalRc -eq 0) { "OK" } else { "FAIL" }) @{ exit = $evalRc; diag = $diagRc }
        continue
    }

    if ($legId -ne "ref_s0" -and $SkipCompleted -and (Test-Path (Join-Path $legDir "best.pth")) -and (Test-Path (Join-Path $legDir "eval_$EvalAnchor.json"))) {
        Write-Host "[skip] $legId already has ckpt+eval (use without -SkipCompleted to retrain)" -ForegroundColor Yellow
        if (-not $SkipDiagnostics) {
            if (-not (Test-Path (Join-Path $legDir "diagnostic_$EvalAnchor.json"))) {
                Invoke-LegDiagnostic -LegId $legId -LegDir $legDir | Out-Null
            }
        }
        if (-not $SkipViz) {
            $vizRc = Invoke-PythonRc @(
                "scripts/viz_t0_r4_sweep_leg.py",
                "--anchor", $EvalAnchor,
                "--leg-dir", $legDir,
                "--max-frames", "10"
            )
            if ($vizRc -ne 0) { Write-Host "[WARN] viz $legId failed (exit=$vizRc)" -ForegroundColor Yellow }
        }
        continue
    }

    Write-Host "[NEW] leg=$legId | $($cat.Hypothesis)" -ForegroundColor Cyan
    $trainArgs = @(
        "-m", "src.training.train_t0_r4_sweep_leg",
        "--recipe", $legId,
        "--val-anchor", $ValAnchor,
        "--out", $ckptRel
    )
    if ($Epochs -gt 0) { $trainArgs += @("--epochs", "$Epochs") }

    $trainLog = Join-Path $legDir "train.log"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python @trainArgs 2>&1 | Tee-Object -FilePath $trainLog
    $trainRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap

    $evalRc = 0
    if ($trainRc -eq 0 -and (Test-Path (Join-Path $legDir "best.pth"))) {
        $evalArgs = @(
            "scripts/eval_t0_r4_sweep_leg.py",
            "--anchor", $EvalAnchor,
            "--times", "0,7,15,22,27,40,53",
            "--leg-dir", $legDir
        )
        if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
        $evalRc = Invoke-PythonRcCheck -Label "sweep eval $legId" -PyArgs $evalArgs
    } else {
        Write-Host "[WARN] skip eval (train failed or no ckpt)" -ForegroundColor Yellow
        $evalRc = $trainRc
    }

    $diagRc = 0
    if ($trainRc -eq 0) {
        $diagRc = Invoke-LegDiagnostic -LegId $legId -LegDir $legDir
    }

    if (-not $SkipViz -and $trainRc -eq 0) {
        $vizArgs = @(
            "scripts/viz_t0_r4_sweep_leg.py",
            "--anchor", $EvalAnchor,
            "--leg-dir", $legDir,
            "--max-frames", "10"
        )
        if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
        $vizRc = Invoke-PythonRc @vizArgs
        if ($vizRc -ne 0) {
            Write-Host "[WARN] viz $legId failed (exit=$vizRc); continuing sweep" -ForegroundColor Yellow
        }
    }

    Invoke-PythonRcCheck -Label "summarize $legId" -PyArgs @(
        "scripts/summarize_t0_r4_arch_sweep_6h.py",
        "--leg-id", $legId,
        "--train-exit", "$trainRc",
        "--hypothesis", $cat.Hypothesis,
        "--manifest-append", $ManifestPath
    ) | Out-Null

    Write-SweepLog $legId $(if ($trainRc -eq 0 -and $evalRc -eq 0) { "OK" } else { "FAIL" }) @{
        train_exit = $trainRc
        eval_exit = $evalRc
        diag_exit = $diagRc
        log = $trainLog
    }
}

Write-SweepLog "summarize" "START" @{}
$summRc = Invoke-PythonRcCheck -Label "summarize all" -PyArgs @(
    "scripts/summarize_t0_r4_arch_sweep_6h.py",
    "--manifest", $ManifestPath
)
Write-SweepLog "summarize" $(if ($summRc -eq 0) { "OK" } else { "WARN" }) @{}

if (-not $SkipDiagnostics) {
    Write-SweepLog "postflight" "START" @{}
    $postRc = Invoke-PythonRcCheck -Label "postflight" -PyArgs @(
        "scripts/diagnose_t0_r4_sweep_postflight.py",
        "--sweep-dir", $SweepDir,
        "--anchor", $EvalAnchor
    )
    Write-SweepLog "postflight" $(if ($postRc -eq 0) { "OK" } else { "WARN" }) @{}
}

Write-Host "[OK] T0 Rung4 arch sweep complete." -ForegroundColor Green
Write-Host "[i] summary: outputs/biochem/sweep_t0_r4_arch_6h/summary.json" -ForegroundColor Cyan
if (-not $SkipDiagnostics) {
    Write-Host "[i] diagnostics: outputs/biochem/sweep_t0_r4_arch_6h/diagnostics_summary.json" -ForegroundColor Cyan
    Write-Host "[i] preflight: outputs/biochem/sweep_t0_r4_arch_6h/preflight.json" -ForegroundColor Cyan
}
exit 0
