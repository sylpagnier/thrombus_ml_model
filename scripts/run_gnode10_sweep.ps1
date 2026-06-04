# GNODE rung 10 sweep: predicted kinematics teacher recipes -> auto-rank -> semi -> final dump+clot-phi.
#
# Phases:
#   1) probe  - short train all legs (default 4ep)
#   2) semi   - top N legs at 8ep (warm-start from probe ckpt)
#   3) final  - winner 12ep + species dump + clot-phi gate
#
# One line (~4-8h depending on GPU; resume-safe):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_sweep.ps1 -Fresh
#
# Quick dry-run catalog:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode10_sweep.ps1 -DryRun

param(
    [switch] $Fresh,
    [switch] $Resume,
    [switch] $DryRun,
    [switch] $SkipFinal,
    [string] $InitCkpt = "",
    [int] $ProbeEpochs = 4,
    [int] $SemiEpochs = 8,
    [int] $FinalEpochs = 12,
    [int] $TopN = 3,
    [int] $ClotEpochs = 35,
    [int] $DumpStride = 72,
    [int] $DumpMinSteps = 4,
    [double] $MinClotF1 = 0.26,
    [double] $MinGtPosFrac = 0.55,
    [string[]] $Legs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode10_env.ps1")
. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

$SweepDir = Join-Path $RepoRoot "outputs\biochem\gnode10_sweep"
$LogDir = Join-Path $SweepDir "logs"
$Manifest = Join-Path $SweepDir "manifest.jsonl"
New-Item -ItemType Directory -Force -Path $SweepDir, $LogDir | Out-Null

if ($Fresh) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $SweepDir
    New-Item -ItemType Directory -Force -Path $SweepDir, $LogDir | Out-Null
}

$InitPath = Resolve-Gnode10InitCkpt -UserPath $InitCkpt
if (-not $InitPath) {
    Write-Host "[ERR] No init checkpoint (after_94). Pass -InitCkpt or run go_gnode_8h_ladder.ps1" -ForegroundColor Red
    exit 1
}

$DefaultLegOrder = @(
    "K1_smoke_tf05",
    "K2_tf07_k25",
    "K3_tf10_k25",
    "K4_kine50",
    "K5_kine15",
    "K6_tbptt12",
    "K7_adr1e4",
    "K8_species_boost",
    "K9_detach_macro",
    "K0_kin_frozen"
)

$LegCatalog = @{
    K1_smoke_tf05 = @{
        Title = "Smoke recipe: TF=0.5, w_kine=0.25, TBPTT=5"
        TeacherForceMin = 0.5
        KineWeight = 0.25
        TrainKinLora = $true
        TbpttWindow = 5
    }
    K2_tf07_k25 = @{
        Title = "Higher TF=0.7, w_kine=0.25"
        TeacherForceMin = 0.7
        KineWeight = 0.25
    }
    K3_tf10_k25 = @{
        Title = "Full species TF=1.0, w_kine=0.25"
        TeacherForceMin = 1.0
        KineWeight = 0.25
    }
    K4_kine50 = @{
        Title = "Strong kine leash w_kine=0.5, TF=0.5"
        TeacherForceMin = 0.5
        KineWeight = 0.5
    }
    K5_kine15 = @{
        Title = "Light kine leash w_kine=0.15, TF=0.5"
        TeacherForceMin = 0.5
        KineWeight = 0.15
    }
    K6_tbptt12 = @{
        Title = "Longer TBPTT window=12 (OOM risk on 4GB)"
        TeacherForceMin = 0.5
        KineWeight = 0.25
        TbpttWindow = 12
    }
    K7_adr1e4 = @{
        Title = "ADR backprop weight=1e-4 (species must stay stable)"
        TeacherForceMin = 0.5
        KineWeight = 0.25
        AdrBackprop = $true
        AdrWeight = "1e-4"
    }
    K8_species_boost = @{
        Title = "FI/Mat boost 4/3"
        TeacherForceMin = 0.5
        KineWeight = 0.25
        FiWeight = 4.0
        MatWeight = 3.0
    }
    K9_detach_macro = @{
        Title = "DETACH_MACRO=1 stability ablation"
        TeacherForceMin = 0.5
        KineWeight = 0.25
        DetachMacro = $true
    }
    K0_kin_frozen = @{
        Title = "Predicted DEQ, kin LoRA off, w_kine=0 (flow-only baseline)"
        TeacherForceMin = 0.5
        KineWeight = 0.0
        TrainKinLora = $false
    }
}

function Write-ManifestRow {
    param([hashtable] $Row)
    $line = ($Row | ConvertTo-Json -Compress) + "`n"
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::AppendAllText($Manifest, $line, $utf8)
}

function Test-LegPhaseDone {
    param([string] $LegId, [string] $Phase)
    if (-not $Resume -or -not (Test-Path $Manifest)) { return $false }
    foreach ($line in [System.IO.File]::ReadAllLines($Manifest)) {
        if (-not $line.Trim()) { continue }
        $row = $line | ConvertFrom-Json
        if ($row.event -eq "leg" -and $row.leg_id -eq $LegId -and $row.phase -eq $Phase -and $row.status -eq "OK") {
            return $true
        }
    }
    return $false
}

function Invoke-Gnode10TrainLeg {
    param(
        [string] $LegId,
        [string] $Phase,
        [int] $Epochs,
        [string] $InitCheckpoint,
        [hashtable] $LegSpec
    )
    $legDir = Join-Path $SweepDir $LegId
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null

    $runNote = "gnode10_${LegId}_${Phase}"
    $logPath = Join-Path $LogDir "${LegId}_${Phase}.log"

    if (Test-LegPhaseDone -LegId $LegId -Phase $Phase) {
        Write-Host "[skip] $LegId / $Phase (manifest OK)" -ForegroundColor DarkGray
        return
    }

    Write-Host "[NEW] $Phase | $LegId | ${Epochs}ep | $($LegSpec.Title)" -ForegroundColor Cyan

    if ($DryRun) {
        Write-ManifestRow @{
            event = "leg"; leg_id = $LegId; phase = $Phase; status = "OK"
            dry_run = $true; epochs = $Epochs; title = $LegSpec.Title
        }
        return
    }

    Clear-Gnode10BiochemEnv
    Set-Gnode10PredictedKineBaseEnv -RunNote $runNote -Epochs $Epochs -OomSafe
    Apply-Gnode10LegOverrides -Leg $LegSpec
    $env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $legDir

    Copy-Item -Force $InitCheckpoint (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth")

    $started = Get-Date
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python -u -m src.training.train_biochem_corrector `
            --new --skip-pretrain --init-from-best `
            --epochs $Epochs --save-best `
            --run-name $runNote 2>&1 | Tee-Object -FilePath $logPath
        $rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
        if ($rc -ne 0) {
            Write-ManifestRow @{
                event = "leg"; leg_id = $LegId; phase = $Phase; status = "FAIL"
                run_note = $runNote; log = $logPath; exit = $rc
            }
            throw "Train failed exit=$rc"
        }

        $runId = ""
        foreach ($line in [System.IO.File]::ReadAllLines($logPath)) {
            if ($line -match "Run log:\s*(.+\\(\d{8}T\d{6}Z))\\run\.jsonl") {
                $runId = $Matches[2]
            }
        }
        if (-not $runId) {
            $runs = Get-ChildItem (Join-Path $RepoRoot "outputs\reports\training\biochem") -Directory |
                Where-Object { $_.Name -match '^\d{8}T\d{6}Z$' } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($runs) { $runId = $runs.Name }
        }

        $ckptBest = Join-Path $legDir "biochem_teacher_best_high_mu.pth"
        if (-not (Test-Path $ckptBest)) {
            $fallback = Join-Path $legDir "biochem_teacher_last.pth"
            if (Test-Path $fallback) { Copy-Item -Force $fallback $ckptBest }
        }

        $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
        Write-ManifestRow @{
            event = "leg"; leg_id = $LegId; phase = $Phase; status = "OK"
            run_note = $runNote; run_id = $runId; ckpt_dir = $legDir
            epochs = $Epochs; minutes = $mins; init = $InitCheckpoint
        }
        Write-Host "[OK]  $LegId / $Phase done (${mins}m) run_id=$runId" -ForegroundColor Green
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Get-Winners {
    param([string] $Phase, [int] $Count)
    $rc = Invoke-PythonRc scripts/score_gnode10_sweep.py `
        --sweep-dir $SweepDir --phase $Phase --top $Count --write-winners
    if ($rc -ne 0) { return @() }
    $winPath = Join-Path $SweepDir "winners_$Phase.json"
    if (-not (Test-Path $winPath)) { return @() }
    $w = Get-Content $winPath -Raw | ConvertFrom-Json
    return @($w.top)
}

function Get-LegSpec {
    param([string] $LegId)
    if (-not $LegCatalog.ContainsKey($LegId)) {
        throw "Unknown leg $LegId"
    }
    return $LegCatalog[$LegId]
}

function Get-ClotEvalSummary {
    param([string] $JsonlPath)
    if (-not (Test-Path $JsonlPath)) { return $null }
    $rows = Get-Content $JsonlPath | ForEach-Object { $_ | ConvertFrom-Json }
    if (-not $rows) { return $null }
    $f1 = @($rows | ForEach-Object { [double]$_.val.clot_f1 })
    return [pscustomobject]@{
        mean_f1 = [math]::Round(($f1 | Measure-Object -Average).Average, 3)
        min_f1  = [math]::Round(($f1 | Measure-Object -Minimum).Minimum, 3)
        n = $f1.Count
    }
}

$legOrder = if ($Legs.Count -gt 0) { $Legs } else { $DefaultLegOrder }

Write-Host "[NEW] GNODE 10 sweep | probe=${ProbeEpochs}ep semi=${SemiEpochs}ep final=${FinalEpochs}ep topN=$TopN" -ForegroundColor Cyan
Write-Host "[i]  Init=$InitPath | Out=$SweepDir" -ForegroundColor DarkGray

# --- Phase 1: probe ---
Write-Host "`n=== PHASE probe (${ProbeEpochs}ep x $($legOrder.Count) legs) ===" -ForegroundColor Yellow
foreach ($legId in $legOrder) {
    Invoke-Gnode10TrainLeg -LegId $legId -Phase "probe" -Epochs $ProbeEpochs `
        -InitCheckpoint $InitPath -LegSpec (Get-LegSpec $legId)
}

if ($DryRun) {
    Write-Host "[OK]  DryRun complete." -ForegroundColor Green
    exit 0
}

Invoke-PythonRc scripts/score_gnode10_sweep.py --sweep-dir $SweepDir --phase probe | Out-Null
$semiCandidates = Get-Winners -Phase "probe" -Count $TopN
if (-not $semiCandidates -or $semiCandidates.Count -eq 0) {
    Write-Host "[ERR] No probe winners scored. Check logs under $LogDir" -ForegroundColor Red
    exit 1
}

Write-Host "`n[i]  Semi-finalists:" -ForegroundColor Cyan
foreach ($c in $semiCandidates) {
    Write-Host "     $($c.leg_id) score=$([math]::Round($c.composite_score,3)) FI=$($c.val_species_fi_log_mae)"
}

# --- Phase 2: semi ---
Write-Host "`n=== PHASE semi (${SemiEpochs}ep x $($semiCandidates.Count) legs) ===" -ForegroundColor Yellow
foreach ($c in $semiCandidates) {
    $legId = [string]$c.leg_id
    $probeCkpt = Join-Path ([string]$c.ckpt_dir) "biochem_teacher_best_high_mu.pth"
    if (-not (Test-Path $probeCkpt)) {
        $probeCkpt = Join-Path ([string]$c.ckpt_dir) "biochem_teacher_last.pth"
    }
    if (-not (Test-Path $probeCkpt)) {
        Write-Host "[WARN] skip semi $legId (no probe ckpt)" -ForegroundColor Yellow
        continue
    }
    Invoke-Gnode10TrainLeg -LegId $legId -Phase "semi" -Epochs $SemiEpochs `
        -InitCheckpoint $probeCkpt -LegSpec (Get-LegSpec $legId)
}

Invoke-PythonRc scripts/score_gnode10_sweep.py --sweep-dir $SweepDir --phase semi | Out-Null
$finalWinner = (Get-Winners -Phase "semi" -Count 1 | Select-Object -First 1)
if (-not $finalWinner) {
    Write-Host "[ERR] No semi winner." -ForegroundColor Red
    exit 1
}

$winnerLeg = [string]$finalWinner.leg_id
$winnerDir = [string]$finalWinner.ckpt_dir
Write-Host "`n[i]  Final winner (semi): $winnerLeg score=$([math]::Round($finalWinner.composite_score,3))" -ForegroundColor Green

if ($SkipFinal) {
    Write-Host "[OK]  SkipFinal set. Leaderboard: $SweepDir\leaderboard.json" -ForegroundColor Green
    exit 0
}

# --- Phase 3: final train ---
Write-Host "`n=== PHASE final (${FinalEpochs}ep + dump + clot-phi) leg=$winnerLeg ===" -ForegroundColor Yellow
$semiCkpt = Join-Path $winnerDir "biochem_teacher_best_high_mu.pth"
if (-not (Test-Path $semiCkpt)) { $semiCkpt = Join-Path $winnerDir "biochem_teacher_last.pth" }

$finalLegId = "${winnerLeg}_final"
Invoke-Gnode10TrainLeg -LegId $finalLegId -Phase "final_train" -Epochs $FinalEpochs `
    -InitCheckpoint $semiCkpt -LegSpec (Get-LegSpec $winnerLeg)

$finalDir = Join-Path $SweepDir $finalLegId
$teacherBest = Join-Path $finalDir "biochem_teacher_best_high_mu.pth"
$teacherLast = Join-Path $finalDir "biochem_teacher_last.pth"
$dumpCkpt = if (Test-Path $teacherBest) { $teacherBest } else { $teacherLast }

if (-not (Test-Path $dumpCkpt)) {
    Write-Host "[ERR] Missing final teacher ckpt under $finalDir" -ForegroundColor Red
    exit 1
}

# Species eval (predicted kine path)
$speciesLog = Join-Path $LogDir "${finalLegId}_species_eval.log"
Invoke-PythonRc scripts/eval_passive_species_anchors.py `
    --checkpoint $dumpCkpt --device cuda --predicted-kine 2>&1 | Tee-Object -FilePath $speciesLog

# Dump
$anchorDir = Join-Path $SweepDir "anchors_stride_$DumpStride"
if (-not (Test-LegPhaseDone -LegId $finalLegId -Phase "final_dump")) {
    Write-Host "[NEW] dump -> $anchorDir" -ForegroundColor Cyan
    Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
        --teacher $dumpCkpt --out-dir $anchorDir --device cuda `
        --time-stride $DumpStride --min-steps $DumpMinSteps --force
    Write-ManifestRow @{
        event = "leg"; leg_id = $finalLegId; phase = "final_dump"; status = "OK"
        anchor_dir = $anchorDir; teacher = $dumpCkpt
    }
}

# Clot-phi preflight (1ep gt+ check)
$preflightLeg = "${finalLegId}_preflight"
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
    -AnchorDir $anchorDir -LegName $preflightLeg -Epochs 1 -SkipViz -SkipEval
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] preflight clot-phi failed (dump may be bad vs June cache)" -ForegroundColor Yellow
}

# Full clot-phi
$clotLeg = "gnode10_${winnerLeg}_clotphi"
$clotDir = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$clotLeg"
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
    -AnchorDir $anchorDir -LegName $clotLeg -Epochs $ClotEpochs -SkipViz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$clotCkpt = Join-Path $clotDir "clot_phi_best.pth"
$evalJson = Join-Path $SweepDir "multi_anchor_$clotLeg.jsonl"
Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $clotCkpt --out $evalJson

$clotSummary = Get-ClotEvalSummary $evalJson
$promotedDir = Join-Path $SweepDir "promoted"
New-Item -ItemType Directory -Force -Path $promotedDir | Out-Null
if (Test-Path $clotCkpt) {
    Copy-Item -Force $clotCkpt (Join-Path $promotedDir "clot_phi_best.pth")
    Copy-Item -Force $dumpCkpt (Join-Path $promotedDir "biochem_teacher_best_high_mu.pth")
}

$gateOk = $false
if ($clotSummary) {
    $gateOk = $clotSummary.min_f1 -ge $MinClotF1
    Write-Host "[i]  clot-phi: mean_f1=$($clotSummary.mean_f1) min_f1=$($clotSummary.min_f1) gate>=$MinClotF1 -> $(if ($gateOk) { 'PASS' } else { 'FAIL' })" -ForegroundColor $(if ($gateOk) { 'Green' } else { 'Yellow' })
}

Write-ManifestRow @{
    event = "leg"; leg_id = $finalLegId; phase = "final_clotphi"; status = $(if ($gateOk) { "OK" } else { "WARN" })
    winner_probe_leg = $winnerLeg
    clot_leg = $clotLeg
    anchor_dir = $anchorDir
    min_f1 = if ($clotSummary) { $clotSummary.min_f1 } else { $null }
    mean_f1 = if ($clotSummary) { $clotSummary.mean_f1 } else { $null }
    promoted = $promotedDir
}

# Viz
if (Test-Path $clotCkpt) {
    Invoke-ClotPhiScatterViz -Checkpoint $clotCkpt -Anchor patient007 -TimeIndex 4 `
        -Out (Join-Path $SweepDir "viz_clotphi_p007.png")
}

Write-Host "`n[OK]  GNODE 10 sweep complete." -ForegroundColor Green
Write-Host "[i]  Manifest: $Manifest" -ForegroundColor DarkGray
Write-Host "[i]  Leaderboard: $SweepDir\leaderboard.json" -ForegroundColor DarkGray
Write-Host "[i]  Promoted: $promotedDir" -ForegroundColor DarkGray
Write-Host "[i]  Compare min_f1 to cached 9.9 (~0.34) and p007 (~0.63); check preflight gt+ ~0.578" -ForegroundColor DarkGray
