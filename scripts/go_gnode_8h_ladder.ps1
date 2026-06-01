# GNODE ladder ~8h queue: 9.4 teacher -> species eval -> dump -> 9.5 clot-phi -> optional 9.6 ADR probe.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gnode_8h_ladder.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gnode_8h_ladder.ps1" -Fresh -AdrProbe

param(
    [switch] $Fresh,
    [switch] $Resume,
    [int] $TeacherEpochs = 8,
    [int] $ClotEpochs = 35,
    [int] $DumpStride = 72,
    [int] $DumpMinSteps = 4,
    [int] $AdrProbeEpochs = 6,
    [switch] $AdrProbe,
    [double] $MinClotF1 = 0.26
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
. (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\gnode_8h_ladder"
$LogDir = Join-Path $OutRoot "logs"
$Manifest = Join-Path $OutRoot "manifest.jsonl"
New-Item -ItemType Directory -Force -Path $OutRoot, $LogDir | Out-Null

if ($Fresh) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $OutRoot
    New-Item -ItemType Directory -Force -Path $OutRoot, $LogDir | Out-Null
}

function Write-Manifest {
    param([string] $Leg, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        leg    = $Leg
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    $line = ($row | ConvertTo-Json -Compress) + "`n"
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::AppendAllText($Manifest, $line, $utf8)
    $color = switch ($Status) {
        "OK" { "Green" }
        "FAIL" { "Red" }
        "WARN" { "Yellow" }
        default { "Cyan" }
    }
    Write-Host "[$Status] $Leg" -ForegroundColor $color
}

function Test-LegDone {
    param([string] $Leg)
    if (-not $Resume -or -not (Test-Path $Manifest)) { return $false }
    foreach ($line in [System.IO.File]::ReadAllLines($Manifest)) {
        if (-not $line.Trim()) { continue }
        $row = $line | ConvertFrom-Json
        if ($row.leg -eq $Leg -and $row.status -eq "OK") { return $true }
    }
    return $false
}

function Invoke-Leg {
    param(
        [string] $Leg,
        [string] $LogName,
        [scriptblock] $Body
    )
    if (Test-LegDone $Leg) {
        Write-Host "[skip] $Leg (already OK in manifest)" -ForegroundColor DarkGray
        return
    }
    $logPath = Join-Path $LogDir "$LogName.log"
    Write-Host "[NEW] === $Leg === (log: $logPath)" -ForegroundColor Cyan
    $started = Get-Date
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Body 2>&1 | Tee-Object -FilePath $logPath
        $rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
        if ($rc -ne 0) {
            Write-Manifest $Leg "FAIL" @{ log = $LogName; exit = $rc }
            throw "Leg $Leg failed (exit $rc)"
        }
        $mins = ((Get-Date) - $started).TotalMinutes
        Write-Manifest $Leg "OK" @{ log = $LogName; minutes = [math]::Round($mins, 1) }
    } catch {
        $mins = ((Get-Date) - $started).TotalMinutes
        if (-not (Test-LegDone $Leg)) {
            Write-Manifest $Leg "FAIL" @{ log = $LogName; minutes = [math]::Round($mins, 1); err = $_.Exception.Message }
        }
        throw
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Set-GnodeTeacherEnv {
    $env:BIOCHEM_PRESET = "passive_transport"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = "gnode_8h_ladder"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    Set-GnodeGtFlowVizEnv
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0.25"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
    $env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"
}

function Copy-TeacherArtifacts {
    param([string] $Tag)
    $dest = Join-Path $OutRoot "checkpoints"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    foreach ($name in @("biochem_teacher_last.pth", "biochem_teacher_best_high_mu.pth")) {
        $src = Join-Path $RepoRoot "outputs\biochem\$name"
        if (Test-Path $src) {
            Copy-Item -Force $src (Join-Path $dest "${Tag}_$name")
        }
    }
}

function Get-ClotEvalSummary {
    param([string] $JsonlPath)
    if (-not (Test-Path $JsonlPath)) { return $null }
    $rows = Get-Content $JsonlPath | ForEach-Object { $_ | ConvertFrom-Json }
    if (-not $rows) { return $null }
    $f1 = @($rows | ForEach-Object { [double]$_.val.clot_f1 })
    $mae = @($rows | ForEach-Object { [double]$_.val.mu_log_mae })
    return [pscustomobject]@{
        mean_f1 = [math]::Round(($f1 | Measure-Object -Average).Average, 3)
        min_f1  = [math]::Round(($f1 | Measure-Object -Minimum).Minimum, 3)
        mean_logMAE = [math]::Round(($mae | Measure-Object -Average).Average, 3)
        n = $f1.Count
    }
}

Write-Host "[NEW] GNODE 8h ladder | teacher=${TeacherEpochs}ep dump_stride=$DumpStride clot=${ClotEpochs}ep" -ForegroundColor Cyan
Write-Host "[i]  OutRoot=$OutRoot" -ForegroundColor DarkGray

# --- 9.4: clot-band teacher ---
Invoke-Leg "9.4_teacher" "9.4_teacher" {
    Set-GnodeTeacherEnv
    Invoke-PythonRcCheck -Label "teacher_94" -PyArgs @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--skip-pretrain", "--init-from-best",
        "--epochs", "$TeacherEpochs", "--save-best", "--run-name", "gnode_8h_teacher"
    )
}
if (-not (Test-LegDone "9.4_teacher")) {
    Copy-TeacherArtifacts "after_94"
} elseif (-not (Test-Path (Join-Path $OutRoot "checkpoints\after_94_biochem_teacher_last.pth"))) {
    Copy-TeacherArtifacts "after_94"
}

$Teacher = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"

# Species table + full-field + clot-band on raw anchor (t=200)
Invoke-Leg "9.4_species_eval" "9.4_species_eval" {
    Invoke-PythonRcCheck -Label "eval_passive_species" -PyArgs @(
        "scripts/eval_passive_species_anchors.py", "--checkpoint", $Teacher, "--device", "cuda"
    )
}
Invoke-Leg "9.4_viz_flow" "9.4_viz_flow" {
    Invoke-BiochemTeacherSnapshot -Checkpoint $Teacher -Anchor patient007 `
        -Out (Join-Path $OutRoot "viz_teacher_flow_p007.png") -Label "gnode8h"
}
Invoke-Leg "9.4_viz_clotband_raw" "9.4_viz_clotband_raw" {
    Invoke-BiochemTeacherClotbandViz -Checkpoint $Teacher -Anchor patient007 -TimeIndex -1 `
        -Out (Join-Path $OutRoot "viz_teacher_clotband_raw_p007_t200.png") -Label "gnode8h"
}

# --- 9.5: dump ---
$AnchorDir = Join-Path $OutRoot "anchors_stride_$DumpStride"
Invoke-Leg "9.5_dump" "9.5_dump" {
    Invoke-PythonRcCheck -Label "dump_teacher_species" -PyArgs @(
        "scripts/dump_teacher_species_to_anchors.py",
        "--teacher", $Teacher,
        "--out-dir", $AnchorDir,
        "--device", "cuda",
        "--time-stride", "$DumpStride",
        "--min-steps", "$DumpMinSteps",
        "--force"
    )
}

Invoke-Leg "9.5_viz_clotband_dump" "9.5_viz_clotband_dump" {
    $ti = 4
    if ($DumpMinSteps -ge 6) { $ti = -1 }
    Invoke-BiochemTeacherClotbandViz -Checkpoint $Teacher -Anchor patient007 `
        -AnchorDir $AnchorDir -TimeIndex $ti `
        -Out (Join-Path $OutRoot "viz_teacher_clotband_dump_p007_t${ti}.png") -Label "gnode8h_dump"
}

# --- 9.5: clot-phi ---
$LegName = "gnode_8h_clotphi"
$ClotCkptDir = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$LegName"
$EvalJson = Join-Path $OutRoot "multi_anchor_$LegName.jsonl"

Invoke-Leg "9.5_clotphi" "9.5_clotphi" {
    $clotArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", ".\scripts\go_clot_phi_from_anchor_dir.ps1",
        "-AnchorDir", $AnchorDir,
        "-LegName", $LegName,
        "-Epochs", "$ClotEpochs",
        "-SkipViz"
    )
    & powershell @clotArgs
}

$ClotCkpt = Join-Path $ClotCkptDir "clot_phi_best.pth"
if (Test-Path $ClotCkpt) {
    Copy-Item -Force $ClotCkpt (Join-Path $OutRoot "clot_phi_best_promoted.pth")
}

Invoke-Leg "9.5_clotphi_eval" "9.5_clotphi_eval" {
    Invoke-PythonRcCheck -Label "eval_clot_phi" -PyArgs @(
        "scripts/eval_clot_phi_multi_anchor.py", "--checkpoint", $ClotCkpt, "--out", $EvalJson
    )
}

$summary = Get-ClotEvalSummary $EvalJson
if ($summary) {
    Write-Host "[i]  clot-phi multi-anchor: mean_f1=$($summary.mean_f1) min_f1=$($summary.min_f1) mean_logMAE=$($summary.mean_logMAE)" -ForegroundColor Cyan
    $gateOk = $summary.min_f1 -ge $MinClotF1
    Write-Manifest "9.5_gate" $(if ($gateOk) { "OK" } else { "WARN" }) @{
        mean_f1 = $summary.mean_f1
        min_f1  = $summary.min_f1
        min_f1_target = $MinClotF1
    }
} else {
    Write-Manifest "9.5_gate" "WARN" @{ note = "missing eval jsonl" }
}

Invoke-Leg "9.5_viz_clotphi" "9.5_viz_clotphi" {
    $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
    $ti = 4
    if ($DumpMinSteps -ge 6) { $ti = -1 }
    Invoke-ClotPhiScatterViz -Checkpoint $ClotCkpt -Anchor patient007 -TimeIndex $ti `
        -Out (Join-Path $OutRoot "viz_clotphi_mlp_p007.png")
    Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue
}

# --- 9.6 optional: ADR backprop probe ---
if ($AdrProbe -and (Test-Path $Teacher)) {
    Invoke-Leg "9.6_adr_teacher" "9.6_adr_teacher" {
        Set-GnodeTeacherEnv
        $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "1"
        $env:BIOCHEM_PASSIVE_ADR_WEIGHT = "1e-4"
        $env:BIOCHEM_RESUME = "1"
        $env:BIOCHEM_TRAIN_MODE = "resume"
        $env:BIOCHEM_INIT_FROM_BEST = "0"
        $env:BIOCHEM_TEACHER_EPOCHS = "$AdrProbeEpochs"
        $env:BIOCHEM_EPOCHS = "$AdrProbeEpochs"
        $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$AdrProbeEpochs"
        Invoke-PythonRcCheck -Label "adr_teacher" -PyArgs @(
            "-m", "src.training.train_biochem_corrector",
            "--epochs", "$AdrProbeEpochs", "--save-best", "--run-name", "gnode_8h_adr_probe"
        )
    }
    Copy-TeacherArtifacts "after_96"
    $TeacherAdr = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
    Invoke-Leg "9.6_viz_clotband_adr" "9.6_viz_clotband_adr" {
        Invoke-BiochemTeacherClotbandViz -Checkpoint $TeacherAdr -Anchor patient007 -TimeIndex -1 `
            -Out (Join-Path $OutRoot "viz_teacher_clotband_after_adr_p007.png") -Label "gnode8h_adr"
    }
}

Write-Host "[OK]  GNODE 8h ladder finished. Manifest: $Manifest" -ForegroundColor Green
Write-Host "[i]  Artifacts under $OutRoot" -ForegroundColor DarkGray
