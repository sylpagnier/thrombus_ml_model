# Reliable python exit codes under PowerShell $ErrorActionPreference=Stop.
# Dot-source: . (Join-Path $PSScriptRoot "_python_rc.ps1")

function Invoke-PythonRc {
    param(
        [Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs,
        [switch] $Quiet
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if ($Quiet) {
        & python -u @PyArgs 2>&1 | Out-Null
    } else {
        & python -u @PyArgs 2>&1 | ForEach-Object { Write-Host $_ }
    }
    $rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
    $ErrorActionPreference = $prevEap
    return $rc
}

function Invoke-PythonRcCheck {
    param(
        [Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs,
        [string] $Label = "python"
    )
    $rc = Invoke-PythonRc @PyArgs
    if ($rc -ne 0) {
        Write-Host "[ERR] $Label failed (exit=$rc)" -ForegroundColor Red
        exit $rc
    }
    return 0
}
