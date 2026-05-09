param(
    [Parameter(Mandatory = $true)]
    [string]$ParamsPath
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$RepoRoot;$env:PYTHONPATH"
python -m pipeline.train_master $ParamsPath
exit $LASTEXITCODE
