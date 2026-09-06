param([string]$EnvFile = ".env.runpod")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv. Create it and install the project with: py -m venv .venv; .venv\Scripts\python -m pip install -e '.[dev]'"
}
& $Python -m compute_echo.cli deployment-readiness --env-file $EnvFile --probe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m compute_echo.cli runpod-demo --env-file $EnvFile
