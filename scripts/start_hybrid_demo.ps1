param([string]$EnvFile = ".env.hybrid")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv. Run: py -m venv .venv; .venv\Scripts\python -m pip install -e '.[dev]'"
}
if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing $EnvFile. Copy config\hybrid.env.example and fill its three credentials."
}
& $Python -m compute_echo.cli hybrid-demo --env-file $EnvFile
