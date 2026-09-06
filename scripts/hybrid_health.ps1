param([int]$Port = 8100)
$ErrorActionPreference = "Stop"
Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" | ConvertTo-Json -Depth 6
