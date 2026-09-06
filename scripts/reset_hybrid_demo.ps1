param([int]$Port = 8100)
$ErrorActionPreference = "Stop"
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/reset" | ConvertTo-Json -Depth 8
