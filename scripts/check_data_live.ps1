param(
    [string]$BaseUrl = "http://localhost:8000",
    [int]$WaitSeconds = 5
)

$ErrorActionPreference = "Stop"

function Get-Json($url) {
    return Invoke-RestMethod -Uri $url -TimeoutSec 20
}

function Fmt-Num($v) {
    if ($null -eq $v) { return "null" }
    return ("{0:N6}" -f [double]$v)
}

Write-Host "=== DATA LIVE CHECK ==="
Write-Host "Base URL: $BaseUrl"
Write-Host "Wait: $WaitSeconds s"
Write-Host ""

$mode = Get-Json "$BaseUrl/api/v1/portfolio/mode"
Write-Host ("demo_mode: {0}" -f $mode.demo_mode)
Write-Host ""

$summary1 = Get-Json "$BaseUrl/api/v1/portfolio/summary"
$perf1 = Get-Json "$BaseUrl/api/v1/portfolio/performance"
$market1 = Get-Json "$BaseUrl/api/v1/market/prices"

Start-Sleep -Seconds $WaitSeconds

$summary2 = Get-Json "$BaseUrl/api/v1/portfolio/summary"
$perf2 = Get-Json "$BaseUrl/api/v1/portfolio/performance"
$market2 = Get-Json "$BaseUrl/api/v1/market/prices"

$btc1 = ($market1.markets | Where-Object { $_.symbol -eq "BTC/USDT" } | Select-Object -First 1).price
$btc2 = ($market2.markets | Where-Object { $_.symbol -eq "BTC/USDT" } | Select-Object -First 1).price

Write-Host "--- Portfolio Summary ---"
Write-Host ("total_value:       {0} -> {1}" -f (Fmt-Num $summary1.total_value), (Fmt-Num $summary2.total_value))
Write-Host ("unrealized_pnl:    {0} -> {1}" -f (Fmt-Num $summary1.unrealized_pnl), (Fmt-Num $summary2.unrealized_pnl))
Write-Host ("daily_return_pct:  {0} -> {1}" -f (Fmt-Num $summary1.daily_return_pct), (Fmt-Num $summary2.daily_return_pct))
Write-Host ("total_return_pct:  {0} -> {1}" -f (Fmt-Num $summary1.total_return_pct), (Fmt-Num $summary2.total_return_pct))
Write-Host ""

Write-Host "--- Performance Metrics ---"
Write-Host ("sharpe_ratio:      {0} -> {1}" -f (Fmt-Num $perf1.sharpe_ratio), (Fmt-Num $perf2.sharpe_ratio))
Write-Host ("win_rate:          {0} -> {1}" -f (Fmt-Num $perf1.win_rate), (Fmt-Num $perf2.win_rate))
Write-Host ("max_drawdown_pct:  {0} -> {1}" -f (Fmt-Num $perf1.max_drawdown_pct), (Fmt-Num $perf2.max_drawdown_pct))
Write-Host ""

Write-Host "--- Market Feed ---"
Write-Host ("BTC/USDT price:    {0} -> {1}" -f (Fmt-Num $btc1), (Fmt-Num $btc2))
Write-Host ("market timestamp:  {0} -> {1}" -f $market1.timestamp, $market2.timestamp)
Write-Host ""

$summaryChanged = ($summary1.total_value -ne $summary2.total_value) -or ($summary1.unrealized_pnl -ne $summary2.unrealized_pnl)
$perfChanged = ($perf1.sharpe_ratio -ne $perf2.sharpe_ratio) -or ($perf1.win_rate -ne $perf2.win_rate) -or ($perf1.max_drawdown_pct -ne $perf2.max_drawdown_pct)
$marketChanged = ($btc1 -ne $btc2)

Write-Host "=== RESULT ==="
Write-Host ("summary_dynamic:   {0}" -f $summaryChanged)
Write-Host ("performance_fixed: {0}" -f (-not $perfChanged))
Write-Host ("market_dynamic:    {0}" -f $marketChanged)

if ($mode.demo_mode -eq $true) {
    Write-Host ""
    Write-Host "NOTE: demo_mode=true -> molti KPI performance sono simulati/fissi."
}
