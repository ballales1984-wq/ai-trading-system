#!/usr/bin/env pwsh
# Auto-commit and push script for AI Trading System
# Automatically names commits based on changed files and pushes to remote

$ErrorActionPreference = "Stop"

function Get-AutoCommitMessage {
    $status = git status --porcelain
    if (-not $status) {
        Write-Host "No changes to commit" -ForegroundColor Yellow
        return $null
    }
    
    $files = $status | ForEach-Object { $_.Substring(3) }
    $fileTypes = @{
        "app" = @()
        "frontend" = @()
        "dashboard" = @()
        "api" = @()
        "config" = @()
        "docs" = @()
        "security" = @()
        "tests" = @()
        "other" = @()
    }
    
    foreach ($file in $files) {
        if ($file -match "^app/") { $fileTypes["app"] += $file }
        elseif ($file -match "^frontend/") { $fileTypes["frontend"] += $file }
        elseif ($file -match "^dashboard/") { $fileTypes["dashboard"] += $file }
        elseif ($file -match "^api/") { $fileTypes["api"] += $file }
        elseif ($file -match "\.(md|txt|rst)$" -and $file -notmatch "^app/") { $fileTypes["docs"] += $file }
        elseif ($file -match "security|scan|audit" -or $file -match "security_scan\.json$") { $fileTypes["security"] += $file }
        elseif ($file -match "^tests/|_test\.py$|test_\.py$") { $fileTypes["tests"] += $file }
        elseif ($file -match "\.env|config\.|settings\.") { $fileTypes["config"] += $file }
        else { $fileTypes["other"] += $file }
    }
    
    # Build commit message based on changes
    $scope = ""
    $type = "chore"
    
    if ($fileTypes["security"].Count -gt 0) {
        $type = "security"
        $scope = "update security scan results"
    }
    elseif ($fileTypes["docs"].Count -gt 0) {
        $type = "docs"
        $scope = "update documentation"
    }
    elseif ($fileTypes["app"].Count -gt 0 -and $fileTypes["frontend"].Count -gt 0) {
        $type = "feat"
        $scope = "update app and frontend"
    }
    elseif ($fileTypes["app"].Count -gt 0) {
        $type = "refactor"
        $scope = "update backend app"
    }
    elseif ($fileTypes["frontend"].Count -gt 0) {
        $type = "feat"
        $scope = "update frontend"
    }
    elseif ($fileTypes["dashboard"].Count -gt 0) {
        $type = "feat"
        $scope = "update dashboard"
    }
    elseif ($fileTypes["tests"].Count -gt 0) {
        $type = "test"
        $scope = "update tests"
    }
    else {
        $scope = "update various files"
    }
    
    $fileCount = $files.Count
    $details = ($files | Select-Object -First 3) -join ", "
    if ($fileCount -gt 3) {
        $details += " and $($fileCount - 3) more"
    }
    
    $message = "$type`: $scope`n`n- Files changed: $details"
    
    return $message
}

function Start-AutoCommitPush {
    Write-Host "`n=== AI Trading System Auto-Commit & Push ===" -ForegroundColor Cyan
    
    # Check if there are changes
    $status = git status --porcelain
    if (-not $status) {
        Write-Host "No changes to commit. Working directory is clean." -ForegroundColor Green
        return
    }
    
    Write-Host "`nChanged files:" -ForegroundColor Yellow
    $status | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    
    # Get commit message
    $commitMessage = Get-AutoCommitMessage
    if (-not $commitMessage) {
        return
    }
    
    Write-Host "`nCommit message:" -ForegroundColor Yellow
    Write-Host $commitMessage -ForegroundColor White
    
    # Stage all changes
    Write-Host "`nStaging all changes..." -ForegroundColor Yellow
    git add -A
    
    # Create commit
    Write-Host "Creating commit..." -ForegroundColor Yellow
    git commit -m $commitMessage
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create commit" -ForegroundColor Red
        return
    }
    
    Write-Host "Commit created successfully" -ForegroundColor Green
    
    # Push to remote
    Write-Host "Pushing to remote..." -ForegroundColor Yellow
    git push
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to push to remote" -ForegroundColor Red
        return
    }
    
    Write-Host "`n=== Successfully committed and pushed! ===" -ForegroundColor Green
    
    # Show recent commits
    Write-Host "`nRecent commits:" -ForegroundColor Cyan
    git log --oneline -3
}

# Run the auto-commit
Start-AutoCommitPush
