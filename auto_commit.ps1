# Auto Commit and Push Script
# Run this script to continuously monitor and auto-commit changes

param(
    [int]$IntervalSeconds = 30,
    [string]$CommitMessage = "chore: auto-save changes",
    [switch]$Push
)

$RepoPath = "c:\ai-trading-system"
$LastCommit = ""

function Get-CurrentCommit {
    $result = git -C $RepoPath rev-parse HEAD 2>$null
    return $result
}

function Commit-Changes {
    $status = git -C $RepoPath status --porcelain 2>$null
    
    if ($status) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Changes detected, committing..."
        
        # Add all changes
        git -C $RepoPath add -A 2>$null
        
        # Get dynamic commit message with timestamp
        $dynamicMsg = "$CommitMessage - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        
        # Commit
        $commit = git -C $RepoPath commit -m "$dynamicMsg" 2>$null
        
        if ($commit) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Committed: $dynamicMsg"
            
            # Push if requested
            if ($Push) {
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Pushing to remote..."
                git -C $RepoPath push 2>$null
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Pushed successfully!"
            }
            
            return $true
        }
    }
    return $false
}

Write-Host "=========================================="
Write-Host "  Auto Commit Script Started"
Write-Host "  Interval: $IntervalSeconds seconds"
Write-Host "  Push: $($Push.ToString())"
Write-Host "=========================================="
Write-Host "Press Ctrl+C to stop"
Write-Host ""

$LastCommit = Get-CurrentCommit

# Main loop - continuously check for changes
while ($true) {
    Start-Sleep -Seconds $IntervalSeconds
    
    # Check if there are changes and commit them
    Commit-Changes
}

# Note: Press Ctrl+C to stop the script
</parameter>
</create_file>
