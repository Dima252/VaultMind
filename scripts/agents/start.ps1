<#
.SYNOPSIS
    Launch Claude Code in every agent worktree.
.DESCRIPTION
    Default: opens one PowerShell window per agent with an interactive Claude session, so you can
    watch each one and approve commands.
    -Headless: runs every agent in the background with `claude -p`, waits for all of them, and
    writes each agent's output to the logs folder. Edits are auto-accepted, and only the listed
    uv/git commands are pre-approved; anything else is denied, not prompted.
.PARAMETER Only
    Run just these agents, e.g. -Only store,eval
.EXAMPLE
    .\scripts\agents\start.ps1
    .\scripts\agents\start.ps1 -Headless
    .\scripts\agents\start.ps1 -Only ingest
#>
param(
    [switch]$Headless,
    [string[]]$Only
)
. "$PSScriptRoot\_common.ps1"

$selected = if ($Only) { $Only } else { $Agents }
foreach ($name in $selected) {
    if ($Agents -notcontains $name) { throw "Unknown agent '$name'. Known: $($Agents -join ', ')" }
    if (-not (Test-Path (Join-Path $WorktreeRoot $name))) { throw "No worktree for '$name'. Run setup.ps1 first." }
}

if (-not $Headless) {
    foreach ($name in $selected) {
        $wt = Join-Path $WorktreeRoot $name
        $prompt = Get-AgentPrompt $name
        Start-Process powershell -WorkingDirectory $wt -ArgumentList @(
            '-NoExit', '-Command',
            "`$Host.UI.RawUI.WindowTitle = 'agent: $name'; claude '$prompt'"
        )
        Write-Host "[open] $name"
    }
    Write-Host "`nWhen every agent has committed: .\scripts\agents\merge.ps1"
    return
}

$allowedTools = @(
    'Read', 'Edit', 'Write', 'Glob', 'Grep',
    'Bash(uv run:*)', 'Bash(uv sync:*)',
    'Bash(git status:*)', 'Bash(git diff:*)', 'Bash(git log:*)',
    'Bash(git add:*)', 'Bash(git commit:*)'
)

New-Item -ItemType Directory -Force $LogDir | Out-Null
$jobs = foreach ($name in $selected) {
    $wt = Join-Path $WorktreeRoot $name
    $log = Join-Path $LogDir "$name.log"
    Write-Host "[run ] $name  (log: $log)"
    Start-Job -Name "agent-$name" -ArgumentList $wt, (Get-AgentPrompt $name), $log, $allowedTools -ScriptBlock {
        param($wt, $prompt, $log, $allowedTools)
        Set-Location $wt
        & claude -p $prompt --permission-mode acceptEdits --allowedTools @allowedTools *>&1 |
            Out-File -FilePath $log -Encoding utf8
        $LASTEXITCODE
    }
}

Write-Host "`nWaiting for $($jobs.Count) agents... (Ctrl+C stops waiting; the jobs keep running)"
$jobs | Wait-Job | Out-Null

foreach ($job in $jobs) {
    $name = $job.Name -replace '^agent-', ''
    $exit = Receive-Job $job
    $wt = Join-Path $WorktreeRoot $name
    $commits = git -C $wt log --oneline "main..agent/$name"
    $status = if ($exit -eq 0 -and $commits) { 'done' } elseif ($exit -eq 0) { 'NO COMMIT' } else { "FAILED (exit $exit)" }
    Write-Host ("[{0,-10}] {1}" -f $status, $name)
    Remove-Job $job
}
Write-Host "`nRead the summaries in $LogDir, then: .\scripts\agents\merge.ps1"
