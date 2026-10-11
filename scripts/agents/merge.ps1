<#
.SYNOPSIS
    Merge every agent branch into main, then run the full check suite.
.PARAMETER Cleanup
    After a successful merge and green checks, remove the worktrees and delete the agent branches.
.EXAMPLE
    .\scripts\agents\merge.ps1
    .\scripts\agents\merge.ps1 -Cleanup
#>
param([switch]$Cleanup)
. "$PSScriptRoot\_common.ps1"
Set-Location $RepoRoot

if ((git branch --show-current) -ne 'main') { throw 'Run this from the main branch.' }
if (git status --porcelain) { throw 'Working tree has uncommitted changes. Commit or stash them first.' }

foreach ($name in $Agents) {
    $branch = "agent/$name"
    git rev-parse --verify --quiet $branch | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "[skip ] $name - no branch"; continue }
    if (-not (git log --oneline "main..$branch")) { Write-Host "[skip ] $name - nothing new"; continue }

    Write-Host "[merge] $branch"
    git merge --no-ff $branch -m "Merge $branch"
    if ($LASTEXITCODE -ne 0) {
        throw "Conflict merging $branch. Resolve it, run 'git commit', then rerun this script."
    }
}

Write-Host "`n[check] running the full suite on main"
Invoke-Native 'uv sync'      { uv sync --quiet }
Invoke-Native 'ruff format'  { uv run ruff format --check . }
Invoke-Native 'ruff check'   { uv run ruff check . }
Invoke-Native 'mypy'         { uv run mypy src tests }
Invoke-Native 'pytest'       { uv run pytest -q }
Write-Host '[check] all green'

if ($Cleanup) {
    foreach ($name in $Agents) {
        $wt = Join-Path $WorktreeRoot $name
        if (Test-Path $wt) { Invoke-Native "worktree remove ($name)" { git worktree remove $wt } }
        git rev-parse --verify --quiet "agent/$name" | Out-Null
        if ($LASTEXITCODE -eq 0) { Invoke-Native "branch delete ($name)" { git branch -d "agent/$name" } }
    }
    Write-Host '[clean] worktrees removed, branches deleted'
}

Write-Host "`nNext: integrate on main with:  claude 'Read scripts/agents/prompts/integrate.md and complete the task it describes.'"
