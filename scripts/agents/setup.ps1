<#
.SYNOPSIS
    Create one git worktree + branch per agent, each with its own virtualenv.
.EXAMPLE
    .\scripts\agents\setup.ps1
#>
. "$PSScriptRoot\_common.ps1"
Set-Location $RepoRoot

git rev-parse --verify --quiet HEAD | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'No commits yet. Commit the scaffold first so every agent starts from the same base.' }
if (git status --porcelain) { throw 'Working tree has uncommitted changes. Commit them first; agents only see committed code.' }

if (-not $Agents) { throw 'No agents listed. Set $Agents in scripts/agents/_common.ps1 first.' }

New-Item -ItemType Directory -Force $WorktreeRoot, $LogDir | Out-Null

foreach ($name in $Agents) {
    $wt = Join-Path $WorktreeRoot $name
    if (Test-Path $wt) {
        Write-Host "[skip] $name - worktree already exists at $wt"
        continue
    }
    Write-Host "[new ] $name -> $wt (branch agent/$name)"
    Invoke-Native "git worktree add ($name)" { git worktree add -b "agent/$name" $wt HEAD }
    Push-Location $wt
    # Plain sync (no torch); each agent runs `uv sync --extra rerank` itself (rule 0 in prompts/_rules.md).
    try { Invoke-Native "uv sync ($name)" { uv sync --quiet } } finally { Pop-Location }
}

Write-Host "`nReady. Next: .\scripts\agents\start.ps1            (one window per agent)"
Write-Host "        or: .\scripts\agents\start.ps1 -Headless  (background, logs in $LogDir)"
