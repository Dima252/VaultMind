# Shared settings for the parallel-agent scripts. Dot-sourced, not run directly.

$ErrorActionPreference = 'Stop'

$RepoRoot     = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$WorktreeRoot = Join-Path (Split-Path $RepoRoot -Parent) 'VaultMind-agents'
$LogDir       = Join-Path $WorktreeRoot 'logs'

# One entry per agent; each has a prompt at scripts/agents/prompts/<name>.md and a branch agent/<name>.
$Agents = @('providers', 'ingest', 'store', 'eval')

function Get-AgentPrompt([string]$Name) {
    "Read scripts/agents/prompts/$Name.md and complete the task it describes."
}

function Invoke-Native([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}
