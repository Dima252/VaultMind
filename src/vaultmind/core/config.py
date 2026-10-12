"""Typed configuration loaded from YAML.

Resolution order: explicit path > $VAULTMIND_CONFIG > config/local.yaml > config/default.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG = Path("config/default.yaml")
LOCAL_CONFIG = Path("config/local.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    vault_dir: Path = Path("./vault")
    data_dir: Path = Path("./data")


class OllamaConfig(_Section):
    host: str = "http://127.0.0.1:11434"
    keep_alive: str = "30m"


class ModelsConfig(_Section):
    llm: str = "qwen3:14b"
    embedder: str = "qwen3-embedding:0.6b"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    num_ctx: int = Field(default=16384, gt=0)
    think: bool = False


class ChunkingConfig(_Section):
    child_tokens: int = Field(default=256, gt=0)
    parent_tokens: int = Field(default=1200, gt=0)
    overlap: float = Field(default=0.15, ge=0, lt=1)


class RetrievalConfig(_Section):
    hybrid: bool = True
    candidates: int = Field(default=20, gt=0)
    top_k: int = Field(default=5, gt=0)
    rerank: bool = True
    rerank_threshold: float = 0.3
    small_to_big: bool = False  # parents measured worse on CUAD (docs/RESULTS.md, Phase 3)


class AgentConfig(_Section):
    loop: bool = True
    query_variants: int = Field(default=0, ge=0)  # measured: +recall, no answer gain, +1.2 s
    rewrite: bool = False  # measured: recovered 0 of 4 rewritten questions (RESULTS.md)
    max_rewrites: int = Field(default=2, ge=0)
    injection_guard: bool = True
    detect_contradictions: bool = True  # runs only when sources span 2+ files
    groundedness_check: bool = False


class Config(_Section):
    paths: PathsConfig = PathsConfig()
    ollama: OllamaConfig = OllamaConfig()
    models: ModelsConfig = ModelsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    agent: AgentConfig = AgentConfig()


def resolve_config_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return path
    if env := os.environ.get("VAULTMIND_CONFIG"):
        return Path(env)
    for candidate in (LOCAL_CONFIG, DEFAULT_CONFIG):
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path | None = None) -> Config:
    resolved = resolve_config_path(path)
    if resolved is None:
        return Config()
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
