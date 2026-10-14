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
    script_guard: bool = True
    max_answer_tokens: int = Field(default=2048, gt=0)


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
    smalltalk: bool = True  # nothing passed on an unclassified message: one short reply
    router: bool = True  # plan and route every message; false: every message is a lookup
    world_knowledge: bool = False  # true: labeled answers from the model's own knowledge (unwired)
    catalog_files: int = Field(default=60, ge=0)  # documents listed in the planner prompt
    max_steps: int = Field(default=6, gt=0)  # tool calls per open task
    observation_tokens: int = Field(default=2000, gt=0)  # per tool result shown to the model
    think_open: bool = True  # thinking on for open-task tool steps only (M3 loop, unwired)


class IngestConfig(_Section):
    watch: bool = True  # index changes in paths.vault_dir while the app runs
    debounce_s: float = Field(default=2.0, ge=0)  # wait for a file to stop changing first
    enrich: bool = True  # write a document card per file in the background, once per content


class UIConfig(_Section):
    native: bool = False  # true: own window (WebView2 on Windows phones Microsoft by itself)
    port: int = Field(default=8730, gt=0, lt=65536)  # served on 127.0.0.1 only
    privacy_check_s: float = Field(default=5.0, gt=0)  # privacy badge refresh interval


class Config(_Section):
    paths: PathsConfig = PathsConfig()
    ollama: OllamaConfig = OllamaConfig()
    models: ModelsConfig = ModelsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    agent: AgentConfig = AgentConfig()
    ingest: IngestConfig = IngestConfig()
    ui: UIConfig = UIConfig()


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
