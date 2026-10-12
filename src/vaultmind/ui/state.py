"""App-wide state shared by every page: the config, the engine once loaded, ingestion events.

Engine loading and ingestion run on worker threads. Pages never get callbacks from those
threads; they poll this state with `ui.timer`, which keeps all UI updates on NiceGUI's loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vaultmind.app.engine import Engine
from vaultmind.core.config import Config
from vaultmind.ingest.queue import JobEvent

Phase = Literal["loading", "ready", "failed"]


@dataclass
class AppState:
    cfg: Config
    config_source: Path | None = None  # the YAML the config came from (None: built-in defaults)
    engine: Engine | None = None
    phase: Phase = "loading"
    message: str = "Starting"  # what's loading, or why it failed
    _jobs: dict[Path, JobEvent] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self) -> None:
        """Load the engine, warm the models up, start ingestion. Blocking: run it off the UI
        loop. Safe to call again after a failure (e.g. Ollama was down): it resumes."""
        self.phase = "loading"
        try:
            if self.engine is None:
                self.message = "Loading models (the reranker takes a few seconds)"
                self.engine = Engine.load(self.cfg)
            self.message = "Warming up Ollama"
            self.engine.warm_up()
            self.message = "Checking your documents"
            self.engine.start_ingest(self.on_job)
        except Exception as exc:
            self.message = str(exc) or type(exc).__name__
            self.phase = "failed"
            return
        self.message = "Ready"
        self.phase = "ready"

    def on_job(self, event: JobEvent) -> None:
        """IngestQueue callback (worker thread): keep the latest event per file."""
        with self._lock:
            self._jobs[event.path] = event

    def jobs(self) -> dict[Path, JobEvent]:
        """Latest ingestion event per file since the app started."""
        with self._lock:
            return dict(self._jobs)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None


_state: AppState | None = None


def init_state(cfg: Config, config_source: Path | None = None) -> AppState:
    global _state
    _state = AppState(cfg, config_source)
    return _state


def state() -> AppState:
    if _state is None:
        raise RuntimeError("app state not initialized; call init_state first")
    return _state
