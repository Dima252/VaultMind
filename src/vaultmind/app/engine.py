"""What the desktop app keeps loaded for its lifetime: models, index, stores, background ingestion.

Load it once at startup (the reranker alone takes ~7 s), then every question reuses it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

from vaultmind.agent.graph import stream_agent
from vaultmind.agent.types import AgentResult
from vaultmind.core.config import Config
from vaultmind.core.providers import LLM, Embedder, Reranker
from vaultmind.ingest.queue import IngestQueue, JobEvent
from vaultmind.ingest.watcher import VaultWatcher
from vaultmind.store.chats import ChatMessage, ChatStore
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry
from vaultmind.store.traces import TraceStore


class Engine:
    def __init__(
        self,
        cfg: Config,
        *,
        llm: LLM,
        embedder: Embedder,
        reranker: Reranker | None,
        index: ChunkIndex,
        registry: Registry,
        traces: TraceStore,
        chats: ChatStore,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.index = index
        self.registry = registry
        self.traces = traces
        self.chats = chats
        self.queue: IngestQueue | None = None
        self.watcher: VaultWatcher | None = None
        self._ask_lock = threading.Lock()

    @classmethod
    def load(cls, cfg: Config) -> Engine:
        """Open the stores and load the models. Raises `EmbedderMismatchError` and
        `RerankerUnavailableError` with messages fit for the user."""
        from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM

        reranker: Reranker | None = None
        if cfg.retrieval.rerank:
            from vaultmind.core.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker.from_config(cfg)
        index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=cfg.models.embedder)
        db = cfg.paths.data_dir / "vaultmind.sqlite"
        return cls(
            cfg,
            llm=OllamaLLM.from_config(cfg),
            embedder=OllamaEmbedder.from_config(cfg),
            reranker=reranker,
            index=index,
            registry=Registry(db),
            traces=TraceStore(db),
            chats=ChatStore(db),
        )

    def warm_up(self) -> None:
        """Load the LLM and embedder into memory now, so the first question isn't slow.
        Raises `OllamaUnavailableError` when Ollama is down."""
        self.embedder.embed(["warm up"])
        self.llm.generate([{"role": "user", "content": "Reply with the single word: ok"}])

    def ask(self, chat_id: str, question: str) -> Iterator[str | ChatMessage]:
        """Answer in the context of a chat: yield answer tokens, then the stored assistant
        message. The run and the exchange are saved before it's yielded. One question at a
        time; on an error nothing is saved."""
        with self._ask_lock:
            result: AgentResult | None = None
            events = stream_agent(
                question,
                self.cfg,
                llm=self.llm,
                embedder=self.embedder,
                index=self.index,
                registry=self.registry,
                reranker=self.reranker,
                history=self.chats.history(chat_id),
            )
            for event in events:
                if isinstance(event, AgentResult):
                    result = event
                else:
                    yield event
            if result is None:
                raise RuntimeError("the agent ended without a result")
            run_id = self.traces.save(result)
            _, answer = self.chats.add_exchange(chat_id, result, run_id)
        yield answer

    def start_ingest(self, on_event: Callable[[JobEvent], None] | None = None) -> None:
        """Start background ingestion: catch up with the vault, then watch it (`ingest.watch`)."""
        if self.queue is not None:
            return
        self.queue = IngestQueue(
            self.cfg,
            embedder=self.embedder,
            index=self.index,
            registry=self.registry,
            on_event=on_event,
        )
        self.queue.start()
        self.queue.scan()
        if self.cfg.ingest.watch:
            self.cfg.paths.vault_dir.mkdir(parents=True, exist_ok=True)
            self.watcher = VaultWatcher(
                self.cfg.paths.vault_dir, self.queue, debounce_s=self.cfg.ingest.debounce_s
            )
            self.watcher.start()

    def close(self) -> None:
        """Stop background work and close the stores. Idempotent."""
        if self.watcher is not None:
            self.watcher.stop()
            self.watcher = None
        if self.queue is not None:
            self.queue.stop()
            self.queue = None
        for store in (self.chats, self.traces, self.registry):
            store.close()
        self.index.close()
