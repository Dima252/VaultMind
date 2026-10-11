"""Cross-encoder reranker (sentence-transformers) implementing the `Reranker` protocol.

Offline: the model is loaded from the local Hugging Face cache only. Downloading it is an
explicit setup step (`vaultmind pull-reranker`), never a side effect of answering a question.
Needs the `rerank` extra: `uv sync --extra rerank`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from vaultmind.core.config import Config

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MAX_LENGTH = 512  # query + one child chunk (~256 tokens) fits with room to spare


class RerankerUnavailableError(Exception):
    """The reranker can't be loaded: the extra isn't installed or the model isn't downloaded."""


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        # Read when huggingface_hub is first imported: never touch the network while answering,
        # and no progress bar when loading weights.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailableError(
                "The reranker needs the 'rerank' extra. Run: uv sync --extra rerank"
            ) from exc

        cuda = torch.cuda.is_available()
        try:
            self._model: Any = CrossEncoder(
                model_name,
                device="cuda" if cuda else "cpu",
                local_files_only=True,
                max_length=MAX_LENGTH,
                model_kwargs={"dtype": torch.float16} if cuda else None,
            )
        except OSError as exc:
            raise RerankerUnavailableError(
                f"Reranker {model_name!r} is not downloaded. Run: vaultmind pull-reranker"
            ) from exc
        self.model_name = model_name
        self._batch_size = batch_size

    @classmethod
    def from_config(cls, cfg: Config) -> CrossEncoderReranker:
        return cls(cfg.models.reranker)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Sigmoid relevance in [0, 1] per passage, same order as `passages`."""
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages], batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


def is_downloaded(model_name: str) -> bool:
    """True if the model's files are in the local Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model_name, "config.json"), str)


def download(model_name: str) -> str:
    """Fetch the model into the Hugging Face cache (the only network step). Returns its path."""
    from huggingface_hub import snapshot_download

    weights_and_tokenizer = ["*.json", "*.safetensors", "*.model", "*.txt"]
    return str(snapshot_download(model_name, allow_patterns=weights_and_tokenizer))
