"""Settings page: vault folder, models, retrieval/agent switches, saved to config/local.yaml."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nicegui import ui
from pydantic import ValidationError

from vaultmind.app.settings import needs_reindex, needs_restart, save_local
from vaultmind.core.config import DEFAULT_CONFIG, LOCAL_CONFIG, Config, load_config
from vaultmind.ui.badge import off_loop
from vaultmind.ui.state import AppState, state

LOCAL_PATH = LOCAL_CONFIG  # where Save writes (a module constant so tests can redirect it)

Kind = Literal["text", "path", "model", "switch", "int", "float"]


@dataclass(frozen=True)
class _Field:
    key: str  # "section.name" in Config
    label: str
    kind: Kind
    help: str
    step: float | None = None


# Help texts follow the comments in config/default.yaml.
_SECTIONS: list[tuple[str, list[_Field]]] = [
    (
        "Documents",
        [
            _Field("paths.vault_dir", "Vault folder", "path", "Watched folder for your documents"),
            _Field(
                "ingest.watch", "Watch the folder", "switch", "Index changes while the app runs"
            ),
            _Field(
                "ingest.debounce_s",
                "Debounce (seconds)",
                "float",
                "Wait for a file to stop changing before indexing it",
                0.5,
            ),
        ],
    ),
    (
        "Models",
        [
            _Field("models.llm", "LLM", "model", "The Ollama model that writes the answers"),
            _Field(
                "models.embedder",
                "Embedder",
                "model",
                "The Ollama embedding model; changing it means rebuilding the index",
            ),
            _Field(
                "models.reranker",
                "Reranker",
                "text",
                "Cross-encoder that grades passages (vaultmind pull-reranker downloads it)",
            ),
            _Field(
                "models.num_ctx",
                "Context window (num_ctx)",
                "int",
                "Ollama's default is small and silently truncates context",
                1024,
            ),
            _Field(
                "models.think",
                "Thinking",
                "switch",
                "Let the LLM reason before free-form answers (slower); never for JSON calls",
            ),
        ],
    ),
    (
        "Retrieval",
        [
            _Field("retrieval.hybrid", "Hybrid search", "switch", "Vector + BM25 fused with RRF"),
            _Field(
                "retrieval.rerank",
                "Rerank",
                "switch",
                "Grade candidates with the reranker (loaded at startup)",
            ),
            _Field("retrieval.candidates", "Candidates", "int", "Retrieved before reranking", 1),
            _Field("retrieval.top_k", "Top k", "int", "Passages passed to the generator", 1),
            _Field(
                "retrieval.rerank_threshold",
                "Rerank threshold",
                "float",
                "Reranker score below this = the passage fails grading",
                0.05,
            ),
            _Field(
                "retrieval.small_to_big",
                "Small to big",
                "switch",
                "Send each hit's parent section to the LLM, not the child chunk",
            ),
        ],
    ),
    (
        "Agent",
        [
            _Field(
                "agent.query_variants",
                "Query variants",
                "int",
                "Extra phrasings searched alongside the question (2: +recall, +1.2 s)",
                1,
            ),
            _Field(
                "agent.rewrite",
                "Rewrite",
                "switch",
                "When nothing passes grading, rewrite the query and search again",
            ),
            _Field("agent.max_rewrites", "Max rewrites", "int", "Rewrites per question", 1),
            _Field(
                "agent.injection_guard",
                "Injection guard",
                "switch",
                "Passages go in delimited data blocks; instruction-like text is flagged",
            ),
            _Field(
                "agent.detect_contradictions",
                "Detect contradictions",
                "switch",
                "One LLM call when sources span 2+ files: report conflicts, cite both",
            ),
            _Field(
                "agent.groundedness_check",
                "Groundedness check",
                "switch",
                "One LLM call: flag answer claims their cited passage doesn't support",
            ),
        ],
    ),
    (
        "Window",
        [
            _Field(
                "ui.native",
                "Native window",
                "switch",
                "Own window; off: open the default browser instead",
            ),
            _Field("ui.port", "Port", "int", "Served on 127.0.0.1 only, never on the network", 1),
            _Field(
                "ui.privacy_check_s",
                "Privacy check interval (seconds)",
                "float",
                "How often the privacy badge checks the app's connections",
                1,
            ),
        ],
    ),
]


def page() -> None:
    """Build the settings page. Works without an engine (e.g. to fix a model name)."""
    _Settings(state()).build()


class _Settings:
    def __init__(self, s: AppState) -> None:
        self.s = s
        self.baseline = _baseline(s)
        self.inputs: dict[str, Any] = {}  # dotted key -> element

    def build(self) -> None:
        values = _flatten(self.baseline.model_dump(mode="json"))
        with ui.column().classes("w-full max-w-3xl p-4 gap-4"):
            ui.label("Settings").classes("text-xl font-semibold")
            ui.label(_source_note(self.s.config_source)).classes("text-sm text-grey-7")
            for title, fields in _SECTIONS:
                with ui.card().classes("w-full"):
                    ui.label(title).classes("text-lg font-semibold")
                    for f in fields:
                        self.inputs[f.key] = self._field(f, values[f.key])
                    if title == "Models":
                        self.models_note = ui.label("Checking which models Ollama has").classes(
                            "text-xs text-grey-7"
                        )
            self.errors = ui.column().classes("gap-1")
            with ui.row().classes("gap-2"):
                ui.button("Save", icon="save", on_click=self._save)
                ui.button("Reset to defaults", on_click=self._reset).props("outline")
            self.messages = ui.column().classes("gap-1")
        ui.timer(0.01, self._load_model_names, once=True)

    def _field(self, f: _Field, value: Any) -> Any:
        if f.kind == "switch":
            with ui.column().classes("gap-0"):
                element: Any = ui.switch(f.label, value=bool(value))
                ui.label(f.help).classes("text-xs text-grey-7 ml-12")
        else:
            if f.kind in ("int", "float"):
                element = ui.number(f.label, value=value, step=f.step)
            else:
                element = ui.input(f.label, value=str(value))
            element.props["hint"] = f.help
            element.classes("w-full")
        return element.mark(f.key)

    async def _load_model_names(self) -> None:
        try:
            names = await off_loop(list_ollama_models, self.s.cfg.ollama.host)
        except Exception:
            self.models_note.text = "Ollama isn't reachable; type the model names."
            return
        if not names:
            self.models_note.text = "Ollama has no models pulled yet; type the model names."
            return
        for _, fields in _SECTIONS:
            for f in fields:
                if f.kind == "model":
                    self.inputs[f.key].set_autocomplete(names)
        self.models_note.text = "Pulled in Ollama: " + ", ".join(names)

    def _collect(self) -> dict[str, Any]:
        """The baseline config with the form's values on top (fields not on the page kept)."""
        data = copy.deepcopy(self.baseline.model_dump(mode="json"))
        kinds = {f.key: f.kind for _, fields in _SECTIONS for f in fields}
        for key, element in self.inputs.items():
            section, name = key.split(".", 1)
            value = element.value
            if kinds[key] == "int" and isinstance(value, float) and value.is_integer():
                value = int(value)
            elif isinstance(value, str):
                value = value.strip()
            data[section][name] = value
        return data

    def _clear_errors(self) -> None:
        self.errors.clear()
        self.messages.clear()
        for element in self.inputs.values():
            if hasattr(element, "error"):
                element.error = None

    def _show_error(self, key: str, message: str) -> None:
        element = self.inputs.get(key)
        if element is not None and hasattr(element, "error"):
            element.error = message
        with self.errors:
            ui.label(f"{key}: {message}").classes("text-negative")

    def _say(self, text: str, kind: str = "info") -> None:
        color = {"warning": "text-warning", "positive": "text-positive"}.get(kind, "")
        with self.messages:
            ui.label(text).classes(color)

    async def _save(self) -> None:
        self._clear_errors()
        try:
            new = Config.model_validate(self._collect())
        except ValidationError as exc:
            for err in exc.errors():
                self._show_error(".".join(str(p) for p in err["loc"]), err["msg"])
            ui.notify("Some settings are invalid; nothing was saved.", type="negative")
            return
        if problem := await off_loop(_check_vault, new.paths.vault_dir):
            self._show_error("paths.vault_dir", problem)
            ui.notify("Some settings are invalid; nothing was saved.", type="negative")
            return
        try:
            save_local(new, LOCAL_PATH)
        except OSError as exc:
            ui.notify(f"Couldn't save {LOCAL_PATH}: {exc}", type="negative")
            return
        ui.notify(f"Saved to {LOCAL_PATH}", type="positive")
        self._report(new)

    def _report(self, new: Config) -> None:
        """Say what the save means for the running app, and apply what can be applied now."""
        s, old = self.s, self.baseline
        if not _uses_local(s.config_source):
            self._say(
                f"VaultMind was started with {s.config_source}, so {LOCAL_PATH} won't be used "
                "until you launch it without --config or VAULTMIND_CONFIG.",
                "warning",
            )
        if needs_reindex(old, new):
            self._say(
                "The embedder changed, so the index must be rebuilt: close VaultMind, delete "
                f"{new.paths.data_dir / 'index'} and start it again.",
                "warning",
            )
        restart = needs_restart(old, new)
        if s.engine is None and s.phase == "failed":
            # Retry builds the engine from state().cfg, so model and path fixes apply there.
            s.cfg = self.baseline = new
            window = [k for k in restart if k.startswith("ui.")]
            self._say("Press Retry to start VaultMind with these settings.", "positive")
            if window:
                self._say("Restart VaultMind to apply: " + ", ".join(window), "warning")
        elif s.phase == "loading":
            self._say("VaultMind is still starting; restart it to apply the changes.", "warning")
        elif restart:
            self._say("Restart VaultMind to apply: " + ", ".join(restart), "warning")
        else:
            s.cfg = self.baseline = new
            if s.engine is not None:
                s.engine.cfg = new  # the next question reads it
            self._say("Applied. The next question uses these settings.", "positive")

    async def _reset(self) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label("Reset every setting on this page to its default?")
            ui.label("Nothing is saved until you press Save.").classes("text-sm text-grey-7")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                ui.button("Reset", on_click=lambda: dialog.submit(True)).mark("confirm")
        confirmed = await dialog
        dialog.delete()
        if not confirmed:
            return
        self._clear_errors()
        defaults = _flatten(Config().model_dump(mode="json"))
        for key, element in self.inputs.items():
            value = defaults[key]
            element.value = value if isinstance(value, bool | int | float) else str(value)
        self._say("Defaults filled in. Press Save to keep them.")


def list_ollama_models(host: str) -> list[str]:
    """Names of the models Ollama has pulled (a localhost call; raises when it's down)."""
    import ollama

    listed = ollama.Client(host=host, timeout=5).list()
    return sorted({m.model for m in listed.models if m.model})


def _baseline(s: AppState) -> Config:
    """The running config, except `ui.native` as the config file has it: `--browser` turns it
    off for this run only, and saving shouldn't make that permanent."""
    try:
        native = load_config(s.config_source).ui.native if s.config_source else Config().ui.native
    except Exception:
        return s.cfg
    return s.cfg.model_copy(update={"ui": s.cfg.ui.model_copy(update={"native": native})})


def _uses_local(source: Path | None) -> bool:
    """Whether the next launch reads local.yaml: unless an explicit file overrides it."""
    if source is None:
        return True
    return any(_same(source, p) for p in (LOCAL_PATH, DEFAULT_CONFIG))


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _source_note(source: Path | None) -> str:
    where = source if source is not None else "the built-in defaults"
    return f"Loaded from {where}. Save writes what differs from the defaults to {LOCAL_PATH}."


def _check_vault(path: Path) -> str | None:
    """None when the vault folder exists or could be created, else what's wrong."""
    if not str(path).strip() or path == Path("."):
        return "Choose a folder for your documents"
    path = path.expanduser()
    if path.exists():
        return None if path.is_dir() else "This is a file, not a folder"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"Can't create this folder: {exc}"
    return None


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    return {f"{s}.{k}": v for s, section in data.items() for k, v in section.items()}
