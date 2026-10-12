"""Architecture rules: only `ui/` touches NiceGUI, and the engine never imports the UI."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "vaultmind"
UI_LIBS = ("nicegui", "webview")


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _engine_modules() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "ui" not in p.relative_to(SRC).parts]


def test_only_ui_imports_nicegui() -> None:
    offenders = [
        str(p.relative_to(SRC))
        for p in _engine_modules()
        if any(name.split(".")[0] in UI_LIBS for name in _imports(p))
    ]
    assert offenders == []


def test_engine_never_imports_ui() -> None:
    allowed = {SRC / "cli.py"}  # `vaultmind ui` launches it
    offenders = [
        str(p.relative_to(SRC))
        for p in _engine_modules()
        if p not in allowed and any(n.startswith("vaultmind.ui") for n in _imports(p))
    ]
    assert offenders == []
