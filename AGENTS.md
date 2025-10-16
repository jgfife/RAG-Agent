# AGENTS.md (Vector DB / RAG Project)

Build & Run:
1. Create venv: `make install` then activate: `. .virtual_environment/bin/activate`.
2. Index + retrieval: `make index` (build) then `make agent` (query). Add `-b` to force rebuild.
3. Chroma demo: `python3 src/chroma.py` (if present) else skip.
4. Clean DB: `make clean` (removes `db/*`). Rebuild after cleaning.
5. Single module validation (pseudo-test): `python3 src/<module>.py` (e.g. `python3 src/indexPipeline.py`).
6. Lightweight syntax check: `python -m py_compile src/*.py`; optional lint if `ruff` installed: `ruff check src`.

Style & Conventions:
7. Python 3.12+; first line (non-shebang) `from __future__ import annotations`.
8. Imports grouped: stdlib, third-party, local; alphabetical within each group.
9. Naming: snake_case functions/vars, PascalCase classes, UPPER_SNAKE constants.
10. Use `pathlib.Path`; accept `str | Path` in public APIs; avoid bare `os.path`.
11. Docstrings (module + public funcs) include Args / Returns / Raises; prefer precise types (`list[dict[str, Any]]`).
12. Error handling: catch specific exceptions, re-raise with context; never silent `except:`.
13. Output: prefer contextual f-strings; avoid bare prints (prefix action/result); shebang only for direct executables.
14. Encoding: always explicit when opening text; binary only for PDFs (`open(path, 'rb')`).
15. Dependencies: keep minimal; no Cursor/Copilot rule files present; follow this guide for agents.
