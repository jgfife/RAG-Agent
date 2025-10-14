# AGENTS.md (Vector DB / RAG Project)

Build & Run:
1. Create env: `make install` (apt packages + pip deps)
2. Activate: `. .virtual_environment/bin/activate`
3. Index pipeline: `make index` (runs `src/indexPipeline.py`)
4. Chroma demo: `python3 src/chroma.py` (or add a make target if missing)
5. Clean DB: `make clean` (wipes `db/chroma/*`)
6. Module check (acts as “single test”): `python3 src/<module>.py`
7. No formal tests; treat each runnable module as validation.

Style & Conventions:
8. Python 3.12+, add `from __future__ import annotations` at top.
9. Imports grouped: stdlib, third-party, local; alphabetical within group.
10. Naming: `snake_case` funcs/vars, `PascalCase` classes, constants UPPER_SNAKE.
11. Use `pathlib.Path` for filesystem paths; type as `str | Path` in APIs.
12. Provide module + public function docstrings (Args / Returns / Raises).
13. Add shebang `#!/usr/bin/env python3` for executable scripts only.
14. Prefer f-strings; avoid bare prints—add context (e.g., prefix action/result).
15. Explicit encodings: `open(path, 'r', encoding='utf-8')`.
16. Error handling: catch specific exceptions (e.g., `FileNotFoundError`), re-raise with context; avoid silent except.*
17. Data structures: use `list[dict[str, Any]]` modern generics.
18. Keep functions < ~40 lines; refactor pipeline stages into helpers.
19. No Cursor / Copilot rule files present—nothing extra to enforce.
20. Keep dependencies minimal; do not pin beyond `requirements.txt` without need.
