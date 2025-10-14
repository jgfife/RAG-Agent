# Agent Instructions for Vector Database Assignment

## Build/Test Commands
- `make install` - Install environment and dependencies (creates .virtual_environment)
- `make chroma` - Run the main ChromaDB demo (equivalent to `make chroma-demo`)
- `. .virtual_environment/bin/activate; python3 src/chroma.py` - Run chroma demo directly
- `make clean` - Delete contents of db/chroma folder for fresh start
- No test framework configured - run demos to verify functionality
- Individual modules can be tested with: `. .virtual_environment/bin/activate; python3 src/<module>.py`

## Code Style Guidelines
- Use Python 3.12+ with type hints (`from __future__ import annotations`)
- Follow PEP 8 formatting with 4-space indentation
- Use `snake_case` for variables/functions, `PascalCase` for classes
- Include docstrings for modules and main functions with Args/Returns sections
- Import order: standard library, third-party, local imports
- Use pathlib.Path for file operations, not string paths
- Use descriptive variable names (e.g., `dbDir`, `collection`, `query_result`)
- Add blank lines to separate logical sections
- Use f-strings for string formatting and descriptive print statements
- Include shebang `#!/usr/bin/env python3` for executable scripts
- Type hints: Use `str | Path` for path parameters, `List[Dict[str, Any]]` for data structures
- Handle encoding explicitly: use `encoding='utf-8'` for file operations
- Error handling: Use explicit exception handling where needed, especially for file I/O and external API calls

## Dependencies
- Core: chromadb>=0.5.0, numpy>=1.26, requests>=2.31.0
- ChromaDB uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Requires python3.12-venv and ffmpeg system packages (via `make install-deb`)

## Project Structure
- `src/` - Main source code (chroma.py, db.py, generate_queries.py)
- `db/chroma/` - ChromaDB persistent storage (auto-created)
- `dataset/` - Data files and generated queries
- Virtual environment in `.virtual_environment/`