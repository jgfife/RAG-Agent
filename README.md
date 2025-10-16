# CS6300 RAG Agent

Retrieval-Augmented Generation prototype indexing academic PDFs into a persistent ChromaDB collection and answering questions via local Ollama (LLM) using contextual chunks.

## 1. Features
- PDF extraction with pypdf (fallback: pdfminer.six if added manually)
- Sentence-aware chunking + overlap (config constants in `indexPipeline.py`)
- Batch upsert to Chroma (persistent under `./db`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Interactive retrieval + prompt construction (`retrievalPipeline.py`)
- Local Ollama completion (HTTP `localhost:11434`)

## 2. Environment Setup
```bash
make install              # creates .virtual_environment + installs pip deps
. .virtual_environment/bin/activate
```
System packages (apt): `python3.12-venv ffmpeg` (auto via `make install-deb`).

## 3. Documents
Place source PDFs in `documents/` (repo root). Only `.pdf` processed; empty dir yields a notice.

## 4. Build / Query
```bash
make index                # rebuild index (forces -b) then prompt for a question
make agent                # reuse existing index, just prompt
make clean                # remove db/* (must rebuild afterward)
```
Example session:
```bash
make index
Enter your question (blank to exit): What is retrieval augmented generation?
```

## 5. Direct Scripts
- Retrieval / interactive: `python3 src/retrievalPipeline.py [-b]`
- (Index helper only used programmatically; do not run `indexPipeline.py` standalone.)

## 6. Lightweight Checks
```bash
python -m py_compile src/*.py         # syntax
ruff check src                        # optional if ruff installed
python3 src/retrievalPipeline.py -b   # pseudo-test (build + query prompt)
```

## 7. Dependencies
See `requirements.txt`: chromadb, sentence-transformers, numpy, requests, pypdf, pinecone (future use). Optional: install `pdfminer.six` for fallback extraction.

## 8. Troubleshooting
- Missing answers: ensure Ollama server running (`ollama run llama3.1` pulls model).
- Empty pages: some PDFs lack extractable text; page skipped.
- Vector store mismatch: run `make clean` then `make index`.
- Import errors: activate venv (`. .virtual_environment/bin/activate`).

## 9. Standards
Refer to `AGENTS.md` for code style (imports grouping, typing, error handling, docstrings, path usage). Keep additions minimal & typed.
