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

## 4. Run Agent Pipeline
Index (build/rebuild) and query are handled by the retrieval pipeline. Use either Make targets or direct script invocation:
```bash
make index      # build (passes -b to force rebuild if collection empty or requested)
make agent      # query using existing index
# OR directly:
python3 src/retrievalPipeline.py -b   # rebuild then prompt for a question
python3 src/retrievalPipeline.py      # reuse existing index
```
Example:
```bash
make agent
Enter your question (blank to exit): What is retrieval augmented generation?
```
To reset state: `make clean` then rebuild.

## 5. Dependencies
See `requirements.txt`: chromadb, sentence-transformers, numpy, requests, pypdf, pinecone (future use). Optional: install `pdfminer.six` for fallback extraction.

## 6. Troubleshooting
- Missing answers: ensure Ollama server running (`ollama run llama3.1` pulls model).
- Empty pages: some PDFs lack extractable text; page skipped.
- Vector store mismatch: run `make clean` then `make index`.
- Import errors: activate venv (`. .virtual_environment/bin/activate`).

## 7. Standards
Refer to `AGENTS.md` for code style (imports grouping, typing, error handling, docstrings, path usage). Keep additions minimal & typed.
