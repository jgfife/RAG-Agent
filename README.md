# CS 6300 RAG Agent

Minimal Retrieval-Augmented Generation indexing prototype. Current functionality: read all PDFs in a `documents/` directory, extract text, and (utility functions prepared) support loading structured data into a ChromaDB collection.

## 1. Environment Setup
```bash
make install            # creates .virtual_environment + installs deps
. .virtual_environment/bin/activate
```
System packages needed (apt): `python3.12-venv ffmpeg` (handled by `make install-deb`).

## 2. Add Documents
Place source PDFs in `documents/` at repo root (create if missing). Only `.pdf` files are processed.

## 3. Run Index Pipeline
```bash
make index              # or: python3 src/indexPipeline.py
```
Outputs extracted text for each PDF bracketed by BEGIN/END markers; returns a mapping (filename -> text).

## 4. Extending to Full RAG
Next steps you can implement: chunking, embedding via sentence-transformers, persisting to Chroma (`chromadb.Client().get_or_create_collection()` + `upsert`), retrieval + prompt assembly. See `load_data_to_chroma` helper for batch logic.

## 5. Dependencies (requirements.txt)
chromadb, sentence-transformers, numpy, requests, pypdf (or PyPDF2 fallback), pypdf/pdfminer.six for extraction.

## 6. Troubleshooting
Missing text? Ensure `pypdf` installed; large PDFs may produce empty strings for pages without extractable text. If neither `pypdf` nor `pdfminer.six` is present a RuntimeError is raised. Use `make clean` to clear any future vector store directory.

See AGENTS.md for coding standards and contribution guidelines.
