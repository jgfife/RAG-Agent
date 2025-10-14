from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, List, Any
import time
import io

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings

PDF_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "documents"

def extractPdfText(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract text and basic metadata from a PDF.

    Returns a dict with keys:
        text: full concatenated text
        pages: list of {index, text, char_count}
        meta: document-level metadata (source_name, source_path, file_size_bytes, modified_time, page_count)
    """
    pdf_path = Path(pdf_path)

    try:
        with open(pdf_path, "rb") as fh:
            file_bytes = fh.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"PDF not found: {pdf_path}") from e

    pages: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    try:
        try:
            from pypdf import PdfReader  # preferred modern package
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(file_bytes))
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            pages.append({
                "index": idx,
                "text": page_text,
                "char_count": len(page_text),
            })
            full_text_parts.append(page_text)
    except ImportError:
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
            overall = pdfminer_extract_text(str(pdf_path))
            pages = [{"index": 0, "text": overall, "char_count": len(overall)}]
            full_text_parts = [overall]
        except ImportError as e:
            raise RuntimeError(
                "No PDF extraction library available. Install 'pypdf' or 'pdfminer.six'."
            ) from e

    full_text = "\n".join(full_text_parts)
    stat = pdf_path.stat()
    meta = {
        "source_name": pdf_path.name,
        "file_size_bytes": stat.st_size,
        "page_count": len(pages),
    }
    return {"text": full_text, "pages": pages, "meta": meta}

def loadDataToChroma(
    collection: chromadb.Collection, 
    data: List[Dict[str, Any]]
) -> int:
    """
    Load structured data into a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection to load data into
        data: List of data items, each with 'id', 'text', and 'meta' keys
        
    Returns:
        Number of items loaded
    """
    
    batch_size = 5000  # Safe batch size under ChromaDB's limit
    total_loaded = 0
    total_start_time = time.time()
    
    # Process in batches to respect ChromaDB's batch size limits
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        ids = [item["id"] for item in batch]
        documents = [item["text"] for item in batch]
        metadatas = [item["meta"] for item in batch]
        
        batch_start_time = time.time()
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        batch_time = time.time() - batch_start_time
        
        total_loaded += len(batch)
        print(f"Loaded batch {i//batch_size + 1}: {total_loaded}/{len(data)} items (batch time: {batch_time:.2f}s)")
    
    total_time = time.time() - total_start_time
    print(f"Total loading time: {total_time:.2f}s")
    
    return collection.count()

def processDocuments(doc_dir: Union[str, Path] = PDF_DIR_DEFAULT) -> List[Dict[str, Any]]:
    """Extract all PDFs into page-level Chroma-ready records.

    Returns a list of dicts each with keys: id, text, meta
    where id is stable per (document,page).
    """
    doc_dir = Path(doc_dir)
    if not doc_dir.exists() or not doc_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {doc_dir}")

    records: List[Dict[str, Any]] = []
    for pdf_path in sorted(doc_dir.glob("*.pdf")):
        try:
            extracted = extractPdfText(pdf_path)
        except Exception as exc:
            print(f"[WARN] Extraction failed for {pdf_path.name}: {exc}")
            continue

        # TODO: better chunking strategy (e.g. overlapping windows)
        doc_meta = extracted["meta"]
        pages = extracted["pages"]
        total = len(pages)
        for page in pages:
            page_number = page["index"] + 1
            text = page["text"]
            rec_meta = {
                **doc_meta,
                "page_number": page_number,
                "page_char_count": page["char_count"],
                "chunk_index": page["index"],
                "total_chunks": total,
            }
            records.append({
                "id": f"{doc_meta['source_name']}#p{page_number}",
                "text": text,
                "meta": rec_meta,
            })
        print(f"Processed {pdf_path.name}: {doc_meta['page_count']} pages, size={doc_meta['file_size_bytes']} bytes")

    return records

def main() -> List[Dict[str, Any]]:
    """Index all PDFs into Chroma at a page granularity and return records."""
    dbDir = Path("./db")
    client = chromadb.PersistentClient(path=str(dbDir), settings=Settings(anonymized_telemetry=False))

    collection = client.get_or_create_collection(
        name="ai_research_docs",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        metadata={"hnsw:space": "cosine"}
    )

    records = processDocuments()
    if not records:
        print("No PDF documents found to index.")
        return []

    existing = collection.count()
    if existing < len(records):
        print(f"\nIndexing {len(records)} page chunks into collection 'ai_research_docs' (currently {existing}).")
        loadDataToChroma(collection, records)
    else:
        print(f"Collection already has {existing} items; skipping re-index.")

    return records


if __name__ == "__main__":
    main()
