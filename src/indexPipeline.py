from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, List, Any
import time
import io
import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings

# Chunking configuration (tunable)
CHUNK_MAX_CHARS = 3000          # Upper bound per chunk
CHUNK_MIN_CHARS = 1200          # Try to reach at least this unless end of page
CHUNK_OVERLAP_CHARS = 200       # Overlap tail of previous chunk
USE_SENTENCE_SPLIT = True        # Whether to split by sentences first (if False, hard splits)
PDF_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "documents"

def extract_pdf_text(pdf_path: Union[str, Path]) -> Dict[str, Any]:
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

def load_data_to_chroma(
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

def split_into_sentences(text: str) -> List[str]:
    if not USE_SENTENCE_SPLIT:
        return [text]
    
    regex = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
    sentences = []
    for part in regex.split(text):
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences or [text]


def build_chunks_from_sentences(sentences: List[str]) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        slen = len(sent)
        # If sentence itself is too large, hard split it
        if slen > CHUNK_MAX_CHARS:
            if current:
                chunks.append(" ".join(current).strip())
                current, current_len = [], 0
            for start in range(0, slen, CHUNK_MAX_CHARS):
                piece = sent[start:start + CHUNK_MAX_CHARS]
                chunks.append(piece.strip())
            continue

        if current_len + slen + 1 <= CHUNK_MAX_CHARS:
            current.append(sent)
            current_len += slen + 1
        else:
            # finalize current
            if current:
                chunk_text = " ".join(current).strip()
                chunks.append(chunk_text)
                # prepare overlap
                if CHUNK_OVERLAP_CHARS > 0:
                    overlap = chunk_text[-CHUNK_OVERLAP_CHARS:]
                    current = [overlap]
                    current_len = len(overlap)
                else:
                    current, current_len = [], 0
            # add new sentence (may start new chunk)
            current.append(sent)
            current_len = sum(len(s) + 1 for s in current)

    if current:
        chunks.append(" ".join(current).strip())

    # Merge tiny tail chunk if below minimum and more than one chunk
    if len(chunks) > 1 and len(chunks[-1]) < CHUNK_MIN_CHARS:
        prev = chunks[-2]
        merged = prev + " " + chunks[-1]
        if len(merged) <= CHUNK_MAX_CHARS:
            chunks[-2] = merged.strip()
            chunks.pop()
    return chunks


def chunk_page_text(page_text: str) -> List[str]:
    sentences = split_into_sentences(page_text)
    return build_chunks_from_sentences(sentences)


def process_documents(doc_dir: Union[str, Path] = PDF_DIR_DEFAULT) -> List[Dict[str, Any]]:
    """Extract all PDFs into chunk-level Chroma-ready records.

    Page texts are further split into overlapping chunks for better retrieval granularity.
    """
    doc_dir = Path(doc_dir)
    if not doc_dir.exists() or not doc_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {doc_dir}")

    records: List[Dict[str, Any]] = []
    for pdf_path in sorted(doc_dir.glob("*.pdf")):
        try:
            extracted = extract_pdf_text(pdf_path)
        except Exception as exc:
            print(f"[WARN] Extraction failed for {pdf_path.name}: {exc}")
            continue

        doc_meta = extracted["meta"]
        pages = extracted["pages"]
        total_pages = len(pages)
        chunk_counter = 0

        for page in pages:
            page_number = page["index"] + 1
            text = page["text"].strip()
            if not text:
                continue
            page_chunks = chunk_page_text(text)
            for local_idx, chunk_text in enumerate(page_chunks):
                chunk_counter += 1
                rec_meta = {
                    **doc_meta,
                    "page_number": page_number,
                    "page_char_count": page["char_count"],
                    "chunk_index": chunk_counter - 1,
                    "page_chunk_index": local_idx,
                    "page_total_chunks": len(page_chunks),
                    "total_pages": total_pages,
                    "approx_chars": len(chunk_text),
                    "overlap_chars": CHUNK_OVERLAP_CHARS,
                }
                records.append({
                    "id": f"{doc_meta['source_name']}#p{page_number}#c{local_idx+1}",
                    "text": chunk_text,
                    "meta": rec_meta,
                })
        print(f"Processed {pdf_path.name}: {doc_meta['page_count']} pages -> {chunk_counter} chunks")

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

    records = process_documents()
    if not records:
        print("No PDF documents found to index.")
        return []

    existing = collection.count()
    if existing < len(records):
        print(f"\nIndexing {len(records)} page chunks into collection 'ai_research_docs' (currently {existing}).")
        load_data_to_chroma(collection, records)
    else:
        print(f"Collection already has {existing} items; skipping re-index.")

    return records


if __name__ == "__main__":
    main()
