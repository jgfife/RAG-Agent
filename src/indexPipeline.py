from pathlib import Path
from typing import Dict
from pathlib import Path
from typing import Dict, Union, List, Any
import time
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings

PDF_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "documents"

def extractPdfText(pdf_path: Union[str, Path]) -> str:
    pdf_path = Path(pdf_path)
    try:
        try:
            from pypdf import PdfReader  # preferred modern package
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages: #TODO: generate metatdata with page numbers
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)
    except ImportError:
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
            return pdfminer_extract_text(str(pdf_path))
        except ImportError as e:
            raise RuntimeError(
                "No PDF extraction library available. Install 'pypdf' or 'pdfminer.six'."
            ) from e

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
        
        # TODO: make ids, documents, and metadatas not be the movie dataset
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

def processDocuments(doc_dir: Union[str, Path] = PDF_DIR_DEFAULT) -> Dict[str, str]:
    """
    Reads every .pdf in the documents folder, extracts text to memory,
    prints each file's text to stdout bracketed by BEGIN/END markers,
    and returns a dict {filename: text}.
    """
    doc_dir = Path(doc_dir)
    if not doc_dir.exists() or not doc_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {doc_dir}")

    results: Dict[str, str] = {}
    for pdf_path in sorted(doc_dir.glob("*.pdf")):
        try:
            text = extractPdfText(pdf_path)
        except Exception as exc:
            text = f"[ERROR extracting {pdf_path.name}: {exc}]"
        results[pdf_path.name] = text

        print(f"--- BEGIN {pdf_path.name} ---")
        print(text)
        print(f"--- END {pdf_path.name} ---\n")

    return results

def main() -> Dict[str, str]:
    """Entry point for the RAG agent.

    Currently: reads all PDFs in the default documents directory
    via `log_all_pdf_texts` and returns a mapping of filename to text.
    Extend this function later to add embedding, vector store loading,
    and retrieval logic.
    """
    dbDir = Path("./db")
    client = chromadb.PersistentClient(path=str(dbDir), settings=Settings(anonymized_telemetry=False))

    # Create or get a collection with an embedding function
    # Note: you can add HNSW params via metadata if desired (implementation-dependent)
    collection = client.get_or_create_collection(
        name="ai_research_docs",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        metadata={"hnsw:space": "cosine"}
    )

    docs = processDocuments()
    if collection.count() != len(docs):
        print(f"\nDataset Loading:")
        print(f"Loading {len(docs)} items into ChromaDB collection 'ai_research_docs'...")
        count = loadDataToChroma(collection, docs)
        print(f"{count} items are contained in the ChromaDB collection 'ai_research_docs'")
    else:
        print(f"\nDataset already loaded")


if __name__ == "__main__":
    main()
