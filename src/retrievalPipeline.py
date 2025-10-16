from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import indexPipeline

def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print Chroma query results (single-query case)."""
    ids_list = results.get("ids", [])
    metas_list = results.get("metadatas", [])
    docs_list = results.get("documents", [])
    dists_list = results.get("distances", []) or results.get("embeddings", [])

    if not ids_list:
        print("No results returned.")
        return

    # Assuming one query -> lists of hits
    ids = ids_list[0]
    metas = metas_list[0] if metas_list else []
    docs = docs_list[0] if docs_list else []
    dists = dists_list[0] if dists_list else []

    print("\nTop Results:")
    for rank, (cid, meta, doc, dist) in enumerate(zip(ids, metas, docs, dists), start=1):
        snippet = (doc[:indexPipeline.CHUNK_MAX_CHARS] + "…") if len(doc) > indexPipeline.CHUNK_MAX_CHARS else doc
        page = meta.get("page_number") if isinstance(meta, dict) else None
        src = meta.get("source_name") if isinstance(meta, dict) else None
        chars = meta.get("approx_chars") if isinstance(meta, dict) else None
        print(f"#{rank} id={cid}")
        if src:
            print(f"   Source: {src}  Page: {page}  Size≈{chars}")
        print(f"   Distance: {dist:.4f}" if isinstance(dist, (int, float)) else f"   Distance: {dist}")
        print(f"   Snippet: {snippet}\n")


def main(argv: List[str] | None = None) -> List[Dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Interactive retrieval over indexed AI research PDFs.")
    parser.add_argument(
        "-b",
        action="store_true",
        help="Rebuild the Chroma index before starting the query prompt.",
    )
    args = parser.parse_args(argv)

    db_dir = Path("./db")
    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name="ai_research_docs",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        metadata={"hnsw:space": "cosine"}
    )

    if args.b or collection.count() == 0:
        print("Building index as requested.")
        indexPipeline.process_documents(collection)

    try:
        user_question = input("Enter your question (blank to exit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        return []

    if not user_question:
        print("No question entered.")
        return []

    results = collection.query(
        query_texts=[user_question],
        n_results=5,
        include=["metadatas", "documents", "distances"],
    )

    # TODO: Assemble a context prompt (with citations if possible).
    # TODO: Generate a final answer using an LLM (local or API).
    print_results(results)

    return []


if __name__ == "__main__":
    main()