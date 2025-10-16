from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests

import indexPipeline


def call_ollama(prompt: str, model: str = "llama3.1", timeout: int = 120) -> str:
    """Invoke the local Ollama API and return the generated text.

    Args:
        prompt: The fully formatted prompt to send to Ollama.
        model: Ollama model name to use, e.g. "llama3.1".
        timeout: Request timeout (seconds) for the HTTP call.

    Returns:
        The text portion of the Ollama response.

    Raises:
        RuntimeError: If the HTTP request fails or Ollama returns an error status.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    data = response.json()
    return data.get("response", "").strip()


def build_ollama_prompt(question: str, results: Dict[str, Any], max_passages: int = 5) -> str:
    """Build a context-rich prompt for an Ollama completion call.

    Args:
        question: User's natural-language query.
        results: Raw Chroma query response.
        max_passages: Maximum number of passages to embed in the prompt.

    Returns:
        A formatted prompt string ready to send to an Ollama completion endpoint.
    """
    ids_list = results.get("ids", [])
    metas_list = results.get("metadatas", [])
    docs_list = results.get("documents", [])
    dists_list = results.get("distances", [])

    if not ids_list:
        return (
            "You are an expert assistant. No supporting documents were retrieved. "
            "Explain that the knowledge base did not yield results and encourage follow-up.\n"
            f"Question: {question}\n"
            "Answer:"
        )

    ids = ids_list[0]
    metas = metas_list[0] if metas_list else []
    docs = docs_list[0] if docs_list else []
    dists = dists_list[0] if dists_list else []

    lines: List[str] = [
        "You are an expert AI assistant answering questions using the provided document chunks.",
        "Use only the context to answer. If the context is insufficient, state that explicitly.",
        "Cite sources in square brackets using the format [source_name pX].",
        "",
        f"Question: {question}",
        "",
        "Context Passages:",
    ]

    for rank, (cid, meta, doc, dist) in enumerate(zip(ids, metas, docs, dists), start=1):
        if rank > max_passages:
            break
        source = "unknown"
        page = "?"
        if isinstance(meta, dict):
            source = meta.get("source_name", source)
            page_num = meta.get("page_number")
            page = str(page_num) if page_num is not None else page
        snippet = " ".join((doc or "").split())
        snippet = snippet[:600]
        dist_display = f"{dist:.3f}" if isinstance(dist, (int, float)) else str(dist)
        lines.append(f"[{rank}] Source: {source} p{page} (distance {dist_display})")
        lines.append(snippet)
        lines.append("")

    lines.extend([
        "Answer Instructions:",
        "- Produce a concise answer grounded in the context above.",
        "- List key supporting points when helpful, each with citations.",
        "- If the answer is unknown, respond with an explicit statement of insufficiency.",
        "",
        "Final Answer:",
    ])

    return "\n".join(lines)

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


def main(argv: List[str] | None = None) -> str:
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
        return ""

    if not user_question:
        print("No question entered.")
        return ""

    results = collection.query(
        query_texts=[user_question],
        n_results=5,
        include=["metadatas", "documents", "distances"],
    )

    prompt = build_ollama_prompt(user_question, results)
    print_results(results)
    print("\n--- Prompt sent to Ollama ---\n")
    print(prompt)

    try:
        ollama_answer = call_ollama(prompt)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return ""

    print("\n--- Ollama Answer ---\n")
    print(ollama_answer)

    return ollama_answer


if __name__ == "__main__":
    main()